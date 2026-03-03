from pathlib import Path
import numpy as np
import wgpu

import PySide6
from rendercanvas.qt import RenderCanvas, loop

from config import *


def load_shader(
    device: wgpu.GPUDevice,
    filename: str = 'shader.wgsl',
    replacements: list[tuple[str, str]] = []
) -> wgpu.GPUShaderModule:

    path = Path(__file__).parent / filename
    if not path.exists():
        raise FileNotFoundError(f'shader source file {path} is missing')

    code = path.read_text()
    for replacement in replacements:
        code = code.replace(replacement[0], replacement[1])

    return device.create_shader_module(code=code)


class DynamicUniformBuffer:
    device: wgpu.GPUDevice
    data_view: memoryview[int]
    data_size: int
    buf: wgpu.GPUBuffer
    staging_buf: wgpu.GPUBuffer

    def __init__(
        self,
        device: wgpu.GPUDevice,
        label: str,
        data: memoryview,
        upload_at_creation: bool = True
    ):
        self.device = device
        self.data_view = data.cast("B")
        self.data_size = (self.data_view.nbytes + 3) & ~3  # 4-byte alignment

        self.buf = device.create_buffer(
            label=label,
            size=self.data_size,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            mapped_at_creation=False
        )

        self.staging_buf = device.create_buffer(
            label=label + " (staging buffer)",
            size=self.data_size,
            usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
            mapped_at_creation=False
        )

        if upload_at_creation:
            self.upload()

    def upload(self):
        self.staging_buf.map_sync(wgpu.MapMode.WRITE)
        self.staging_buf.write_mapped(self.data_view)
        self.staging_buf.unmap()

        cmd_encoder = self.device.create_command_encoder()
        cmd_encoder.copy_buffer_to_buffer(
            self.staging_buf,
            0,
            self.buf,
            0,
            self.data_size
        )
        self.device.queue.submit([cmd_encoder.finish()])
        self.device.queue.on_submitted_work_done_sync()


# returns the start (inclusive) and end (exclusive) offset (in bytes) of given
# fields (unioned) in a numpy data structure.
# NOTE: the fields must be in the same order as in the original numpy.dtype.
def field_offset_in_numpy_dtype(
    dtype: np.dtype,
    fields: list[str],
    alignment: int = 1
) -> tuple[int, int]:
    start: int = -1
    end: int = -1
    head: int = 0
    for field in dtype.fields:
        if field == fields[0] and start == -1:
            start = head
        if field == fields[-1]:
            end = head + dtype[field].itemsize
            # don't break here! we want to store the total size in "head".
        head += dtype[field].itemsize

    if start >= end:
        raise IndexError(
            "make sure the fields exist and are in the same order as in the "
            "original numpy.dtype"
        )

    if alignment > 1:
        start = start // alignment * alignment
        end = (end + alignment - 1) // alignment * alignment

    end = min(end, head)

    return (start, end)


def main():
    adapter = wgpu.gpu.request_adapter_sync(
        power_preference=wgpu.PowerPreference.high_performance
    )
    device = adapter.request_device_sync()

    canvas = RenderCanvas(
        size=(
            int(RENDER_RES[0] * DISPLAY_SCALE),
            int(RENDER_RES[1] * DISPLAY_SCALE)
        ),
        title=WINDOW_TITLE,
        update_mode="continuous",
        max_fps=60,
        vsync=True,
    )

    context = canvas.get_wgpu_context()
    surface_format = context.get_preferred_format(adapter)
    context.configure(device=device, format=surface_format)

    # shaders
    sim_shader = load_shader(
        device,
        'sim.wgsl',
        [(
            "// [user-functions]",
            selected_sim_params.initial_value_function +
            selected_sim_params.update_value_function
        )]
    )
    render_shader = load_shader(device, 'render.wgsl')
    display_shader = load_shader(device, 'display.wgsl')

    # create double buffered 3D textures for the simulation

    def create_wave_texture(label):
        return device.create_texture(
            label=label,
            size=selected_sim_params.grid_res,
            dimension=wgpu.TextureDimension.d3,
            format=wgpu.TextureFormat.rg32float,
            usage=(
                wgpu.TextureUsage.TEXTURE_BINDING |
                wgpu.TextureUsage.STORAGE_BINDING |
                wgpu.TextureUsage.COPY_DST
            ),
        )

    wave_grid_a = create_wave_texture("wave_grid_a")
    wave_grid_b = create_wave_texture("wave_grid_b")

    wave_grid_a_view = wave_grid_a.create_view(label="wave_grid_a_view")
    wave_grid_b_view = wave_grid_b.create_view(label="wave_grid_b_view")

    # render target
    render_target = device.create_texture(
        label="render target",
        size=RENDER_RES,
        format=wgpu.TextureFormat.rgba8unorm,
        usage=(
            wgpu.TextureUsage.RENDER_ATTACHMENT |
            wgpu.TextureUsage.TEXTURE_BINDING |
            wgpu.TextureUsage.COPY_SRC
        )
    )
    render_target_view = render_target.create_view(label="render target view")

    # texture sampler
    linear_sampler = device.create_sampler(
        label="linear_sampler",
        address_mode_u=wgpu.AddressMode.clamp_to_edge,
        address_mode_v=wgpu.AddressMode.clamp_to_edge,
        address_mode_w=wgpu.AddressMode.clamp_to_edge,
        mag_filter=wgpu.FilterMode.linear,
        min_filter=wgpu.FilterMode.linear,
        mipmap_filter=wgpu.FilterMode.linear
    )

    # uniform buffer for compute pipeline

    sim_uniform_dtype = np.dtype([
        ("grid_res", np.int32, (3,)),
        ("cell_size", np.float32),
        ("grid_dims", np.float32, (3,)),
        ("wave_speed", np.float32),
        ("remove_reflections", np.int32),
        ("damp_fac", np.float32),
        ("timestep", np.float32),
        ("iter", np.int32),
        ("time", np.float32),
        ("max_timestep", np.float32),
        ("min_wavelength", np.float32),
        ("max_freq", np.float32),
        ("impedance_matching_coefficient", np.float32),
        ("_pad", np.int32, (3,)),
    ])

    sim_uniform = np.zeros((), dtype=sim_uniform_dtype)
    sim_uniform["grid_res"] = selected_sim_params.grid_res
    sim_uniform["cell_size"] = selected_sim_params.cell_size
    sim_uniform["grid_dims"] = selected_sim_limits.grid_dims
    sim_uniform["wave_speed"] = selected_sim_params.wave_speed
    sim_uniform["remove_reflections"] = int(
        selected_sim_params.remove_reflections
    )
    sim_uniform["damp_fac"] = selected_sim_params.damp_fac
    sim_uniform["timestep"] = selected_sim_limits.resolved_timestep
    sim_uniform["iter"] = 0
    sim_uniform["time"] = 0.
    sim_uniform["max_timestep"] = selected_sim_limits.max_timestep
    sim_uniform["min_wavelength"] = selected_sim_limits.min_wavelength
    sim_uniform["max_freq"] = selected_sim_limits.max_freq
    sim_uniform["impedance_matching_coefficient"] = \
        selected_sim_limits.impedance_matching_coefficient

    sim_uniform_buffer = DynamicUniformBuffer(
        device=device,
        label="sim_uniform_buffer",
        data=memoryview(sim_uniform),
        upload_at_creation=True
    )

    # uniform buffer for render pipeline

    render_uniform_dtype = np.dtype([
        ("res", np.int32, (2,)),
        ("bg_col", np.float32, (3,)),
        ("tint_positive", np.float32, (3,)),
        ("tint_negative", np.float32, (3,)),
        ("brightness", np.float32),
        ("n_samples_per_pixel", np.int32),
        ("raymarch_step", np.float32),
        ("raymarch_step_jitter", np.float32),
        ("cam_pos", np.float32, (3,)),
        ("cam_lookat", np.float32, (3,)),
        ("cam_world_up", np.float32, (3,)),
        ("cam_fov_degrees", np.float32),
        ("apply_flim", np.int32),
        ("sim_grid_res", np.int32, (3,)),
        ("sim_grid_dims", np.float32, (3,)),
    ])

    render_uniform = np.zeros((), dtype=render_uniform_dtype)
    render_uniform["res"] = RENDER_RES
    render_uniform["bg_col"] = selected_sim_params.render_bg_col
    render_uniform["tint_positive"] = selected_sim_params.render_tint_positive
    render_uniform["tint_negative"] = selected_sim_params.render_tint_negative
    render_uniform["brightness"] = selected_sim_params.render_brightness
    render_uniform["n_samples_per_pixel"] = selected_sim_params.render_n_samples_per_pixel
    render_uniform["raymarch_step"] = selected_sim_params.render_raymarch_step
    render_uniform["raymarch_step_jitter"] = selected_sim_params.render_raymarch_step_jitter
    render_uniform["apply_flim"] = int(selected_sim_params.render_apply_flim)
    render_uniform["sim_grid_res"] = selected_sim_params.grid_res
    render_uniform["sim_grid_dims"] = selected_sim_limits.grid_dims

    render_uniform_buffer = DynamicUniformBuffer(
        device=device,
        label="render_uniform_buffer",
        data=memoryview(render_uniform),
        upload_at_creation=True
    )

    # uniform buffer for display pipeline

    display_uniform_dtype = np.dtype([
        ("srgb_surface", np.int32, (4,)),
    ])

    display_uniform = np.zeros((), dtype=display_uniform_dtype)
    display_uniform["srgb_surface"][0] = int('srgb' in surface_format.lower())

    display_uniform_buffer = device.create_buffer_with_data(
        data=display_uniform,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    # bind group layouts

    compute_bgl = device.create_bind_group_layout(
        entries=[
            wgpu.BindGroupLayoutEntry(
                binding=0,
                visibility=wgpu.ShaderStage.COMPUTE,
                buffer=wgpu.BufferBindingLayout()
            ),
            wgpu.BindGroupLayoutEntry(
                binding=1,
                visibility=wgpu.ShaderStage.COMPUTE,
                storage_texture=wgpu.StorageTextureBindingLayout(
                    access=wgpu.StorageTextureAccess.read_only,
                    format=wgpu.TextureFormat.rg32float,
                    view_dimension=wgpu.TextureViewDimension.d3
                )
            ),
            wgpu.BindGroupLayoutEntry(
                binding=2,
                visibility=wgpu.ShaderStage.COMPUTE,
                storage_texture=wgpu.StorageTextureBindingLayout(
                    access=wgpu.StorageTextureAccess.write_only,
                    format=wgpu.TextureFormat.rg32float,
                    view_dimension=wgpu.TextureViewDimension.d3
                )
            ),
        ]
    )

    render_bgl = device.create_bind_group_layout(
        entries=[
            wgpu.BindGroupLayoutEntry(
                binding=0,
                visibility=wgpu.ShaderStage.FRAGMENT,
                buffer=wgpu.BufferBindingLayout()
            ),
            wgpu.BindGroupLayoutEntry(
                binding=1,
                visibility=wgpu.ShaderStage.FRAGMENT,
                storage_texture=wgpu.StorageTextureBindingLayout(
                    access=wgpu.StorageTextureAccess.read_only,
                    format=wgpu.TextureFormat.rg32float,
                    view_dimension=wgpu.TextureViewDimension.d3
                )
            ),
        ]
    )

    display_bgl = device.create_bind_group_layout(
        entries=[
            wgpu.BindGroupLayoutEntry(
                binding=0,
                visibility=wgpu.ShaderStage.FRAGMENT,
                buffer=wgpu.BufferBindingLayout()
            ),
            wgpu.BindGroupLayoutEntry(
                binding=1,
                visibility=wgpu.ShaderStage.FRAGMENT,
                texture=wgpu.TextureBindingLayout()
            ),
            wgpu.BindGroupLayoutEntry(
                binding=2,
                visibility=wgpu.ShaderStage.FRAGMENT,
                sampler=wgpu.SamplerBindingLayout()
            ),
        ]
    )

    # pipelines

    compute_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(
            bind_group_layouts=[compute_bgl]
        ),
        compute=wgpu.ProgrammableStage(
            module=sim_shader,
            entry_point="cs_main"
        ),
    )

    render_pipeline = device.create_render_pipeline(
        layout=device.create_pipeline_layout(
            bind_group_layouts=[render_bgl]
        ),
        vertex=wgpu.VertexState(
            module=render_shader,
            entry_point="vs_main"
        ),
        fragment=wgpu.FragmentState(
            module=render_shader,
            entry_point="fs_main",
            targets=[wgpu.ColorTargetState(format=render_target.format)],
        ),
        primitive=wgpu.PrimitiveState(
            topology=wgpu.PrimitiveTopology.triangle_list
        ),
    )

    display_pipeline = device.create_render_pipeline(
        layout=device.create_pipeline_layout(
            bind_group_layouts=[display_bgl]
        ),
        vertex=wgpu.VertexState(
            module=display_shader,
            entry_point="vs_main"
        ),
        fragment=wgpu.FragmentState(
            module=display_shader,
            entry_point="fs_main",
            targets=[wgpu.ColorTargetState(format=surface_format)],
        ),
        primitive=wgpu.PrimitiveState(
            topology=wgpu.PrimitiveTopology.triangle_list
        ),
    )

    # simulation state
    sim_state = WaveSimState(selected_sim_limits.resolved_timestep)

    # per-frame logic
    def draw():
        nonlocal canvas, sim_state

        # advance the simulation
        for _ in range(selected_sim_params.n_sim_steps_per_frame):
            # update iter and time in the uniform buffer
            sim_uniform["iter"] = sim_state.iter
            sim_uniform["time"] = sim_state.time
            sim_uniform_buffer.upload()

            # run the compute shader

            cmd_encoder = device.create_command_encoder()

            input_grid_view = \
                wave_grid_a_view if sim_state.use_a_as_input else wave_grid_b_view
            output_grid_view = \
                wave_grid_b_view if sim_state.use_a_as_input else wave_grid_a_view

            cpass = cmd_encoder.begin_compute_pass()
            cpass.set_pipeline(compute_pipeline)
            cpass.set_bind_group(0, device.create_bind_group(
                layout=compute_bgl,
                entries=[
                    wgpu.BindGroupEntry(
                        binding=0,
                        resource=wgpu.BufferBinding(
                            buffer=sim_uniform_buffer.buf
                        )
                    ),
                    wgpu.BindGroupEntry(
                        binding=1,
                        resource=input_grid_view
                    ),
                    wgpu.BindGroupEntry(
                        binding=2,
                        resource=output_grid_view
                    ),
                ],
            ))

            dispatch = (
                (selected_sim_params.grid_res[0] + 7) // 8,
                (selected_sim_params.grid_res[1] + 7) // 8,
                (selected_sim_params.grid_res[2] + 3) // 4,
            )
            cpass.dispatch_workgroups(*dispatch)
            cpass.end()

            device.queue.submit([cmd_encoder.finish()])

            # update simulation state
            sim_state.advance()

        # update camera state in the render uniform buffer

        cam_state = selected_sim_params.render_camera_function(
            selected_sim_params,
            selected_sim_limits,
            sim_state
        )

        render_uniform["cam_pos"] = cam_state.pos
        render_uniform["cam_lookat"] = cam_state.lookat
        render_uniform["cam_world_up"] = cam_state.world_up
        render_uniform["cam_fov_degrees"] = cam_state.fov_degrees

        render_uniform_buffer.upload()

        # render a frame

        cmd_encoder = device.create_command_encoder()

        # render pass

        render_grid_view = \
            wave_grid_a_view if sim_state.use_a_as_input else wave_grid_b_view

        rpass = cmd_encoder.begin_render_pass(
            color_attachments=[
                wgpu.RenderPassColorAttachment(
                    view=render_target_view,
                    load_op=wgpu.LoadOp.clear,
                    store_op=wgpu.StoreOp.store,
                    clear_value=(0., 0., 0., 1.),
                )
            ]
        )

        rpass.set_pipeline(render_pipeline)
        rpass.set_bind_group(0, device.create_bind_group(
            layout=render_bgl,
            entries=[
                wgpu.BindGroupEntry(
                    binding=0,
                    resource=wgpu.BufferBinding(
                        buffer=render_uniform_buffer.buf
                    )
                ),
                wgpu.BindGroupEntry(
                    binding=1,
                    resource=render_grid_view
                ),
            ],
        ))
        rpass.draw(6, 1, 0, 0)
        rpass.end()

        # display pass

        current_swapchain_view = (
            context.get_current_texture()
            .create_view()
        )

        dpass = cmd_encoder.begin_render_pass(
            color_attachments=[
                wgpu.RenderPassColorAttachment(
                    view=current_swapchain_view,
                    load_op=wgpu.LoadOp.clear,
                    store_op=wgpu.StoreOp.store,
                    clear_value=(0., 0., 0., 1.),
                )
            ]
        )

        dpass.set_pipeline(display_pipeline)
        dpass.set_bind_group(0, device.create_bind_group(
            layout=display_bgl,
            entries=[
                wgpu.BindGroupEntry(
                    binding=0,
                    resource=wgpu.BufferBinding(buffer=display_uniform_buffer)
                ),
                wgpu.BindGroupEntry(
                    binding=1,
                    resource=render_target_view
                ),
                wgpu.BindGroupEntry(
                    binding=2,
                    resource=linear_sampler
                ),
            ],
        ))
        dpass.draw(6, 1, 0, 0)
        dpass.end()

        device.queue.submit([cmd_encoder.finish()])

        # update window title
        canvas.set_title(WINDOW_TITLE.format(
            sim_state.time,
            sim_state.iter
        ))

    canvas.request_draw(draw)
    loop.run()


if __name__ == "__main__":
    main()
