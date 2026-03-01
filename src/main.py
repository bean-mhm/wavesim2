from pathlib import Path
import numpy as np
import wgpu

import PySide6
from rendercanvas.qt import RenderCanvas, loop

GRID_RES = (64, 64, 64)
RENDER_RES = (2160, 1080)
DISPLAY_SCALE = .6


def load_shader(
        device: wgpu.GPUDevice,
        filename: str = 'shader.wgsl'
) -> wgpu.GPUShaderModule:

    path = Path(__file__).parent / filename
    if not path.exists():
        raise FileNotFoundError(f'shader source file {path} is missing')

    return device.create_shader_module(code=path.read_text())


def main():
    adapter = wgpu.gpu.request_adapter_sync(
        power_preference=wgpu.PowerPreference.high_performance
    )
    device = adapter.request_device_sync()

    canvas = RenderCanvas(
        size=(int(RENDER_RES[0] * DISPLAY_SCALE),
              int(RENDER_RES[1] * DISPLAY_SCALE)),
        title="wavesim2",
        update_mode="continuous",
        max_fps=60,
        vsync=True,
    )

    context = canvas.get_wgpu_context()
    surface_format = context.get_preferred_format(adapter)
    context.configure(device=device, format=surface_format)

    # shaders
    sim_shader = load_shader(device, 'sim.wgsl')
    render_shader = load_shader(device, 'render.wgsl')
    display_shader = load_shader(device, 'display.wgsl')

    # create double buffered 3D textures for the simulation

    def create_wave_texture(label):
        return device.create_texture(
            label=label,
            size=GRID_RES,
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

    sim_param_dtype = np.dtype([
        ("grid_size", np.uint32, (3,)),
        ("_pad", np.int32),
    ])

    sim_params = np.zeros((), dtype=sim_param_dtype)
    sim_params["grid_size"] = GRID_RES

    sim_params_buffer = device.create_buffer_with_data(
        data=sim_params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    # uniform buffer for render pipeline

    render_params_dtype = np.dtype([
        ("tint", np.float32, (3,)),
        ("res", np.int32, (2,)),
        ("n_samples", np.int32),
        ("raymarch_min_step", np.float32),
        ("raymarch_max_step", np.float32),
        ("cam_pos", np.float32, (3,)),
        ("cam_lookat", np.float32, (3,)),
        ("cam_fov", np.float32),
        ("wall_time", np.float32),
        ("sim_time", np.float32),
        ("_pad", np.float32, (3,)),
    ])

    render_params = np.zeros((), dtype=render_params_dtype)
    render_params["tint"] = (.2, .8, .1)
    render_params["res"] = RENDER_RES
    render_params["n_samples"] = 4
    render_params["raymarch_min_step"] = .01
    render_params["raymarch_max_step"] = .025
    render_params["cam_pos"] = (0., -1.25, .1)
    render_params["cam_lookat"] = (0., 0., 0.)
    render_params["cam_fov"] = 80.
    render_params["wall_time"] = 0.
    render_params["sim_time"] = 0.

    render_params_buffer = device.create_buffer_with_data(
        data=render_params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    # uniform buffer for display pipeline

    display_param_dtype = np.dtype([
        ("srgb_surface", np.uint32, (4,)),
    ])

    display_params = np.zeros((), dtype=display_param_dtype)
    display_params["srgb_surface"][0] = (
        1 if 'srgb' in surface_format.lower() else 0
    )

    display_params_buffer = device.create_buffer_with_data(
        data=display_params,
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
    use_a_as_input = True
    total_sim_iterations = 0

    # per-frame logic
    def draw():
        nonlocal use_a_as_input, total_sim_iterations

        encoder = device.create_command_encoder()

        # compute pass (simulation)
        n_sim_steps_this_frame = 1
        for _ in range(n_sim_steps_this_frame):
            input_grid_view = wave_grid_a_view if use_a_as_input else wave_grid_b_view
            output_grid_view = wave_grid_b_view if use_a_as_input else wave_grid_a_view

            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(compute_pipeline)
            cpass.set_bind_group(0, device.create_bind_group(
                layout=compute_bgl,
                entries=[
                    wgpu.BindGroupEntry(
                        binding=0,
                        resource=wgpu.BufferBinding(buffer=sim_params_buffer)
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
                (GRID_RES[0] + 7) // 8,
                (GRID_RES[1] + 7) // 8,
                (GRID_RES[2] + 3) // 4,
            )
            cpass.dispatch_workgroups(*dispatch)
            cpass.end()

            use_a_as_input = not use_a_as_input
            total_sim_iterations += 1

        # render pass

        render_grid_view = wave_grid_a_view if use_a_as_input else wave_grid_b_view

        rpass = encoder.begin_render_pass(
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
                    resource=wgpu.BufferBinding(buffer=render_params_buffer)
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

        dpass = encoder.begin_render_pass(
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
                    resource=wgpu.BufferBinding(buffer=display_params_buffer)
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

        device.queue.submit([encoder.finish()])

    canvas.request_draw(draw)
    loop.run()


if __name__ == "__main__":
    main()
