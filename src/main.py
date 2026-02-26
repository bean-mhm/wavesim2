from pathlib import Path
import numpy as np
import wgpu

import PySide6
from rendercanvas.qt import RenderCanvas, loop

GRID_RES = (64, 64, 64)
WINDOW_SIZE = (640, 480)
WINDOW_TITLE = "wavesim2"


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
        power_preference="high-performance"
    )
    device = adapter.request_device_sync()

    canvas = RenderCanvas(
        size=WINDOW_SIZE,
        title=WINDOW_TITLE,
        update_mode="continuous",
        max_fps=60,
        vsync=True,
    )

    context = canvas.get_wgpu_context()
    surface_format = context.get_preferred_format(adapter)
    context.configure(device=device, format=surface_format)

    sim_shader = load_shader(device, 'sim.wgsl')
    render_shader = load_shader(device, 'render.wgsl')

    # create double buffered 3D textures for the simulation

    def create_wave_texture(label):
        return device.create_texture(
            size=GRID_RES,
            dimension="3d",
            format="rg32float",
            usage=(
                wgpu.TextureUsage.TEXTURE_BINDING |
                wgpu.TextureUsage.STORAGE_BINDING |
                wgpu.TextureUsage.COPY_DST
            ),
            label=label,
        )

    wave_grid_a = create_wave_texture("wave_grid_a")
    wave_grid_b = create_wave_texture("wave_grid_b")

    wave_grid_a_view = wave_grid_a.create_view(dimension="3d")
    wave_grid_b_view = wave_grid_b.create_view(dimension="3d")

    # uniform buffer for compute pipeline

    sim_param_dtype = np.dtype([
        ("grid_size", np.uint32, (3,)),
        ("_pad", np.uint32),
    ])

    sim_params = np.zeros((), dtype=sim_param_dtype)
    sim_params["grid_size"] = GRID_RES

    sim_param_buffer = device.create_buffer_with_data(
        data=sim_params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    # uniform buffer for render pipeline

    render_param_dtype = np.dtype([
        ("tint", np.float32, (4,)),
    ])

    render_params = np.zeros((), dtype=render_param_dtype)
    render_params["tint"] = (.2, .8, .1, 1.)

    render_param_buffer = device.create_buffer_with_data(
        data=render_params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    # bind group layouts

    compute_bgl = device.create_bind_group_layout(
        entries=[
            dict(binding=0, visibility=wgpu.ShaderStage.COMPUTE, buffer={}),
            dict(
                binding=1,
                visibility=wgpu.ShaderStage.COMPUTE,
                storage_texture=dict(
                    access="read-only",
                    format="rg32float",
                    view_dimension="3d"
                )
            ),
            dict(
                binding=2,
                visibility=wgpu.ShaderStage.COMPUTE,
                storage_texture=dict(
                    access="write-only",
                    format="rg32float",
                    view_dimension="3d"
                )
            ),
        ]
    )

    render_bgl = device.create_bind_group_layout(
        entries=[
            dict(binding=0, visibility=wgpu.ShaderStage.FRAGMENT, buffer={}),
            dict(
                binding=1,
                visibility=wgpu.ShaderStage.FRAGMENT,
                storage_texture=dict(
                    access="read-only",
                    format="rg32float",
                    view_dimension="3d"
                )
            ),
        ]
    )

    compute_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(
            bind_group_layouts=[compute_bgl]
        ),
        compute=dict(module=sim_shader, entry_point="cs_main"),
    )

    render_pipeline = device.create_render_pipeline(
        layout=device.create_pipeline_layout(
            bind_group_layouts=[render_bgl]
        ),
        vertex=dict(module=render_shader, entry_point="vs_main"),
        fragment=dict(
            module=render_shader,
            entry_point="fs_main",
            targets=[dict(format=surface_format)],
        ),
        primitive=dict(topology="triangle-list"),
    )

    # simulation state
    use_a_as_input = True
    total_sim_iterations = 0

    # per-frame logic
    def draw():
        nonlocal use_a_as_input, total_sim_iterations

        encoder = device.create_command_encoder()

        n_sim_steps_this_frame = 1

        for _ in range(n_sim_steps_this_frame):
            input_grid_view = wave_grid_a_view if use_a_as_input else wave_grid_b_view
            output_grid_view = wave_grid_b_view if use_a_as_input else wave_grid_a_view

            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(compute_pipeline)
            cpass.set_bind_group(0, device.create_bind_group(
                layout=compute_bgl,
                entries=[
                    dict(binding=0, resource=dict(buffer=sim_param_buffer)),
                    dict(binding=1, resource=input_grid_view),
                    dict(binding=2, resource=output_grid_view),
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

        current_view = (
            context.get_current_texture()
            .create_view()
        )

        render_grid_view = wave_grid_a_view if use_a_as_input else wave_grid_b_view

        rpass = encoder.begin_render_pass(
            color_attachments=[
                dict(
                    view=current_view,
                    load_op="clear",
                    store_op="store",
                    clear_value=(0., 0., 0., 1.),
                )
            ]
        )

        rpass.set_pipeline(render_pipeline)
        rpass.set_bind_group(0, device.create_bind_group(
            layout=render_bgl,
            entries=[
                dict(binding=0, resource=dict(buffer=render_param_buffer)),
                dict(binding=1, resource=render_grid_view),
            ],
        ))
        rpass.draw(6, 1, 0, 0)
        rpass.end()

        device.queue.submit([encoder.finish()])

    canvas.request_draw(draw)
    loop.run()


if __name__ == "__main__":
    main()
