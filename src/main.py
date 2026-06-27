from copy import deepcopy
from functools import reduce
import operator
import numpy as np
from PIL import Image

import wgpu
from rendercanvas.pyside6 import RenderCanvas, loop
from rendercanvas.contexts import WgpuContext

from common import *
from config import *


# global variables (sue me)
device: wgpu.GPUDevice | None = None
cpu_visible_buf: wgpu.GPUBuffer | None = None
readback_dest_buf: wgpu.GPUBuffer | None = None
render_target: wgpu.GPUTexture | None = None
render_target_view: wgpu.GPUTextureView | None = None

# WGSL constants
wgsl_sim_constants = f"""
const GRID_RES = vec3i{str(selected_sim_params.grid_res)};
const CELL_SIZE = {selected_sim_params.cell_size};
const GRID_DIMS = vec3f{str(selected_sim_limits.grid_dims)};
const WAVE_SPEED = {selected_sim_params.wave_speed};
const REMOVE_REFLECTIONS = {str(selected_sim_params.remove_reflections).lower()};
const TIMESTEP = {selected_sim_limits.resolved_timestep};
const MAX_TIMESTEP = {selected_sim_limits.max_timestep};
const MIN_WAVELENGTH = {selected_sim_limits.min_wavelength};
const MAX_FREQ = {selected_sim_limits.max_freq};
const IMPEDANCE_MATCHING_COEFFICIENT = {selected_sim_limits.impedance_matching_coefficient};
    """
wgsl_render_constants = f"""
const RENDER_MODE_RAYMARCHING = {int(RenderMode.Raymarching)};
const RENDER_MODE_SLICE = {int(RenderMode.Slice)};
    """


def prepare_render_target(min_res: tuple[int, int]):
    global device, render_target, render_target_view

    if render_target is not None \
            and render_target.size[0] >= min_res[0] \
            and render_target.size[1] >= min_res[1]:
        return

    old_size = (0, 0)
    if render_target is not None:
        old_size = deepcopy(render_target.size)

    # the width must be divisible by 64 because bytes_per_row must be divisible
    # by 256 when copying the pixels to a buffer for readback (e.g. when
    # exporting a frame).
    new_size = (
        make_divisible_by(max(old_size[0], min_res[0]), 64),
        max(old_size[1], min_res[1])
    )

    render_target = device.create_texture(
        label="render target",
        size=(*new_size, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=(
            wgpu.TextureUsage.RENDER_ATTACHMENT |
            wgpu.TextureUsage.TEXTURE_BINDING |
            wgpu.TextureUsage.COPY_SRC
        )
    )
    render_target_view = render_target.create_view(label="render target view")


render_pipeline_map = dict[
    # mapping from
    tuple[
        bool,  # uses averaging buffer
        str,  # WGSL user functions (shade_cell)
        str  # WGSL UserData struct fields (for sending custom data from CPU)
    ],

    # to
    tuple[
        wgpu.GPUBindGroupLayout,  # bind group layout
        wgpu.GPURenderPipeline  # pipeline
    ]
]()
MAX_RENDER_PIPELINE_MAP_LEN: int = 64


def get_render_pipeline(
    uses_avg_buf: bool,
    wgsl_user_functions: str,
    user_data_decl: str,
    have_user_data: bool
) -> tuple[wgpu.GPUBindGroupLayout, wgpu.GPURenderPipeline]:
    """
    return render pipeline matching given grid format and WGSL user functions
    (shade_cell). create new one if not already existing.
    """

    global render_pipeline_map

    render_grid_format = \
        wgpu.TextureFormat.r32float if uses_avg_buf \
        else wgpu.TextureFormat.rg32float

    key = (uses_avg_buf, wgsl_user_functions)
    if key in render_pipeline_map.keys():
        return render_pipeline_map[key]

    if len(render_pipeline_map.keys()) >= MAX_RENDER_PIPELINE_MAP_LEN:
        dict_remove_n_oldest(
            render_pipeline_map,
            MAX_RENDER_PIPELINE_MAP_LEN // 2
        )

    shader = load_shader(
        device,
        "render.wgsl",
        [
            (
                "// [constants]",
                wgsl_sim_constants
                + wgsl_render_constants
                + f"""
const RENDER_USES_AVERAGING_BUFFER = {str(uses_avg_buf).lower()};
                """
            ),
            (
                "[render-grid-format]",
                render_grid_format
            ),
            (
                "// [user-data-decl]",
                user_data_decl.replace("@binding(N)", "@binding(2)")
            ),
            (
                "// [colormaps]",
                load_text("colormaps.wgsl")
            ),
            (
                "// [common-header]",
                selected_sim_params.wgsl_common_header
            ),
            (
                "// [user-functions]",
                wgsl_user_functions
            )
        ]
    )

    entries = [
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
                format=render_grid_format,
                view_dimension=wgpu.TextureViewDimension.d3
            )
        ),
    ]
    if have_user_data:
        entries.append(wgpu.BindGroupLayoutEntry(
            binding=2,
            visibility=wgpu.ShaderStage.FRAGMENT,
            buffer=wgpu.BufferBindingLayout(
                type=wgpu.BufferBindingType.read_only_storage
            )
        ))
    bgl = device.create_bind_group_layout(entries=entries)

    pipeline = device.create_render_pipeline(
        layout=device.create_pipeline_layout(
            bind_group_layouts=[bgl]
        ),
        vertex=wgpu.VertexState(
            module=shader,
            entry_point="vs_main"
        ),
        fragment=wgpu.FragmentState(
            module=shader,
            entry_point="fs_main",
            targets=[wgpu.ColorTargetState(format=render_target.format)],
        ),
        primitive=wgpu.PrimitiveState(
            topology=wgpu.PrimitiveTopology.triangle_list
        ),
    )

    render_pipeline_map[key] = (bgl, pipeline)
    return (bgl, pipeline)


def main():
    global device

    canvas = RenderCanvas(
        size=(960, 540),
        title=WINDOW_TITLE,
        update_mode="fastest",
        vsync=True,
    )

    adapter = wgpu.gpu.request_adapter_sync(
        power_preference=wgpu.PowerPreference.high_performance,
        canvas=canvas
    )
    device = adapter.request_device_sync()

    def retrieve_context() -> tuple[WgpuContext, str]:
        c = canvas.get_wgpu_context()
        surface_format = c.get_preferred_format(adapter)
        c.configure(device=device, format=surface_format)
        return c, surface_format

    context, surface_format = retrieve_context()

    # invoke the on_start() callback if provided
    if selected_sim_params.on_start is not None:
        selected_sim_params.on_start(
            selected_sim_params,
            selected_sim_limits
        )

    # shaders

    user_data_decl: str = "// no user data"
    if selected_sim_params.user_data_fields:
        user_data_decl = \
            f"""
struct UserData {{
    {selected_sim_params.user_data_fields}
}}
@group(0) @binding(N)
var<storage, read> user_data: UserData;
            """

    sim_shader = load_shader(
        device,
        "sim.wgsl",
        [
            (
                "// [constants]",
                wgsl_sim_constants
            ),
            (
                "// [user-data-decl]",
                user_data_decl.replace("@binding(N)", "@binding(3)")
            ),
            (
                "// [common-header]",
                selected_sim_params.wgsl_common_header
            ),
            (
                "// [user-functions]",
                selected_sim_params.initial_value_function
                + selected_sim_params.update_value_function
                + selected_sim_params.speed_fac_function
                + selected_sim_params.damp_fac_function
            )
        ]
    )

    averaging_shader = load_shader(
        device,
        "averaging.wgsl",
        [
            (
                "// [constants]",
                f"""
const AVERAGING_MIX_FAC_PER_DT = {selected_sim_limits.averaging_mix_fac_per_dt};
                """
            )
        ]
    )

    display_shader = load_shader(
        device,
        "display.wgsl",
        [
            (
                "// [constants]",
                f"""
// if surface format applies sRGB OETF internally
const SRGB_SURFACE = {str("srgb" in surface_format.lower()).lower()};
                """
            )
        ]
    )

    grid_copy_shader = load_shader(
        device,
        "grid_to_buffer_copy.wgsl",
        [
            (
                "[source-grid-format]",
                wgpu.TextureFormat.rg32float
            ),
            (
                "[single-channel]",
                "false"
            )
        ]
    )

    grid_copy_single_channel_shader = load_shader(
        device,
        "grid_to_buffer_copy.wgsl",
        [
            (
                "[source-grid-format]",
                wgpu.TextureFormat.r32float
            ),
            (
                "[single-channel]",
                "true"
            )
        ]
    )

    # create double buffered 3D textures for the simulation

    def create_wave_texture(
            label: str,
            format: wgpu.TextureFormat = wgpu.TextureFormat.rg32float
    ) -> tuple[wgpu.GPUTexture, wgpu.GPUTextureView]:
        global device
        t = device.create_texture(
            label=label,
            size=selected_sim_params.grid_res,
            dimension=wgpu.TextureDimension.d3,
            format=format,
            usage=(
                wgpu.TextureUsage.TEXTURE_BINDING |
                wgpu.TextureUsage.STORAGE_BINDING |
                wgpu.TextureUsage.COPY_DST |
                wgpu.TextureUsage.COPY_SRC
            ),
        )
        v = t.create_view(label=label + " (view)")
        return t, v

    wave_grid_a, wave_grid_a_view = create_wave_texture("wave_grid_a")
    wave_grid_b, wave_grid_b_view = create_wave_texture("wave_grid_b")

    # create averaging buffer if needed
    wave_grid_avg = None
    wave_grid_avg_view = None
    if selected_sim_params.averaging:
        wave_grid_avg, wave_grid_avg_view = create_wave_texture(
            "wave_grid_avg",
            wgpu.TextureFormat.r32float
        )

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

    # uniform buffer for simulation pipeline (also reused for averaging)

    sim_uniform_dtype = np.dtype([
        ("iter", np.int32),
        ("time", np.float32),
        ("wall_time", np.float32),
    ])

    sim_uniform = np.zeros((), dtype=sim_uniform_dtype)
    sim_uniform_buffer = CpuToGpuBuffer(
        device=device,
        label="sim_uniform_buffer",
        usage=wgpu.BufferUsage.UNIFORM,
        data_view=memoryview(sim_uniform)
    )

    # uniform buffer for render pipeline

    render_uniform_dtype = np.dtype([
        ("res", np.int32, (2,)),
        ("total_res", np.int32, (2,)),
        ("mode", np.int32),
        ("__pad0", np.int32, (3,)),
        ("pmin", np.int32, (3,)),
        ("__pad1", np.int32),
        ("pmax", np.int32, (3,)),
        ("__pad2", np.int32),
        ("region_span", np.int32, (3,)),
        ("__pad3", np.int32),
        ("pmin_world", np.float32, (3,)),
        ("__pad4", np.int32),
        ("pmax_world", np.float32, (3,)),
        ("__pad5", np.int32),
        ("slice_quad_origin", np.float32, (3,)),
        ("__pad6", np.int32),
        ("slice_quad_right", np.float32, (3,)),
        ("__pad7", np.int32),
        ("slice_quad_up", np.float32, (3,)),
        ("slice_aspect_ratio", np.float32),
        ("bg_col", np.float32, (3,)),
        ("n_samples_per_pixel", np.int32),
        ("raymarch_step", np.float32),
        ("raymarch_step_jitter", np.float32),
        ("use_trilinear", np.int32),
        ("__pad8", np.int32),
        ("cam_pos", np.float32, (3,)),
        ("__pad9", np.int32),
        ("cam_lookat", np.float32, (3,)),
        ("__pad10", np.int32),
        ("cam_world_up", np.float32, (3,)),
        ("cam_fov_degrees", np.float32),
        ("apply_flim", np.int32),
        ("iter", np.int32),
        ("time", np.float32),
        ("wall_time", np.float32),
    ])

    render_uniform = np.zeros((), dtype=render_uniform_dtype)
    render_uniform_buffer = CpuToGpuBuffer(
        device=device,
        label="render_uniform_buffer",
        usage=wgpu.BufferUsage.UNIFORM,
        data_view=memoryview(render_uniform)
    )

    # uniform buffer for display pipeline

    display_uniform_dtype = np.dtype([
        ("render_res", np.float32, (2,)),
        ("display_res", np.float32, (2,)),
    ])

    display_uniform = np.zeros((), dtype=display_uniform_dtype)
    display_uniform_buffer = CpuToGpuBuffer(
        device=device,
        label="display_uniform_buffer",
        usage=wgpu.BufferUsage.UNIFORM,
        data_view=memoryview(display_uniform)
    )

    # uniform buffer for grid copy pipeline

    grid_copy_uniform_dtype = np.dtype([
        ("pmin", np.int32, (3,)),
        ("__pad0", np.int32),
        ("read_res", np.int32, (3,)),
        ("__pad1", np.int32),
    ])

    grid_copy_uniform = np.zeros((), dtype=grid_copy_uniform_dtype)
    grid_copy_uniform_buffer = CpuToGpuBuffer(
        device=device,
        label="grid_copy_uniform_buffer",
        usage=wgpu.BufferUsage.UNIFORM,
        data_view=memoryview(grid_copy_uniform)
    )

    # user data buffer
    user_data_buffer: CpuToGpuBuffer | None = None
    if selected_sim_params.user_data_fields:
        if not selected_sim_params.user_data:
            raise ValueError(
                "user_data_fields is provided but user_data is not"
            )
        user_data_buffer = CpuToGpuBuffer(
            device=device,
            label="user_data_buffer",
            usage=wgpu.BufferUsage.STORAGE,
            data_view=selected_sim_params.user_data
        )

    # bind group layouts

    entries = [
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
    if user_data_buffer:
        entries.append(wgpu.BindGroupLayoutEntry(
            binding=3,
            visibility=wgpu.ShaderStage.COMPUTE,
            buffer=wgpu.BufferBindingLayout(
                type=wgpu.BufferBindingType.read_only_storage
            )
        ))
    sim_bgl = device.create_bind_group_layout(entries=entries)

    averaging_bgl = device.create_bind_group_layout(
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
                    access=wgpu.StorageTextureAccess.read_write,
                    format=wgpu.TextureFormat.r32float,
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

    grid_copy_bgl = device.create_bind_group_layout(
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
                buffer=wgpu.BufferBindingLayout(
                    type=wgpu.BufferBindingType.storage
                )
            ),
        ]
    )

    grid_copy_single_channel_bgl = device.create_bind_group_layout(
        entries=[
            wgpu.BindGroupLayoutEntry(
                binding=0,
                visibility=wgpu.ShaderStage.COMPUTE,
                buffer=wgpu.BufferBindingLayout(
                    type=wgpu.BufferBindingType.uniform
                )
            ),
            wgpu.BindGroupLayoutEntry(
                binding=1,
                visibility=wgpu.ShaderStage.COMPUTE,
                storage_texture=wgpu.StorageTextureBindingLayout(
                    access=wgpu.StorageTextureAccess.read_only,
                    format=wgpu.TextureFormat.r32float,
                    view_dimension=wgpu.TextureViewDimension.d3
                )
            ),
            wgpu.BindGroupLayoutEntry(
                binding=2,
                visibility=wgpu.ShaderStage.COMPUTE,
                buffer=wgpu.BufferBindingLayout(
                    type=wgpu.BufferBindingType.storage
                )
            ),
        ]
    )

    # pipelines

    sim_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(
            bind_group_layouts=[sim_bgl]
        ),
        compute=wgpu.ProgrammableStage(
            module=sim_shader,
            entry_point="cs_main"
        ),
    )

    averaging_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(
            bind_group_layouts=[averaging_bgl]
        ),
        compute=wgpu.ProgrammableStage(
            module=averaging_shader,
            entry_point="cs_main"
        )
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

    grid_copy_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(
            bind_group_layouts=[grid_copy_bgl]
        ),
        compute=wgpu.ProgrammableStage(
            module=grid_copy_shader,
            entry_point="cs_main"
        ),
    )

    grid_copy_single_channel_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(
            bind_group_layouts=[grid_copy_single_channel_bgl]
        ),
        compute=wgpu.ProgrammableStage(
            module=grid_copy_single_channel_shader,
            entry_point="cs_main"
        ),
    )

    def readback_grid(
        grid_view: wgpu.GPUTextureView,
        region: Aabb
    ) -> np.ndarray:
        global device, cpu_visible_buf, readback_dest_buf

        pmin, pmax = region.resolve(selected_sim_params.grid_res)

        # see if grid is r32f (averaging buffer) or rg32f (regular wave grid)
        if grid_view.texture.format == wgpu.TextureFormat.r32float:
            single_channel = True
        elif grid_view.texture.format == wgpu.TextureFormat.rg32float:
            single_channel = False
        else:
            raise ValueError("unsupported grid format for readback")

        # calculate total size
        read_res = region.span(selected_sim_params.grid_res)
        n_cells = reduce(operator.mul, read_res)
        n_numbers = n_cells * 2  # current and previous value for each
        if single_channel:
            n_numbers = n_cells  # just current value
        n_bytes = n_numbers * np.float32().nbytes

        # prepare destination and staging buffers
        readback_dest_buf, cpu_visible_buf = prepare_buffers(
            device,
            [readback_dest_buf, cpu_visible_buf],
            ["readback_dest_buf", "cpu_visible_buf"],
            [
                wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
            ],
            [n_bytes, n_bytes],
        )

        # start GPU commands
        cmd_encoder = device.create_command_encoder()

        # command: update uniform buffer
        grid_copy_uniform["pmin"] = pmin
        grid_copy_uniform["read_res"] = read_res
        grid_copy_uniform_buffer.push_upload_command(cmd_encoder)

        # command: run compute shader for grid-to-buffer copy
        cpass = cmd_encoder.begin_compute_pass()
        if single_channel:
            cpass.set_pipeline(grid_copy_single_channel_pipeline)
            cpass.set_bind_group(0, device.create_bind_group(
                layout=grid_copy_single_channel_bgl,
                entries=[
                    wgpu.BindGroupEntry(
                        binding=0,
                        resource=wgpu.BufferBinding(
                            buffer=grid_copy_uniform_buffer.buf
                        )
                    ),
                    wgpu.BindGroupEntry(
                        binding=1,
                        resource=grid_view
                    ),
                    wgpu.BindGroupEntry(
                        binding=2,
                        resource=readback_dest_buf
                    ),
                ]
            ))
        else:
            cpass.set_pipeline(grid_copy_pipeline)
            cpass.set_bind_group(0, device.create_bind_group(
                layout=grid_copy_bgl,
                entries=[
                    wgpu.BindGroupEntry(
                        binding=0,
                        resource=wgpu.BufferBinding(
                            buffer=grid_copy_uniform_buffer.buf
                        )
                    ),
                    wgpu.BindGroupEntry(
                        binding=1,
                        resource=grid_view
                    ),
                    wgpu.BindGroupEntry(
                        binding=2,
                        resource=readback_dest_buf
                    ),
                ]
            ))

        workgroup_count = (
            (read_res[0] + 7) // 8,
            (read_res[1] + 7) // 8,
            (read_res[2] + 3) // 4,
        )
        cpass.dispatch_workgroups(*workgroup_count)
        cpass.end()

        # command: copy from destination buffer to staging (CPU-visible) buffer
        cmd_encoder.copy_buffer_to_buffer(
            readback_dest_buf, 0,
            cpu_visible_buf, 0,
            n_bytes
        )

        # submit the commands
        device.queue.submit([cmd_encoder.finish()])

        # read back the staging buffer
        cpu_visible_buf.map_sync(wgpu.MapMode.READ)
        buf_copy = cpu_visible_buf.read_mapped(
            0, n_bytes, copy=True
        )
        cpu_visible_buf.unmap()

        # reinterpret as a numpy.ndarray
        if single_channel:
            return np.ndarray(
                (read_res[2], read_res[1], read_res[0]),
                dtype=np.float32,
                buffer=buf_copy
            )
        else:
            return np.ndarray(
                (read_res[2], read_res[1], read_res[0], 2),
                dtype=np.float32,
                buffer=buf_copy
            )

    # simulation state
    sim_state = WaveSimState(selected_sim_limits.resolved_timestep)
    prev_sim_state = deepcopy(sim_state)

    # per-frame logic
    def draw():
        global device, render_target, render_target_view, cpu_visible_buf
        nonlocal canvas, context, surface_format, sim_state, prev_sim_state, \
            user_data_buffer

        # advance the simulation

        # update the uniform buffer
        sim_uniform["iter"] = sim_state.iter
        sim_uniform["time"] = sim_state.time
        sim_uniform["wall_time"] = sim_state.wall_time
        sim_uniform_buffer.upload()

        cmd_encoder = device.create_command_encoder()

        # run compute shader for the simulation

        input_grid_view = \
            wave_grid_a_view if sim_state.use_a_as_input else wave_grid_b_view
        output_grid_view = \
            wave_grid_b_view if sim_state.use_a_as_input else wave_grid_a_view

        cpass = cmd_encoder.begin_compute_pass()
        cpass.set_pipeline(sim_pipeline)

        entries = [
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
        ]
        if user_data_buffer:
            entries.append(wgpu.BindGroupEntry(
                binding=3,
                resource=user_data_buffer.buf
            ))
        cpass.set_bind_group(0, device.create_bind_group(
            layout=sim_bgl,
            entries=entries
        ))

        workgroup_count = (
            (selected_sim_params.grid_res[0] + 7) // 8,
            (selected_sim_params.grid_res[1] + 7) // 8,
            (selected_sim_params.grid_res[2] + 3) // 4,
        )
        cpass.dispatch_workgroups(*workgroup_count)
        cpass.end()

        # run compute shader for averaging
        if selected_sim_params.averaging:
            apass = cmd_encoder.begin_compute_pass()
            apass.set_pipeline(averaging_pipeline)
            apass.set_bind_group(0, device.create_bind_group(
                layout=averaging_bgl,
                entries=[
                    wgpu.BindGroupEntry(
                        binding=0,
                        resource=wgpu.BufferBinding(
                            buffer=sim_uniform_buffer.buf
                        )
                    ),
                    wgpu.BindGroupEntry(
                        binding=1,
                        resource=output_grid_view
                    ),
                    wgpu.BindGroupEntry(
                        binding=2,
                        resource=wave_grid_avg_view
                    ),
                ]
            ))
            apass.dispatch_workgroups(*workgroup_count)
            apass.end()

        device.queue.submit([cmd_encoder.finish()])

        # update simulation state
        prev_sim_state = deepcopy(sim_state)
        sim_state.advance()

        # update window title
        canvas.set_title(WINDOW_TITLE.format(
            prev_sim_state.time,
            prev_sim_state.iter
        ))

        def try_get_averaging_grid_view():
            if selected_sim_params.averaging:
                return wave_grid_avg_view
            raise ValueError(
                "there is no averaging buffer because averaging is disabled"
            )

        # user callback
        render_commands, display_render_idx, new_user_data, should_stop = \
            selected_sim_params.on_update(
                selected_sim_params,
                selected_sim_limits,
                prev_sim_state,
                lambda region, read_averaging_grid:
                    readback_grid(
                        try_get_averaging_grid_view() if read_averaging_grid
                        else output_grid_view,
                        region
                    )
            )

        # update user data buffer if needed
        if new_user_data:
            user_data_buffer.set_data_view(new_user_data)
            user_data_buffer.upload()

        # process render commands
        for render_command_idx in range(len(render_commands)):
            render_cmd = render_commands[render_command_idx]

            # don't waste time if we're not gonna display or export it
            if display_render_idx != render_command_idx \
                    and render_cmd.export_path is None:
                continue

            # resize render target if it's not large enough
            prepare_render_target(render_cmd.res)

            # visible region
            pmin, pmax, span = render_cmd.region.resolve_with_span(
                selected_sim_params.grid_res
            )

            # which slice to render in slice mode
            slice_quad, slice_aspect_ratio = render_cmd.slice.resolve(
                selected_sim_limits
            )

            # verify render mode
            if render_cmd.mode not in [
                RenderMode.Raymarching,
                RenderMode.Slice
            ]:
                raise ValueError("unsupported render mode")

            # update uniform buffer
            render_uniform["res"] = render_cmd.res
            render_uniform["total_res"] = render_target.size[:2]
            render_uniform["mode"] = int(render_cmd.mode)
            render_uniform["pmin"] = pmin
            render_uniform["pmax"] = pmax
            render_uniform["region_span"] = span
            render_uniform["pmin_world"] = selected_sim_limits.icoord_to_world(
                tuple([pmin[i] - .5 for i in range(len(pmin))])
            )
            render_uniform["pmax_world"] = selected_sim_limits.icoord_to_world(
                tuple([pmax[i] + .5 for i in range(len(pmax))])
            )
            render_uniform["slice_quad_origin"] = slice_quad.origin
            render_uniform["slice_quad_right"] = slice_quad.right
            render_uniform["slice_quad_up"] = slice_quad.up
            render_uniform["slice_aspect_ratio"] = slice_aspect_ratio
            render_uniform["bg_col"] = render_cmd.bg_col
            render_uniform["n_samples_per_pixel"] = render_cmd.n_samples_per_pixel
            render_uniform["raymarch_step"] = render_cmd.raymarch_step
            render_uniform["raymarch_step_jitter"] = render_cmd.raymarch_step_jitter
            render_uniform["use_trilinear"] = int(render_cmd.use_trilinear)
            render_uniform["cam_pos"] = render_cmd.cam.pos
            render_uniform["cam_lookat"] = render_cmd.cam.lookat
            render_uniform["cam_world_up"] = render_cmd.cam.world_up
            render_uniform["cam_fov_degrees"] = render_cmd.cam.fov_degrees
            render_uniform["apply_flim"] = int(render_cmd.apply_flim)
            render_uniform["iter"] = prev_sim_state.iter
            render_uniform["time"] = prev_sim_state.time
            render_uniform["wall_time"] = prev_sim_state.wall_time
            render_uniform_buffer.upload()

            # see if we can and should read averaging buffer and get a matching
            # render pipeline and bind group layout.

            use_avg_buf = \
                render_cmd.try_use_averaging_buffer \
                and selected_sim_params.averaging

            render_bgl, render_pipeline = get_render_pipeline(
                use_avg_buf,
                render_cmd.shade_cell_function,
                user_data_decl,
                user_data_buffer is not None
            )

            # command buffer
            cmd_encoder = device.create_command_encoder()
            cmd_submitted = False

            # render pass

            render_grid_view = wave_grid_a_view
            if prev_sim_state.use_a_as_input:
                render_grid_view = wave_grid_b_view
            if use_avg_buf:
                render_grid_view = wave_grid_avg_view

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

            entries = [
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
            ]
            if user_data_buffer:
                entries.append(wgpu.BindGroupEntry(
                    binding=2,
                    resource=user_data_buffer.buf
                ))
            rpass.set_bind_group(0, device.create_bind_group(
                layout=render_bgl,
                entries=entries
            ))

            rpass.draw(6, 1, 0, 0)
            rpass.end()

            # export to image file if needed
            if render_cmd.export_path:
                n_bytes = (
                    render_target.width
                    * render_target.height
                    * 4 * 1  # n. color channels * n. bytes per channel
                )
                n_bytes_per_row = (
                    render_target.width
                    * 4 * 1  # n. color channels * n. bytes per channel
                )

                # prepare staging buffer
                cpu_visible_buf = prepare_buffers(
                    device,
                    [cpu_visible_buf],
                    ["cpu_visible_buf"],
                    [wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST],
                    [n_bytes],
                )[0]

                # add command to copy from render target to staging buffer
                cmd_encoder.copy_texture_to_buffer(
                    wgpu.TexelCopyTextureInfo(texture=render_target),
                    wgpu.TexelCopyBufferInfo(
                        buffer=cpu_visible_buf,
                        bytes_per_row=n_bytes_per_row,
                        rows_per_image=render_target.height
                    ),
                    copy_size=render_target.size
                )

                # submit command buffer
                device.queue.submit([cmd_encoder.finish()])
                cmd_submitted = True

                # read the staging buffer
                cpu_visible_buf.map_sync(wgpu.MapMode.READ)
                buf_copy = cpu_visible_buf.read_mapped(
                    0, n_bytes, copy=True
                )
                cpu_visible_buf.unmap()

                # reinterpret as a numpy.ndarray, crop to render resolution, and
                # convert from RGBA to RGB.
                pixels = np.ndarray(
                    (render_target.height, render_target.width, 4),
                    dtype=np.uint8,
                    buffer=buf_copy
                )[0:render_cmd.res[1], 0:render_cmd.res[0], :3]

                # export
                if (
                    str(render_cmd.export_path).lower().endswith(".jpg")
                    or str(render_cmd.export_path).lower().endswith(".jpeg")
                ):
                    Image.fromarray(pixels, "RGB").save(
                        render_cmd.export_path,
                        subsampling=0,
                        quality=95
                    )
                else:
                    Image.fromarray(pixels, "RGB").save(
                        render_cmd.export_path
                    )

            # display pass

            if display_render_idx != render_command_idx:
                if not cmd_submitted:
                    device.queue.submit([cmd_encoder.finish()])
                    cmd_submitted = True
                continue

            window_physical_size = (
                render_cmd.res[0] * DISPLAY_SCALE,
                render_cmd.res[1] * DISPLAY_SCALE
            )
            window_physical_size_int = (
                int(window_physical_size[0]),
                int(window_physical_size[1])
            )
            window_logical_size = (
                window_physical_size[0] / canvas.get_pixel_ratio(),
                window_physical_size[1] / canvas.get_pixel_ratio()
            )
            if np.abs(
                canvas.get_logical_size()[0] - window_logical_size[0]
            ) > 1. or np.abs(
                canvas.get_logical_size()[1] - window_logical_size[1]
            ) > 1.:
                canvas.set_logical_size(*window_logical_size)
                context, surface_format = retrieve_context()

            uniform_render_res_int = tuple(
                display_uniform["render_res"].astype(int)
            )
            uniform_display_res_int = tuple(
                display_uniform["display_res"].astype(int)
            )
            if uniform_render_res_int != render_cmd.res \
                    or uniform_display_res_int != window_physical_size_int:
                display_uniform["render_res"] = np.asarray(
                    render_cmd.res,
                    np.float32
                )
                display_uniform["display_res"] = np.asarray(
                    window_physical_size_int,
                    np.float32
                )
                display_uniform_buffer.upload()

            current_swapchain_view = (
                context.get_current_texture()
                .create_view()
            )

            if cmd_submitted:
                cmd_encoder = device.create_command_encoder()
                cmd_submitted = False

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
                        resource=display_uniform_buffer.buf
                    ),
                    wgpu.BindGroupEntry(
                        binding=1,
                        resource=render_target_view
                    ),
                    wgpu.BindGroupEntry(
                        binding=2,
                        resource=linear_sampler
                    ),
                ]
            ))
            dpass.draw(6, 1, 0, 0)
            dpass.end()

            device.queue.submit([cmd_encoder.finish()])

        # stop if needed
        if should_stop:
            canvas.request_draw(lambda _: None)
            canvas.close()

    canvas.request_draw(draw)
    loop.run()


if __name__ == "__main__":
    main()
