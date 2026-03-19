import time
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable
import numpy as np
import wgpu


# store source code for shaders with replacements applied. useful for debugging,
# especially when we need to go to a certain line number.
STORE_RESOLVED_SHADER_CODE = False


class WaveSimState:
    # current iteration
    iter: int

    # simulation time
    time: float

    # real world time
    wall_time: float
    _wall_time_start_ns: int

    # for double buffering (alternating between buffer A and B)
    use_a_as_input: bool

    # for calculating time based on iter
    timestep: float

    def __init__(self, timestep: float):
        self.iter = 0
        self.time = 0.
        self.wall_time = 0.
        self.use_a_as_input = True
        self.timestep = timestep
        self._wall_time_start_ns = time.time_ns()

    def advance(self, n_steps: int = 1):
        for _ in range(n_steps):
            self.use_a_as_input = not self.use_a_as_input
            self.iter += 1
            self.time = self.iter * self.timestep
        self.wall_time = float(time.time_ns() - self._wall_time_start_ns) / 1e9


@dataclass
class CameraState:
    pos: tuple[float, float, float] = (0., -1., 0.)
    lookat: tuple[float, float, float] = (0., 0., 0.)
    world_up: tuple[float, float, float] = (0., 0., 1.)
    fov_degrees: float = 90.


WaveSimReadbackFunction = Callable[
    [
        tuple[int, int, int] | None,  # pmin
        tuple[int, int, int] | None,  # pmax
        bool  # read_averaging_grid
    ],
    np.ndarray  # returned ndarray
]


@dataclass
class WaveSimParams:
    # render resolution
    render_res: tuple[int, int]

    # simulation grid resolution
    # NOTE: if (and only if) the third axis (z) equals 1, 2D simulation and
    # rendering logic will be used instead of 3D, and the computed limits (e.g.
    # max stable timestep) will also assume a 2D simulation. this only applies
    # if the third axis (z) is 1, not the other two.
    grid_res: tuple[int, int, int]

    # simulation grid cell (voxel) size
    cell_size: float

    # wave propagation speed
    wave_speed: float

    # remove reflections at the grid boundaries
    remove_reflections: bool

    # delta time. if negative, will be a factor of the maximum stable timestep,
    # e.g. -0.5 means 0.5 * max_timestep. the final value can be found in
    # WaveSimLimits.resolved_timestep.
    timestep: float

    # advance the simulation this many iterations before rendering each frame.
    n_sim_steps_per_frame: int

    # user-provided WGSL code added to the simulation compute shader and the
    # fragment shader used for rendering. useful for sharing code between
    # simulation-related functions and the shade_cell_function.
    wgsl_common_header: str

    # WGSL function used in the compute shader that defines how the grid is
    # initialized in iteration 0. the function signature must not be modified.
    # see "sim.wgsl" and sim_test below for an example.
    initial_value_function: str

    # WGSL function used in the compute shader for applying custom excitations
    # or modifications to the grid/field. it must return the new current
    # ("curr") value for every cell. the function signature must not be
    # modified. see "sim.wgsl" and sim_test below for an example.
    update_value_function: str

    # WGSL function used in the compute shader returning the wave propagation
    # speed factor for every cell, useful for defining obstacles, reflectors, or
    # lenses (refractors). must return values in the [0, 1] range for a stable
    # simulation.
    speed_fac_function: str

    # WGSL function used in the compute shader returning the dampening factor
    # per second for every cell. this factor must be in the [0, 1] range and
    # defines how fast the wave decays in every cell. for example, a dampening
    # factor of 0.8 for a cell means the time derivative of its wave value will
    # be scaled by 0.8 every second in simulation time or by (0.8^timestep)
    # every iteration.
    damp_fac_function: str

    # NOTE:
    # the following WGSL constants, uniform values, and functions are available
    # in the simulation compute shader and therefore the WGSL functions above:
    #
    # math constants
    #   PI, TAU, HALF_PI: f32
    #
    # simulation grid resolution
    #   GRID_RES: vec3i
    #
    # same as (GRID_RES.z == 1)
    #   IS_2D: bool
    #
    # simulation grid cell (voxel) size
    #   CELL_SIZE: f32
    #
    # dimensions of the simulation grid (cell_size * grid_res)
    #   GRID_DIMS: vec3f
    #
    # wave propagation speed
    #   WAVE_SPEED: f32
    #
    # remove reflections at the grid boundaries
    #   REMOVE_REFLECTIONS: bool
    #
    # resolved timestep
    #   TIMESTEP: f32
    #
    # maximum stable timestep
    #   MAX_TIMESTEP: f32
    #
    # minimum stable wavelength
    #   MIN_WAVELENGTH: f32
    #
    # maximum stable frequency
    #   MAX_FREQ: f32
    #
    # simulation iteration
    #   ubo.iter: i32
    #
    # simulation time
    #   ubo.time: f32
    #
    # real world time
    #   ubo.wall_time: f32
    #
    # used internally for removing reflections at the grid boundaries
    #   IMPEDANCE_MATCHING_COEFFICIENT: f32
    #
    # this function converts 3D indices in the grid to physical coordinates in
    # the simulation world based on GRID_RES and GRID_DIMS. it's centered around
    # the origin (0, 0, 0) so if e.g. GRID_DIMS=(1, 1, 1) then the coordinates
    # will range from (-0.5, -0.5, -0.5) to (+0.5, +0.5, +0.5).
    #   fn icoord_to_world(icoord: vec3i) -> vec3f {...}
    #
    # this is the data stored in every cell:
    #   struct WaveValue {
    #       curr: f32, // value in the current iteration
    #       prev: f32, // value in the previous iteration
    #   };
    #
    # this function fetches the data stored in an arbitrary cell
    #   fn grid_fetch(icoord: vec3i) -> WaveValue {...}

    # apply exponential smoothing to the field intensity (v^2, as opposed to
    # v which is the raw amplitude) and use that for rendering instead of the
    # raw amplitude.
    # NOTE: this will average out the squares of the cell values, so if your
    # shade_cell function uses a colormap that squares the value internally,
    # make sure it only squares it if the AVERAGING constant is set to true.
    averaging: bool

    # when averaging is enabled, the averaging buffer will approach the newest
    # values by 63% (1 - 1/e) every averaging_time_constant seconds in
    # simulation time.
    averaging_time_constant: float

    # rendering: background color
    render_bg_col: tuple[float, float, float]

    # WGSL function used in the fragment shader returning a color for every grid
    # cell based on its stored value (v) and coordinates (icoord) as well as the
    # constants and uniform values mentioned below. this function is useful for
    # colorizing the grid values with colormaps and adding visual indicators for
    # obstacles, lenses, etc.
    shade_cell_function: str

    # NOTE:
    # the following WGSL constants, uniform values, and functions are available
    # in the fragment shader and therefore in the shade_cell_function:
    #
    # math constants
    #   PI, TAU, HALF_PI: f32
    #
    # render target resolution
    #   RES: vec2i
    #
    # same as averaging
    #   AVERAGING: bool
    #
    # same as averaging_time_constant
    #   AVERAGING_TIME_CONSTANT: AbstractFloat
    #
    # same as render_bg_col
    #   BG_COL: vec3f
    #
    # same as render_n_samples_per_pixel
    #   N_SAMPLES_PER_PIXEL: AbstractInt
    #
    # same as render_raymarch_step
    #   RAYMARCH_STEP: AbstractFloat
    #
    # same as render_raymarch_step_jitter
    #   RAYMARCH_STEP_JITTER: AbstractFloat
    #
    # same as render_use_trilinear
    #   USE_TRILINEAR: bool
    #
    # same as render_apply_flim
    #   APPLY_FLIM: bool
    #
    # simulation grid resolution
    #   GRID_RES: vec3i
    #
    # same as (GRID_RES.z == 1)
    #   IS_2D: bool
    #
    # simulation grid cell (voxel) size
    #   CELL_SIZE: f32
    #
    # dimensions of the simulation grid (cell_size * grid_res)
    #   GRID_DIMS: vec3f
    #
    # wave propagation speed
    #   WAVE_SPEED: f32
    #
    # remove reflections at the grid boundaries
    #   REMOVE_REFLECTIONS: bool
    #
    # resolved timestep
    #   TIMESTEP: f32
    #
    # maximum stable timestep
    #   MAX_TIMESTEP: f32
    #
    # minimum stable wavelength
    #   MIN_WAVELENGTH: f32
    #
    # maximum stable frequency
    #   MAX_FREQ: f32
    #
    # simulation iteration
    #   ubo.iter: i32
    #
    # simulation time
    #   ubo.time: f32
    #
    # real world time
    #   ubo.wall_time: f32
    #
    # used internally for removing reflections at the grid boundaries
    #   IMPEDANCE_MATCHING_COEFFICIENT: f32
    #
    # this function converts 3D indices in the grid to physical coordinates in
    # the simulation world based on GRID_RES and GRID_DIMS. it's centered around
    # the origin (0, 0, 0) so if e.g. GRID_DIMS=(1, 1, 1) then the coordinates
    # will range from (-0.5, -0.5, -0.5) to (+0.5, +0.5, +0.5).
    #   fn icoord_to_world(icoord: vec3i) -> vec3f {...}

    # rendering: number of jittered samples per pixel for anti-aliasing
    render_n_samples_per_pixel: int

    # rendering: step size for ray marching. not used in 2D simulations.
    render_raymarch_step: float

    # rendering: step size jitter for ray marching. not used in 2D simulations.
    render_raymarch_step_jitter: float

    # rendering: use trilinear interpolation/filtering for sampling the
    # simulation grid instead of nearest-neighbor. for 2D simulations, this
    # enables bilinear filtering.
    render_use_trilinear: bool

    # rendering: Python function returning the camera state per frame. not used
    # in 2D simulations.
    render_camera_function: Callable[
        [WaveSimParams, WaveSimLimits, WaveSimState],
        CameraState
    ]

    # rendering: whether to apply the flim view transform for prettier results.
    # see https://github.com/bean-mhm/flim
    render_apply_flim: bool

    # a user-provided callback called after every iteration.
    # example signature:
    #   def sim_on_update(
    #       params: WaveSimParams,
    #       limits: WaveSimLimits,
    #       state: WaveSimState,
    #       readback_function: WaveSimReadbackFunction
    #   )
    on_update: Callable[
        [
            WaveSimParams,
            WaveSimLimits,
            WaveSimState,

            # a function you may call to read back the simulation grid data to
            # the CPU as a numpy.ndarray and do with it what you will.
            #
            # you must pass as arguments two tuples pmin and pmax describing
            # which region to read from the grid. pmin and pmax are 3D integer
            # vectors which describe the minimum and maximum indices in each
            # axis.
            #
            # the third argument is a boolean defining whether to read from the
            # averaging buffer instead of the original grid. this can only be
            # enabled if averaging=True.
            #
            # NOTE: just like Python itself, negative indices will start from
            # the end of the axis and move backward, e.g. -1 is the last element
            # in the axis.
            #
            # NOTE: if pmin is None, (0, 0, 0) will be used, and if pmax is
            # None, (grid_res.x - 1, ..., ...) will be used. therefore, you can
            # set them both to None to read the whole grid.
            #
            # NOTE: in the simulation grid, each cell contains two numbers: its
            # current value and its previous value. therefore, the last axis in
            # the returned ndarray's shape will be 2. you can access the first
            # element (index 0) for the current value and the second one for the
            # previous value. this is NOT true for the averaging buffer, as it
            # only has a single channel.
            #
            # NOTE: calling this every iteration will slow down the simulation.
            WaveSimReadbackFunction
        ],
        None
    ] | None = None


class WaveSimLimits:
    # dimensions of the simulation grid (cell_size * grid_res)
    grid_dims: tuple[float, float, float]

    # resolved timestep
    resolved_timestep: float

    # maximum stable timestep
    max_timestep: float

    # minimum stable wavelength
    min_wavelength: float

    # maximum stable frequency
    max_freq: float

    # used internally for removing reflections at the grid boundaries
    impedance_matching_coefficient: float

    # used in the averaging shader for exponential smoothing
    averaging_mix_fac_per_dt: float

    def __init__(self, params: WaveSimParams):
        dimensionality = 2 if params.grid_res[2] == 1 else 3
        sqrt_dimensionality = np.sqrt(float(dimensionality))

        self.grid_dims = (
            params.cell_size * params.grid_res[0],
            params.cell_size * params.grid_res[1],
            params.cell_size * params.grid_res[2]
        )

        self.max_timestep = \
            params.cell_size / (params.wave_speed * sqrt_dimensionality)

        self.resolved_timestep = params.timestep
        if self.resolved_timestep < 0:
            self.resolved_timestep = \
                -self.resolved_timestep * self.max_timestep

        self.min_wavelength = params.cell_size * sqrt_dimensionality * 8.
        self.max_freq = params.wave_speed / self.min_wavelength

        self.impedance_matching_coefficient = \
            (params.wave_speed * self.resolved_timestep - params.cell_size) \
            / (params.wave_speed * self.resolved_timestep + params.cell_size)

        if params.averaging:
            self.averaging_mix_fac_per_dt = 1. - np.exp(
                -self.resolved_timestep / params.averaging_time_constant
            )
        else:
            self.averaging_mix_fac_per_dt = 1.


def constant_initial_value_function(v: float) -> str:
    return f"""
fn initial_value(icoord: vec3i) -> WaveValue {{
    return WaveValue({v}, {v});
}}
    """


def constant_speed_fac_function(fac: float) -> str:
    return f"""
fn speed_fac(icoord: vec3i, v: WaveValue) -> f32 {{
    return {fac};
}}
    """


def constant_damp_fac_function(fac: float) -> str:
    return f"""
fn damp_fac(icoord: vec3i, v: WaveValue) -> f32 {{
    return {fac};
}}
    """


def load_text(filename_relative: str) -> str:
    path = Path(__file__).parent / filename_relative
    if not path.exists():
        raise FileNotFoundError(f"file {path} is missing")
    return path.read_text()


def load_shader(
    device: wgpu.GPUDevice,
    filename_relative: str = "shader.wgsl",
    replacements: list[tuple[str, str]] = []
) -> wgpu.GPUShaderModule:

    path = Path(__file__).parent / filename_relative
    if not path.exists():
        raise FileNotFoundError(f"shader source file {path} is missing")

    code = path.read_text()
    for replacement in replacements:
        code = code.replace(replacement[0], replacement[1])

    # store the version with replacements applied (useful for debugging)
    if STORE_RESOLVED_SHADER_CODE:
        resolved_path = \
            Path(__file__).parent / ".debug" / \
            ("resolved_" + filename_relative)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved_path, "w") as f:
            f.write(code)
            f.close()

    return device.create_shader_module(code=code)


# (re)create given buffers to make sure they exist and have enough size
def prepare_buffers(
    device: wgpu.GPUDevice,
    bufs: list[wgpu.GPUBuffer | None],
    labels: list[str],
    usage_flags: list[wgpu.flags.BufferUsageFlags],
    min_sizes: list[int]
) -> list[wgpu.GPUBuffer]:
    if len(bufs) != len(labels) or len(bufs) != len(usage_flags) \
            or len(bufs) != len(min_sizes):
        raise IndexError("provided lists must have identical sizes")

    for i in range(len(bufs)):
        if bufs[i] is not None and bufs[i].size >= min_sizes[i]:
            continue
        bufs[i] = device.create_buffer(
            label=labels[i],
            size=min_sizes[i],
            usage=usage_flags[i],
            mapped_at_creation=False
        )

    return bufs


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

    # only pushes GPU commands, does not run them
    def push_upload_command(self, cmd_encoder: wgpu.GPUCommandEncoder):
        self.staging_buf.map_sync(wgpu.MapMode.WRITE)
        self.staging_buf.write_mapped(self.data_view)
        self.staging_buf.unmap()

        cmd_encoder.copy_buffer_to_buffer(
            self.staging_buf,
            0,
            self.buf,
            0,
            self.data_size
        )

    def upload(self):
        cmd_encoder = self.device.create_command_encoder()
        self.push_upload_command(cmd_encoder)
        self.device.queue.submit([cmd_encoder.finish()])


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
