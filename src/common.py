import time
from pathlib import Path
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import NamedTuple
from enum import IntEnum
from copy import deepcopy
import numpy as np
import wgpu


WINDOW_TITLE = "wavesim2 (time: {:.3f}, iter: {})"


class StoreShaderWhen(IntEnum):
    Never = 0
    OnError = 1
    Always = 2


# store source code for shaders with replacements applied. useful for debugging,
# especially when we need to go to a certain line number.
STORE_RESOLVED_SHADER_CODE = StoreShaderWhen.OnError


class WaveSimState:
    iter: int
    "current iteration"

    time: float
    "simulation time in seconds"

    wall_time: float
    "real world time in seconds"
    _wall_time_start_ns: int

    use_a_as_input: bool
    "for double buffering (alternating between buffer A and B)"

    timestep: float
    "for calculating time based on iter"

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


@dataclass
class Aabb:
    """
    integer bounds for a 3D axis-aligned bounding box.
    pmin and pmax are 3D integer vectors describing the minimum and maximum
    indices in each axis.

    just like Python itself, negative indices will start from the end of the
    axis (depending on the resolution) and move backward, e.g. -1 is the last
    element in the axis.

    if pmin or pmax is None, (0, 0, 0) and (-1, -1, -1) will be used,
    respectively. therefore, you can set them both to None to cover the entire
    grid.
    """

    pmin: tuple[int, int, int] | None = None
    pmax: tuple[int, int, int] | None = None

    def resolve(self, resolution: tuple[int, int, int]) \
            -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """
        returns valid pmin and pmax indices based on given resolution. raises
        IndexError on incorrect ordering or out-of-bounds indices.
        """

        # use default values if None
        pmin_r = (0, 0, 0)
        pmax_r = (-1, -1, -1)
        if self.pmin is not None:
            pmin_r = deepcopy(self.pmin)
        if self.pmax is not None:
            pmax_r = deepcopy(self.pmax)

        # wrap negatives back to the end of the axis
        for axis in range(3):
            if pmin_r[axis] < 0:
                wrapped = pmin_r[axis] + resolution[axis]
                pmin_r = pmin_r[:axis] + (wrapped,) + pmin_r[axis+1:]
            if pmax_r[axis] < 0:
                wrapped = pmax_r[axis] + resolution[axis]
                pmax_r = pmax_r[:axis] + (wrapped,) + pmax_r[axis+1:]

        for axis in range(3):
            # ordering: ensure min <= max
            if pmin_r[axis] > pmax_r[axis]:
                raise IndexError(
                    f"every component in pmin ({str(pmin_r)}) must be less "
                    f"than or equal to its counterpart in pmax ({str(pmax_r)})"
                )

            # bounds checking
            if pmin_r[axis] < 0:
                raise IndexError(
                    f"pmin is out of bounds (pmin={str(pmin_r)}, resolution="
                    f"{str(resolution)})"
                )
            if pmax_r[axis] >= resolution[axis]:
                raise IndexError(
                    f"pmax_r[{axis}] is out of bounds (pmax={str(pmax_r)}, "
                    f"resolution={str(resolution)})"
                )

        return (pmin_r, pmax_r)

    def resolve_with_span(self, resolution: tuple[int, int, int]) \
            -> tuple[tuple[int, int, int],
                     tuple[int, int, int],
                     tuple[int, int, int]]:
        pmin, pmax = self.resolve(resolution)
        span = tuple([pmax[i] - pmin[i] + 1 for i in range(len(pmin))])
        return (pmin, pmax, span)

    def span(self, resolution: tuple[int, int, int]) -> tuple[int, int, int]:
        pmin, pmax = self.resolve(resolution)
        return tuple([pmax[i] - pmin[i] + 1 for i in range(len(pmin))])


@dataclass
class Quadrilateral:
    "a 4-sided polygon, usually in the shape of a rectangle or parallelogram"

    origin: tuple[float, float, float] = (0., 0., 0.)
    right: tuple[float, float, float] = (1., 0., 0.)
    up: tuple[float, float, float] = (0., 1., 0.)


@dataclass
class GridSlice:
    """
    a 2D slice of a 3D grid.

    example for selecting the XZ plane at Y=14

    .. code-block:: py
        GridSlice(
            x_axis=0,  # select the X axis of the grid for the slice's X axis
            x_range=(0, -1),  # cover the whole range

            y_axis=2,  # select the Z axis of the grid for the slice's Y axis
            y_range=(0, 49),  # select the first 50 rows (0 <= Z <= 49)

            unused_axis_index=14,  # set the unused axis of the grid (Y) to 14
        )

    - the ranges are inclusive on both ends.
    - the start and end indices can be in the opposite order, e.g. (50, 20) is
    allowed.
    """

    x_axis: int = 0
    "which 3D axis to use for the slice's X (horizontal) axis (0, 1, 2 for XYZ)"

    x_range: tuple[int, int] = (0, -1)
    """
    map the X (horizontal) axis of the slice to this range in the 3D grid's
    X/Y/Z axis (depending on x_axis above).
    """

    y_axis: int = 1
    "which 3D axis to use for the slice's Y (vertical) axis (0, 1, 2 for XYZ)"

    y_range: tuple[int, int] = (0, -1)
    """
    map the Y (vertical) axis of the slice to this range in the 3D grid's
    X/Y/Z axis (depending on y_axis above).
    """

    unused_axis_index: int = 0
    """
    constant index to use for the remaining 3D axis (different from both
    x_axis and y_axis).
    """

    def resolve(self, sim_limits: WaveSimLimits) -> tuple[Quadrilateral, float]:
        """
        return a quad representing the slice in 3D space (which is easier to use
        in a shader) and its aspect ratio (width / height). performs sanity
        checks and raises errors if the numbers don't add up.
        """

        # verify axes
        if self.x_axis not in range(3) or self.y_axis not in range(3):
            raise IndexError(
                "axis must be one of 0, 1, or 2 for X, Y, and Z, respectively ("
                f"x_axis={self.x_axis}, y_axis={self.y_axis})"
            )
        if self.x_axis == self.y_axis:
            raise ValueError(
                f"x_axis and y_axis must be different (both are {self.x_axis})"
            )

        # redundant copies to make the code more readable

        grid_res = sim_limits.params.grid_res
        cell_size = sim_limits.params.cell_size

        x_axis = self.x_axis
        y_axis = self.y_axis
        x_axis_name = "xyz"[self.x_axis]
        y_axis_name = "xyz"[self.y_axis]

        x_start = self.x_range[0]
        x_end = self.x_range[1]
        y_start = self.y_range[0]
        y_end = self.y_range[1]

        # wrap negatives
        if x_start < 0:
            x_start += grid_res[x_axis]
        if x_end < 0:
            x_end += grid_res[x_axis]
        if y_start < 0:
            y_start += grid_res[y_axis]
        if y_end < 0:
            y_end += grid_res[y_axis]

        # bounds checking
        if x_start < 0 or x_start >= grid_res[x_axis]:
            raise IndexError(
                f"invalid range ({x_axis_name} start = {x_start}, {grid_res=}, "
                f"{self=})."
            )
        if x_end < 0 or x_end >= grid_res[x_axis]:
            raise IndexError(
                f"invalid range ({x_axis_name} end = {x_end}, {grid_res=}, "
                f"{self=})."
            )
        if y_start < 0 or y_start >= grid_res[y_axis]:
            raise IndexError(
                f"invalid range ({y_axis_name} start = {y_start}, {grid_res=}, "
                f"{self=})."
            )
        if y_end < 0 or y_end >= grid_res[y_axis]:
            raise IndexError(
                f"invalid range ({y_axis_name} end = {y_end}, {grid_res=}, "
                f"{self=})."
            )

        # offset by 0.5 to go from cell center to edge
        x_start -= asymmetric_sign(x_end - x_start) * .5
        x_end += asymmetric_sign(x_end - x_start) * .5
        y_start -= asymmetric_sign(y_end - y_start) * .5
        y_end += asymmetric_sign(y_end - y_start) * .5

        # make quad

        unused_axis = list(filter(
            lambda axis: axis != x_axis and axis != y_axis,
            range(3)
        ))[0]

        origin = [0.] * 3
        origin[x_axis] = x_start
        origin[y_axis] = y_start
        origin[unused_axis] = self.unused_axis_index
        origin = sim_limits.icoord_to_world(tuple(origin))

        right = [0.] * 3
        right[x_axis] = (x_end - x_start) * cell_size

        up = [0.] * 3
        up[y_axis] = (y_end - y_start) * cell_size

        quad = Quadrilateral(origin, right, up)

        # calculate aspect ratio
        aspect_ratio = np.abs(x_end - x_start) / np.abs(y_end - y_start)

        return (quad, aspect_ratio)


class RenderMode(IntEnum):
    Raymarching = 0
    "3D volumetric rendering"

    Slice = 1
    "basic 2D slice"


@dataclass
class RenderCommand:
    """
    a set of parameters defining how to render a simulation frame.

    NOTE:
    the following WGSL constants, uniform values, and functions are available in
    the fragment shader and therefore in shade_cell() and colormaps.wgsl:

    - math constants
        PI, TAU, HALF_PI: f32

    - whether the averaging buffer is used for rendering
        RENDER_USES_AVERAGING_BUFFER: bool

    - simulation grid resolution
        GRID_RES: vec3i

    - simulation grid cell (voxel) size
        CELL_SIZE: f32

    - dimensions of the simulation grid (cell_size * grid_res)
        GRID_DIMS: vec3f

    - wave propagation speed
        WAVE_SPEED: f32

    - remove reflections at the grid boundaries
        REMOVE_REFLECTIONS: bool

    - resolved timestep
        TIMESTEP: f32

    - maximum stable timestep
        MAX_TIMESTEP: f32

    - minimum stable wavelength
        MIN_WAVELENGTH: f32

    - maximum stable frequency
        MAX_FREQ: f32

    - used internally for removing reflections at the grid boundaries
        IMPEDANCE_MATCHING_COEFFICIENT: f32

    - render resolution
        ubo.res: vec2i

    - render target resolution (implementation detail)
        ubo.total_res: vec2i

    - render mode, one of [RENDER_MODE_RAYMARCHING, RENDER_MODE_SLICE]
        ubo.mode: i32

    - same as RenderCommand.region.resolve_with_span(grid_res)[0]
        ubo.pmin: vec3i

    - same as RenderCommand.region.resolve_with_span(grid_res)[1]
        ubo.pmax: vec3i

    - same as RenderCommand.region.resolve_with_span(grid_res)[2]
        ubo.region_span: vec3i

    - bottom-back-left corner of the region in world-space coordinates
        ubo.pmin_world: vec3f

    - top-front-right corner of the region in world-space coordinates
        ubo.pmax_world: vec3f

    - same as RenderCommand.slice.resolve(...)[0].origin
        ubo.slice_quad_origin: vec3f

    - same as RenderCommand.slice.resolve(...)[0].right
        ubo.slice_quad_right: vec3f

    - same as RenderCommand.slice.resolve(...)[0].up
        ubo.slice_quad_up: vec3f

    - same as RenderCommand.slice.resolve(...)[1]
        ubo.slice_aspect_ratio: f32

    - same as RenderCommand.bg_col
        ubo.bg_col: vec3f

    - same as RenderCommand.n_samples_per_pixel
        ubo.n_samples_per_pixel: i32

    - same as RenderCommand.raymarch_step
        ubo.raymarch_step: f32

    - same as RenderCommand.raymarch_step_jitter
        ubo.raymarch_step_jitter: f32

    - same as int(RenderCommand.use_trilinear)
        ubo.use_trilinear: i32

    - same as RenderCommand.cam.pos
        ubo.cam_pos: vec3f

    - same as RenderCommand.cam.lookat
        ubo.cam_lookat: vec3f

    - same as RenderCommand.cam.world_up
        ubo.cam_world_up: vec3f

    - same as RenderCommand.cam.fov_degrees
        ubo.cam_fov_degrees: f32

    - same as int(RenderCommand.apply_flim)
        ubo.apply_flim: i32

    - simulation iteration
        ubo.iter: i32

    - simulation time
        ubo.time: f32

    - real world time
        ubo.wall_time: f32

    - this function converts 3D indices in the grid to physical coordinates in
      the simulation world based on GRID_RES and GRID_DIMS. it's centered around
      the origin (0, 0, 0) so if e.g. GRID_DIMS=(1, 1, 1) then the coordinates
      will range from (-0.5, -0.5, -0.5) to (+0.5, +0.5, +0.5).
        fn icoord_to_world(icoord: vec3i) -> vec3f {...}

    - custom data sent from the CPU (see `WaveSimParams.user_data_fields`)
        user_data.X
    """

    res: tuple[int, int] = (800, 600)
    "image resolution"

    mode: RenderMode = RenderMode.Raymarching
    "render mode"

    region: Aabb = field(default_factory=Aabb)
    """
    3D AABB (axis-aligned bounding box) describing the visible region. any grid
    cell outside of this region will not be rendered.
    """

    slice: GridSlice = field(default_factory=GridSlice)
    "which 2D slice to render in slice mode"

    try_use_averaging_buffer: bool = True
    """
    use the averaging buffer if available. if averaging if disabled in
    WaveSimParams, the original simulation grids will be used instead.
    NOTE: the averaging buffer works on squares of the values of the original
    grid (intensity vs. raw amplitude) so if your shade_cell() function or
    colormap squares the value internally, make sure it only does so if the
    RENDER_USES_AVERAGING_BUFFER constant is set to false (to avoid squaring
    twice).
    """

    bg_col: tuple[float, float, float] = (.02, .005, .02)
    "background color, additively blended in"

    shade_cell_function: str = """
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {
    return colormap_simple(v * 5.);
}"""
    """
    WGSL function used in the fragment shader returning a color for every grid
    cell based on its stored value (v) and coordinates (icoord) as well as
    other constants or uniform values available in the fragment shader
    (mentioned above). this function is useful for colorizing the grid values
    with colormaps and adding visual indicators for obstacles, lenses, etc.
    NOTE: try to avoid modifying this function every frame, as it may cause a
    recompilation of the fragment shader and render pipeline.
    """

    n_samples_per_pixel: int = 4
    "the number of jittered samples per pixel for anti-aliasing"

    raymarch_step: float = .02
    "step size for raymarching"

    raymarch_step_jitter: float = .015
    "step size jitter for raymarching"

    use_trilinear: bool = True
    """
    use trilinear (or bilinear for 2D) interpolation/filtering when sampling the
    simulation grid instead of nearest-neighbor.
    """

    cam: CameraState = field(default_factory=CameraState)
    "3D camera state. not used in 2D."

    apply_flim: bool = True
    """
    rendering: whether to apply the flim view transform for prettier results.
    see https://github.com/bean-mhm/flim
    """

    export_png_path: Path | None = None
    """
    optional path for exporting a PNG file.
    NOTE: if you need to export a PNG sequence, you must manually format the
    path (e.g. include the frame number) in your WaveSimParams.on_update
    function.
    """


WaveSimReadbackFunction = Callable[
    [
        Aabb,  # region
        bool  # read_averaging_grid
    ],
    np.ndarray  # returned ndarray
]
"""
a function you may call to read back the simulation grid data to the CPU as a
`numpy.ndarray` and do with it what you will.

Args:
    region (Aabb):
        an integer axis-aligned bounding box describing which region to read
        from the grid.

    read_averaging_grid (bool):
        whether to read from the averaging buffer instead of the original grid.
        this can only be enabled if averaging=True.

NOTE: in the simulation grid, each cell contains two numbers: its current value
(`curr`) and its value in the previous iteration (`prev`). therefore, the last
axis in the returned ndarray will have a size of 2. you can access the first
element (index 0) for `curr` and the second one for `prev`. this is NOT true for
the averaging buffer, as it only has a single channel.

NOTE: calling this every iteration will slow down the simulation.
"""


class WaveSimOnUpdateReturn(NamedTuple):
    render_commands: list[RenderCommand] = []
    "render commands"

    display_render_idx: int = 0
    """
    index of the render command whose output should be displayed on the window
    (negative if none).
    """

    new_user_data: memoryview | None = None
    """
    optional new value for user_data in the compute and fragment shaders. see
    `WaveSimParams.user_data_fields` and `WaveSimParams.user_data` to learn
    more.

    NOTE: resizing is allowed and this is allowed to have a different size than
    the original `WaveSimParams.user_data` but only if your
    `WaveSimParams.user_data_fields` ends with a variable-length array.
    """

    should_stop: bool = False
    "whether to stop the simulation"


@dataclass
class WaveSimParams:
    """
    wave simulation parameters.

    NOTE:
    the following WGSL constants, uniform values, and functions are available
    in the simulation compute shader and therefore in user-provided WGSL
    functions.

    - math constants
        PI, TAU, HALF_PI: f32

    - simulation grid resolution
        GRID_RES: vec3i

    - simulation grid cell (voxel) size
        CELL_SIZE: f32

    - dimensions of the simulation grid (cell_size * grid_res)
        GRID_DIMS: vec3f

    - wave propagation speed
        WAVE_SPEED: f32

    - remove reflections at the grid boundaries
        REMOVE_REFLECTIONS: bool

    - resolved timestep
        TIMESTEP: f32

    - maximum stable timestep
        MAX_TIMESTEP: f32

    - minimum stable wavelength
        MIN_WAVELENGTH: f32

    - maximum stable frequency
        MAX_FREQ: f32

    - used internally for removing reflections at the grid boundaries
        IMPEDANCE_MATCHING_COEFFICIENT: f32

    - simulation iteration
        ubo.iter: i32

    - simulation time
        ubo.time: f32

    - real world time
        ubo.wall_time: f32

    - this function converts 3D indices in the grid to physical coordinates in
      the simulation world based on GRID_RES and GRID_DIMS. it's centered around
      the origin (0, 0, 0) so if e.g. GRID_DIMS=(1, 1, 1) then the coordinates
      will range from (-0.5, -0.5, -0.5) to (+0.5, +0.5, +0.5).
        fn icoord_to_world(icoord: vec3i) -> vec3f {...}

    - this is the data stored in every cell:
        struct WaveValue {
            curr: f32, // value in the current iteration
            prev: f32, // value in the previous iteration
        };

    - this function fetches the data stored in an arbitrary cell
        fn grid_fetch(icoord: vec3i) -> WaveValue {...}

    - custom data sent from the CPU (see `WaveSimParams.user_data_fields` to
      learn more)
        user_data.X
    """

    grid_res: tuple[int, int, int]
    """
    simulation grid resolution. this is always a tuple of 3 integers, and for
    a 2D simulation, at least one axis must be set to 1 (e.g. (500, 500, 1)).
    """

    cell_size: float
    "simulation grid cell (voxel) size"

    wave_speed: float
    "wave propagation speed"

    remove_reflections: bool
    "remove reflections at the grid boundaries"

    timestep: float
    """
    delta time. if negative, will be a factor of the maximum stable timestep,
    e.g. -0.5 means 0.5 * max_timestep. the final value can be found in
    WaveSimLimits.resolved_timestep.
    """

    wgsl_common_header: str
    """
    user-provided WGSL code added to the simulation compute shader and the
    fragment shader used for rendering. useful for sharing code between
    simulation-related functions (below) and shade_cell().
    """

    initial_value_function: str
    """
    WGSL function used in the compute shader that defines how the grid is
    initialized in iteration 0. the function signature must not be modified.
    see `sim.wgsl` and `config.py`.
    """

    update_value_function: str
    """
    WGSL function used in the compute shader for applying custom excitations
    or modifications to the grid/field. it must return the new current
    ("curr") value for every cell. the function signature must not be
    modified. see `sim.wgsl` and `config.py`.
    """

    speed_fac_function: str
    """
    WGSL function used in the compute shader returning the wave propagation
    speed factor for every cell, useful for defining obstacles, reflectors, or
    lenses (refractors). must return values in the [0, 1] range for a stable
    simulation.
    """

    damp_fac_function: str
    """
    WGSL function used in the compute shader returning the dampening factor
    per second for every cell. this factor must be in the [0, 1] range and
    defines how fast the wave decays in every cell. for example, a dampening
    factor of 0.8 for a cell means the time derivative of its wave value will
    be scaled by 0.8 every second in simulation time or by (0.8^timestep)
    every iteration.
    """

    user_data_fields: str | None
    """
    optional string containing WGSL fields for sending custom data from the CPU,
    available in both the compute and fragment shaders (and therefore
    `wgsl_common_header`, `initial_value_function`, `shade_cell_function` in
    `RenderCommand`, etc.) by the name `user_data` (for example,
    `user_data.my_custom_parameter`).

    the syntax is exactly the same as inside any other struct block in WGSL.
    example:
    ```
        light_sensor: f32,
        microphone_input: array<f32>  // OK: ends with a variable-length array
    ```
    """

    user_data: memoryview | None
    """
    initial user data. only required if `user_data_fields` is provided. note
    that you can update this throughout the simulation by setting
    `new_user_data` when returning a `WaveSimOnUpdateReturn` from your
    `on_update` function.

    NOTE:
    please read about
    [WGSL memory alignment](https://www.w3.org/TR/WGSL/#memory-layouts) so you
    can choose the right type, size, and ordering for your fields. also make
    sure the total size is a multiple of 4 bytes.
    """

    averaging: bool
    """
    apply exponential smoothing to the field intensity (v^2, as opposed to
    v which is the raw amplitude) in a separate "averaging buffer".
    """

    averaging_time_constant: float
    """
    when averaging is enabled, the averaging buffer will approach the newest
    values by 63% (1 - 1/e) every averaging_time_constant seconds in
    simulation time.
    """

    on_start: Callable[
        [WaveSimParams, WaveSimLimits],
        None
    ] | None
    """
    an optional user-provided callback called just before the simulation starts.
    """

    on_update: Callable[
        [
            WaveSimParams,
            WaveSimLimits,
            WaveSimState,
            WaveSimReadbackFunction
        ],
        WaveSimOnUpdateReturn
    ]
    """
    a user-provided callback called after every iteration. it must return a
    list of `RenderCommand`s, an integer defining which of those renders to
    display (negative if none), and a boolean defining whether to stop the
    simulation.
    example signature:
    ```
        def sim_on_update(
            params: WaveSimParams,
            limits: WaveSimLimits,
            state: WaveSimState,
            readback_function: WaveSimReadbackFunction
        ) -> WaveSimOnUpdateReturn
    ```
    """

    def __deepcopy__(self, memo):
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result

        for name, value in self.__dict__.items():
            # do not deepcopy memoryview objects
            if type(value) == memoryview:
                setattr(result, name, value)
            else:
                setattr(result, name, deepcopy(value, memo))

        return result


class WaveSimLimits:
    params: WaveSimParams

    grid_dims: tuple[float, float, float]
    "dimensions of the simulation grid (cell_size * grid_res)"

    resolved_timestep: float
    "resolved timestep"

    max_timestep: float
    "maximum stable timestep"

    min_wavelength: float
    "minimum stable wavelength"

    max_freq: float
    "maximum stable frequency"

    impedance_matching_coefficient: float
    "used internally for removing reflections at the grid boundaries"

    averaging_mix_fac_per_dt: float
    "used in the averaging shader for exponential smoothing"

    def __init__(self, params: WaveSimParams):
        self.params = deepcopy(params)

        if any(map(lambda v: v < 1, params.grid_res)):
            raise ValueError(
                f"grid resolution {str(params.grid_res)} must be positive and "
                "non-zero in every axis."
            )

        dimensionality = 3
        if params.grid_res[0] == 1:
            dimensionality -= 1
        if params.grid_res[1] == 1:
            dimensionality -= 1
        if params.grid_res[2] == 1:
            dimensionality -= 1

        sqrt_dimensionality = np.sqrt(float(dimensionality))
        if dimensionality == 0:
            sqrt_dimensionality = 1e-9

        self.grid_dims = (
            params.cell_size * params.grid_res[0],
            params.cell_size * params.grid_res[1],
            params.cell_size * params.grid_res[2]
        )

        self.max_timestep = \
            params.cell_size / (params.wave_speed * sqrt_dimensionality)

        self.resolved_timestep = params.timestep
        if self.resolved_timestep < 0:
            self.resolved_timestep = -self.resolved_timestep * self.max_timestep

        self.min_wavelength = params.cell_size * sqrt_dimensionality * 8.
        self.max_freq = params.wave_speed / self.min_wavelength

        self.impedance_matching_coefficient = (
            params.wave_speed * self.resolved_timestep - params.cell_size
        ) / (params.wave_speed * self.resolved_timestep + params.cell_size)

        if params.averaging:
            self.averaging_mix_fac_per_dt = 1. - np.exp(
                -self.resolved_timestep / params.averaging_time_constant
            )
        else:
            self.averaging_mix_fac_per_dt = 1.

    def icoord_to_world(
        self,
        p: tuple[int, int, int]
    ) -> tuple[float, float, float]:
        return tuple([
            (
                (float(p[i]) + .5)
                / float(self.params.grid_res[i])
                - .5
            ) * self.grid_dims[i]
            for i in range(len(p))
        ])

    def recalculate(self, params: WaveSimParams):
        self.__init__(params)


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
        resolved_path = Path(__file__).parent / ".debug" /  \
            ("resolved_" + filename_relative)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved_path, "w") as f:
            f.write(code)
            f.close()

    module: wgpu.GPUShaderModule | None = None
    err: Exception | None = None
    try:
        module = device.create_shader_module(code=code)
    except Exception as e:
        err = RuntimeError(
            f"failed to compile shader \"{filename_relative}\": {e}"
        )

    if (err is None and STORE_RESOLVED_SHADER_CODE == StoreShaderWhen.Always) \
            or (err is not None and STORE_RESOLVED_SHADER_CODE != StoreShaderWhen.Never):
        resolved_path = Path(__file__).parent / ".debug" /  \
            ("resolved_" + filename_relative)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved_path, "w") as f:
            f.write(code)
            f.close()

    if err is None:
        return module
    else:
        raise err


def prepare_buffers(
    device: wgpu.GPUDevice,
    bufs: list[wgpu.GPUBuffer | None],
    labels: list[str],
    usage_flags: list[wgpu.flags.BufferUsageFlags],
    min_sizes: list[int]
) -> list[wgpu.GPUBuffer]:
    """
    (re)create given buffers to make sure they exist and have enough size
    """

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


class CpuToGpuBuffer:
    """
    a wrapper for creating wgpu buffers where we only send data from the CPU to
    the GPU and not the other way around.

    Args:

        device (wgpu.GPUDevice):
            wgpu device

        label (str):
            buffer label used in wgpu error messages for debugging

        usage (wgpu.BufferUsage):
            buffer usage. one of `wgpu.BufferUsage.INDEX`,
            `wgpu.BufferUsage.VERTEX`, `wgpu.BufferUsage.UNIFORM`, or
            `wgpu.BufferUsage.STORAGE`.

        data_view (memoryview):
            a memory view to the initial data to write to the buffer.

            NOTE: to avoid headaches around memory alignment, make sure the size
            is a multiple of 4 bytes.
    """

    _device: wgpu.GPUDevice
    _label: str
    _usage: wgpu.BufferUsage

    _data_view_i32: memoryview[int] | None = None
    _data_size: int = 0
    _buf: wgpu.GPUBuffer | None = None
    _staging_buf: wgpu.GPUBuffer | None = None

    @property
    def buf(self) -> wgpu.GPUBuffer:
        """the underlying wgpu buffer"""
        return self._buf

    def __init__(
        self,
        device: wgpu.GPUDevice,
        label: str,
        usage: wgpu.BufferUsage,
        data_view: memoryview
    ):
        self._device = device
        self._label = label
        self._usage = usage
        self.set_data_view(data_view)
        self.upload()

    def set_data_view(self, data_view: memoryview):
        """
        change the memory view for your data. you don't need to call this every
        time you change your data. it should only be called when the actual
        object you're using to store the values (e.g. numpy struct) is now a
        different object.
        """

        # prevent empty views
        if data_view.nbytes < 1:
            raise ValueError("set_data_view was provided an empty data_view")

        # force 4-byte alignment
        if data_view.nbytes % 4 != 0:
            raise ValueError(
                "the size of data_view must be a multiple of 4 bytes"
            )

        # recreate GPU buffers if needed
        if not self._data_view_i32 or (
            self._data_view_i32
            and data_view.nbytes != self._data_size
        ):
            self._buf = self._device.create_buffer(
                label=self._label,
                size=data_view.nbytes,
                usage=self._usage | wgpu.BufferUsage.COPY_DST,
                mapped_at_creation=False
            )
            self._staging_buf = self._device.create_buffer(
                label=self._label + " (staging buffer)",
                size=data_view.nbytes,
                usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
                mapped_at_creation=False
            )

        self._data_view_i32 = data_view.cast("B")
        self._data_size = data_view.nbytes

    # only pushes GPU commands, does not run them
    def push_upload_command(self, cmd_encoder: wgpu.GPUCommandEncoder):
        self._staging_buf.map_sync(wgpu.MapMode.WRITE)
        self._staging_buf.write_mapped(self._data_view_i32)
        self._staging_buf.unmap()

        cmd_encoder.copy_buffer_to_buffer(
            self._staging_buf,
            0,
            self._buf,
            0,
            self._data_size
        )

    def upload(self):
        cmd_encoder = self._device.create_command_encoder()
        self.push_upload_command(cmd_encoder)
        self._device.queue.submit([cmd_encoder.finish()])


def field_offset_in_numpy_dtype(
    dtype: np.dtype,
    fields: list[str],
    alignment: int = 1
) -> tuple[int, int]:
    """
    returns the start (inclusive) and end (exclusive) offset (in bytes) of given
    fields (unioned) in a numpy data structure.

    NOTE: the fields must be in the same order as in the original numpy.dtype.
    """
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


def asymmetric_sign(v: float) -> float:
    return 1. if v >= 0. else -1.


def dict_remove_n_oldest(d: dict, n: int):
    """remove the first n entries from a dict (oldest by insertion order)"""
    keys = list(d.keys())[:n]
    for key in keys:
        d.pop(key)
