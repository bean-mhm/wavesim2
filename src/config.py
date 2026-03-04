from typing import NamedTuple
from collections.abc import Callable
import numpy as np

RENDER_RES = (2160, 1080)
DISPLAY_SCALE = .6
WINDOW_TITLE = "wavesim2 (time: {:.3f}, iter: {})"

# store source code for shaders with replacements applied. useful for debugging,
# especially when we need to go to a certain line number.
STORE_RESOLVED_SHADER_CODE = False


class WaveSimState:
    # current iteration
    iter: int

    # simulation time
    time: float

    # for double buffering (alternating between buffer A and B)
    use_a_as_input: bool

    # for calculating time based on iter
    timestep: float

    def __init__(self, timestep: float):
        self.iter = 0
        self.time = 0.
        self.use_a_as_input = True
        self.timestep = timestep

    def advance(self, n_steps: int = 1):
        for _ in range(n_steps):
            self.use_a_as_input = not self.use_a_as_input
            self.iter += 1
            self.time = self.iter * self.timestep


class CameraState(NamedTuple):
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    world_up: tuple[float, float, float]
    fov_degrees: float


class WaveSimParams(NamedTuple):
    # simulation grid resolution
    grid_res: tuple[int, int, int]

    # simulation grid cell (voxel) size
    cell_size: float

    # wave propagation speed
    wave_speed: float

    # remove reflections at the grid boundaries
    remove_reflections: bool

    # dampening factor per second (stiff near zero and loose at 1)
    damp_fac: float

    # delta time. if negative, will be a factor of the maximum stable timestep,
    # e.g. -0.5 means 0.5 * max_timestep. the final value can be found in
    # WaveSimLimits.resolved_timestep.
    timestep: float

    # advance the simulation this many iterations before rendering each frame.
    n_sim_steps_per_frame: int

    # WGSL function used in the compute shader that defines how the grid is
    # initialized in iteration 0. the function signature must not be modified.
    # see "sim.wgsl" and sim_test below for an example.
    initial_value_function: str

    # WGSL function used in the compute shader for applying custom excitations
    # or modifications to the grid/field. it must return the new current
    # ("curr") value for every cell. the function signature must not be
    # modified. see "sim.wgsl" and sim_test below for an example.
    update_value_function: str

    # NOTE:
    # the following constants and uniform values are accessible in the WGSL
    # functions above:
    #
    # math constants
    #   PI, TAU, HALF_PI: f32
    #
    # simulation grid resolution
    #   GRID_RES: vec3i
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
    # dampening factor per second (stiff near zero and loose at 1)
    #   DAMP_FAC: f32
    #
    # dampening factor per timestep
    #   DAMP_FAC_PER_DT: f32
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
    # used internally for removing reflections at the grid boundaries
    #   IMPEDANCE_MATCHING_COEFFICIENT: f32
    #
    # this is the data stored in every cell:
    #   struct WaveValue {
    #       curr: f32, // value in the current iteration
    #       prev: f32, // value in the previous iteration
    #   };

    # rendering: background color
    render_bg_col: tuple[float, float, float]

    # rendering: volume color in positive areas
    render_tint_positive: tuple[float, float, float]

    # rendering: volume color in negative areas
    render_tint_negative: tuple[float, float, float]

    # rendering: volume brightness multiplier
    render_brightness: float

    # rendering: number of jittered samples per pixel for anti-aliasing
    render_n_samples_per_pixel: int

    # rendering: step size for ray marching
    render_raymarch_step: float

    # rendering: step size jitter for ray marching
    render_raymarch_step_jitter: float

    # rendering: use trilinear interpolation/filtering for sampling the
    # simulation grid instead of nearest-neighbor.
    render_use_trilinear: bool

    # rendering: Python function returning the camera state per frame
    render_camera_function: Callable[
        [WaveSimParams, WaveSimLimits, WaveSimState],
        CameraState
    ]

    # rendering: whether to apply the flim view transform for prettier results.
    # see https://github.com/bean-mhm/flim
    render_apply_flim: bool


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

    def __init__(self, params: WaveSimParams):
        self.grid_dims = (
            params.cell_size * params.grid_res[0],
            params.cell_size * params.grid_res[1],
            params.cell_size * params.grid_res[2]
        )

        self.max_timestep = \
            params.cell_size / (params.wave_speed * np.sqrt(3.))

        self.resolved_timestep = params.timestep
        if self.resolved_timestep < 0:
            self.resolved_timestep = \
                -self.resolved_timestep * self.max_timestep

        self.min_wavelength = params.cell_size * np.sqrt(3.) * 8.
        self.max_freq = params.wave_speed / self.min_wavelength

        self.impedance_matching_coefficient = \
            (params.wave_speed * self.resolved_timestep - params.cell_size) \
            / (params.wave_speed * self.resolved_timestep + params.cell_size)


sim_test = WaveSimParams(
    grid_res=(100, 100, 100),
    cell_size=.01,
    wave_speed=.05,
    remove_reflections=True,
    damp_fac=.95,
    timestep=-.5,
    n_sim_steps_per_frame=1,
    initial_value_function="""
fn initial_value(icoord: vec3i) -> WaveValue {
    return WaveValue(0, 0);
}
    """,
    update_value_function="""
fn update_value(icoord: vec3i, v: WaveValue) -> f32 {
    // initial pulse at the center
    if (all(icoord == GRID_RES / 2)) {
        let freq = .9 * MAX_FREQ;
        return 10. * sin(TAU * ubo.time * freq);
    }
    return v.curr;
}
    """,
    render_bg_col=(.02, .005, .02),
    render_tint_positive=(1., .4, 0.),
    render_tint_negative=(0., .3, 1.),
    render_brightness=5.,
    render_n_samples_per_pixel=4,
    render_raymarch_step=.018,
    render_raymarch_step_jitter=.015,
    render_use_trilinear=True,
    render_camera_function=lambda params, limits, state:
    CameraState(
        pos=(0., -1.25, .03),
        lookat=(0., 0., 0.),
        world_up=(0., 0., 1.),
        fov_degrees=70.
    ),
    render_apply_flim=True
)


# choose which simulation to run from above
selected_sim_params = sim_test
selected_sim_limits = WaveSimLimits(selected_sim_params)
