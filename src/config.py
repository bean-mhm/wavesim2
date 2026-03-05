from typing import NamedTuple
from collections.abc import Callable
import numpy as np

DISPLAY_SCALE = 1.5
WINDOW_TITLE = "wavesim2 (time: {:.3f}, iter: {})"

# store source code for shaders with replacements applied. useful for debugging,
# especially when we need to go to a certain line number.
STORE_RESOLVED_SHADER_CODE = False

# WGSL colormap functions used in the fragment shader, accessible from
# WaveSimParams.render_shade_cell_function.
WGSL_COLORMAPS = """
    fn colormap_simple(v: f32) -> vec3f {
        return mix(
            vec3f(0, .3, 1) * -v,
            vec3f(1, .4, 0) * v,
            saturate(sign(v))
        );
    }

    fn colormap_grayscale_positive_only(v: f32) -> vec3f {
        return vec3f(max(v, 0.));
    }

    fn colormap_grayscale_abs(v: f32) -> vec3f {
        return vec3f(abs(v));
    }

    fn colormap_squared_jet(v: f32) -> vec3f {
        return vec3f(1, 0, .5) * v * v;
    }

    fn colormap_squared_grayscale(v: f32) -> vec3f {
        return vec3f(v * v);
    }
"""


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
    pos: tuple[float, float, float] = (0., -1., 0.)
    lookat: tuple[float, float, float] = (0., 0., 0.)
    world_up: tuple[float, float, float] = (0., 0., 1.)
    fov_degrees: float = 90.


class WaveSimParams(NamedTuple):
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

    # WGSL function used in the compute shader returning the wave propagation
    # speed factor for every cell, useful for defining obstacles, reflectors, or
    # lenses (refractors). must return values in the [0, 1] range for a stable
    # simulation.
    speed_fac_function: str

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
    #
    # for more advanced use cases, you've also got this function:
    #   fn grid_fetch(icoord: vec3i) -> WaveValue {...}

    # rendering: background color
    render_bg_col: tuple[float, float, float]

    # WGSL function used in the fragment shader returning a color for every grid
    # cell based on its stored value (v) and coordinates (icoord) as well as the
    # constants and uniform values mentioned below. this function is useful for
    # colorizing the grid values with colormaps and adding visual indicators for
    # obstacles, lenses, etc.
    #
    # NOTE:
    # the following WGSL constants and uniform values are accessible in the
    # fragment shader and therefore this function:
    #
    # math constants
    #   PI, TAU, HALF_PI: f32
    #
    # simulation grid resolution
    #   SIM_GRID_RES: vec3i
    #
    # same as (GRID_RES.z == 1)
    #   SIM_IS_2D: bool
    #
    # dimensions of the simulation grid (cell_size * grid_res)
    #   SIM_GRID_DIMS: vec3f
    #
    # simulation iteration
    #   ubo.sim_iter: i32
    #
    # simulation time
    #   ubo.sim_time: f32
    render_shade_cell_function: str

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


sim_test = WaveSimParams(
    render_res=(1280, 720),
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
    if (all(icoord == GRID_RES / 2)) {
        let freq = .9 * MAX_FREQ;
        return 10. * sin(TAU * ubo.time * freq);
    }
    return v.curr;
}
    """,
    speed_fac_function="""
fn speed_fac(icoord: vec3i, v: WaveValue) -> f32 {
    if (all(abs(icoord - vec3i(85, 50, 50)) <= vec3i(6))) {
        return 0.;
    }
    return 1.;
}
    """,
    render_bg_col=(.02, .005, .02),
    render_shade_cell_function="""
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {
    return colormap_simple(v);
}
    """,
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

sim_2d_planar_wave = WaveSimParams(
    render_res=(500, 500),
    grid_res=(500, 500, 1),
    cell_size=.001,
    wave_speed=.1,
    remove_reflections=True,
    damp_fac=.95,
    timestep=-.5,
    n_sim_steps_per_frame=2,
    initial_value_function="""
fn initial_value(icoord: vec3i) -> WaveValue {
    return WaveValue(0, 0);
}
    """,
    update_value_function="""
fn update_value(icoord: vec3i, v: WaveValue) -> f32 {
    if (icoord.x == GRID_RES.x / 5
        && abs(f32(icoord.y) / f32(GRID_RES.y) - .5) < .2) {
        let freq = .9 * MAX_FREQ;
        let v_new = 3. * sin(TAU * ubo.time * freq);
        return mix(
            v.curr,
            v_new,
            remap_clamp(ubo.time, 2., 2.5, 1., 0.)
        );
    }
    return v.curr;
}
    """,
    speed_fac_function="""
fn speed_fac(icoord: vec3i, v: WaveValue) -> f32 {
    if (all(abs(icoord - vec3i(400, 250, 0)) <= vec3i(6))) {
        return 0.;
    }
    return 1.;
}
    """,
    render_bg_col=(.01, 0., .02),
    render_shade_cell_function="""
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {
    return colormap_squared_grayscale(v * .5);
}
    """,
    render_n_samples_per_pixel=16,
    render_raymarch_step=0,
    render_raymarch_step_jitter=0,
    render_use_trilinear=True,
    render_camera_function=lambda params, limits, state: CameraState(),
    render_apply_flim=True
)


# choose which simulation to run from above
selected_sim_params = sim_2d_planar_wave
selected_sim_limits = WaveSimLimits(selected_sim_params)
