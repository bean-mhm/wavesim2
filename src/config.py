import time
from dataclasses import dataclass
from collections.abc import Callable
from copy import deepcopy
import numpy as np

DISPLAY_SCALE = 1.
WINDOW_TITLE = "wavesim2 (time: {:.3f}, iter: {})"

# store source code for shaders with replacements applied. useful for debugging,
# especially when we need to go to a certain line number.
STORE_RESOLVED_SHADER_CODE = False

# WGSL colormap functions used in the fragment shader, accessible from
# WaveSimParams.render_shade_cell_function.
WGSL_COLORMAPS = """
    fn colormap_simple(v: f32) -> vec3f {
        return mix(
            vec3f(0, .4, 1) * -v,
            vec3f(1, .4, 0) * v,
            saturate(sign(v))
        );
    }

    fn colormap_exp(v: f32, rgb_fac: vec3f) -> vec3f {
        var sq = v;
        if (!AVERAGING) {
            sq *= sq;
        }
        return vec3f(
            1. - exp(-sq * rgb_fac.r),
            1. - exp(-sq * rgb_fac.g),
            1. - exp(-sq * rgb_fac.b)
        );
    }

    fn colormap_fire(v: f32) -> vec3f {
        return 1.5 * colormap_exp(v, vec3f(1., .15, .01));
    }

    fn colormap_grayscale_positive_only(v: f32) -> vec3f {
        return vec3f(max(v, 0.));
    }

    fn colormap_grayscale_abs(v: f32) -> vec3f {
        return vec3f(abs(v));
    }

    fn colormap_grayscale_squared(v: f32) -> vec3f {
        var sq = v;
        if (!AVERAGING) {
            sq *= sq;
        }
        return vec3f(sq);
    }

    // https://www.desmos.com/calculator/n4mfhffj1n
    fn _jetski_f(x: f32, v_: f32) -> f32 {
        var v = v_;
        if (abs(v) < .0001) { v = .0001; }
        let p = pow(2., v);
        return (1. - pow(p, -x)) / (1. - 1. / p);
    }

    // https://www.shadertoy.com/view/DdcyRf
    fn colormap_jetski(v: f32) -> vec3f {
        var x = v;
        if (!AVERAGING) {
            x *= x;
        }

        let t = .6 + .8 * x;

        // https://www.desmos.com/calculator/sdqk904uu9
        let tone = 9. * vec3f(
            cos(6.283 * t),
            cos(6.283 * (t - .333)),
            cos(6.283 * (t - .667))
        );

        x = smoothstep(-.04, 1., x);
        let c = vec3f(
            _jetski_f(x, tone.r),
            _jetski_f(x, tone.g),
            _jetski_f(x, tone.b)
        );

        return c;
    }
"""


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
    # the following WGSL constants, uniform values, and functions are accessible
    # in the simulation compute shader and therefore also the WGSL functions
    # above:
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
    render_shade_cell_function: str

    # NOTE:
    # the following WGSL constants, uniform values, and functions are accessible
    # in the fragment shader and therefore in render_shade_cell_function:
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


# basic 3D simulation with a sine wave source at the center
sim1_basic = WaveSimParams(
    render_res=(1280, 720),
    grid_res=(101, 101, 101),
    cell_size=.01,
    wave_speed=.05,
    remove_reflections=False,
    damp_fac=.9,
    timestep=-.5,
    n_sim_steps_per_frame=1,
    initial_value_function="""
fn initial_value(icoord: vec3i) -> WaveValue {
    return WaveValue(0, 0);
}
    """,
    update_value_function="""
fn update_value(icoord: vec3i, v: WaveValue) -> f32 {
    // sine wave source at the center
    if (all(icoord == GRID_RES / 2)) {
        let freq = .9 * MAX_FREQ;
        let amp = remap_clamp(ubo.time, 8., 10., 10., 0.);
        return amp * sin(TAU * ubo.time * freq);
    }
    return v.curr;
}
    """,
    speed_fac_function="""
fn speed_fac(icoord: vec3i, v: WaveValue) -> f32 {
    return 1.;
}
    """,
    averaging=False,
    averaging_time_constant=0.,
    render_bg_col=(.02, .005, .02),
    render_shade_cell_function="""
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {
    return colormap_simple(v * 4.);
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


# same as sim #1 but with a box-shaped obstacle
sim2_box_obstacle = deepcopy(sim1_basic)
sim2_box_obstacle.speed_fac_function = """
fn speed_fac(icoord: vec3i, v: WaveValue) -> f32 {
    // add a small box-shaped obstacle
    if (all(abs(icoord - vec3i(86, 50, 50)) <= vec3i(10, 20, 10))) {
        return 0.;
    }
    return 1.;
}
"""


# same as sim #2 but with a rotating camera around the origin
sim3_rotating_camera = deepcopy(sim2_box_obstacle)
sim3_rotating_camera.render_camera_function = \
    lambda params, limits, state: \
    CameraState(
        pos=(
            1.2 * np.cos(2. * np.pi * .07 * state.wall_time),
            1.2 * np.sin(2. * np.pi * .07 * state.wall_time),
            .03
        ),
        lookat=(0., 0., 0.),
        world_up=(0., 0., 1.),
        fov_degrees=80.
    )


# make the obstacle more noticable by modifying render_shade_cell_function
sim4_highlighted_obstacle = deepcopy(sim3_rotating_camera)
sim4_highlighted_obstacle.render_shade_cell_function = """
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {
    var col = colormap_simple(v * 4.);
    if (all(abs(icoord - vec3i(86, 50, 50)) <= vec3i(7, 20, 7))) {
        col = col * .8 + vec3f(.3, 0, .35);
    }
    return col;
}
"""


# remove reflections at the boundaries
sim5_remove_reflections = sim4_highlighted_obstacle.__replace__(
    remove_reflections=True
)


# replace the sine wave source with a constant but moving source
sim6_moving_source = deepcopy(sim5_remove_reflections)
sim6_moving_source.update_value_function = """
fn update_value(icoord: vec3i, v: WaveValue) -> f32 {
    let coord = icoord_to_world(icoord);

    let source = .05 * vec3f(
        0,
        cos(TAU * .2 * MAX_FREQ * ubo.time),
        sin(TAU * .2 * MAX_FREQ * ubo.time)
    );

    let dist = distance(coord, source);
    
    let strength = .5 * smoothstep(0., 4., ubo.time);

    return mix(
        v.curr,
        strength,
        smoothstep(.03, .01, dist)
    );
}
"""


# two moving sources with opposite values
sim7_opposite_sources = deepcopy(sim6_moving_source)
sim7_opposite_sources.update_value_function = """
fn update_value(icoord: vec3i, v: WaveValue) -> f32 {
    let coord = icoord_to_world(icoord);

    let angle = TAU * .25 * MAX_FREQ * ubo.time;
    let psource = .05 * vec3f(
        0,
        cos(angle),
        sin(angle)
    );
    let nsource = .05 * vec3f(
        0,
        cos(angle + HALF_PI),
        sin(angle + HALF_PI)
    );

    let pdist = distance(coord, psource);
    let ndist = distance(coord, nsource);
    
    let strength = .5 * smoothstep(0., 5., ubo.time);

    var v_new = mix(
        v.curr,
        strength,
        smoothstep(.03, .01, pdist)
    );
    v_new = mix(
        v_new,
        -strength,
        smoothstep(.03, .01, ndist)
    );
    return v_new;
}
"""


# a 2D simulation with a planar wave source and a circular lens
sim8_2d_planar_wave_with_lens = WaveSimParams(
    render_res=(900, 750),
    grid_res=(600, 500, 1),
    cell_size=.001,
    wave_speed=.1,
    remove_reflections=True,
    damp_fac=.9,
    timestep=-.5,
    n_sim_steps_per_frame=1,
    initial_value_function="""
fn initial_value(icoord: vec3i) -> WaveValue {
    return WaveValue(0, 0);
}
    """,
    update_value_function="""
fn update_value(icoord: vec3i, v: WaveValue) -> f32 {
    if (icoord.x != GRID_RES.x / 6
        || abs(f32(icoord.y) / f32(GRID_RES.y) - .5) > .3) {
        return v.curr;
    }

    let freq = .9 * MAX_FREQ;
    let v_new = .8 * sin(TAU * ubo.time * freq);

    let mix_factor = remap01(ubo.time, 0., 1.) * remap01(ubo.time, 5.5, 4.5);
    return mix(
        v.curr,
        v_new,
        mix_factor
    );
}
    """,
    speed_fac_function="""
fn speed_fac(icoord: vec3i, v: WaveValue) -> f32 {
    // circular lens
    const LENS_CENTER = vec2f(0);
    const LENS_IOR = 1.05;
    let coord = icoord_to_world(icoord).xy;
    return remap_clamp(
        distance(coord, LENS_CENTER),
        .048,
        .046,
        1.,
        1. / LENS_IOR
    );
}
    """,
    averaging=False,
    averaging_time_constant=0.,
    render_bg_col=(0., 0., 0.),
    render_shade_cell_function="""
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {
    return colormap_jetski(v);
}
    """,
    render_n_samples_per_pixel=16,
    render_raymarch_step=0,
    render_raymarch_step_jitter=0,
    render_use_trilinear=True,
    render_camera_function=lambda params, limits, state: CameraState(),
    render_apply_flim=True
)


# enable expoential smoothing over time
sim9_2d_lens_with_averaging = sim8_2d_planar_wave_with_lens.__replace__(
    averaging=True,
    averaging_time_constant=1.
)


# a 2D simulation with a planar wave source and a wall with a small hole
single_slit_condition = "abs(coord.x) < .001 && abs(coord.y) > .005"
sim10_2d_single_slit = WaveSimParams(
    render_res=(960, 640),
    grid_res=(960, 640, 1),
    cell_size=.0005,
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
    if (icoord.x != GRID_RES.x / 5
        || abs(f32(icoord.y) / f32(GRID_RES.y) - .5) > .3) {
        return v.curr;
    }

    let freq = .6 * MAX_FREQ;
    let v_new = .9 * sin(TAU * ubo.time * freq);

    let mix_factor = remap01(ubo.time, 0., 1.) * remap01(ubo.time, 4., 3.5);
    return mix(
        v.curr,
        v_new,
        mix_factor
    );
}
    """,
    speed_fac_function=f"""
fn speed_fac(icoord: vec3i, v: WaveValue) -> f32 {{
    let coord = icoord_to_world(icoord).xy;
    if ({single_slit_condition}) {{
        return 0.;
    }}
    return 1.;
}}
    """,
    averaging=True,
    averaging_time_constant=1.,
    render_bg_col=(0., 0., 0.),
    render_shade_cell_function=f"""
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {{
    let coord = icoord_to_world(icoord).xy;
    if ({single_slit_condition}) {{
        return vec3f(1);
    }}
    return colormap_jetski(v);
}}
    """,
    render_n_samples_per_pixel=16,
    render_raymarch_step=0,
    render_raymarch_step_jitter=0,
    render_use_trilinear=True,
    render_camera_function=lambda params, limits, state: CameraState(),
    render_apply_flim=True
)


# a 2D simulation with a planar wave source and a wall with a small hole
double_slit_condition = """
    abs(coord.x) < .001
    && abs(coord.y - .02) > .003
    && abs(coord.y + .02) > .003
"""
sim11_2d_double_slit = WaveSimParams(
    render_res=(960, 640),
    grid_res=(960, 640, 1),
    cell_size=.0005,
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
    if (icoord.x != GRID_RES.x / 5
        || abs(f32(icoord.y) / f32(GRID_RES.y) - .5) > .3) {
        return v.curr;
    }

    let freq = .8 * MAX_FREQ;
    let v_new = .9 * sin(TAU * ubo.time * freq);

    let mix_factor = remap01(ubo.time, 0., 1.) * remap01(ubo.time, 6., 5.);
    return mix(
        v.curr,
        v_new,
        mix_factor
    );
}
    """,
    speed_fac_function=f"""
fn speed_fac(icoord: vec3i, v: WaveValue) -> f32 {{
    let coord = icoord_to_world(icoord).xy;
    if ({double_slit_condition}) {{
        return 0.;
    }}
    return 1.;
}}
    """,
    averaging=True,
    averaging_time_constant=1.,
    render_bg_col=(0., 0., 0.),
    render_shade_cell_function=f"""
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {{
    let coord = icoord_to_world(icoord).xy;
    if ({double_slit_condition}) {{
        return vec3f(1);
    }}
    return colormap_jetski(v);
}}
    """,
    render_n_samples_per_pixel=16,
    render_raymarch_step=0,
    render_raymarch_step_jitter=0,
    render_use_trilinear=True,
    render_camera_function=lambda params, limits, state: CameraState(),
    render_apply_flim=True
)


# 3D simulation with a planar wave source and a spherical lens
lens_params = """
    const LENS_CENTER = vec3f(-.05, 0, 0);
    const LENS_RADIUS = .16;
    const LENS_IOR = 1.1;
"""
sim12_3d_planar_with_lens = WaveSimParams(
    render_res=(1280, 640),
    grid_res=(400, 150, 200),
    cell_size=.003,
    wave_speed=.05,
    remove_reflections=True,
    damp_fac=.85,
    timestep=-.5,
    n_sim_steps_per_frame=2,
    initial_value_function="""
fn initial_value(icoord: vec3i) -> WaveValue {
    return WaveValue(0, 0);
}
    """,
    update_value_function="""
fn update_value(icoord: vec3i, v: WaveValue) -> f32 {
    // planar sine wave source at the center

    let coord = icoord_to_world(icoord);
    if (icoord.x != GRID_RES.x / 8) {
        return v.curr;
    }
    if (any(abs(coord.yz) > vec2f(.2, .25))) {
        return v.curr;
    }

    let v_new = 2.5 * sin(TAU * ubo.time * MAX_FREQ);
    return mix(
        v.curr,
        v_new,
        remap01(ubo.time, 0., 4.) * remap01(ubo.time, 32., 28.)
    );
}
    """,
    speed_fac_function=f"""
fn speed_fac(icoord: vec3i, v: WaveValue) -> f32 {{
    {lens_params}
    let coord = icoord_to_world(icoord);
    return remap_clamp(
        distance(coord, LENS_CENTER),
        LENS_RADIUS,
        LENS_RADIUS - .002,
        1.,
        1. / LENS_IOR
    );
}}
    """,
    averaging=True,
    averaging_time_constant=1.,
    render_bg_col=(.02, .005, .02),
    render_shade_cell_function=f"""
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {{
    // highlight the edges of the volume cube
    const EDGE_THICKNESS = 3;
    var n_edge = 0;
    if (icoord.x < EDGE_THICKNESS || icoord.x >= GRID_RES.x - EDGE_THICKNESS) {{
        n_edge++;
    }}
    if (icoord.y < EDGE_THICKNESS || icoord.y >= GRID_RES.y - EDGE_THICKNESS) {{
        n_edge++;
    }}
    if (icoord.z < EDGE_THICKNESS || icoord.z >= GRID_RES.z - EDGE_THICKNESS) {{
        n_edge++;
    }}
    if (n_edge >= 2) {{
        return vec3f(.5, 0, 3);
    }}

    var col = colormap_fire(v);

    // highlight the lens
    {lens_params}
    let coord = icoord_to_world(icoord);
    if (distance(coord, LENS_CENTER) < LENS_RADIUS) {{
        col += vec3f(0, .05, .12);
    }}

    return col;
}}
    """,
    render_n_samples_per_pixel=1,
    render_raymarch_step=.02,
    render_raymarch_step_jitter=.015,
    render_use_trilinear=True,
    render_camera_function=lambda params, limits, state:
    CameraState(
        pos=(
            .07 * np.cos(2. * np.pi * .12 * state.wall_time),
            -.68,
            .01 * np.sin(2. * np.pi * .284 * state.wall_time),
        ),
        lookat=(0., 0., 0.),
        world_up=(0., 0., 1.),
        fov_degrees=60.
    ),
    render_apply_flim=True
)


# choose which simulation to run from above
selected_sim_params = sim12_3d_planar_with_lens
selected_sim_limits = WaveSimLimits(selected_sim_params)
