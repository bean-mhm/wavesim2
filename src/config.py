from copy import deepcopy
import numpy as np

from common import *


DISPLAY_SCALE = 1.
WINDOW_TITLE = "wavesim2 (time: {:.3f}, iter: {})"


# basic 3D simulation with a sine wave source at the center
sim1_basic = WaveSimParams(
    render_res=(1280, 720),
    grid_res=(101, 101, 101),
    cell_size=.01,
    wave_speed=.05,
    remove_reflections=False,
    timestep=-.5,
    n_sim_steps_per_frame=1,
    wgsl_common_header="",
    initial_value_function=constant_initial_value_function(0.),
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
    speed_fac_function=constant_speed_fac_function(1.),
    damp_fac_function=constant_damp_fac_function(.9),
    averaging=False,
    averaging_time_constant=0.,
    render_bg_col=(.02, .005, .02),
    shade_cell_function="""
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {
    return colormap_simple(v * 5.);
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
sim2_box_obstacle.damp_fac_function = """
fn damp_fac(icoord: vec3i, v: WaveValue) -> f32 {
    // add a small box-shaped obstacle
    if (all(abs(icoord - vec3i(86, 50, 50)) <= vec3i(10, 20, 10))) {
        return 0.;
    }
    return .9;
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


# make the obstacle more noticable by modifying shade_cell_function
sim4_highlighted_obstacle = deepcopy(sim3_rotating_camera)
sim4_highlighted_obstacle.shade_cell_function = """
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


# a 2D simulation with a planar wave source and a biconvex lens
sim8_2d_planar_wave_with_lens = WaveSimParams(
    render_res=(1200, 700),
    grid_res=(600, 350, 1),
    cell_size=.001,
    wave_speed=.1,
    remove_reflections=True,
    timestep=-.5,
    n_sim_steps_per_frame=1,
    wgsl_common_header="""
const LENS_IOR = 1.15;

fn lens_sdf(coord: vec2f) -> f32 {
    return max(
        distance(coord, vec2f(-.16, 0)) - .1,
        distance(coord, vec2f(0., 0)) - .12
    );
}
    """,
    initial_value_function=constant_initial_value_function(0.),
    update_value_function="""
fn update_value(icoord: vec3i, v: WaveValue) -> f32 {
    if (icoord.x != GRID_RES.x / 6
        || abs(f32(icoord.y) / f32(GRID_RES.y) - .5) > .3) {
        return v.curr;
    }

    let freq = .9 * MAX_FREQ;
    let v_new = .8 * sin(TAU * ubo.time * freq);

    let mix_factor = remap01(ubo.time, 0., 1.) * remap01(ubo.time, 6., 5.);
    return mix(
        v.curr,
        v_new,
        mix_factor
    );
}
    """,
    speed_fac_function="""
fn speed_fac(icoord: vec3i, v: WaveValue) -> f32 {
    let coord = icoord_to_world(icoord).xy;
    return remap_clamp(
        lens_sdf(coord),
        .001,
        -.001,
        1.,
        1. / LENS_IOR
    );
}
    """,
    damp_fac_function=constant_damp_fac_function(.9),
    averaging=False,
    averaging_time_constant=0.,
    render_bg_col=(0., 0., 0.),
    shade_cell_function="""
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {
    var col = colormap_jetski(v);

    let coord = icoord_to_world(icoord).xy;
    let signed_dist = lens_sdf(coord);
    col = mix(
        col,
        col * vec3f(1.1, .05, .1),
        remap_clamp(signed_dist, .001, -.001, 0., .35)
    );

    return col;
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


# add more damping away from the center to avoid artifacts from removing
# boundary reflections.
sim10_2d_lens_smooth_damping = deepcopy(sim9_2d_lens_with_averaging)
sim10_2d_lens_smooth_damping.damp_fac_function = """
fn damp_fac(icoord: vec3i, v: WaveValue) -> f32 {
    let coord = icoord_to_world(icoord).xy;

    // more damping near the edges
    return remap_clamp(
        max(abs(coord.x), abs(coord.y)),
        .2, .3,
        .9, .05
    );
}
"""


# a 2D simulation with a planar wave source and a wall with a small hole
sim11_2d_single_slit = WaveSimParams(
    render_res=(960, 640),
    grid_res=(960, 640, 1),
    cell_size=.0005,
    wave_speed=.1,
    remove_reflections=True,
    timestep=-.5,
    n_sim_steps_per_frame=2,
    wgsl_common_header="""
fn inside_wall(coord: vec2f) -> bool {
    return abs(coord.x) < .001 && abs(coord.y) > .005;
}
    """,
    initial_value_function=constant_initial_value_function(0.),
    update_value_function="""
fn update_value(icoord: vec3i, v: WaveValue) -> f32 {
    if (icoord.x != GRID_RES.x / 3
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
    speed_fac_function=constant_speed_fac_function(1.),
    damp_fac_function="""
fn damp_fac(icoord: vec3i, v: WaveValue) -> f32 {
    let coord = icoord_to_world(icoord).xy;

    // 0 inside the wall (full damping because it's an obstacle)
    if (inside_wall(coord)) {
        return 0.;
    }

    // more damping away from the center to prevent artifacts from boundary
    // reflections removal.
    return remap_clamp(
        length(coord),
        .12, .2,
        .95, .01
    );
}
    """,
    averaging=True,
    averaging_time_constant=1.,
    render_bg_col=(0., 0., 0.),
    shade_cell_function="""
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {
    let coord = icoord_to_world(icoord).xy;
    if (inside_wall(coord)) {
        return vec3f(1);
    }
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


# a 2D double-slit simulation
sim12_2d_double_slit = WaveSimParams(
    render_res=(960, 640),
    grid_res=(960, 640, 1),
    cell_size=.0005,
    wave_speed=.1,
    remove_reflections=True,
    timestep=-.5,
    n_sim_steps_per_frame=2,
    wgsl_common_header="""
fn inside_wall(coord: vec2f) -> bool {
    return abs(coord.x) < .001
        && abs(coord.y - .02) > .003
        && abs(coord.y + .02) > .003;
}
    """,
    initial_value_function=constant_initial_value_function(0.),
    update_value_function="""
fn update_value(icoord: vec3i, v: WaveValue) -> f32 {
    if (icoord.x != GRID_RES.x / 3
        || abs(f32(icoord.y) / f32(GRID_RES.y) - .5) > .3) {
        return v.curr;
    }

    let freq = .8 * MAX_FREQ;
    let v_new = 1.1 * sin(TAU * ubo.time * freq);

    let mix_factor = remap01(ubo.time, 0., 1.) * remap01(ubo.time, 6., 5.);
    return mix(
        v.curr,
        v_new,
        mix_factor
    );
}
    """,
    speed_fac_function=constant_speed_fac_function(1.),
    damp_fac_function="""
fn damp_fac(icoord: vec3i, v: WaveValue) -> f32 {
    let coord = icoord_to_world(icoord).xy;

    // 0 inside the wall (full damping because it's an obstacle)
    if (inside_wall(coord)) {
        return 0.;
    }

    // more damping away from the center to prevent artifacts from boundary
    // reflections removal.
    return remap_clamp(
        length(coord),
        .12, .2,
        .95, .01
    );
}
    """,
    averaging=True,
    averaging_time_constant=1.,
    render_bg_col=(0., 0., 0.),
    shade_cell_function="""
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {
    let coord = icoord_to_world(icoord).xy;
    if (inside_wall(coord)) {
        return vec3f(1);
    }
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


# a 2D double-slit simulation where we read back the grid data to the CPU and
# graph the rightmost column using matplotlib.
def sim13_on_update(
    params: WaveSimParams,
    limits: WaveSimLimits,
    state: WaveSimState,
    readback_function: WaveSimReadbackFunction
):
    rightmost_column = readback_function(
        (-1, 300, 0),
        (-1, -300, 0),
        True
    )[0, :, 0]  # reshape from (1, N, 1) to (N,)
    print(f"{rightmost_column}\n")


sim13_cpu_readback = deepcopy(sim12_2d_double_slit)
sim13_cpu_readback.on_update = sim13_on_update


# 3D simulation with a planar wave source and a spherical lens
sim14_3d_planar_with_lens = WaveSimParams(
    render_res=(1280, 640),
    grid_res=(400, 200, 200),
    cell_size=.003,
    wave_speed=.05,
    remove_reflections=True,
    timestep=-.5,
    n_sim_steps_per_frame=2,
    wgsl_common_header="""
const LENS_CENTER = vec3f(-.05, 0, 0);
const LENS_RADIUS = .16;
const LENS_IOR = 1.1;
    """,
    initial_value_function=constant_initial_value_function(0.),
    update_value_function="""
fn update_value(icoord: vec3i, v: WaveValue) -> f32 {
    let coord = icoord_to_world(icoord);
    if (icoord.x != GRID_RES.x / 6) {
        return v.curr;
    }
    if (any(abs(coord.yz) > vec2f(.2, .25))) {
        return v.curr;
    }

    let v_new = 7. * sin(TAU * ubo.time * MAX_FREQ);
    return mix(
        v.curr,
        v_new,
        remap01(ubo.time, 0., 4.) * remap01(ubo.time, 32., 28.)
    );
}
    """,
    speed_fac_function="""
fn speed_fac(icoord: vec3i, v: WaveValue) -> f32 {
    let coord = icoord_to_world(icoord);
    return remap_clamp(
        distance(coord, LENS_CENTER),
        LENS_RADIUS,
        LENS_RADIUS - .002,
        1.,
        1. / LENS_IOR
    );
}
    """,
    damp_fac_function="""
fn damp_fac(icoord: vec3i, v: WaveValue) -> f32 {
    let coord = icoord_to_world(icoord);

    // more damping away from the center
    return remap_clamp(
        length(coord),
        .2, .6,
        .9, .2
    );
}
    """,
    averaging=True,
    averaging_time_constant=1.,
    render_bg_col=(.02, .005, .02),
    shade_cell_function="""
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {
    // highlight the edges of the volume cube
    const EDGE_THICKNESS = 3;
    var n_edge = 0;
    if (icoord.x < EDGE_THICKNESS || icoord.x >= GRID_RES.x - EDGE_THICKNESS) {
        n_edge++;
    }
    if (icoord.y < EDGE_THICKNESS || icoord.y >= GRID_RES.y - EDGE_THICKNESS) {
        n_edge++;
    }
    if (icoord.z < EDGE_THICKNESS || icoord.z >= GRID_RES.z - EDGE_THICKNESS) {
        n_edge++;
    }
    if (n_edge >= 2) {
        return vec3f(.5, 0, 3);
    }

    var col = colormap_fire(v);

    // highlight the lens
    let coord = icoord_to_world(icoord);
    if (distance(coord, LENS_CENTER) < LENS_RADIUS) {
        col += vec3f(0, .05, .12);
    }

    return col;
}
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
selected_sim_params = sim13_cpu_readback
selected_sim_limits = WaveSimLimits(selected_sim_params)
