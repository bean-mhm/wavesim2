from copy import deepcopy
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from common import *


DISPLAY_SCALE = 1.


basic_render_cmd = RenderCommand(
    res=(1280, 720),
    mode=RenderMode.Raymarching,
    region=Aabb(),
    try_use_averaging_buffer=True,
    bg_col=(.02, .005, .02),
    shade_cell_function="""
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {
    return colormap_simple(v * 5.);
}
        """,
    n_samples_per_pixel=4,
    raymarch_step=.02,
    raymarch_step_jitter=.015,
    use_trilinear=True,
    cam=CameraState(
        pos=(0., -1.25, .03),
        lookat=(0., 0., 0.),
        world_up=(0., 0., 1.),
        fov_degrees=70.
    ),
    apply_flim=True,
    export_path=None
)


def sim_on_update_basic(
    params: WaveSimParams,
    limits: WaveSimLimits,
    state: WaveSimState,
    readback_function: WaveSimReadbackFunction
) -> WaveSimOnUpdateReturn:
    return WaveSimOnUpdateReturn(
        render_commands=[basic_render_cmd],
        display_render_idx=0,
        new_user_data=None,
        should_stop=False
    )


def sim_on_update_rotating_camera(
    params: WaveSimParams,
    limits: WaveSimLimits,
    state: WaveSimState,
    readback_function: WaveSimReadbackFunction
) -> WaveSimOnUpdateReturn:
    render_cmd = basic_render_cmd.__replace__(cam=CameraState(
        pos=(
            1.2 * np.cos(2. * np.pi * .07 * state.wall_time),
            1.2 * np.sin(2. * np.pi * .07 * state.wall_time),
            .03
        ),
        lookat=(0., 0., 0.),
        world_up=(0., 0., 1.),
        fov_degrees=80.
    ))

    return WaveSimOnUpdateReturn(
        render_commands=[render_cmd],
        display_render_idx=0,
        new_user_data=None,
        should_stop=False
    )


def sim_on_update_highlight_obstacle(
    params: WaveSimParams,
    limits: WaveSimLimits,
    state: WaveSimState,
    readback_function: WaveSimReadbackFunction
) -> WaveSimOnUpdateReturn:
    render_cmd = basic_render_cmd.__replace__(
        cam=CameraState(
            pos=(
                1.2 * np.cos(2. * np.pi * .07 * state.wall_time),
                1.2 * np.sin(2. * np.pi * .07 * state.wall_time),
                .03
            ),
            lookat=(0., 0., 0.),
            world_up=(0., 0., 1.),
            fov_degrees=80.
        ),
        shade_cell_function="""
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {
    var col = colormap_simple(v * 4.);
    if (all(abs(icoord - vec3i(86, 50, 50)) <= vec3i(7, 20, 7))) {
        col = col * .8 + vec3f(.3, 0, .35);
    }
    return col;
}
        """
    )

    return WaveSimOnUpdateReturn(
        render_commands=[render_cmd],
        display_render_idx=0,
        new_user_data=None,
        should_stop=False
    )


def sim_on_update_2d_lens(
    params: WaveSimParams,
    limits: WaveSimLimits,
    state: WaveSimState,
    readback_function: WaveSimReadbackFunction
) -> WaveSimOnUpdateReturn:
    render_cmd = RenderCommand(
        res=(1200, 700),
        mode=RenderMode.Slice,
        slice=GridSlice(),
        bg_col=(0., 0., 0.),
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
        n_samples_per_pixel=1
    )

    return WaveSimOnUpdateReturn(
        render_commands=[render_cmd],
        display_render_idx=0,
        new_user_data=None,
        should_stop=False
    )


def sim_on_update_2d_slit(
    params: WaveSimParams,
    limits: WaveSimLimits,
    state: WaveSimState,
    readback_function: WaveSimReadbackFunction
) -> WaveSimOnUpdateReturn:
    render_cmd = RenderCommand(
        res=(1080, 720),
        mode=RenderMode.Slice,
        slice=GridSlice(),
        bg_col=(0., 0., 0.),
        shade_cell_function="""
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {
    let coord = icoord_to_world(icoord).xy;
    if (inside_wall(coord)) {
        return vec3f(1);
    }
    return colormap_jetski(v);
}
        """,
        n_samples_per_pixel=1
    )

    return WaveSimOnUpdateReturn(
        # only render on even iterations
        render_commands=[render_cmd] if state.iter % 2 == 0 else [],
        display_render_idx=0,
        new_user_data=None,
        should_stop=False
    )


plot_vars: dict = {}


def setup_plot(n_points: int, title: str = ""):
    global plot_vars

    # enable interactive mode
    plt.ion()

    plot_vars["n"] = n_points
    plot_vars["title"] = title
    fig, plot_vars["ax"] = plt.subplots()
    if title:
        plot_vars["ax"].set_title(title)
    x = np.linspace(0, n_points - 1, n_points, dtype=np.float32)
    y = np.zeros_like(x)

    # initial line
    plot_vars["line"] = plot_vars["ax"].plot(x, y)[0]


def plot(y: np.ndarray, title: str = ""):
    if not plot_vars.keys() or plot_vars["n"] != y.size \
            or plot_vars["title"] != title:
        setup_plot(y.size, title)
    for _ in range(plot_vars["n"]):
        plot_vars["line"].set_ydata(y)  # update y data
        plot_vars["ax"].relim()  # update limits
        plot_vars["ax"].autoscale()  # rescale
        plt.draw()  # draw


def sim_on_update_cpu_readback(
    params: WaveSimParams,
    limits: WaveSimLimits,
    state: WaveSimState,
    readback_function: WaveSimReadbackFunction
) -> WaveSimOnUpdateReturn:
    # plot every 40 iterations
    if state.iter % 40 == 0:
        # read back the rightmost column of the grid to the CPU as a numpy.ndarray
        read_averaging_grid = True
        rightmost_column = readback_function(
            Aabb(
                pmin=(-1, 0, 0),
                pmax=(-1, -1, 0)
            ),
            read_averaging_grid
        )

        # reshape from (1, N, 1) to (N,)
        rightmost_column = rightmost_column[0, :, 0]

        # plot
        plot(
            rightmost_column[100:-100],
            "right-most column (wait for it)"
        )

    # make render command
    render_cmd = RenderCommand(
        res=(1080, 720),
        mode=RenderMode.Slice,
        slice=GridSlice(),
        bg_col=(0., 0., 0.),
        shade_cell_function="""
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {
    let coord = icoord_to_world(icoord).xy;
    if (inside_wall(coord)) {
        return vec3f(1);
    }
    return colormap_jetski(v);
}
        """,
        n_samples_per_pixel=1
    )

    return WaveSimOnUpdateReturn(
        # render on even iterations
        render_commands=[render_cmd] if state.iter % 2 == 0 else [],
        display_render_idx=0,
        new_user_data=None,
        should_stop=False
    )


def sim_on_update_2d_speed_mask(
    params: WaveSimParams,
    limits: WaveSimLimits,
    state: WaveSimState,
    readback_function: WaveSimReadbackFunction
) -> WaveSimOnUpdateReturn:
    render_cmd = RenderCommand(
        res=(1360, 340),
        mode=RenderMode.Slice,
        slice=GridSlice(),
        bg_col=(.025, 0., .025),
        shade_cell_function="""
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {
    return colormap_blood(v);
}
        """,
        n_samples_per_pixel=1
    )

    return WaveSimOnUpdateReturn(
        # only render every N iterations
        render_commands=[render_cmd] if state.iter % 10 == 0 else [],
        display_render_idx=0,
        new_user_data=None,
        should_stop=False
    )


def sim_on_update_3d_planar_with_lens(
    params: WaveSimParams,
    limits: WaveSimLimits,
    state: WaveSimState,
    readback_function: WaveSimReadbackFunction
) -> WaveSimOnUpdateReturn:
    render_cmd = basic_render_cmd.__replace__(
        cam=CameraState(
            pos=(
                .07 * np.cos(2. * np.pi * .12 * state.wall_time),
                -.68,
                .01 * np.sin(2. * np.pi * .284 * state.wall_time),
            ),
            lookat=(0., 0., 0.),
            world_up=(0., 0., 1.),
            fov_degrees=60.
        ),
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
        """
    )

    return WaveSimOnUpdateReturn(
        # render every 3 iterations
        render_commands=[render_cmd] if state.iter % 3 == 0 else [],
        display_render_idx=0,
        new_user_data=None,
        should_stop=False
    )


def sim_on_update_3d_hexagonal_diffraction(
    params: WaveSimParams,
    limits: WaveSimLimits,
    state: WaveSimState,
    readback_function: WaveSimReadbackFunction
) -> WaveSimOnUpdateReturn:
    render_cmd = RenderCommand(
        res=(1280, 640),
        mode=RenderMode.Raymarching,
        region=Aabb(),
        try_use_averaging_buffer=True,
        bg_col=(0, 0, 0),
        shade_cell_function="""
fn shade_cell(icoord: vec3i, v: f32) -> vec3f {
    // wall
    let coord = icoord_to_world(icoord);
    if (wall_sdf(coord) < 0.) {
        return 3. * vec3f(.5, 1, .2);
    }

    var col = colormap_blood(10. * v);

    let sensor_plane_x = i32(floor(mix(
        f32(GRID_RES.x) * .25,
        f32(GRID_RES.x) * .99,
        cos(TAU * .05 * ubo.wall_time) * .5 + .5
    )));
    if (icoord.x == sensor_plane_x) {
        return 100. * col;
    } else {
        return .1 * col;
    }
}
        """,
        n_samples_per_pixel=2,
        raymarch_step=.02,
        raymarch_step_jitter=.015,
        use_trilinear=True,
        cam=CameraState(
            pos=(
                3.,
                -1.,
                .01 * np.sin(2. * np.pi * .237 * state.wall_time),
            ),
            lookat=(
                1.,
                0.,
                0.
            ),
            world_up=(0., 0., 1.),
            fov_degrees=20.
        ),
        apply_flim=True,
        export_path=None
    )

    return WaveSimOnUpdateReturn(
        # render every 8 iterations
        render_commands=[render_cmd] if state.iter % 8 == 0 else [],
        display_render_idx=0,
        new_user_data=None,
        should_stop=False
    )


# basic 3D simulation with a sine wave source at the center
sim1_basic = WaveSimParams(
    grid_res=(101, 101, 101),
    cell_size=.01,
    wave_speed=.05,
    remove_reflections=False,
    timestep=-.5,
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
    user_data_fields=None,
    user_data=None,
    averaging=False,
    averaging_time_constant=0.,
    on_start=None,
    on_update=sim_on_update_basic
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


# same as sim #2 but with a spinning camera
sim3_rotating_camera = sim2_box_obstacle.__replace__(
    on_update=sim_on_update_rotating_camera
)


# make the obstacle more noticable by modifying shade_cell_function
sim4_highlighted_obstacle = sim3_rotating_camera.__replace__(
    on_update=sim_on_update_highlight_obstacle
)


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
    grid_res=(600, 350, 1),
    cell_size=.001,
    wave_speed=.1,
    remove_reflections=True,
    timestep=-.5,
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
    user_data_fields=None,
    user_data=None,
    averaging=False,
    averaging_time_constant=0.,
    on_start=None,
    on_update=sim_on_update_2d_lens
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
    grid_res=(960, 640, 1),
    cell_size=.0005,
    wave_speed=.1,
    remove_reflections=True,
    timestep=-.8,
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
    user_data_fields=None,
    user_data=None,
    averaging=True,
    averaging_time_constant=1.,
    on_start=None,
    on_update=sim_on_update_2d_slit
)


# 2D double-slit simulation
sim12_2d_double_slit = deepcopy(sim11_2d_single_slit)
sim12_2d_double_slit.wgsl_common_header = """
fn inside_wall(coord: vec2f) -> bool {
    return abs(coord.x) < .001
        && abs(coord.y - .02) > .003
        && abs(coord.y + .02) > .003;
}
"""


# an example where we read back the rightmost column of the grid to the CPU as a
# numpy.ndarray and graph it using matplotlib.
sim13_cpu_readback_graph = sim12_2d_double_slit.__replace__(
    on_update=sim_on_update_cpu_readback
)


# load speed mask image for the next simulation
speed_mask_path = Path(__file__).parent / "speed-mask.png"
speed_mask = np.asarray(
    Image.open(speed_mask_path).convert("RGB")
)[::-1, :].astype(np.float32) / 255.
speed_mask_res = (speed_mask.shape[1], speed_mask.shape[0])
speed_mask_n_pixels = speed_mask_res[0] * speed_mask_res[1]

# in the speed mask, the red channel stores wave speed factor and the green
# channel stores 1 if that pixel is a wave source. here we find pixels whose
# green channels are large enough and make a nice organized list.
speed_mask_wave_sources = np.column_stack(
    np.where(speed_mask[:, :, 1] > .99)  # green channel > 0.99
)[:, ::-1]

# sort wave sources from left to right
speed_mask_wave_sources = speed_mask_wave_sources[
    np.argsort(speed_mask_wave_sources[:, 0])
]

# wave source list in WGSL
speed_mask_wave_sources_wgsl: str = ""
for i, wave_source in enumerate(speed_mask_wave_sources):
    if i != 0:
        speed_mask_wave_sources_wgsl += ", "
    speed_mask_wave_sources_wgsl += \
        f"vec2i({int(wave_source[0])}, {int(wave_source[1])})"

# extract the red channel for speed factor
speed_mask = np.ascontiguousarray(speed_mask[:, :, 0])

# 2D simulation where we send a custom image to use as the wave speed mask,
# resulting in waves filling up a certain piece of text.
sim14_send_user_data = WaveSimParams(
    grid_res=(speed_mask_res[0], speed_mask_res[1], 1),
    cell_size=.001,
    wave_speed=1.,
    remove_reflections=False,
    timestep=-.95,
    wgsl_common_header=f"""
const IMG_RES = vec2i{speed_mask_res};
const N_WAVE_SOURCES = {len(speed_mask_wave_sources)};
const WAVE_SOURCES = array({speed_mask_wave_sources_wgsl});
    """,
    initial_value_function=constant_initial_value_function(0.),
    update_value_function="""
fn update_value(icoord: vec3i, v: WaveValue) -> f32 {
    for (var i: i32 = 0; i < N_WAVE_SOURCES; i++) {
        let wave_source = WAVE_SOURCES[i];
        if (any(icoord.xy != wave_source)) {
            continue;
        }

        let v_new = 1.5 * sin(TAU * ubo.time * .9 * MAX_FREQ);
        return mix(
            v.curr,
            v_new,
            remap01(ubo.time - .05 * f32(i), 0., .2)
        );
    }
    return v.curr;
}
    """,
    speed_fac_function="""
fn speed_fac(icoord: vec3i, v: WaveValue) -> f32 {
    let pixel_index = (icoord.x * 1) + (icoord.y * IMG_RES.x);
    return user_data.pixels[pixel_index];
}
    """,
    damp_fac_function=constant_damp_fac_function(1.),
    user_data_fields=f"""
pixels: array<f32, {speed_mask_n_pixels}>,
    """,
    user_data=speed_mask.data,
    averaging=True,
    averaging_time_constant=.002,
    on_start=None,
    on_update=sim_on_update_2d_speed_mask
)


# 3D simulation with a planar wave source and a spherical lens
sim15_3d_planar_with_lens = WaveSimParams(
    grid_res=(400, 200, 200),
    cell_size=.003,
    wave_speed=.05,
    remove_reflections=True,
    timestep=-.5,
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
    user_data_fields=None,
    user_data=None,
    averaging=True,
    averaging_time_constant=1.,
    on_start=None,
    on_update=sim_on_update_3d_planar_with_lens
)


# 3D point source diffracting through a hexagonal hole
sim16_3d_hexagonal_diffraction = WaveSimParams(
    grid_res=(1000, 150, 150),
    cell_size=.004,
    wave_speed=.5,
    remove_reflections=True,
    timestep=-.7,
    wgsl_common_header="""
fn hexagon_sdf(p: vec2f, radius: f32) -> f32 {
    const k = vec3f(-sqrt(3.) / 2., .5, 1. / sqrt(3.));

    var p2 = abs(p);
    p2 -= 2. * min(dot(k.xy, p2), 0.) * k.xy;
    p2 -= vec2f(
        clamp(p2.x, -k.z * radius, k.z * radius),
        radius
    );

    return length(p2) * sign(p2.y);
}

fn wall_sdf(p: vec3f) -> f32 {
    return max(
        abs(p.x + 1.75) - .005,
        -hexagon_sdf(p.yz, .23)
    );
}
    """,
    initial_value_function=constant_initial_value_function(0.),
    update_value_function="""
fn update_value(icoord: vec3i, v: WaveValue) -> f32 {
    let coord = icoord_to_world(icoord);
    if (icoord.x != 50) {
        return v.curr;
    }
    if (any(abs(coord.yz) > vec2f(.25, .25))) {
        return v.curr;
    }

    let v_new = 2.2 * sin(TAU * ubo.time * .9 * MAX_FREQ);
    return mix(
        v.curr,
        v_new,
        remap01(ubo.time, 0., 4.) * remap01(ubo.time, 30., 26.)
    );
}
    """,
    speed_fac_function="""
fn speed_fac(icoord: vec3i, v: WaveValue) -> f32 {
    let coord = icoord_to_world(icoord);
    let signed_dist = wall_sdf(coord);
    return remap01(signed_dist, -.004, .004);
}
    """,
    damp_fac_function=constant_damp_fac_function(.95),
    user_data_fields=None,
    user_data=None,
    averaging=True,
    averaging_time_constant=.5,
    on_start=None,
    on_update=sim_on_update_3d_hexagonal_diffraction
)


# choose which simulation to run from above
selected_sim_params = sim12_2d_double_slit
selected_sim_limits = WaveSimLimits(selected_sim_params)
