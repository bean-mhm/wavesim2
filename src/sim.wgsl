// the following will be replaced with constants
// [constants]

// NOTE: must be synced with averaging.wgsl
struct UniformBuf {
    // simulation iteration
    iter: i32,

    // simulation time
    time: f32,
};

@group(0) @binding(0)
var<uniform> ubo: UniformBuf;

@group(0) @binding(1)
var input_grid: texture_storage_3d<rg32float, read>;

@group(0) @binding(2)
var output_grid: texture_storage_3d<rg32float, write>;

const PI = 3.1415926535897932384626433832;
const TAU = 6.283185307179586476925286766;
const HALF_PI = 1.57079632679489661923132169163;

fn remap(
    x: f32,
    inp_start: f32,
    inp_end: f32,
    out_start: f32,
    out_end: f32
) -> f32 {
    return out_start
        + ((out_end - out_start) / (inp_end - inp_start)) * (x - inp_start);
}

fn remap2(
    x: vec2<f32>,
    inp_start: f32,
    inp_end: f32,
    out_start: f32,
    out_end: f32
) -> vec2<f32> {
    return out_start
        + ((out_end - out_start) / (inp_end - inp_start)) * (x - inp_start);
}

fn remap3(
    x: vec3<f32>,
    inp_start: f32,
    inp_end: f32,
    out_start: f32,
    out_end: f32
) -> vec3<f32> {
    return out_start
        + ((out_end - out_start) / (inp_end - inp_start)) * (x - inp_start);
}

fn remap4(
    x: vec4<f32>,
    inp_start: f32,
    inp_end: f32,
    out_start: f32,
    out_end: f32
) -> vec4<f32> {
    return out_start
        + ((out_end - out_start) / (inp_end - inp_start)) * (x - inp_start);
}

fn remap_clamp(
    x: f32,
    inp_start: f32,
    inp_end: f32,
    out_start: f32,
    out_end: f32
) -> f32 {
    let t = saturate((x - inp_start) / (inp_end - inp_start));
    return out_start + t * (out_end - out_start);
}

fn remap2_clamp(
    x: vec2<f32>,
    inp_start: f32,
    inp_end: f32,
    out_start: f32,
    out_end: f32
) -> vec2<f32> {
    let t = saturate((x - inp_start) / (inp_end - inp_start));
    return out_start + t * (out_end - out_start);
}

fn remap3_clamp(
    x: vec3<f32>,
    inp_start: f32,
    inp_end: f32,
    out_start: f32,
    out_end: f32
) -> vec3<f32> {
    let t = saturate((x - inp_start) / (inp_end - inp_start));
    return out_start + t * (out_end - out_start);
}

fn remap4_clamp(
    x: vec4<f32>,
    inp_start: f32,
    inp_end: f32,
    out_start: f32,
    out_end: f32
) -> vec4<f32> {
    let t = saturate((x - inp_start) / (inp_end - inp_start));
    return out_start + t * (out_end - out_start);
}

fn remap01(
    x: f32,
    inp_start: f32,
    inp_end: f32
) -> f32 {
    return saturate((x - inp_start) / (inp_end - inp_start));
}

fn remap2_01(
    x: vec2<f32>,
    inp_start: f32,
    inp_end: f32
) -> vec2<f32> {
    return saturate((x - inp_start) / (inp_end - inp_start));
}

fn remap3_01(
    x: vec3<f32>,
    inp_start: f32,
    inp_end: f32
) -> vec3<f32> {
    return saturate((x - inp_start) / (inp_end - inp_start));
}

fn remap4_01(
    x: vec4<f32>,
    inp_start: f32,
    inp_end: f32
) -> vec4<f32> {
    return saturate((x - inp_start) / (inp_end - inp_start));
}

fn icoord_to_world(icoord: vec3i) -> vec3f {
    let p_norm = (vec3f(icoord) + .5) / vec3f(GRID_RES);
    return (p_norm - .5) * GRID_DIMS;
}

struct WaveValue {
    curr: f32, // value in the current iteration
    prev: f32, // value in the previous iteration
};

fn grid_fetch(icoord: vec3i) -> WaveValue {
    let data = textureLoad(input_grid, icoord).xy;
    return WaveValue(data.x, data.y);
}

fn grid_write(icoord: vec3i, v: WaveValue) {
    textureStore(output_grid, icoord, vec4f(v.curr, v.prev, 0, 0));
}

// the following will be replaced with the definitions for user-provided
// functions initial_value, update_value, and speed_fac. see "config.py".
// [user-functions]

@compute @workgroup_size(8, 8, 4)
fn cs_main(@builtin(global_invocation_id) gid_u: vec3u) {
    let icoord = vec3i(gid_u);
    if (any(icoord >= GRID_RES)) {
        return;
    }

    var v = grid_fetch(icoord);

    // initial values
    if (ubo.iter == 0) {
        grid_write(icoord, initial_value(icoord));
        return;
    }

    // this is approximate and works best when the wave is perpendicular to
    // the boundary face. I genuinely have no clue how this works and I frankly
    // got help from an LLM. sue me.
    // https://chatgpt.com/share/6995fcad-7f98-800b-91c2-d86b4f1551ac
    if (REMOVE_REFLECTIONS) {
        var boundary: bool = false;
        if (icoord.x == 0) {
            boundary = true;
            let neighbor = grid_fetch(icoord + vec3i(1, 0, 0));
            v.curr =
                neighbor.prev
                + IMPEDANCE_MATCHING_COEFFICIENT
                * (neighbor.curr - v.curr);
        }
        if (icoord.x == GRID_RES.x - 1) {
            boundary = true;
            let neighbor = grid_fetch(icoord + vec3i(-1, 0, 0));
            v.curr =
                neighbor.prev
                + IMPEDANCE_MATCHING_COEFFICIENT
                * (neighbor.curr - v.curr);
        }
        if (icoord.y == 0) {
            boundary = true;
            let neighbor = grid_fetch(icoord + vec3i(0, 1, 0));
            v.curr =
                neighbor.prev
                + IMPEDANCE_MATCHING_COEFFICIENT
                * (neighbor.curr - v.curr);
        }
        if (icoord.y == GRID_RES.y - 1) {
            boundary = true;
            let neighbor = grid_fetch(icoord + vec3i(0, -1, 0));
            v.curr =
                neighbor.prev
                + IMPEDANCE_MATCHING_COEFFICIENT
                * (neighbor.curr - v.curr);
        }
        if (icoord.z == 0 && !IS_2D) {
            boundary = true;
            let neighbor = grid_fetch(icoord + vec3i(0, 0, 1));
            v.curr =
                neighbor.prev
                + IMPEDANCE_MATCHING_COEFFICIENT
                * (neighbor.curr - v.curr);
        }
        if (icoord.z == GRID_RES.z - 1 && !IS_2D) {
            boundary = true;
            let neighbor = grid_fetch(icoord + vec3i(0, 0, -1));
            v.curr =
                neighbor.prev
                + IMPEDANCE_MATCHING_COEFFICIENT
                * (neighbor.curr - v.curr);
        }

        if (boundary) {
            v.prev = v.curr;
            grid_write(icoord, v);
            return;
        }
    }

    // the terms "next" and "prev" here refer to spatial offsets and have
    // nothing to do with the terms "curr" and "prev" used for temporal offsets.
    var next_in_x = 0.;
    var prev_in_x = 0.;
    var next_in_y = 0.;
    var prev_in_y = 0.;
    var next_in_z = 0.;
    var prev_in_z = 0.;
    if ((icoord.x + 1) < GRID_RES.x) {
        next_in_x = grid_fetch(icoord + vec3i(1, 0, 0)).curr;
    }
    if ((icoord.x - 1) >= 0) {
        prev_in_x = grid_fetch(icoord + vec3i(-1, 0, 0)).curr;
    }
    if ((icoord.y + 1) < GRID_RES.y) {
        next_in_y = grid_fetch(icoord + vec3i(0, 1, 0)).curr;
    }
    if ((icoord.y - 1) >= 0) {
        prev_in_y = grid_fetch(icoord + vec3i(0, -1, 0)).curr;
    }
    if ((icoord.z + 1) < GRID_RES.z) {
        next_in_z = grid_fetch(icoord + vec3i(0, 0, 1)).curr;
    }
    if ((icoord.z - 1) >= 0) {
        prev_in_z = grid_fetch(icoord + vec3i(0, 0, -1)).curr;
    }
    
    // d2u/dx2 (but without the dx2 because it'll be applied below)
    let grad_x = next_in_x - v.curr - v.curr + prev_in_x;
    let grad_y = next_in_y - v.curr - v.curr + prev_in_y;
    var grad_z: f32 = 0.;
    if (!IS_2D) {
        grad_z = next_in_z - v.curr - v.curr + prev_in_z;
    }

    // propagation speed
    let c = WAVE_SPEED * speed_fac(icoord, v);
    let c2 = c * c;

    // d2u/dt2
    let acc =
        (grad_x + grad_y + grad_z)  // Laplacian
        * c2  // c^2
        / (CELL_SIZE * CELL_SIZE);  // dx^2

    // du/dt
    var vel = (v.curr - v.prev) / TIMESTEP;
    vel += (acc * TIMESTEP);
    vel *= DAMP_FAC_PER_DT;

    // u
    v.prev = v.curr;
    v.curr += (vel * TIMESTEP);

    // custom excitations
    v.curr = update_value(icoord, v);

    grid_write(icoord, v);
}
