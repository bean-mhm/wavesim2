struct UniformBufRaw {
    d0: vec4f,
    d1: vec4f,
    d2: vec4f,
    d3: vec4f,
    d4: vec4f,
};

struct UniformBuf {
    // simulation grid resolution
    grid_res: vec3i,

    // simulation grid cell (voxel) size
    cell_size: f32,

    // dimensions of the simulation grid
    grid_dims: vec3f,

    // wave propagation speed
    wave_speed: f32,

    // remove reflections at the grid boundaries
    remove_reflections: bool,

    // dampening factor per second (stiff near zero and loose at 1)
    damp_fac: f32,

    // delta time
    timestep: f32,

    // simulation iteration
    iter: i32,

    // simulation time
    time: f32,

    // maximum stable timestep
    max_timestep: f32,

    // minimum stable wavelength
    min_wavelength: f32,

    // maximum stable frequency
    max_freq: f32,

    // used internally for removing reflections at the grid boundaries
    impedance_matching_coefficient: f32,
};

@group(0) @binding(0)
var<uniform> ubo_raw: UniformBufRaw;
var<private> ubo: UniformBuf;

fn extract_uniforms() {
    ubo.grid_res = vec3i(
        bitcast<i32>(ubo_raw.d0.x),
        bitcast<i32>(ubo_raw.d0.y),
        bitcast<i32>(ubo_raw.d0.z)
    );
    ubo.cell_size = ubo_raw.d0.w;
    ubo.grid_dims = ubo_raw.d1.xyz;
    ubo.wave_speed = ubo_raw.d1.w;
    ubo.remove_reflections = (bitcast<i32>(ubo_raw.d2.x) != 0);
    ubo.damp_fac = ubo_raw.d2.y;
    ubo.timestep = ubo_raw.d2.z;
    ubo.iter = bitcast<i32>(ubo_raw.d2.w);
    ubo.time = ubo_raw.d3.x;
    ubo.max_timestep = ubo_raw.d3.y;
    ubo.min_wavelength = ubo_raw.d3.z;
    ubo.max_freq = ubo_raw.d3.w;
    ubo.impedance_matching_coefficient = ubo_raw.d4.x;
}

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
// functions initial_value() and update_value().
// [user-functions]

@compute @workgroup_size(8, 8, 4)
fn cs_main(@builtin(global_invocation_id) gid_u: vec3u) {
    extract_uniforms();
    let icoord = vec3i(gid_u);

    if (icoord.x >= ubo.grid_res.x ||
        icoord.y >= ubo.grid_res.y ||
        icoord.z >= ubo.grid_res.z) {
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
    if (ubo.remove_reflections) {
        var boundary: bool = false;
        if (icoord.x == 0) {
            boundary = true;
            let neighbor = grid_fetch(icoord + vec3i(1, 0, 0));
            v.curr =
                neighbor.prev
                + ubo.impedance_matching_coefficient
                * (neighbor.curr - v.curr);
        }
        if (icoord.x == ubo.grid_res.x - 1) {
            boundary = true;
            let neighbor = grid_fetch(icoord + vec3i(1, 0, 0));
            v.curr =
                neighbor.prev
                + ubo.impedance_matching_coefficient
                * (neighbor.curr - v.curr);
        }
        if (icoord.y == 0) {
            boundary = true;
            let neighbor = grid_fetch(icoord + vec3i(0, 1, 0));
            v.curr =
                neighbor.prev
                + ubo.impedance_matching_coefficient
                * (neighbor.curr - v.curr);
        }
        if (icoord.y == ubo.grid_res.y - 1) {
            boundary = true;
            let neighbor = grid_fetch(icoord + vec3i(0, 1, 0));
            v.curr =
                neighbor.prev
                + ubo.impedance_matching_coefficient
                * (neighbor.curr - v.curr);
        }
        if (icoord.z == 0) {
            boundary = true;
            let neighbor = grid_fetch(icoord + vec3i(0, 0, 1));
            v.curr =
                neighbor.prev
                + ubo.impedance_matching_coefficient
                * (neighbor.curr - v.curr);
        }
        if (icoord.z == ubo.grid_res.z - 1) {
            boundary = true;
            let neighbor = grid_fetch(icoord + vec3i(0, 0, 1));
            v.curr =
                neighbor.prev
                + ubo.impedance_matching_coefficient
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
    if ((icoord.x + 1) < ubo.grid_res.x) {
        next_in_x = grid_fetch(icoord + vec3i(1, 0, 0)).curr;
    }
    if ((icoord.x - 1) >= 0) {
        prev_in_x = grid_fetch(icoord + vec3i(-1, 0, 0)).curr;
    }
    if ((icoord.y + 1) < ubo.grid_res.y) {
        next_in_y = grid_fetch(icoord + vec3i(0, 1, 0)).curr;
    }
    if ((icoord.y - 1) >= 0) {
        prev_in_y = grid_fetch(icoord + vec3i(0, -1, 0)).curr;
    }
    if ((icoord.z + 1) < ubo.grid_res.z) {
        next_in_z = grid_fetch(icoord + vec3i(0, 0, 1)).curr;
    }
    if ((icoord.z - 1) >= 0) {
        prev_in_z = grid_fetch(icoord + vec3i(0, 0, -1)).curr;
    }
    
    // d2u/dx2 (but without the dx2 because it'll be applied below)
    let grad_x = next_in_x - v.curr - v.curr + prev_in_x;
    let grad_y = next_in_y - v.curr - v.curr + prev_in_y;
    let grad_z = next_in_z - v.curr - v.curr + prev_in_z;

    // d2u/dt2
    let acc =
        (grad_x + grad_y + grad_z)  // Laplacian
        * (ubo.wave_speed * ubo.wave_speed)  // c^2
        / (ubo.cell_size * ubo.cell_size);  // dx^2

    // du/dt
    var vel = (v.curr - v.prev) / ubo.timestep;
    vel += (acc * ubo.timestep);
    vel *= pow(ubo.damp_fac, ubo.timestep);

    // u
    v.prev = v.curr;
    v.curr += (vel * ubo.timestep);

    // custom excitations
    v.curr = update_value(icoord, v);

    grid_write(icoord, v);
}
