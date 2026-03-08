// the following will be replaced with constants
// [constants]

// NOTE: must be synced with sim.wgsl
struct UniformBuf {
    // simulation iteration
    iter: i32,

    // simulation time
    time: f32,
};

@group(0) @binding(0)
var<uniform> ubo: UniformBuf;

@group(0) @binding(1)
var wave_grid: texture_storage_3d<rg32float, read>;

@group(0) @binding(2)
var wave_grid_avg: texture_storage_3d<r32float, read_write>;

fn wave_grid_fetch(icoord: vec3i) -> f32 {
    return textureLoad(wave_grid, icoord).x;
}

fn avg_grid_fetch(icoord: vec3i) -> f32 {
    return textureLoad(wave_grid_avg, icoord).x;
}

fn avg_grid_write(icoord: vec3i, v: f32) {
    textureStore(wave_grid_avg, icoord, vec4f(v, 0, 0, 0));
}

@compute @workgroup_size(8, 8, 4)
fn cs_main(@builtin(global_invocation_id) gid_u: vec3u) {
    let icoord = vec3i(gid_u);
    let grid_res = vec3i(textureDimensions(wave_grid));
    if (any(icoord >= grid_res)) {
        return;
    }

    let new_amp = wave_grid_fetch(icoord); // raw amplitude
    let new_ = new_amp * new_amp; // intensity

    // first iteration
    if (ubo.iter == 0) {
        avg_grid_write(icoord, new_);
        return;
    }

    // exponential smoothing
    let curr = avg_grid_fetch(icoord);
    avg_grid_write(
        icoord,
        mix(curr, new_, AVERAGING_MIX_FAC_PER_DT)
    );
}
