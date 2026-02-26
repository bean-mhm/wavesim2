struct SimParams {
    grid_size: vec3<u32>,
    _pad: u32,
};

@group(0) @binding(0)
var<uniform> sim_params: SimParams;

@group(0) @binding(1)
var input_grid: texture_storage_3d<rg32float, read>;

@group(0) @binding(2)
var output_grid: texture_storage_3d<rg32float, write>;

@compute @workgroup_size(8, 8, 4)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= sim_params.grid_size.x ||
        gid.y >= sim_params.grid_size.y ||
        gid.z >= sim_params.grid_size.z) {
        return;
    }

    var v: f32 = textureLoad(input_grid, gid).x;
    v = fract(v + 0.02);
    textureStore(output_grid, gid, vec4<f32>(v, v, v, v));
}
