struct UniformBuf {
    pmin: vec3i,
    read_res: vec3i, // pmax - pmin + 1
};

@group(0) @binding(0)
var<uniform> ubo: UniformBuf;

@group(0) @binding(1)
var source_grid: texture_storage_3d<[source-grid-format], read>;

@group(0) @binding(2)
var<storage, read_write> destination_buffer: array<f32>;

@compute @workgroup_size(8, 8, 4)
fn cs_main(@builtin(global_invocation_id) gid_u: vec3u) {
    let icoord_dst = vec3i(gid_u);
    let icoord_src = ubo.pmin + icoord_dst;
    if (any(icoord_dst >= ubo.read_res)) {
        return;
    }

    let dst_idx =
        icoord_dst.x * (1)
        + icoord_dst.y * (ubo.read_res.x)
        + icoord_dst.z * (ubo.read_res.x * ubo.read_res.y);

    let cell_data: vec4f = textureLoad(source_grid, icoord_src);

    if ([single-channel]) {
        destination_buffer[dst_idx] = cell_data[0];
    } else {
        destination_buffer[dst_idx * 2 + 0] = cell_data[0];
        destination_buffer[dst_idx * 2 + 1] = cell_data[1];
    }
}
