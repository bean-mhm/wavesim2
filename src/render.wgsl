struct RenderParams {
    tint: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> render_params: RenderParams;

@group(0) @binding(1)
var render_grid: texture_storage_3d<rg32float, read>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var pos = array<vec2<f32>, 6>(
        vec2(-1., -1.),
        vec2( 1., -1.),
        vec2(-1.,  1.),
        vec2(-1.,  1.),
        vec2( 1., -1.),
        vec2( 1.,  1.)
    );

    var out: VsOut;
    out.pos = vec4(pos[vid], 0., 1.);
    out.uv = pos[vid] * 0.5 + vec2(0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let v: f32 = textureLoad(render_grid, vec3<i32>(0, 0, 0)).x;
    return vec4<f32>(v * render_params.tint.rgb, 1);
}
