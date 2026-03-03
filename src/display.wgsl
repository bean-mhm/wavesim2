struct UniformBuf {
    srgb_surface: vec4i, // if surface format applies sRGB OETF internally
};

@group(0) @binding(0)
var<uniform> ubo: UniformBuf;

@group(0) @binding(1)
var render_target: texture_2d<f32>;

@group(0) @binding(2)
var linear_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    const pos = array<vec2f, 6>(
        vec2f(-1., -1.),
        vec2f( 1., -1.),
        vec2f(-1.,  1.),
        vec2f(-1.,  1.),
        vec2f( 1., -1.),
        vec2f( 1.,  1.)
    );

    var out: VsOut;
    out.pos = vec4f(pos[vid], 0., 1.);
    out.uv = pos[vid] * .5 + .5;
    out.uv.y = 1. - out.uv.y;
    return out;
}

// OETF: Linear BT.709 I-D65 -> sRGB
fn linear_bt709_id65_to_srgb(v_in: vec3f) -> vec3f {
    let v = saturate(v_in);
    return mix(
        12.92 * v,
        pow(v, vec3f(1. / 2.4)) * 1.055 - .055,
        step(vec3f(.0031308), v)
    );
}

// EOTF: sRGB -> Linear BT.709 I-D65
fn srgb_to_linear_bt709_id65(v_in: vec3f) -> vec3f {
    let v = saturate(v_in);
    return mix(
        v * .07739938080495356,
        pow((v + .055) * .9478672985781990521327, vec3f(2.4)),
        step(vec3f(.040449936), v)
    );
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    var col = textureSample(
        render_target,
        linear_sampler,
        in.uv
    ).rgb;
    if (ubo.srgb_surface[0] != 0)
    {
        col = srgb_to_linear_bt709_id65(col);
    }
    return vec4f(col, 1);
}
