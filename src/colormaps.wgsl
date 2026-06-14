// WGSL colormap functions used in the fragment shader, available in
// shade_cell().

fn colormap_simple(v: f32) -> vec3f {
    return select(
        vec3f(0, .5, 1) * -v,
        vec3f(1, .35, 0) * v,
        v > 0.
    );
}

fn colormap_exp(v: f32, rgb_fac: vec3f) -> vec3f {
    var sq = v;
    if (!RENDER_USES_AVERAGING_BUFFER) {
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

fn colormap_blood(v: f32) -> vec3f {
    return 1.7 * colormap_exp(v, vec3f(1., .02, .01));
}

fn colormap_grayscale_positive_only(v: f32) -> vec3f {
    return vec3f(max(v, 0.));
}

fn colormap_grayscale_abs(v: f32) -> vec3f {
    return vec3f(abs(v));
}

fn colormap_grayscale_squared(v: f32) -> vec3f {
    var sq = v;
    if (!RENDER_USES_AVERAGING_BUFFER) {
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
    if (!RENDER_USES_AVERAGING_BUFFER) {
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
