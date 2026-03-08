// the following will be replaced with constants
// [constants]

struct UniformBufRaw {
    d0: vec4f,
    d1: vec4f,
    d2: vec4f,
    d3: vec4f,
};

struct UniformBuf {
    cam_pos: vec3f,
    cam_lookat: vec3f,
    cam_world_up: vec3f,
    cam_fov_degrees: f32,
    iter: i32,
    time: f32,
    wall_time: f32,
};

@group(0) @binding(0)
var<uniform> ubo_raw: UniformBufRaw;
var<private> ubo: UniformBuf;

fn extract_uniforms() {
    // extract uniform data (i hate stupid padding rules)
    ubo.cam_pos = ubo_raw.d0.xyz;
    ubo.cam_lookat = vec3f(ubo_raw.d0.w, ubo_raw.d1.xy);
    ubo.cam_world_up = vec3f(ubo_raw.d1.zw, ubo_raw.d2.x);
    ubo.cam_fov_degrees = ubo_raw.d2.y;
    ubo.iter = bitcast<i32>(ubo_raw.d2.z);
    ubo.time = ubo_raw.d2.w;
    ubo.wall_time = ubo_raw.d3.x;
}

@group(0) @binding(1)
var render_grid: texture_storage_3d<[render-grid-format], read>;

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
    return out;
}

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

fn spherical_to_cartesian(s: vec3f) -> vec3f {
    let sin_theta = sin(s.y);
    return s.x * vec3f(
        sin_theta * cos(s.z),
        sin_theta * sin(s.z),
        cos(s.y)
    );
}

struct Ray {
    orig: vec3f,
    dir: vec3f,
};

struct Hit {
    hit: bool,
    tmin: f32,
    tmax: f32,
};

// https://www.reddit.com/r/opengl/comments/8ntzz5/comment/dzyqwgr
fn ray_aabb(box_min: vec3f, box_max: vec3f, r: Ray) -> Hit {
    let inv_dir = 1. / r.dir;
    let tbot = inv_dir * (box_min - r.orig);
    let ttop = inv_dir * (box_max - r.orig);
    let tmin = min(ttop, tbot);
    let tmax = max(ttop, tbot);
    var t = max(tmin.xx, tmin.yz);
    let t0 = max(t.x, t.y);
    t = min(tmax.xx, tmax.yz);
    let t1 = min(t.x, t.y);
    var hit: Hit;
    hit.tmin = t0;
    hit.tmax = t1;
    hit.hit = t1 > max(t0, 0.);
    return hit;
}

/*__________ hash function collection _________*/
// sources: https://nullprogram.com/blog/2018/07/31/
//          https://www.shadertoy.com/view/WttXWX

fn triple32(x_in: u32) -> u32 {
    var x = x_in;
    x ^= x >> 17;
    x *= 0xed5ad4bb;
    x ^= x >> 11;
    x *= 0xac4c1b51;
    x ^= x >> 15;
    x *= 0x31848bab;
    x ^= x >> 14;
    return x;
}

fn hash_1uu(v: u32) -> u32 {
    return triple32(bitcast<u32>(v));
}

fn hash_2uu(v: vec2u) -> u32 {
    return triple32(bitcast<u32>(v.x) + triple32(bitcast<u32>(v.y)));
}

fn hash_3uu(v: vec3u) -> u32 {
    return triple32(bitcast<u32>(v.x) + triple32(bitcast<u32>(v.y) + triple32(bitcast<u32>(v.z))));
}

fn hash_4uu(v: vec4u) -> u32 {
    return triple32(bitcast<u32>(v.x) + triple32(bitcast<u32>(v.y) + triple32(bitcast<u32>(v.z) + triple32(bitcast<u32>(v.w)))));
}

fn hash_1iu(v: i32) -> u32 {
    return triple32(bitcast<u32>(v));
}

fn hash_2iu(v: vec2i) -> u32 {
    return triple32(bitcast<u32>(v.x) + triple32(bitcast<u32>(v.y)));
}

fn hash_3iu(v: vec3i) -> u32 {
    return triple32(bitcast<u32>(v.x) + triple32(bitcast<u32>(v.y) + triple32(bitcast<u32>(v.z))));
}

fn hash_4iu(v: vec4i) -> u32 {
    return triple32(bitcast<u32>(v.x) + triple32(bitcast<u32>(v.y) + triple32(bitcast<u32>(v.z) + triple32(bitcast<u32>(v.w)))));
}

fn hash_1fu(v: f32) -> u32 {
    return triple32(bitcast<u32>(v));
}

fn hash_2fu(v: vec2f) -> u32 {
    return triple32(bitcast<u32>(v.x) + triple32(bitcast<u32>(v.y)));
}

fn hash_3fu(v: vec3f) -> u32 {
    return triple32(bitcast<u32>(v.x) + triple32(bitcast<u32>(v.y) + triple32(bitcast<u32>(v.z))));
}

fn hash_4fu(v: vec4f) -> u32 {
    return triple32(bitcast<u32>(v.x) + triple32(bitcast<u32>(v.y) + triple32(bitcast<u32>(v.z) + triple32(bitcast<u32>(v.w)))));
}

fn hash_1ui(v: u32) -> i32 {
    return i32(hash_1uu(v));
}

fn hash_2ui(v: vec2u) -> i32 {
    return i32(hash_2uu(v));
}

fn hash_3ui(v: vec3u) -> i32 {
    return i32(hash_3uu(v));
}

fn hash_4ui(v: vec4u) -> i32 {
    return i32(hash_4uu(v));
}

fn hash_1ii(v: i32) -> i32 {
    return i32(hash_1iu(v));
}

fn hash_2ii(v: vec2i) -> i32 {
    return i32(hash_2iu(v));
}

fn hash_3ii(v: vec3i) -> i32 {
    return i32(hash_3iu(v));
}

fn hash_4ii(v: vec4i) -> i32 {
    return i32(hash_4iu(v));
}

fn hash_1fi(v: f32) -> i32 {
    return i32(hash_1fu(v));
}

fn hash_2fi(v: vec2f) -> i32 {
    return i32(hash_2fu(v));
}

fn hash_3fi(v: vec3f) -> i32 {
    return i32(hash_3fu(v));
}

fn hash_4fi(v: vec4f) -> i32 {
    return i32(hash_4fu(v));
}

fn hash_1uf(v: u32) -> f32 {
    return f32(hash_1uu(v)) / 4294967295.;
}

fn hash_2uf(v: vec2u) -> f32 {
    return f32(hash_2uu(v)) / 4294967295.;
}

fn hash_3uf(v: vec3u) -> f32 {
    return f32(hash_3uu(v)) / 4294967295.;
}

fn hash_4uf(v: vec4u) -> f32 {
    return f32(hash_4uu(v)) / 4294967295.;
}

fn hash_1if(v: i32) -> f32 {
    return f32(hash_1iu(v)) / 4294967295.;
}

fn hash_2if(v: vec2i) -> f32 {
    return f32(hash_2iu(v)) / 4294967295.;
}

fn hash_3if(v: vec3i) -> f32 {
    return f32(hash_3iu(v)) / 4294967295.;
}

fn hash_4if(v: vec4i) -> f32 {
    return f32(hash_4iu(v)) / 4294967295.;
}

fn hash_1ff(v: f32) -> f32 {
    return f32(hash_1fu(v)) / 4294967295.;
}

fn hash_2ff(v: vec2f) -> f32 {
    return f32(hash_2fu(v)) / 4294967295.;
}

fn hash_3ff(v: vec3f) -> f32 {
    return f32(hash_3fu(v)) / 4294967295.;
}

fn hash_4ff(v: vec4f) -> f32 {
    return f32(hash_4fu(v)) / 4294967295.;
}

/*_______________________________________________

flim - Filmic Color Transform

this is a port of flim v1.2.0 for GLSL/Shadertoy.

input color space: Linear BT.709 I-D65
output color space: Linear BT.709 I-D65 / sRGB (depends on arguments)

author:
  bean (beans_please on Shadertoy)

original repo:
  https://github.com/bean-mhm/flim

original shader:
  https://www.shadertoy.com/view/dd2yDz

_______________________________________________*/

// parameters

const flim_pre_exposure = 4.3;
const flim_pre_formation_filter = vec3f(1.);
const flim_pre_formation_filter_strength = 0.;

const flim_extended_gamut_red_scale = 1.05;
const flim_extended_gamut_green_scale = 1.12;
const flim_extended_gamut_blue_scale = 1.045;
const flim_extended_gamut_red_rot = .5;
const flim_extended_gamut_green_rot = 2.;
const flim_extended_gamut_blue_rot = .1;
const flim_extended_gamut_red_mul = 1.;
const flim_extended_gamut_green_mul = 1.;
const flim_extended_gamut_blue_mul = 1.;

const flim_sigmoid_log2_min = -10.;
const flim_sigmoid_log2_max = 22.;
const flim_sigmoid_toe_x = .44;
const flim_sigmoid_toe_y = .28;
const flim_sigmoid_shoulder_x = .591;
const flim_sigmoid_shoulder_y = .779;

const flim_negative_film_exposure = 6.;
const flim_negative_film_density = 5.;

const flim_print_backlight = vec3f(1);
const flim_print_film_exposure = 6.;
const flim_print_film_density = 27.5;

const flim_luminance_weights = vec3f(.3, .5, .2);
const flim_black_point = -1.; // -1 = auto
const flim_post_formation_filter = vec3f(1);
const flim_post_formation_filter_strength = 0.;
const flim_midtone_saturation = 1.02;

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

const flim_luminance_weights_norm =
    flim_luminance_weights / (
        flim_luminance_weights.x 
        + flim_luminance_weights.y
        + flim_luminance_weights.z
    );

fn flim_wrap(v: f32, start: f32, end: f32) -> f32 {
    return start + ((v - start) % (end - start));
}

fn flim_remap(
    v: f32,
    inp_start: f32,
    inp_end: f32,
    out_start: f32,
    out_end: f32
) -> f32 {
    return out_start
        + ((out_end - out_start) / (inp_end - inp_start)) * (v - inp_start);
}

fn flim_remap_clamp(
    v: f32,
    inp_start: f32,
    inp_end: f32,
    out_start: f32,
    out_end: f32
) -> f32 {
    let t = saturate((v - inp_start) / (inp_end - inp_start));
    return out_start + t * (out_end - out_start);
}

fn flim_remap01(
    v: f32,
    inp_start: f32,
    inp_end: f32
) -> f32 {
    return saturate((v - inp_start) / (inp_end - inp_start));
}

fn flim_rgb_to_hsv(rgb: vec3f) -> vec3f {
    let cmax = max(rgb[0], max(rgb[1], rgb[2]));
    let cmin = min(rgb[0], min(rgb[1], rgb[2]));
    let cdelta = cmax - cmin;

    var h = 0.;
    var s = 0.;
    let v = cmax;

    if (cmax != 0.) {
        s = cdelta / cmax;
    }

    if (s != 0.) {
        let c = (vec3f(cmax) - rgb.xyz) / cdelta;

        if (rgb.x == cmax) {
            h = c[2] - c[1];
        }
        else if (rgb.y == cmax) {
            h = 2. + c[0] - c[2];
        }
        else {
            h = 4. + c[1] - c[0];
        }

        h /= 6.;

        if (h < 0.) {
            h += 1.;
        }
    }

    return vec3f(h, s, v);
}

fn flim_hsv_to_rgb(hsv: vec3f) -> vec3f {
    var h = fract(hsv[0]);
    var s = hsv[1];
    var v = hsv[2];
    var rgb = vec3f(v, v, v);
    if (s != 0.) {
        h *= 6.;
        var i: i32 = i32(floor(h));
        let f = h - f32(i);
        rgb = vec3f(f, f, f);
        let p = v * (1. - s);
        let q = v * (1. - (s * f));
        let t = v * (1. - (s * (1. - f)));

        if (i == 0) {
            rgb = vec3f(v, t, p);
        }
        else if (i == 1) {
            rgb = vec3f(q, v, p);
        }
        else if (i == 2) {
            rgb = vec3f(p, v, t);
        }
        else if (i == 3) {
            rgb = vec3f(p, q, v);
        }
        else if (i == 4) {
            rgb = vec3f(t, p, v);
        }
        else {
            rgb = vec3f(v, p, q);
        }
    }

    return rgb;
}

fn flim_adjust_hsv(col: vec3f, hue: f32, sat: f32, value: f32) -> vec3f {
    var hsv: vec3f = flim_rgb_to_hsv(col);

    hsv[0] = fract(hsv[0] + hue + .5);
    hsv[1] = saturate(hsv[1] * sat);
    hsv[2] = hsv[2] * value;

    return flim_hsv_to_rgb(hsv);
}

fn flim_rgb_avg(col: vec3f) -> f32 {
    return dot(col, vec3f(1. / 3.));
}

fn flim_rgb_sum(col: vec3f) -> f32 {
    return dot(col, vec3f(1.));
}

fn flim_rgb_max(col: vec3f) -> f32 {
    return max(max(col.x, col.y), col.z);
}

fn flim_rgb_min(col: vec3f) -> f32 {
    return min(min(col.x, col.y), col.z);
}

fn flim_rgb_uniform_offset(col: vec3f, black_point: f32, white_point: f32) -> vec3f {
    let mono = dot(col, flim_luminance_weights_norm);
    
    // avoid division by zero
    if (abs(mono) < .0001) {
        return col;
    }
    
    let mono2: f32 = flim_remap01(
        mono,
        min(black_point, .999),
        1. - min(white_point, .999)
    );
    return col * (mono2 / mono);
}

fn flim_rgb_sweep(hue: f32) -> vec3f {
    let hue2: f32 = flim_wrap(hue * 360., 0., 360.);

    var col: vec3f = vec3f(1, 0, 0);
    col = mix(col, vec3f(1, 1, 0), flim_remap01(hue2, 0., 60.));
    col = mix(col, vec3f(0, 1, 0), flim_remap01(hue2, 60., 120.));
    col = mix(col, vec3f(0, 1, 1), flim_remap01(hue2, 120., 180.));
    col = mix(col, vec3f(0, 0, 1), flim_remap01(hue2, 180., 240.));
    col = mix(col, vec3f(1, 0, 1), flim_remap01(hue2, 240., 300.));
    col = mix(col, vec3f(1, 0, 0), flim_remap01(hue2, 300., 360.));
    
    return col;
}

fn flim_rgb_exposure_sweep_test(uv0to1: vec2f) -> vec3f {
    let hue = 1. - uv0to1.y;
    let exposure = flim_remap(uv0to1.x, 0., 1., -5., 10.);
    return flim_rgb_sweep(hue) * pow(2., exposure);
}

// https://www.desmos.com/calculator/khkztixyeu
fn flim_super_sigmoid(
    v_: f32,
    toe_x_: f32,
    toe_y_: f32,
    shoulder_x_: f32,
    shoulder_y_: f32
) -> f32 {
    // clip
    var v = saturate(v_);
    var toe_x = saturate(toe_x_);
    var toe_y = saturate(toe_y_);
    var shoulder_x = saturate(shoulder_x_);
    var shoulder_y = saturate(shoulder_y_);

    // calculate straight line slope
    let slope = (shoulder_y - toe_y) / (shoulder_x - toe_x);

    // toe
    if (v < toe_x) {
        let toe_pow = slope * toe_x / toe_y;
        return toe_y * pow(v / toe_x, toe_pow);
    }

    // straight line
    if (v < shoulder_x) {
        let intercept = toe_y - (slope * toe_x);
        return slope * v + intercept;
    }

    // shoulder
    let shoulder_pow =
        -slope / (
            ((shoulder_x - 1.) / pow(1. - shoulder_x, 2.))
            * (1. - shoulder_y)
        );
    return saturate(
        (1. - pow(1. - (v - shoulder_x) / (1. - shoulder_x), shoulder_pow))
        * (1. - shoulder_y)
        + shoulder_y
    );
}

fn flim_dye_mix_factor(mono: f32, max_density: f32) -> f32 {
    // log2 and map range
    let offset = pow(2., flim_sigmoid_log2_min);
    var fac = flim_remap01(
        log2(mono + offset),
        flim_sigmoid_log2_min,
        flim_sigmoid_log2_max
    );

    // calculate amount of exposure from 0 to 1
    fac = flim_super_sigmoid(
        fac,
        flim_sigmoid_toe_x,
        flim_sigmoid_toe_y,
        flim_sigmoid_shoulder_x,
        flim_sigmoid_shoulder_y
    );

    // calculate dye density
    fac *= max_density;

    // mix factor
    fac = pow(2., -fac);

    // clip and return
    return saturate(fac);
}

fn flim_rgb_color_layer(
    col: vec3f,
    sensitivity_tone: vec3f,
    dye_tone: vec3f,
    max_density: f32
) -> vec3f {
    // normalize
    let sensitivity_tone_norm =
        sensitivity_tone / flim_rgb_sum(sensitivity_tone);
    let dye_tone_norm = dye_tone / flim_rgb_max(dye_tone);

    // dye mix factor
    let mono = dot(col, sensitivity_tone_norm);
    let mix_fac = flim_dye_mix_factor(mono, max_density);

    // dye mixing
    return mix(dye_tone_norm, vec3f(1), mix_fac);
}

fn flim_rgb_develop(col_: vec3f, exposure: f32, max_density: f32) -> vec3f {
    // exposure
    var col = col_ * pow(2., exposure);

    // blue-sensitive layer
    var result = flim_rgb_color_layer(
        col,
        vec3f(0, 0, 1),
        vec3f(1, 1, 0),
        max_density
    );

    // green-sensitive layer
    result *= flim_rgb_color_layer(
        col,
        vec3f(0, 1, 0),
        vec3f(1, 0, 1),
        max_density
    );

    // red-sensitive layer
    result *= flim_rgb_color_layer(
        col,
        vec3f(1, 0, 0),
        vec3f(0, 1, 1),
        max_density
    );

    return result;
}

fn flim_gamut_extension_mat_row(
    primary_hue: f32,
    scale: f32,
    rotate: f32,
    mul: f32
) -> vec3f {
    var result = flim_hsv_to_rgb(vec3(
        flim_wrap(primary_hue + (rotate / 360.), 0., 1.),
        1. / scale,
        1.
    ));
    result /= flim_rgb_sum(result);
    result *= mul;
    return result;
}

fn flim_gamut_extension_mat(
    red_scale: f32,
    green_scale: f32,
    blue_scale: f32,
    red_rot: f32,
    green_rot: f32,
    blue_rot: f32,
    red_mul: f32,
    green_mul: f32,
    blue_mul: f32
) -> mat3x3f {
    var m: mat3x3f;
    m[0] = flim_gamut_extension_mat_row(
        0.,
        red_scale,
        red_rot,
        red_mul
    );
    m[1] = flim_gamut_extension_mat_row(
        1. / 3.,
        green_scale,
        green_rot,
        green_mul
    );
    m[2] = flim_gamut_extension_mat_row(
        2. / 3.,
        blue_scale,
        blue_rot,
        blue_mul
    );
    return m;
}

fn flim_negative_and_print(col_: vec3f, backlight_ext: vec3f) -> vec3f {
    // develop negative
    var col = flim_rgb_develop(
        col_,
        flim_negative_film_exposure,
        flim_negative_film_density
    );

    // backlight
    col *= backlight_ext;

    // develop print
    col = flim_rgb_develop(
        col,
        flim_print_film_exposure,
        flim_print_film_density
    );

    return col;
}

fn flim_inverse_mat3(m: mat3x3f) -> mat3x3f {
    let a = m[0].x;
    let b = m[1].x;
    let c = m[2].x;

    let d = m[0].y;
    let e = m[1].y;
    let f = m[2].y;

    let g = m[0].z;
    let h = m[1].z;
    let i = m[2].z;

    let A =  (e * i - f * h);
    let B = -(d * i - f * g);
    let C =  (d * h - e * g);

    let D = -(b * i - c * h);
    let E =  (a * i - c * g);
    let F = -(a * h - b * g);

    let G =  (b * f - c * e);
    let H = -(a * f - c * d);
    let I =  (a * e - b * d);

    let inv_det = 1. / (a * A + b * B + c * C);

    // determinant must not be zero
    return mat3x3f(
        vec3f(A, D, G) * inv_det,
        vec3f(B, E, H) * inv_det,
        vec3f(C, F, I) * inv_det
    );
}

// apply flim
// - set "convert_to_srgb" to false if you manually apply the
//   sRGB OETF after applying flim.
fn flim(col_: vec3f, convert_to_srgb: bool) -> vec3f {
    // eliminate negative values
    var col = max(col_, vec3f(0.));

    // pre-Exposure
    col *= pow(2., flim_pre_exposure);

    // clip very large values for float precision issues
    col = min(col, vec3f(5000.));

    // gamut extension matrix (Linear BT.709)
    let extend_mat = flim_gamut_extension_mat(
        flim_extended_gamut_red_scale,
        flim_extended_gamut_green_scale,
        flim_extended_gamut_blue_scale,
        flim_extended_gamut_red_rot,
        flim_extended_gamut_green_rot,
        flim_extended_gamut_blue_rot,
        flim_extended_gamut_red_mul,
        flim_extended_gamut_green_mul,
        flim_extended_gamut_blue_mul
    );
    let extend_mat_inv = flim_inverse_mat3(extend_mat);

    // backlight in the extended gamut
    let backlight_ext = flim_print_backlight * extend_mat;

    // upper limit in the print (highlight cap)
    const big = 10000000.;
    let white_cap = flim_negative_and_print(vec3f(big), backlight_ext);

    // pre-formation filter
    col *= mix(
        vec3f(1),
        flim_pre_formation_filter,
        flim_pre_formation_filter_strength
    );

    // convert to the extended gamut
    col *= extend_mat;

    // negative & print
    col = flim_negative_and_print(col, backlight_ext);
    
    // white cap
    col /= white_cap;
    
    // black cap (-1 = auto)
    if (flim_black_point == -1.) {
        var black_cap = flim_negative_and_print(vec3f(0.), backlight_ext);
        black_cap /= white_cap;
        
        col = flim_rgb_uniform_offset(
            col,
            dot(black_cap, flim_luminance_weights_norm),
            0.
        );
    }
    else {
        col = flim_rgb_uniform_offset(col, flim_black_point / 1000., 0.);
    }

    // convert back from the extended gamut and clip out-of-gamut triplets
    col *= extend_mat_inv;
    col = max(col, vec3f(0.));

    // post-formation filter
    col *= mix(
        vec3f(1),
        flim_post_formation_filter,
        flim_post_formation_filter_strength
    );

    // clip
    col = saturate(col);

    // midtone saturation
    let mono = dot(col, flim_luminance_weights_norm);
    let midtone_fac = max(1. - (abs(mono - .5) / .45), 0.);
    col = mix(
        col,
        flim_adjust_hsv(col, .5, flim_midtone_saturation, 1.),
        midtone_fac
    );

    // clip
    col = saturate(col);

    // OETF
    if (convert_to_srgb) {
        col = linear_bt709_id65_to_srgb(col);
    }

    return col;
}

/*________________ end of flim ________________*/

fn screen_to_uv(frag_coord: vec2f) -> vec2f {
    return (2. * frag_coord - vec2f(RES)) /
           f32(min(RES.x, RES.y));
}

fn icoord_to_world(icoord: vec3i) -> vec3f {
    let p_norm = (vec3f(icoord) + .5) / vec3f(GRID_RES);
    return (p_norm - .5) * GRID_DIMS;
}

// the following will be replaced with colormap functions. see "config.py".
// [colormaps]

// the following will be replaced with a user-provided definition for the
// function shade_cell. see "config.py".
// [user-functions]

fn grid_fetch(icoord: vec3i) -> vec3f {
    return shade_cell(
        icoord,
        textureLoad(render_grid, icoord).x
    );
}

fn grid_sample(p: vec3f) -> vec3f {
    // de-center and normalize to [0, 1]
    let p_norm = p / GRID_DIMS + .5;

    if (USE_TRILINEAR && IS_2D) {
        // bottom left cell index (float)
        let fcoord = p_norm.xy * vec2f(GRID_RES.xy) - .5;

        // bottom left cell index (integer)
        let icoord_bl = vec2i(floor(fcoord));

        // blending weights
        let weights = fract(fcoord);

        // handle out-of-bounds
        if (any(icoord_bl >= GRID_RES.xy)
            || any((icoord_bl + 1) < vec2i(0))) {
            return grid_fetch(vec3i(icoord_bl, 0));
        }
        
        // fetch all 4 corners
        let v00 = grid_fetch(vec3i(icoord_bl, 0));
        let v01 = grid_fetch(vec3i(icoord_bl + vec2i(0, 1), 0));
        let v10 = grid_fetch(vec3i(icoord_bl + vec2i(1, 0), 0));
        let v11 = grid_fetch(vec3i(icoord_bl + vec2i(1, 1), 0));

        // interpolate
        return mix(
            mix(
                v00,
                v10,
                weights.x
            ),
            mix(
                v01,
                v11,
                weights.x
            ),
            weights.y
        );
    }
    else if (USE_TRILINEAR && !IS_2D) {
        // bottom back left cell index (float)
        let fcoord = p_norm * vec3f(GRID_RES) - .5;

        // bottom back left cell index (integer)
        let icoord_bbl = vec3i(floor(fcoord));

        // blending weights
        let weights = fract(fcoord);

        // handle out-of-bounds
        if (any(icoord_bbl >= GRID_RES)
            || any((icoord_bbl + 1) < vec3i(0))) {
            return grid_fetch(icoord_bbl);
        }
        
        // fetch all 8 corners
        let v000 = grid_fetch(icoord_bbl);
        let v001 = grid_fetch(icoord_bbl + vec3i(0, 0, 1));
        let v010 = grid_fetch(icoord_bbl + vec3i(0, 1, 0));
        let v011 = grid_fetch(icoord_bbl + vec3i(0, 1, 1));
        let v100 = grid_fetch(icoord_bbl + vec3i(1, 0, 0));
        let v101 = grid_fetch(icoord_bbl + vec3i(1, 0, 1));
        let v110 = grid_fetch(icoord_bbl + vec3i(1, 1, 0));
        let v111 = grid_fetch(icoord_bbl + vec3i(1, 1, 1));

        // interpolate
        return mix(
            mix(
                mix(
                    v000,
                    v100,
                    weights.x
                ),
                mix(
                    v010,
                    v110,
                    weights.x
                ),
                weights.y
            ),
            mix(
                mix(
                    v001,
                    v101,
                    weights.x
                ),
                mix(
                    v011,
                    v111,
                    weights.x
                ),
                weights.y
            ),
            weights.z
        );
    } else {
        // get 3D indices in the grid
        let icoord = vec3i(floor(
            p_norm * vec3f(GRID_RES)
        ));
        return grid_fetch(icoord);
    }
}

fn render(frag_coord: vec2f) -> vec3f {
    if (IS_2D) {
        let scale = min(
            f32(RES.x) / GRID_DIMS.x,
            f32(RES.y) / GRID_DIMS.y
        );
        let frag_coord_centered = frag_coord - .5 * vec2f(RES);
        let p = frag_coord_centered / scale;
        let col = grid_sample(vec3f(p, 0));
        return BG_COL + col;
    }

    let uv = screen_to_uv(frag_coord);

    // setup camera
    let cam_zoom = ubo.cam_fov_degrees / 90.;
    let cam_forward = normalize(
        ubo.cam_lookat - ubo.cam_pos
    );
    let cam_right = normalize(cross(cam_forward, ubo.cam_world_up));
    let cam_up = cross(cam_right, cam_forward);

    // generate ray
    var ray: Ray;
    ray.orig = ubo.cam_pos;
    ray.dir = normalize(
        cam_forward
        + cam_right * (uv.x * cam_zoom)
        + cam_up * (uv.y * cam_zoom)
    );

    // intersect container box
    var hit = ray_aabb(
        -.5 * GRID_DIMS,
        .5 * GRID_DIMS,
        ray
    );

    // shade
    var col = vec3f(0);
    if (hit.hit) {
        // initial t along the ray
        var step_size =
            RAYMARCH_STEP
            + RAYMARCH_STEP_JITTER
            * (hash_3ff(vec3f(frag_coord, -777.)) - .5);
        var t = hit.tmin + step_size;

        // keep stepping forward until we reach hit.tmax
        loop {
            if (t >= hit.tmax) {
                break;
            }

            // point along the ray inside the volume
            let p = ray.orig + t * ray.dir;

            // sample the 3D volume
            var sample = grid_sample(p);

            // collect the sample
            col += sample * step_size;

            // step forward
            step_size =
                RAYMARCH_STEP
                + RAYMARCH_STEP_JITTER
                * (hash_3ff(vec3f(frag_coord, -777. + t)) - .5);
            t += step_size;
        }
    }

    return BG_COL + col;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    extract_uniforms();

    let frag_coord = in.uv * vec2f(RES);

    // multisampling
    var col = vec3f(0);
    for (var i: i32 = 0; i < N_SAMPLES_PER_PIXEL; i++) {
        let rand = hash_3ff(vec3f(frag_coord, f32(i + 800)));
        let jitter = vec2f(rand, hash_1ff(rand)) - .5;

        col += render(frag_coord.xy + jitter);
    }
    col /= f32(N_SAMPLES_PER_PIXEL);
    
    // flim
    if (APPLY_FLIM) {
        col = flim(col, false);
    }

    // sRGB OETF
    col = linear_bt709_id65_to_srgb(col);

    // dithering
    col += (hash_3ff(vec3f(frag_coord, 92.147)) - .5) / 256.;
    col = saturate(col);
    
    // output
    return vec4f(col, 1);
}
