// Screen-space ambient occlusion and its blur, plus the full-screen passes
// that share its bind group layout: transparency composite, copy, section cap.

struct Uniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    eye: vec4<f32>,
    lights: array<vec4<f32>, 3>,
    params: vec4<f32>,
    // x: sample radius in world units, y: how far an occluder has to stand
    // clear -- in depth, and above the tangent plane -- z/w: target size.
    ssao: vec4<f32>,
    // x: tan(fovy/2) * aspect, y: tan(fovy/2), z: exposure, w: unused.
    proj_info: vec4<f32>,
    // Some fields are unused here but must be declared: a missing one silently
    // shifts every field after it. Must match shader.wgsl
    // (`the_shaders_agree_about_the_uniform_block`).
    section: vec4<f32>,
    cap: vec4<f32>,
    light_vp: mat4x4<f32>,
    ground: vec4<f32>,
    sh: array<vec4<f32>, 9>,
    kernel: array<vec4<f32>, 16>,
    // xy: how far this frame of a supersampled view is shifted, in normalised
    // device coordinates. Zero on the first frame, which is the one drawn
    // while the view moves.
    jitter: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
// rgb: view-space normal, a: view-space depth (positive, 0 where nothing was drawn).
@group(0) @binding(1) var gbuf: texture_2d<f32>;
@group(0) @binding(2) var ao_in: texture_2d<f32>;

// The prepass that fills `gbuf` lives in shader.wgsl, because it draws the same
// geometry through the same bind group layout as the shaded pass.

// -- full-screen passes ----------------------------------------------------

@vertex
fn vs_fullscreen(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    // One oversized triangle covering the viewport.
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

/// View-space position of the fragment at `frag_xy` with linear depth `depth`.
///
/// Undoes this frame's supersampling `jitter` (device y runs up, pixel y
/// down), matching the shifted matrix the samples are projected through.
fn view_pos(frag_xy: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = frag_xy / u.ssao.zw * 2.0 - 1.0 - vec2<f32>(u.jitter.x, -u.jitter.y);
    let dir = vec3<f32>(ndc.x * u.proj_info.x, -ndc.y * u.proj_info.y, -1.0);
    return dir * depth;
}

/// Kernel rotation for a pixel, as a fraction of a turn.
///
/// The sixteen pixels of a 4x4 tile get sixteen evenly spaced angles, so the
/// 4x4 blur averages every orientation exactly once and leaves no residual
/// pattern (hash noise does, and shows as cross-hatching). Bit-reversed order
/// gives neighbouring pixels widely separated angles.
fn rotation_turns(px: vec2<i32>) -> f32 {
    let i = u32((px.y & 3) * 4 + (px.x & 3));
    let scrambled = ((i & 1u) << 3u) | ((i & 2u) << 1u) | ((i & 4u) >> 1u) | ((i & 8u) >> 3u);
    return f32(scrambled) / 16.0;
}

@fragment
fn fs_ssao(@builtin(position) frag: vec4<f32>) -> @location(0) f32 {
    let px = vec2<i32>(frag.xy);
    let g = textureLoad(gbuf, px, 0);
    let depth = g.w;
    if (depth <= 1e-5) {
        return 1.0; // background
    }
    let n = g.xyz;
    let origin = view_pos(frag.xy, depth);

    // Rotate the kernel per pixel so the blur below averages sixteen distinct
    // orientations: 16 samples do the work of 256.
    let ang = rotation_turns(px) * 6.2831853;
    let rv = vec3<f32>(cos(ang), sin(ang), 0.0);
    let t = normalize(rv - n * dot(rv, n));
    let b = cross(n, t);
    let tbn = mat3x3<f32>(t, b, n);

    let radius = u.ssao.x;
    var occlusion = 0.0;
    for (var k = 0u; k < 16u; k = k + 1u) {
        let sample_v = origin + (tbn * u.kernel[k].xyz) * radius;
        let clip = u.proj * vec4<f32>(sample_v, 1.0);
        if (clip.w <= 0.0) {
            continue;
        }
        let suv = (clip.xy / clip.w) * vec2<f32>(0.5, -0.5) + 0.5;
        if (any(suv < vec2<f32>(0.0)) || any(suv > vec2<f32>(1.0))) {
            continue;
        }
        let spx = vec2<i32>(suv * u.ssao.zw);
        let scene_depth = textureLoad(gbuf, spx, 0).w;
        if (scene_depth <= 1e-5) {
            continue; // background never occludes
        }
        let sample_depth = -sample_v.z;
        // Ignore occluders far outside the radius, or a thin wall in the
        // foreground would cast occlusion onto whatever is behind it.
        let range = smoothstep(0.0, 1.0, radius / max(abs(sample_depth - scene_depth), 1e-6));
        // Only what rises above the surface's tangent plane occludes it. The
        // depth test reads the scene at the centre of the sample's pixel, not
        // on its ray, so alone it lets a slanted face occlude itself
        // (`occlusion_leaves_a_convex_solid_alone`).
        let scene_v = view_pos(vec2<f32>(spx) + 0.5, scene_depth);
        if (scene_depth <= sample_depth - u.ssao.y && dot(scene_v - origin, n) > u.ssao.y) {
            occlusion = occlusion + range;
        }
    }
    return clamp(1.0 - occlusion / 16.0, 0.0, 1.0);
}

@fragment
fn fs_blur(@builtin(position) frag: vec4<f32>) -> @location(0) f32 {
    let px = vec2<i32>(frag.xy);
    let center = textureLoad(gbuf, px, 0).w;
    if (center <= 1e-5) {
        return 1.0;
    }
    let size = vec2<i32>(u.ssao.zw);
    var sum = 0.0;
    var weight = 0.0;
    // 4x4 box, rejecting samples from a different surface so occlusion does not
    // bleed across silhouettes. The tolerance is loose enough to keep all 16
    // taps on a steeply raking surface; dropped taps leave visible noise.
    let tolerance = max(center * 0.05, u.ssao.x);
    for (var dy = -2; dy <= 1; dy = dy + 1) {
        for (var dx = -2; dx <= 1; dx = dx + 1) {
            let q = clamp(px + vec2<i32>(dx, dy), vec2<i32>(0), size - 1);
            let d = textureLoad(gbuf, q, 0).w;
            if (d > 1e-5 && abs(d - center) < tolerance) {
                sum = sum + textureLoad(ao_in, q, 0).r;
                weight = weight + 1.0;
            }
        }
    }
    if (weight == 0.0) {
        return textureLoad(ao_in, px, 0).r;
    }
    return sum / weight;
}

// -- transparency composite -------------------------------------------------

/// Lays the accumulated transparency over the opaque frame.
///
/// Shares the occlusion passes' bind group layout, so `gbuf` is the
/// accumulation buffer here and `ao_in` the revealage.
@fragment
fn fs_composite(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(pos.xy);
    let accum = textureLoad(gbuf, p, 0);
    let reveal = textureLoad(ao_in, p, 0).r;
    // Nothing transparent covered this pixel: leave the opaque frame alone
    // rather than blending a divide-by-nothing over it.
    if (reveal > 0.9999) {
        discard;
    }
    // The weights divide out, leaving the average colour; the revealage is how
    // much of the frame behind still shows. The colour is already tone-mapped
    // and encoded, the same space as the opaque frame.
    let avg = accum.rgb / max(accum.a, 1.0e-5);
    return vec4<f32>(avg, 1.0 - reveal);
}

// -- supersampling ------------------------------------------------------------

/// A texel straight through: the resolved frame into the running average (the
/// blend state averages) and the average onto the screen. The formats' sRGB
/// encoding keeps the average in linear light, like the multisample resolve.
@fragment
fn fs_copy(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(gbuf, vec2<i32>(pos.xy), 0);
}

// -- section cap ------------------------------------------------------------

/// How far the cap is eroded, in pixels.
///
/// A leak through a crack in the mesh is a thin stroke; a cut face is an area.
/// A minimum over this radius removes anything thinner than twice it, pulling
/// the cut face in by the same amount.
const CAP_ERODE: i32 = 2;

/// Lays the section cap over the frame, eroded.
///
/// Both textures of the shared post layout are bound to the same mask here;
/// only `gbuf` is read.
@fragment
fn fs_cap_fill(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(pos.xy);
    let size = vec2<i32>(textureDimensions(gbuf));
    var cover = 1.0;
    for (var dy = -CAP_ERODE; dy <= CAP_ERODE; dy = dy + 1) {
        for (var dx = -CAP_ERODE; dx <= CAP_ERODE; dx = dx + 1) {
            let q = clamp(p + vec2<i32>(dx, dy), vec2<i32>(0), size - 1);
            cover = min(cover, textureLoad(gbuf, q, 0).r);
        }
    }
    if (cover <= 0.0) {
        discard;
    }

    // Lit facing along the plane's normal, in the same studio as everything
    // else.
    let n = normalize(u.section.xyz);
    let irradiance = max(cap_irradiance(n), vec3<f32>(0.0));
    let l = normalize(u.lights[0].xyz);
    let direct = max(dot(n, l), 0.0) * u.lights[0].w * 0.35;
    let shaded = CAP_COLOR * (irradiance + vec3<f32>(direct)) * u.proj_info.z;
    return vec4<f32>(cap_encode(cap_tonemap(shaded)), cover);
}

/// A shade or two off neutral and darker than any part is likely to be, so the
/// model reads against it rather than merging into it.
const CAP_COLOR: vec3<f32> = vec3<f32>(0.44, 0.40, 0.36);

fn cap_irradiance(nw: vec3<f32>) -> vec3<f32> {
    // World direction into the studio's frame, as in shader.wgsl.
    let v = (u.view * vec4<f32>(nw, 0.0)).xyz;
    let n = vec3<f32>(v.x, -v.z, v.y);
    return u.sh[0].rgb * 0.282095
        + u.sh[1].rgb * (0.488603 * n.y)
        + u.sh[2].rgb * (0.488603 * n.z)
        + u.sh[3].rgb * (0.488603 * n.x)
        + u.sh[4].rgb * (1.092548 * n.x * n.y)
        + u.sh[5].rgb * (1.092548 * n.y * n.z)
        + u.sh[6].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0))
        + u.sh[7].rgb * (1.092548 * n.x * n.z)
        + u.sh[8].rgb * (0.546274 * (n.x * n.x - n.y * n.y));
}

fn cap_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn cap_encode(c: vec3<f32>) -> vec3<f32> {
    if (u.params.x < 0.5) {
        return c;
    }
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}
