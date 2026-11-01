// Shaded surfaces and the edge overlay.

struct Uniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    eye: vec4<f32>,
    // xyz: direction *towards* the light in world space, w: intensity.
    // Directions are camera-relative, so the model stays lit while orbiting.
    lights: array<vec4<f32>, 3>,
    // x: 1.0 when the target format is not sRGB and we must encode ourselves.
    // y: fraction of its eye distance by which an edge line is pulled forward.
    // z: 1.0 when surfaces are hidden, so edges must supply their own contrast.
    // w: exponent applied to sampled occlusion; 0 means occlusion is off.
    params: vec4<f32>,
    ssao: vec4<f32>,
    // x: tan(fovy/2) * aspect, y: tan(fovy/2), z: exposure,
    // w: how much occlusion dims direct light as well as ambient.
    proj_info: vec4<f32>,
    // Section plane: xyz a unit normal, w the offset along it. Everything with
    // dot(n, p) > w is cut away. A zero normal means no section at all.
    section: vec4<f32>,
    // x: radius of the quad the cap is drawn on, big enough to cover the model.
    // yzw: the middle of the model, which is where that quad is centred.
    cap: vec4<f32>,
    // World to the shadow light's clip space.
    light_vp: mat4x4<f32>,
    // x: height of the ground plane, y: how far it reaches before it fades out,
    // z: 1.0 when it is drawn at all, w: one shadow map texel in clip units.
    ground: vec4<f32>,
    // Irradiance of the studio as nine spherical-harmonic coefficients, which
    // is the whole of the diffuse lighting and needs no texture.
    sh: array<vec4<f32>, 9>,
    kernel: array<vec4<f32>, 16>,
    // xy: how far this frame of a supersampled view is shifted, in normalised
    // device coordinates. Zero on the first frame, which is the one drawn
    // while the view moves.
    jitter: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var ao_tex: texture_2d<f32>;
@group(0) @binding(2) var env_tex: texture_cube<f32>;
@group(0) @binding(3) var env_smp: sampler;
@group(0) @binding(4) var shadow_tex: texture_depth_2d;
@group(0) @binding(5) var shadow_smp: sampler_comparison;
@group(0) @binding(6) var gbuf_tex: texture_2d<f32>;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(perspective, centroid) world: vec3<f32>,
    @location(1) @interpolate(perspective, centroid) normal: vec3<f32>,
    @location(2) @interpolate(perspective, centroid) color: vec3<f32>,
    @location(3) @interpolate(perspective, centroid) alpha: f32,
};

/// Unfolds a normal from the two fixed-point numbers the vertex carries.
///
/// Must match `oct_encode`/`oct_decode` in `tess.rs`; nothing checks the two
/// against each other, and a slight disagreement shades everything wrong.
fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    let f = e * 2.0 - 1.0;
    let z = 1.0 - abs(f.x) - abs(f.y);
    let t = max(-z, 0.0);
    let s = vec2<f32>(select(t, -t, f.x >= 0.0), select(t, -t, f.y >= 0.0));
    return normalize(vec3<f32>(f + s, z));
}

/// sRGB -> linear, for the display-referred vertex colour. Done per vertex so
/// the linear value is what gets interpolated.
fn decode(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

// Per-instance placement: the three rows of a model matrix with the
// translation in `.w`, a colour multiplier and a paint. One copy of a mesh
// serves every placement of it.
//
// Every vertex stage that consumes mesh geometry declares this block, or its
// pass draws the geometry unplaced, at the origin
// (`every_mesh_stage_takes_the_instance_block`).
//
// The single-model path draws one exact-identity instance, and identity must
// be exact: `dot((1,0,0,0), p) + 0` is `p.x` to the bit, which keeps the
// stored reference views valid.
struct Inst {
    @location(3) r0: vec4<f32>,
    @location(4) r1: vec4<f32>,
    @location(5) r2: vec4<f32>,
    @location(6) tint: vec4<f32>,
    // A colour laid over the part's own, rgb in linear light, and in `a` how
    // far towards it: a selection highlight, or a colour-by-component mode.
    // Zero is no paint, exactly -- see `paint_over`.
    @location(7) paint: vec4<f32>,
}

/// Blends a colour towards the paint by its amount.
///
/// Written as `c + (p - c) * a` rather than `mix` so an amount of exactly zero
/// gives `c` back to the bit; the stored views are drawn unpainted.
fn paint_over(c: vec3<f32>, p: vec4<f32>) -> vec3<f32> {
    return c + (p.rgb - c) * p.a;
}

fn inst_point(m: Inst, p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(m.r0.xyz, p) + m.r0.w,
        dot(m.r1.xyz, p) + m.r1.w,
        dot(m.r2.xyz, p) + m.r2.w,
    );
}

// Directions ignore the translation. Not renormalised: fragment stages
// normalise after interpolation, and renormalising a unit identity normal
// would move its bits.
fn inst_dir(m: Inst, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(m.r0.xyz, v), dot(m.r1.xyz, v), dot(m.r2.xyz, v));
}

@vertex
fn vs_surface(
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec2<f32>,
    @location(2) col: vec4<f32>,
    inst: Inst,
) -> VOut {
    let world = inst_point(inst, pos);
    var o: VOut;
    o.clip = u.view_proj * vec4<f32>(world, 1.0);
    o.world = world;
    o.normal = inst_dir(inst, oct_decode(nrm));
    o.color = paint_over(decode(col.rgb * inst.tint.rgb), inst.paint);
    o.alpha = col.a * inst.tint.a;
    return o;
}

/// True where the section plane has cut this point away.
///
/// Every pass asks the same way: the shaded surface, the edge overlay and the
/// occlusion prepass, or the cut face is darkened by geometry no longer there.
fn clipped(p: vec3<f32>) -> bool {
    let n = u.section.xyz;
    return dot(n, n) > 0.5 && dot(n, p) > u.section.w;
}

fn encode(c: vec3<f32>) -> vec3<f32> {
    if (u.params.x < 0.5) {
        return c;
    }
    // Linear -> sRGB, for targets that do not convert for us.
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

/// Narkowicz's fit of the ACES filmic curve. Without a tone curve, a white part
/// under three lights clips to flat 1.0 and loses every bit of its shape.
fn tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_surface(i: VOut) -> @location(0) vec4<f32> {
    if (clipped(i.world)) {
        discard;
    }
    return vec4<f32>(encode(tonemap(shade(i))), 1.0);
}

/// The lit colour of a surface fragment, before the tone curve.
///
/// Shared with the transparent pass, which lights glass the same way and then
/// divides by its accumulated weight, so it needs the value before `tonemap`.
fn shade(i: VOut) -> vec3<f32> {
    let view = normalize(u.eye.xyz - i.world);

    // Two-sided shading: a face never shows black because its file disagrees
    // about which way the surface points.
    var n = normalize(i.normal);
    if (dot(n, n) < 0.5) {
        n = view;
    }
    if (dot(n, view) < 0.0) {
        n = -n;
    }

    // Occlusion is one sample per pixel, the MSAA target's dimensions, so the
    // pixel coordinate indexes it directly. The exponent is the intensity:
    // raw occlusion on convex CAD geometry clusters near 1.0, and a power
    // curve makes the shallow end visible.
    var ao = 1.0;
    if (u.params.w > 0.0) {
        let a = clamp(textureLoad(ao_tex, vec2<i32>(i.clip.xy), 0).r, 0.0, 1.0);
        ao = pow(a, u.params.w);
    }

    // The studio's irradiance in the direction the surface faces. A diffuse
    // response is a cosine lobe with almost no energy above the second band,
    // so nine coefficients hold all of it.
    let irradiance = max(sh_irradiance(to_studio(n)), vec3<f32>(0.0)) * ao;

    // Reflection, by the split-sum approximation: the environment prefiltered
    // by roughness, times an analytic fit of the rest. It lets a machined
    // surface reflect a shape that bends as the surface curves, which
    // analytic lights cannot.
    let r = reflect(-view, n);
    let ndv = clamp(dot(n, view), 0.0, 1.0);
    let mip = ROUGHNESS * f32(ENV_MIPS - 1u);
    let reflected = textureSampleLevel(env_tex, env_smp, to_studio(r), mip).rgb;
    // Occlusion on the reflection too, but softened: a crevice reflects less of
    // the room, though not in proportion to how enclosed it is.
    let spec = reflected * env_brdf(ndv) * mix(1.0, ao, 0.7);

    // One analytic light on top, for crispness the small prefiltered cube map
    // cannot give. The key alone: fill and rim are in the environment already.
    let l = normalize(u.lights[0].xyz);
    let h = normalize(l + view);
    let direct = max(dot(n, l), 0.0) * u.lights[0].w * 0.35
        + pow(max(dot(n, h), 0.0), 90.0) * u.lights[0].w * 0.25;

    // Contact darkening: keep some occlusion on the direct term too, or deep
    // bores stay suspiciously bright under the key light.
    let lit = irradiance + vec3<f32>(direct * mix(1.0, ao, u.proj_info.w));
    return (i.color * lit + spec) * u.proj_info.z;
}

/// Roughness of the one material: sharp enough that a panel stays a panel in
/// the reflection, dull enough not to mirror the studio at full contrast.
const ROUGHNESS: f32 = 0.30;
/// Reflectance at normal incidence: a dielectric's 4%, for painted or
/// anodised parts rather than bare metal.
const F0: f32 = 0.04;
const ENV_MIPS: u32 = 6u;

/// The other half of the split sum: how much of the prefiltered environment a
/// surface at this viewing angle actually reflects.
///
/// Karis's analytic fit of the environment BRDF, in place of a lookup table.
/// It includes geometric shadowing, so it does not climb to 1 - roughness at
/// grazing incidence as plain Fresnel-Schlick does.
fn env_brdf(ndv: f32) -> f32 {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.040, -0.040);
    let r = ROUGHNESS * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * ndv)) * r.x + r.y;
    return F0 * (-1.04 * a004 + r.z) + (1.04 * a004 + r.w);
}

/// World direction into the studio's frame.
///
/// The studio is anchored to the camera, like the analytic lights, so a part
/// never falls dark when orbited; reflections still sweep across a curved
/// surface as it turns.
///
/// View space is x right, y up, -z forward. The studio is x right, y into the
/// screen, z up.
fn to_studio(d: vec3<f32>) -> vec3<f32> {
    let v = (u.view * vec4<f32>(d, 0.0)).xyz;
    return vec3<f32>(v.x, -v.z, v.y);
}

/// Irradiance from the nine coefficients, by Ramamoorthi and Hanrahan.
fn sh_irradiance(n: vec3<f32>) -> vec3<f32> {
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

// -- transparency -----------------------------------------------------------
//
// Weighted blended order-independent transparency, after McGuire and Bavoil.
//
// Sorting cannot work here: the unit is the triangle, the order changes with
// the camera, and intersecting surfaces have no order at all. Every fragment
// accumulates with a weight falling off with depth, divided out at the end,
// giving the same picture whatever order the parts arrive in.

struct OitOut {
    @location(0) accum: vec4<f32>,
    @location(1) reveal: f32,
};

@fragment
fn fs_transparent(i: VOut) -> OitOut {
    if (clipped(i.world)) {
        discard;
    }
    let a = clamp(i.alpha, 0.0, 1.0);
    // Tone-curved and encoded here, not in the composite, so it blends in the
    // same space as the opaque frame beneath.
    let lit = encode(tonemap(shade(i)));

    // The weight: nearer fragments count for more, cubed to fall off fast.
    // The clamps keep a half-float buffer from overflowing near and
    // underflowing far. Depth is reversed, so `z` is the usual `1 - z`.
    let z = clamp(i.clip.z, 0.0, 1.0);
    let w = clamp(a * 3.0e3 * pow(z, 3.0), 1.0e-2, 3.0e3);

    var o: OitOut;
    // Premultiplied, because the composite divides by the summed alpha.
    o.accum = vec4<f32>(lit * a, a) * w;
    o.reveal = a;
    return o;
}

// -- silhouette -------------------------------------------------------------
//
// The outline a smooth surface has and its B-rep does not: a fillet turning
// away from the camera has a horizon mid-face, at no model edge, and with
// tangent edges suppressed it would have no outline at all.
//
// Found in the depth buffer rather than the mesh, which would mean walking
// every mesh edge and rebuilding a line buffer whenever the camera moves.

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // One oversized triangle rather than two: no seam down the diagonal.
    let p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0),
    );
    return vec4<f32>(p[vi], 0.0, 1.0);
}

/// Linear view depth from the prepass, or 0 where nothing was drawn.
fn view_depth(p: vec2<i32>) -> f32 {
    let size = vec2<i32>(textureDimensions(gbuf_tex));
    return textureLoad(gbuf_tex, clamp(p, vec2<i32>(0), size - 1), 0).w;
}

@fragment
fn fs_silhouette(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(pos.xy);
    let d0 = view_depth(p);
    // Only on the surface; the far side would double the line onto the floor.
    if (d0 <= 0.0) {
        discard;
    }

    // The second difference, not the first: the first is large across any
    // steeply slanted surface, the second is zero across a flat run and the
    // size of the step across a discontinuity. Background reads as 0, so the
    // outer edge comes out at about one whole depth.
    let l = view_depth(p + vec2<i32>(-1, 0));
    let r = view_depth(p + vec2<i32>(1, 0));
    let u = view_depth(p + vec2<i32>(0, -1));
    let d = view_depth(p + vec2<i32>(0, 1));
    let edge = max(abs(l + r - 2.0 * d0), abs(u + d - 2.0 * d0)) / d0;

    // Soft rather than binary: the prepass is single-sample, so a hard test
    // would put a staircase next to B-rep edges the multisampling has smoothed.
    let a = smoothstep(SILHOUETTE_LO, SILHOUETTE_HI, edge);
    if (a <= 0.0) {
        discard;
    }
    return vec4<f32>(encode(SILHOUETTE_COLOR), a);
}

/// Where a depth step stops being curvature and starts being an edge, as a
/// fraction of the distance to the eye. Tuned by eye: lower and a chamfer
/// outlines itself, higher and the line between overlapping parts goes.
const SILHOUETTE_LO: f32 = 0.010;
const SILHOUETTE_HI: f32 = 0.060;
/// The same near-black the B-rep overlay uses, so the two read as one drawing.
const SILHOUETTE_COLOR: vec3<f32> = vec3<f32>(0.06, 0.07, 0.09);

// -- ground and shadow ------------------------------------------------------

/// The model as the light sees it. Position only: the shadow map holds depth
/// and nothing else.
@vertex
fn vs_shadow(@location(0) pos: vec3<f32>, inst: Inst) -> @builtin(position) vec4<f32> {
    return u.light_vp * vec4<f32>(inst_point(inst, pos), 1.0);
}

/// A square of floor under the model, built from the vertex index.
@vertex
fn vs_ground(@builtin(vertex_index) vi: u32) -> COut {
    let corner = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let c = corner[vi] * u.ground.y;
    let p = vec3<f32>(u.cap.y + c.x, u.cap.z + c.y, u.ground.x);
    var o: COut;
    o.clip = u.view_proj * vec4<f32>(p, 1.0);
    o.world = p;
    return o;
}

/// How much of the light reaches `p`, from 0 in full shadow to 1 in the open.
///
/// Four taps on a diagonal cross: the comparison sampler already blends the
/// four texels around each tap, so they cover sixteen texels, as smoothly as a
/// 3x3 box at under half the cost.
fn lit_fraction(p: vec3<f32>) -> f32 {
    let c = u.light_vp * vec4<f32>(p, 1.0);
    if (c.w <= 0.0) {
        return 1.0;
    }
    let ndc = c.xyz / c.w;
    // Outside the light's box there is nothing to cast, which is the far floor.
    if (any(abs(ndc.xy) > vec2<f32>(1.0)) || ndc.z > 1.0) {
        return 1.0;
    }
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    // Wide enough to soften the shadow map's staircase on a large assembly,
    // narrow enough that a bolt still casts a bolt-sized shadow.
    let r = u.ground.w * 1.5;
    var sum = 0.0;
    sum += textureSampleCompare(shadow_tex, shadow_smp, uv + vec2<f32>(-r, -r), ndc.z);
    sum += textureSampleCompare(shadow_tex, shadow_smp, uv + vec2<f32>(r, -r), ndc.z);
    sum += textureSampleCompare(shadow_tex, shadow_smp, uv + vec2<f32>(-r, r), ndc.z);
    sum += textureSampleCompare(shadow_tex, shadow_smp, uv + vec2<f32>(r, r), ndc.z);
    return sum * 0.25;
}

/// The floor: the studio's own light, darkened where the model stands in it.
///
/// It fades out at its rim: a hard edge reads as a disc, and an endless floor
/// takes the model's contrast.
@fragment
fn fs_ground(i: COut) -> @location(0) vec4<f32> {
    // Not clipped by the section: the floor is the room, not the model.
    let up = vec3<f32>(0.0, 0.0, 1.0);
    // Never fully dark; the fill always gets underneath.
    let shade = mix(0.28, 1.0, lit_fraction(i.world));
    let lit = max(sh_irradiance(to_studio(up)), vec3<f32>(0.0)) * shade;
    let color = encode(tonemap(GROUND_COLOR * lit * u.proj_info.z));
    // Fade by distance from the middle of the model, over the outer third.
    let d = length(i.world.xy - u.cap.yz) / u.ground.y;
    let alpha = 1.0 - smoothstep(0.10, 0.48, d);
    // The floor writes depth, so where it has faded out it must not occlude.
    if (alpha < 0.004) {
        discard;
    }
    return vec4<f32>(color, alpha);
}

/// A shade or two off neutral and darker than any part is likely to be, so the
/// model reads against it rather than merging into it.
const GROUND_COLOR: vec3<f32> = vec3<f32>(0.075, 0.075, 0.082);

// -- section cap ------------------------------------------------------------
//
// Two draws. The first re-runs the model geometry to count, in the stencil,
// which pixels have the plane passing through material; the second lays the cut
// over them.

/// Counts the geometry on the far side of the plane from the eye.
///
/// For a closed mesh both sides carry the same parity, but a count along the
/// ray from the eye reads "inside at the plane" XOR "inside at the eye", since
/// nothing behind the eye is counted: with the camera inside a wall the cap
/// inverts. Counting from the plane outwards ends at infinity, where nothing is
/// inside anything, so the answer depends on the geometry alone.
///
/// The returned colour goes nowhere -- every channel is masked off -- but the
/// attachment has to be declared to match the pass.
@fragment
fn fs_cap_mark(i: VOut) -> @location(0) vec4<f32> {
    let n = u.section.xyz;
    let eye_cut = dot(n, n) > 0.5 && dot(n, u.eye.xyz) > u.section.w;
    if (clipped(i.world) == eye_cut) {
        discard;
    }
    return vec4<f32>(0.0);
}

struct COut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
};

/// A quad lying in the section plane, larger than the model.
///
/// Built from the vertex index, so it follows the uniform and is never stale.
/// Drawn as real geometry so the rasteriser gives the plane's own depth and
/// multisampling smooths the cut's outline.
@vertex
fn vs_cap(@builtin(vertex_index) vi: u32) -> COut {
    let n = normalize(u.section.xyz);
    // Any two axes perpendicular to the normal. Picking the world axis the
    // normal leans on least keeps the cross product well conditioned.
    var up = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(n.z) > 0.9) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    let corner = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    var o_world: vec3<f32>;
    let c = corner[vi] * u.cap.x;
    // Centred on the model dropped onto the plane, so a part far from the
    // world origin is still covered.
    let mid = u.cap.yzw;
    let origin = mid - n * (dot(n, mid) - u.section.w);
    let p = origin + t * c.x + b * c.y;
    o_world = p;
    var o: COut;
    o.clip = u.view_proj * vec4<f32>(p, 1.0);
    o.world = o_world;
    return o;
}

/// Coverage only: where the plane passes through material.
@fragment
fn fs_cap_mask() -> @location(0) f32 {
    return 1.0;
}

/// One neutral colour for every cut, since the stencil cannot say which solid
/// a pixel belongs to; warm and darker than bare metal so it reads at a glance
/// as not a surface of the part.
const CAP_COLOR: vec3<f32> = vec3<f32>(0.44, 0.40, 0.36);

// -- depth + normal prepass -------------------------------------------------
//
// Feeds the ambient-occlusion pass. Here rather than in ssao.wgsl to share
// this file's bind group layout and vertex buffer with the shaded pass.

struct GOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) vnormal: vec3<f32>,
    @location(1) vdepth: f32,
    @location(2) world: vec3<f32>,
};

@vertex
fn vs_gbuffer(@location(0) pos: vec3<f32>, @location(1) nrm: vec2<f32>, inst: Inst) -> GOut {
    let world = inst_point(inst, pos);
    let vp = u.view * vec4<f32>(world, 1.0);
    var o: GOut;
    o.clip = u.proj * vp;
    o.vnormal = (u.view * vec4<f32>(inst_dir(inst, oct_decode(nrm)), 0.0)).xyz;
    o.vdepth = -vp.z;
    o.world = world;
    return o;
}

@fragment
fn fs_gbuffer(i: GOut) -> @location(0) vec4<f32> {
    if (clipped(i.world)) {
        discard;
    }
    var n = i.vnormal;
    if (dot(n, n) < 1e-12) {
        n = vec3<f32>(0.0, 0.0, 1.0);
    }
    n = normalize(n);
    // Match the two-sided surface shading, which turns a normal round when it
    // faces away from the *eye*, not the view axis. Under perspective `n.z < 0`
    // is wrong off-axis: the far wall of a bore seen from above would turn
    // into the solid and read as fully occluded.
    let to_eye = (u.view * vec4<f32>(u.eye.xyz - i.world, 0.0)).xyz;
    if (dot(n, to_eye) < 0.0) {
        n = -n;
    }
    return vec4<f32>(n, max(i.vdepth, 1e-6));
}

struct EOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    // The part's paint, so a selected part's edges are highlighted with it.
    @location(1) paint: vec4<f32>,
    // How far across the line this fragment is, in pixels from its centre.
    // Linear: distance from a screen line is affine in screen position.
    @location(2) @interpolate(linear) across: f32,
};

// -- edge lines -------------------------------------------------------------
//
// Drawn as quads rather than hardware lines. A one-pixel hardware line under
// multisampling is covered by a varying number of samples along its length
// and looks dashed; a quad covers each pixel by its distance from the line's
// centre, so the weight is even.
//
// Six vertices a segment, pulled from the edge line list by index.

/// Every edge point, three floats apiece: scalars, since `vec3<f32>` in
/// storage pads to sixteen bytes.
@group(1) @binding(0) var<storage, read> edge_pts: array<f32>;

fn edge_point(i: u32) -> vec3<f32> {
    return vec3<f32>(edge_pts[3u * i], edge_pts[3u * i + 1u], edge_pts[3u * i + 2u]);
}

/// Half an edge line's width, in pixels. Its profile across is that width
/// seen through a one-pixel box filter: solid within `LINE_HALF - 0.5` of the
/// centre and falling to nothing at `LINE_HALF + 0.5`, which is how far the
/// quad reaches.
const LINE_HALF: f32 = 0.6;

/// Slides a point towards the eye by a small fraction of its distance, so an
/// edge wins the depth test against the face it borders.
///
/// Moving along the eye ray leaves the screen position alone, and a step
/// proportional to distance holds at every zoom (a constant clip-space offset
/// would let buried edges punch through far from the camera). It is affine in
/// the point, so near-plane clipping before or after it agrees.
fn pulled(world: vec3<f32>) -> vec3<f32> {
    let to_eye = u.eye.xyz - world;
    if (length(to_eye) > 1e-20) {
        return world + to_eye * u.params.y;
    }
    return world;
}

fn edge_clip(world: vec3<f32>, nudge: bool) -> vec4<f32> {
    return u.view_proj * vec4<f32>(select(world, pulled(world), nudge), 1.0);
}

/// Where a quad's corner goes at a joint with the next or previous segment:
/// along the bisector of the two, as far as keeps it the line's half-width
/// from this segment. Both quads then share the corner, with no wedge open
/// outside the bend and no doubled rim inside it.
///
/// `forward` says the neighbour comes after `at`; `other` is its far end.
fn miter(nrm: vec2<f32>, at: vec2<f32>, other: vec4<f32>, half: vec2<f32>, forward: bool) -> vec2<f32> {
    if (other.z > other.w) {
        return nrm;
    }
    let so = other.xy / other.w * half;
    let d = select(at - so, so - at, forward);
    let len = length(d);
    if (len < 1e-6) {
        return nrm;
    }
    let m = nrm + vec2<f32>(-d.y, d.x) / len;
    let ml = length(m);
    if (ml < 1e-6) {
        return nrm;
    }
    let k = dot(m / ml, nrm);
    // Past a turn of about 120 degrees a mitre reaches further than twice the
    // width; a corner that sharp keeps its square end instead.
    if (k < 0.5) {
        return nrm;
    }
    return m / ml / k;
}

/// One corner of the quad a line segment is drawn as.
fn edge_quad(vi: u32, inst: Inst, nudge: bool) -> EOut {
    let seg = vi / 6u;
    // Two triangles over four corners: 0 and 1 at the segment's start, 2 and
    // 3 at its end, the odd ones on the left.
    let corners = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
    let corner = corners[vi % 6u];
    let at_end = corner >= 2u;
    let side = select(-1.0, 1.0, (corner & 1u) == 1u);

    let ia = 2u * seg;
    var wa = inst_point(inst, edge_point(ia));
    var wb = inst_point(inst, edge_point(ia + 1u));
    var ca = edge_clip(wa, nudge);
    var cb = edge_clip(wb, nudge);

    var o: EOut;
    o.paint = inst.paint;
    o.across = 0.0;
    // A segment reaching behind the eye would divide by a negative w, so it
    // is cut at the near plane first; entirely behind, it collapses to a
    // point off screen. Depth is reversed: the near plane is where z = w.
    let ea = ca.z - ca.w;
    let eb = cb.z - cb.w;
    if (ea > 0.0 && eb > 0.0) {
        o.clip = vec4<f32>(2.0, 2.0, 0.5, 1.0);
        o.world = wa;
        return o;
    }
    var cut_a = false;
    var cut_b = false;
    if (ea > 0.0) {
        let t = ea / (ea - eb);
        ca = mix(ca, cb, t);
        wa = mix(wa, wb, t);
        cut_a = true;
    } else if (eb > 0.0) {
        let t = eb / (eb - ea);
        cb = mix(cb, ca, t);
        wb = mix(wb, wa, t);
        cut_b = true;
    }

    // Pixels per unit of normalised device coordinates. Screen y is flipped,
    // a reflection, which preserves distances and right angles.
    let half = u.ssao.zw * 0.5;
    let sa = ca.xy / ca.w * half;
    let sb = cb.xy / cb.w * half;
    let d = sb - sa;
    let len = length(d);
    // Seen end-on a segment has no screen direction and any will do: its quad
    // is a square the width of the line.
    let dir = select(vec2<f32>(1.0, 0.0), d / max(len, 1e-6), len > 1e-6);
    let nrm = vec2<f32>(-dir.y, dir.x);

    // Consecutive segments of one polyline share their joint point bit for bit
    // (the tessellator writes one value twice), so an exact compare says the
    // line goes on.
    var offset = nrm;
    let count = arrayLength(&edge_pts) / 3u;
    if (at_end && !cut_b && ia + 3u < count) {
        if (all(edge_point(ia + 2u) == edge_point(ia + 1u))) {
            let next = edge_clip(inst_point(inst, edge_point(ia + 3u)), nudge);
            offset = miter(nrm, sb, next, half, true);
        }
    } else if (!at_end && !cut_a && ia >= 2u) {
        if (all(edge_point(ia - 1u) == edge_point(ia))) {
            let prev = edge_clip(inst_point(inst, edge_point(ia - 2u)), nudge);
            offset = miter(nrm, sa, prev, half, false);
        }
    }

    let here = select(ca, cb, at_end);
    let reach = LINE_HALF + 0.5;
    // Undo the supersampling shift: the quad's coverage already antialiases
    // the line, and jittering it too would blur it.
    let at = here.xy + offset * (side * reach) / half * here.w - u.jitter.xy * here.w;
    o.clip = vec4<f32>(at, here.z, here.w);
    // Section against where the edge really is, not where it was nudged to,
    // or an edge in the plane would depend on which side the camera is.
    o.world = select(wa, wb, at_end);
    o.across = side * reach;
    return o;
}

/// An edge of a face, pulled towards the eye so it wins against the face.
@vertex
fn vs_edge(@builtin(vertex_index) vi: u32, inst: Inst) -> EOut {
    return edge_quad(vi, inst, true);
}

/// A loose curve, drawn where it actually is.
///
/// A `GEOMETRIC_SET` curve borders no face, so it is not pulled towards the
/// eye; nudged, a curve lying along a sheet would shimmer through it.
@vertex
fn vs_wire(@builtin(vertex_index) vi: u32, inst: Inst) -> EOut {
    return edge_quad(vi, inst, false);
}

/// The rasteriser's own one-pixel lines, for an adapter that cannot read a
/// storage buffer from a vertex shader.
@vertex
fn vs_edge_line(@location(0) pos: vec3<f32>, inst: Inst) -> EOut {
    let world = inst_point(inst, pos);
    var o: EOut;
    o.clip = edge_clip(world, true);
    o.world = world;
    o.paint = inst.paint;
    o.across = 0.0;
    return o;
}

@vertex
fn vs_wire_line(@location(0) pos: vec3<f32>, inst: Inst) -> EOut {
    let world = inst_point(inst, pos);
    var o: EOut;
    o.clip = edge_clip(world, false);
    o.world = world;
    o.paint = inst.paint;
    o.across = 0.0;
    return o;
}

@fragment
fn fs_edge(i: EOut) -> @location(0) vec4<f32> {
    if (clipped(i.world)) {
        discard;
    }
    let cover = clamp(LINE_HALF + 0.5 - abs(i.across), 0.0, 1.0);
    if (cover <= 0.0) {
        discard;
    }
    // Nearly black reads best over lit surfaces, but a standalone wireframe
    // needs to be lighter than the background instead.
    let over_surfaces = vec3<f32>(0.06, 0.07, 0.09);
    let standalone = vec3<f32>(0.64, 0.70, 0.80);
    let wire = u.params.z > 0.5;
    let c = select(over_surfaces, standalone, wire);
    // A painted part's edges take a dark shade of its paint, or the line would
    // vanish into the face it bounds; a standalone wireframe takes it whole.
    let shade = select(0.35, 1.0, wire);
    let painted = paint_over(c, vec4<f32>(i.paint.rgb * shade, i.paint.a));
    return vec4<f32>(encode(painted), cover);
}

// What is under one pixel, for picking and measuring.
//
// Drawn into a small square target with `view_proj` already shifted to put the
// pixel under the cursor at its centre, so the buffer is small whatever the
// window size. It asks `clipped` like the shaded pass, so a face the section
// has cut away cannot be picked.
struct PickOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) @interpolate(flat) vid: u32,
};

fn pick_vertex(pos: vec3<f32>, vi: u32, inst: Inst) -> PickOut {
    let world = inst_point(inst, pos);
    var o: PickOut;
    o.clip = u.view_proj * vec4<f32>(world, 1.0);
    o.world = world;
    // Any vertex of the triangle will do: it identifies the part.
    o.vid = vi;
    return o;
}

@vertex
fn vs_pick(@location(0) pos: vec3<f32>, @builtin(vertex_index) vi: u32, inst: Inst) -> PickOut {
    return pick_vertex(pos, vi, inst);
}

/// The same, for a part drawn from its refined tessellation. Those vertices are
/// numbered in their own buffers, so the index carries the top bit and
/// `Gpu::pick` maps it back to a vertex of the part.
@vertex
fn vs_pick_refined(@location(0) pos: vec3<f32>, @builtin(vertex_index) vi: u32, inst: Inst) -> PickOut {
    var o = pick_vertex(pos, vi, inst);
    o.vid = vi | 0x80000000u;
    return o;
}

@fragment
fn fs_pick(i: PickOut) -> @location(0) vec4<u32> {
    if (clipped(i.world)) {
        discard;
    }
    // Plus one: the target clears to zero, which means a miss.
    return vec4<u32>(
        i.vid + 1u,
        bitcast<u32>(i.world.x),
        bitcast<u32>(i.world.y),
        bitcast<u32>(i.world.z),
    );
}
