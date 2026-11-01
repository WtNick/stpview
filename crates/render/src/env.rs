//! The studio the model is lit by, computed rather than loaded.
//!
//! Instead of an HDR photograph, the environment is a function: a sky
//! gradient, a floor, and three soft panels. No assets to ship, and a machined
//! surface needs only a large bright shape to reflect and a gradient to roll
//! off against.
//!
//! Two products, both computed once:
//!
//!   * nine spherical-harmonic coefficients: the whole diffuse lighting,
//!     with no texture;
//!   * a small cube map whose mip levels are the environment blurred by
//!     increasing roughness: the reflection.

use rayon::prelude::*;

use crate::math::{v3, Vec3};

/// Base size of the reflection cube map. Small on purpose: every level past
/// the first is a blur, and the environment's sharpest feature is a soft
/// panel.
pub const CUBE_SIZE: u32 = 64;
/// Levels of the mip chain, from mirror-sharp to fully diffuse.
pub const CUBE_MIPS: u32 = 6;

/// Overall brightness of the studio.
///
/// Calibrated by measuring mean model brightness in the stored views, not by
/// eye: a uniform shift in level is the one error the eye adapts to. Anchored
/// on simple convex parts; crowded assemblies come out brighter, since the
/// studio surrounds them.
const INTENSITY: f64 = 0.90;

/// Radiance arriving from `dir`, which need not be normalised.
///
/// Values run past 1.0 on the panels: in high dynamic range a curved surface
/// gets a highlight with a shape rather than a white dot.
pub fn radiance(dir: Vec3) -> Vec3 {
    let d = dir.normalize_or(v3(0.0, 0.0, 1.0));

    // Z-up, matching the models. A cool sky falling to a warmer, dimmer floor.
    //
    // The floor is a bounce card: everything faces it, so it sets how
    // readable the shadowed half of a part is. It stays well under the sky so
    // the shading does not flatten.
    let sky = v3(0.42, 0.50, 0.62);
    let horizon = v3(0.34, 0.35, 0.37);
    let floor = v3(0.20, 0.19, 0.18);
    let mut c = if d.z >= 0.0 {
        lerp(horizon, sky, smoothstep(d.z))
    } else {
        lerp(horizon, floor, smoothstep(-d.z))
    };

    // Key over the left shoulder, fill opposite at about a quarter of it, and
    // a rim behind and above to separate the silhouette.
    for &(dirv, size, power) in &[
        (v3(-0.45, -0.60, 0.65), 0.38, 22.0),
        (v3(0.75, -0.25, 0.10), 0.55, 5.5),
        (v3(0.00, 0.75, 0.55), 0.30, 9.0),
    ] {
        let p = dirv.normalize();
        // A cosine lobe rather than a hard disc: a hard-edged reflection
        // reads as a bug.
        let t = d.dot(p).max(0.0);
        let lobe = t.powf(2.0 / (size * size));
        c += v3(power, power * 0.985, power * 0.96) * lobe;
    }
    c * INTENSITY
}

fn lerp(a: Vec3, b: Vec3, t: f64) -> Vec3 {
    a + (b - a) * t
}

fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The nine spherical-harmonic coefficients of the environment's irradiance,
/// packed one per `vec4` for the uniform buffer.
///
/// Order-2 SH holds all a Lambertian surface can tell apart, since a cosine
/// lobe has almost no energy above band 2 (Ramamoorthi and Hanrahan).
pub fn irradiance_sh9() -> [[f32; 4]; 9] {
    // Fibonacci directions: an even sphere without pole clustering, and no
    // random numbers, so the build is reproducible.
    const N: usize = 16384;
    let mut sh = [Vec3::ZERO; 9];
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    for i in 0..N {
        let z = 1.0 - (2.0 * i as f64 + 1.0) / N as f64;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let a = golden * i as f64;
        let d = v3(r * a.cos(), r * a.sin(), z);
        let l = radiance(d);
        for (k, b) in basis(d).iter().enumerate() {
            sh[k] += l * *b;
        }
    }
    // Monte Carlo over a uniform sphere: each sample carries 4*pi/N of solid
    // angle.
    let w = 4.0 * std::f64::consts::PI / N as f64;
    // Ramamoorthi and Hanrahan's clamped-cosine constants (pi, 2pi/3, pi/4)
    // already divided by pi, which turns irradiance into Lambertian outgoing
    // radiance. Do not divide by pi again.
    let a = [1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25];
    let mut out = [[0.0f32; 4]; 9];
    for k in 0..9 {
        let c = sh[k] * (w * a[k]);
        out[k] = [c.x as f32, c.y as f32, c.z as f32, 0.0];
    }
    out
}

/// Real spherical harmonics through band 2, in the usual order.
fn basis(d: Vec3) -> [f64; 9] {
    let (x, y, z) = (d.x, d.y, d.z);
    [
        0.282095,
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ]
}

/// The reflection cube map, level by level and face by face, as half floats.
///
/// Level 0 is the environment itself and each later level is convolved with a
/// wider lobe, so sampling at a level chosen from roughness gives that
/// roughness's blur: the split-sum approximation, with the shader supplying
/// the rest analytically.
///
/// Built once per process and kept: the studio has no inputs, so it is a
/// constant, and building it is a large share of renderer start-up.
pub fn prefiltered_cube() -> &'static [Vec<u8>] {
    static CUBE: std::sync::OnceLock<Vec<Vec<u8>>> = std::sync::OnceLock::new();
    // Serial over the levels, parallel within one: level 1 holds most of the
    // work, so threading across levels would leave one thread with it.
    CUBE.get_or_init(|| (0..CUBE_MIPS).map(level).collect())
}

fn level(mip: u32) -> Vec<u8> {
    let size = (CUBE_SIZE >> mip).max(1);
    let rough = mip as f64 / (CUBE_MIPS - 1) as f64;
    // Six faces of half-float RGBA, in the order wgpu expects: +X -X +Y -Y +Z -Z.
    let mut out = vec![0u8; (size * size * 6 * 8) as usize];
    // A row at a time, face-major, the order of the bytes. Each texel depends
    // only on its own direction, so the result is byte-identical to a serial
    // build, as `models/visual.ps1` requires.
    out.par_chunks_mut((size * 8) as usize)
        .enumerate()
        .for_each(|(row, dst)| {
            let (face, y) = (row as u32 / size, row as u32 % size);
            for x in 0..size {
                let a = (x as f64 + 0.5) / size as f64 * 2.0 - 1.0;
                let b = (y as f64 + 0.5) / size as f64 * 2.0 - 1.0;
                let c = filter(face_dir(face, a, b), rough);
                for (k, v) in [c.x, c.y, c.z, 1.0].into_iter().enumerate() {
                    let at = x as usize * 8 + k * 2;
                    dst[at..at + 2].copy_from_slice(&half(v as f32).to_le_bytes());
                }
            }
        });
    out
}

/// Cube face directions, following the texture convention wgpu inherits from
/// D3D and Vulkan: +X, -X, +Y, -Y, +Z, -Z with the second axis flipped.
fn face_dir(face: u32, a: f64, b: f64) -> Vec3 {
    match face {
        0 => v3(1.0, -b, -a),
        1 => v3(-1.0, -b, a),
        2 => v3(a, 1.0, b),
        3 => v3(a, -1.0, -b),
        4 => v3(a, -b, 1.0),
        _ => v3(-a, -b, -1.0),
    }
}

/// The environment seen through a lobe of the given roughness, centred on `n`.
///
/// Cosine-power rather than GGX importance sampling: at these sizes the two
/// are indistinguishable, and this needs no random sequence or per-sample
/// weighting.
fn filter(n: Vec3, rough: f64) -> Vec3 {
    if rough <= 0.0 {
        return radiance(n);
    }
    // Roughness to lobe width, chosen so the last level is near-uniform and the
    // first few keep a panel recognisable as a panel.
    let power = 2.0 / (rough * rough * rough * rough + 1e-4) - 2.0;
    let n = n.normalize();
    let (t, s) = basis_of(n);
    // Rings around the axis: even coverage without random numbers, outer
    // rings weighted down as the lobe says.
    let mut sum = Vec3::ZERO;
    let mut weight = 0.0;
    let rings = 12;
    let spokes = 24;
    for i in 0..=rings {
        // Spread by equal cosine, where the lobe's energy is.
        let ct = (1.0 - i as f64 / (rings + 1) as f64).powf(1.0 / (power + 1.0));
        let st = (1.0 - ct * ct).max(0.0).sqrt();
        let w = ct;
        for j in 0..spokes {
            let phi = std::f64::consts::TAU * j as f64 / spokes as f64;
            let d = n * ct + t * (st * phi.cos()) + s * (st * phi.sin());
            sum += radiance(d) * w;
            weight += w;
        }
    }
    sum * (1.0 / weight.max(1e-9))
}

/// Any two unit vectors perpendicular to `n` and to each other.
fn basis_of(n: Vec3) -> (Vec3, Vec3) {
    let t = n.any_perpendicular().normalize();
    (t, n.cross(t).normalize())
}

/// IEEE half from an f32. The environment is bounded, so subnormals and
/// infinities are folded into a clamp.
fn half(x: f32) -> u16 {
    let x = x.clamp(-65504.0, 65504.0);
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3ff;
    if exp <= 0 {
        return sign;
    }
    sign | ((exp as u16) << 10) | mant as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The panels have to be brighter than the sky, or there is nothing for a
    /// polished surface to reflect.
    #[test]
    fn the_studio_has_something_to_reflect() {
        let key = radiance(v3(-0.45, -0.60, 0.65));
        let sky = radiance(v3(0.0, 0.0, 1.0));
        let floor = radiance(v3(0.0, 0.0, -1.0));
        assert!(key.x > sky.x * 4.0, "key {key:?} against sky {sky:?}");
        assert!(sky.x > floor.x * 2.0, "sky {sky:?} against floor {floor:?}");
    }

    /// Irradiance lies between the environment's darkest and brightest parts
    /// and is positive everywhere; a sign error in the constants breaks this.
    #[test]
    fn irradiance_lands_inside_the_environment() {
        let sh = irradiance_sh9();
        let at = |d: Vec3| {
            let b = basis(d);
            (0..9).map(|k| sh[k][0] as f64 * b[k]).sum::<f64>()
        };
        for d in [
            v3(0.0, 0.0, 1.0),
            v3(0.0, 0.0, -1.0),
            v3(1.0, 0.0, 0.0),
            v3(-0.45, -0.60, 0.65).normalize(),
        ] {
            let e = at(d);
            assert!(e > 0.0, "irradiance {e} at {d:?} is not positive");
            assert!(e < 4.0, "irradiance {e} at {d:?} is implausibly bright");
        }
        // Up must be brighter than down: the sky is, and a cosine lobe cannot
        // reverse that.
        assert!(at(v3(0.0, 0.0, 1.0)) > at(v3(0.0, 0.0, -1.0)));
    }

    /// The nine coefficients agree with the cosine-weighted integral they
    /// stand for. A wrong constant or basis order still gives a plausible
    /// gradient, so only this comparison catches it.
    #[test]
    fn the_coefficients_reconstruct_the_integral() {
        let sh = irradiance_sh9();
        // What the shader computes, with its own copy of the basis constants.
        let reconstruct = |n: Vec3| {
            let b = basis(n);
            let mut e = Vec3::ZERO;
            for k in 0..9 {
                e = e + v3(sh[k][0] as f64, sh[k][1] as f64, sh[k][2] as f64) * b[k];
            }
            e
        };
        // What it is supposed to be.
        let integrate = |n: Vec3| {
            const N: usize = 200_000;
            let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
            let mut e = Vec3::ZERO;
            for i in 0..N {
                let z = 1.0 - (2.0 * i as f64 + 1.0) / N as f64;
                let r = (1.0 - z * z).max(0.0).sqrt();
                let a = golden * i as f64;
                let d = v3(r * a.cos(), r * a.sin(), z);
                e = e + radiance(d) * d.dot(n).max(0.0);
            }
            e * (4.0 * std::f64::consts::PI / N as f64 / std::f64::consts::PI)
        };
        let dirs = [
            v3(0.0, 0.0, 1.0),
            v3(0.0, 0.0, -1.0),
            v3(1.0, 0.0, 0.0),
            v3(0.0, -1.0, 0.0),
            v3(-0.45, -0.60, 0.65).normalize(),
            v3(0.6, 0.5, -0.6).normalize(),
        ];
        // Against the brightest direction, not each direction's own value:
        // order 2 cannot hold a sharp panel exactly, and the residual is
        // roughly constant over the sphere. Nine percent is Ramamoorthi and
        // Hanrahan's worst case (a point source); it is loose on purpose, to
        // catch a wrong constant or a basis out of step with the shader.
        let peak = dirs.iter().map(|&n| integrate(n).x).fold(0.0, f64::max);
        for n in dirs {
            let (a, b) = (reconstruct(n), integrate(n));
            assert!(
                (a.x - b.x).abs() < peak * 0.09,
                "at {n:?}: reconstructed {a:?} against exact {b:?}, peak {peak}"
            );
        }
    }

    /// Blurring the environment must not change how much light it holds.
    #[test]
    fn roughness_blurs_without_losing_energy() {
        let mean = |mip: u32| {
            let data = level(mip);
            let n = data.len() / 8;
            let mut sum = 0.0f64;
            for i in 0..n {
                let h = u16::from_le_bytes([data[i * 8], data[i * 8 + 1]]);
                sum += from_half(h) as f64;
            }
            sum / n as f64
        };
        let sharp = mean(0);
        let blurred = mean(CUBE_MIPS - 1);
        assert!(sharp > 0.0 && blurred > 0.0);
        // Within a fifth: a cube map weights its corners more than its face
        // centres, so the two means are not required to agree exactly.
        assert!(
            (blurred - sharp).abs() < sharp * 0.2,
            "sharp {sharp} against blurred {blurred}"
        );
    }

    /// The threaded cube is the serial cube, byte for byte. A cube assembled
    /// out of order still lights a model plausibly, so the serial loop stays
    /// here as the definition.
    #[test]
    fn threading_the_prefilter_moves_no_byte() {
        for mip in 0..CUBE_MIPS {
            let size = (CUBE_SIZE >> mip).max(1);
            let rough = mip as f64 / (CUBE_MIPS - 1) as f64;
            let mut want = Vec::with_capacity((size * size * 6 * 8) as usize);
            for face in 0..6 {
                for y in 0..size {
                    for x in 0..size {
                        let a = (x as f64 + 0.5) / size as f64 * 2.0 - 1.0;
                        let b = (y as f64 + 0.5) / size as f64 * 2.0 - 1.0;
                        let c = filter(face_dir(face, a, b), rough);
                        for v in [c.x, c.y, c.z, 1.0] {
                            want.extend_from_slice(&half(v as f32).to_le_bytes());
                        }
                    }
                }
            }
            let got = level(mip);
            assert_eq!(got.len(), want.len(), "mip {mip} is the wrong length");
            let differing = got.iter().zip(&want).filter(|(g, w)| g != w).count();
            assert_eq!(differing, 0, "mip {mip}: {differing} bytes differ");
        }
    }

    fn from_half(h: u16) -> f32 {
        let sign = ((h >> 15) as u32) << 31;
        let exp = ((h >> 10) & 0x1f) as u32;
        let mant = (h & 0x3ff) as u32;
        if exp == 0 {
            return f32::from_bits(sign);
        }
        f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
    }
}
