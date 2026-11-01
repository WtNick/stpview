//! The render test suite, compiled against the parent module.

    use super::*;

    /// Where a model point lands, in normalised device coordinates.
    fn ndc_of(vp: &Mat4, p: Vec3) -> [f32; 2] {
        let m = &vp.0;
        let c: Vec<f32> = (0..4)
            .map(|r| {
                m[0][r] * p.x as f32 + m[1][r] * p.y as f32 + m[2][r] * p.z as f32 + m[3][r]
            })
            .collect();
        assert!(c[3] > 0.0, "point is behind the camera");
        [c[0] / c[3], c[1] / c[3]]
    }

    fn pixel_of(vp: &Mat4, p: Vec3, w: f32, h: f32) -> [f32; 2] {
        let n = ndc_of(vp, p);
        [(n[0] * 0.5 + 0.5) * w, (0.5 - n[1] * 0.5) * h]
    }

    /// The pick pass draws the model with the projection shifted so that one
    /// window pixel fills the middle of the pick square, at one pick pixel per
    /// window pixel. A wrong shift picks another pixel; a wrong scale searches
    /// the wrong neighbourhood.
    #[test]
    fn pick_transform_frames_the_pixel_it_was_given() {
        let mut b = Aabb::EMPTY;
        b.add_point(v3(-30.0, -20.0, -10.0));
        b.add_point(v3(30.0, 20.0, 10.0));
        let mut cam = Camera::new(&b);
        cam.orbit(0.7, -0.3);
        let (w, h) = (1200.0f32, 900.0f32);
        let vp = cam.view_proj(w / h);

        for p in [v3(7.0, -3.0, 4.0), v3(-25.0, 18.0, -8.0), v3(0.0, 0.0, 0.0)] {
            let pixel = pixel_of(&vp, p, w, h);
            let s = pick_transform(w, h, pixel[0], pixel[1], PICK_SIZE as f32).mul(&vp);
            let at = ndc_of(&s, p);
            // Half a window pixel is half a pick pixel, 1/PICK_SIZE in NDC,
            // plus float slack for a point on the edge of its pixel.
            let slack = 1.0 / PICK_SIZE as f32 + 1e-5;
            assert!(
                at[0].abs() <= slack && at[1].abs() <= slack,
                "{p:?} lands at {at:?}, not within {slack} of the centre"
            );

            // Its neighbour one window pixel to the right lands one pick pixel
            // to the right; at constant depth this is exact.
            let (right, _, fwd) = cam.basis();
            let depth = (p - cam.eye()).dot(fwd);
            let per_pixel = 2.0 * (cam.fov_y as f64 * 0.5).tan() * depth / h as f64;
            let q = p + right * per_pixel;
            assert!((pixel_of(&vp, q, w, h)[0] - pixel[0] - 1.0).abs() < 1e-2);
            let next = ndc_of(&s, q);
            assert!(
                (next[0] - at[0] - 2.0 / PICK_SIZE as f32).abs() < 1e-3
                    && (next[1] - at[1]).abs() < 1e-3,
                "one window pixel moved to {next:?} from {at:?}"
            );
        }
    }

    #[test]
    fn ssao_kernel_is_a_packed_hemisphere() {
        let k = ssao_kernel();
        for (i, s) in k.iter().enumerate() {
            let v = v3(s[0] as f64, s[1] as f64, s[2] as f64);
            assert!(v.z >= 0.0, "sample {i} points into the surface: {v:?}");
            let len = v.length();
            assert!(len > 0.0 && len <= 1.0 + 1e-6, "sample {i} length {len}");
        }
        // Later samples must reach further out than the first few, so the
        // kernel covers both fine detail and the wider cavity.
        let near = k[..4].iter().map(|s| s[2].abs()).fold(0.0f32, f32::max);
        let far = k[12..].iter().map(|s| s[2].abs()).fold(0.0f32, f32::max);
        assert!(far > near, "kernel is not graded outwards ({near} vs {far})");
    }

    #[test]
    fn kernel_is_deterministic() {
        assert_eq!(ssao_kernel(), ssao_kernel());
    }

    /// Both shaders declare the same uniform block, the size of the one Rust
    /// fills. A field one file leaves out is not an error: every field after it
    /// silently shifts, and the result can still look plausible.
    #[test]
    fn the_shaders_agree_about_the_uniform_block() {
        /// The `struct Uniforms { ... }` body, with comments and whitespace
        /// taken out.
        fn block(src: &str) -> String {
            let from = src.find("struct Uniforms {").expect("no Uniforms struct");
            let rest = &src[from..];
            let to = rest.find("};").expect("unterminated Uniforms struct");
            rest[..to]
                .lines()
                .map(|l| l.split("//").next().unwrap_or("").trim())
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
                .replace(' ', "")
        }
        let a = block(include_str!("../shader.wgsl"));
        let b = block(include_str!("../ssao.wgsl"));
        assert_eq!(a, b, "shader.wgsl and ssao.wgsl declare different uniforms");

        // The size check catches a field added on the Rust side alone.
        // Split on the commas between fields, not the ones inside `array<..,N>`.
        let mut fields: Vec<String> = Vec::new();
        let (mut depth, mut at) = (0i32, String::new());
        for c in a.trim_start_matches("structUniforms{").chars() {
            match c {
                '<' => depth += 1,
                '>' => depth -= 1,
                ',' if depth == 0 => {
                    fields.push(std::mem::take(&mut at));
                    continue;
                }
                _ => {}
            }
            at.push(c);
        }
        fields.push(at);

        let mut slots = 0usize;
        for field in &fields {
            let Some((_, ty)) = field.split_once(':') else { continue };
            slots += if ty.starts_with("mat4x4") {
                4
            } else if ty.starts_with("vec4") {
                1
            } else if let Some(n) = ty.strip_prefix("array<vec4<f32>,") {
                n.trim_end_matches('>').parse::<usize>().expect("array length")
            } else {
                panic!("unhandled uniform field type {ty:?}");
            };
        }
        assert_eq!(
            slots * 16,
            std::mem::size_of::<Uniforms>(),
            "the shaders declare {slots} vec4s, which is not what Rust writes"
        );
    }

    /// Every vertex stage that consumes mesh geometry takes the instance
    /// block, and the block is the size Rust writes. A stage without `inst`
    /// would still validate and draw its pass unplaced, at the origin; wgpu
    /// checks the Rust side, nothing else checks the shader side.
    #[test]
    fn every_mesh_stage_takes_the_instance_block() {
        let src = include_str!("../shader.wgsl");
        for stage in ["vs_surface", "vs_gbuffer", "vs_shadow", "vs_pick", "vs_wire", "vs_edge"] {
            let at = src.find(&format!("fn {stage}(")).unwrap_or_else(|| panic!("{stage} missing"));
            // Up to the return arrow: the parameter list holds `@location(N)`
            // parens of its own, so the first `)` is not the signature's end.
            let sig = &src[at..at + src[at..].find("->").expect("unterminated signature")];
            assert!(
                sig.contains("inst: Inst"),
                "{stage} does not take the instance block: {sig}"
            );
        }
        // Three rows, a tint and a paint, as Rust lays them out.
        assert_eq!(std::mem::size_of::<InstanceRaw>(), 56);
        let from = src.find("struct Inst {").expect("no Inst struct");
        let body = &src[from..from + src[from..].find('}').expect("unterminated Inst")];
        for loc in [
            "@location(3) r0: vec4<f32>",
            "@location(4) r1: vec4<f32>",
            "@location(5) r2: vec4<f32>",
            "@location(6) tint: vec4<f32>",
            "@location(7) paint: vec4<f32>",
        ] {
            assert!(body.contains(loc), "Inst is missing {loc:?}");
        }
    }

    /// Runs coalesce only across parts that share a paint, and with no paint
    /// they are the plain unpainted runs, so an unpainted frame issues the same
    /// draw calls.
    #[test]
    fn runs_coalesce_within_a_paint_and_break_between_them() {
        let part = |tris: Range<u32>| Part {
            name: String::new(),
            node: None,
            alpha: 1.0,
            cappable: true,
            color: [0.5; 3],
            tris,
            verts: 0..0,
            wire: 0..0,
            sharp: 0..0,
            tangent: 0..0,
            bbox: Aabb::EMPTY,
            faces: Vec::new(),
        };
        let parts = vec![part(0..3), part(3..9), part(9..12), part(12..15)];
        let pick = |p: &Part| p.tris.clone();
        assert_eq!(runs(&parts, &[], pick), vec![0..15]);
        let red = [255, 0, 0, 160];
        let keys = [NO_PAINT, red, red, NO_PAINT];
        assert_eq!(
            painted_runs(&parts, &[], &keys, pick),
            vec![(NO_PAINT, 0..3), (red, 3..12), (NO_PAINT, 12..15)]
        );
        // A hidden part in the middle of a painted run splits it.
        assert_eq!(
            painted_runs(&parts, &[true, true, false, true], &keys, pick),
            vec![(NO_PAINT, 0..3), (red, 3..9), (NO_PAINT, 12..15)]
        );
        // And a list shorter than the parts leaves the tail unpainted.
        assert_eq!(
            painted_runs(&parts, &[], &[red], pick),
            vec![(red, 0..3), (NO_PAINT, 3..15)]
        );
    }

    /// Every corner of the model lands inside the light's box; anything outside
    /// is clipped from the shadow map and casts nothing. Covers boxes far from
    /// the origin and of extreme aspect.
    #[test]
    fn the_light_box_holds_the_whole_model() {
        let boxes = [
            Aabb { min: v3(-1.0, -1.0, -1.0), max: v3(1.0, 1.0, 1.0) },
            // Far from the origin, which catches a projection built about the
            // world rather than about the model.
            Aabb { min: v3(4000.0, -20.0, 900.0), max: v3(4100.0, 60.0, 1000.0) },
            // A long thin part, and a flat one.
            Aabb { min: v3(0.0, 0.0, 0.0), max: v3(500.0, 3.0, 3.0) },
            Aabb { min: v3(-80.0, -80.0, 0.0), max: v3(80.0, 80.0, 0.5) },
        ];
        for b in boxes {
            let (vp, ground_z, ground_radius) = shadow_frame(&b);
            for i in 0..8 {
                let p = v3(
                    if i & 1 == 0 { b.min.x } else { b.max.x },
                    if i & 2 == 0 { b.min.y } else { b.max.y },
                    if i & 4 == 0 { b.min.z } else { b.max.z },
                );
                let m = vp.0;
                let mut c = [0.0f32; 4];
                for r in 0..4 {
                    c[r] = m[0][r] * p.x as f32
                        + m[1][r] * p.y as f32
                        + m[2][r] * p.z as f32
                        + m[3][r];
                }
                assert!(c[3] > 0.0, "corner {p:?} is behind the light");
                let (x, y, z) = (c[0] / c[3], c[1] / c[3], c[2] / c[3]);
                assert!(
                    x.abs() <= 1.0 && y.abs() <= 1.0 && (0.0..=1.0).contains(&z),
                    "corner {p:?} of {b:?} lands at {x}, {y}, {z}, outside the light's box"
                );
            }
            // And the floor is under the model, not through it.
            assert!(ground_z <= b.min.z as f32, "floor at {ground_z} is not below {b:?}");
            assert!(ground_radius > b.size().length() as f32 * 0.5);
        }
    }

    /// A still view's frames start at the pixel centres (frame 0 is the one
    /// drawn while moving) and then cover the pixel evenly, distinct samples.
    #[test]
    fn a_supersampled_view_starts_at_the_pixel_centres_and_covers_the_pixel() {
        assert_eq!(jitter(0), [0.0, 0.0]);
        let all: Vec<[f32; 2]> = (0..SUPERSAMPLE).map(jitter).collect();
        for (i, j) in all.iter().enumerate() {
            assert!(
                j.iter().all(|c| (-0.5..0.5).contains(c)),
                "sample {i} at {j:?} is outside its pixel"
            );
        }
        // Even: each half of the pixel, either way, gets its share of any
        // leading run of them, give or take one.
        for n in [4usize, 8, SUPERSAMPLE as usize] {
            for axis in 0..2 {
                let low = all[..n].iter().filter(|j| j[axis] < 0.0).count();
                assert!(
                    low.abs_diff(n / 2) <= 1,
                    "{low} of the first {n} samples fall in one half of the pixel"
                );
            }
        }
        // And no two alike, or a sample would be spent twice.
        for a in 0..all.len() {
            for b in a + 1..all.len() {
                assert_ne!(all[a], all[b], "samples {a} and {b} are the same point");
            }
        }
    }

    /// A run drawn around holes yields every index outside them once, in
    /// order, wherever the holes sit relative to the run.
    #[test]
    fn a_run_is_drawn_around_its_holes() {
        let holes = [4..6, 10..12, 12..15, 20..30];
        assert_eq!(around(&holes, 0..40), vec![0..4, 6..10, 15..20, 30..40]);
        assert_eq!(around(&holes, 5..11), vec![6..10]);
        assert_eq!(around(&holes, 21..29), Vec::<Range<u32>>::new());
        assert_eq!(around(&holes, 16..19), vec![16..19]);
        assert_eq!(around(&[], 3..9), vec![3..9]);
    }
