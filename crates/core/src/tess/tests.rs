//! Tests for the tessellator, written against the parent module's
//! re-exports.

    use super::*;
    use crate::geom::{Axis2, TWO_PI};
    use crate::math::v3;
    use crate::model::{Bound, EdgeUse};

    fn unit_axis() -> Axis2 {
        Axis2::new(Vec3::ZERO, None, None)
    }

    /// Total triangle area of a mesh.
    fn area(m: &FaceMesh) -> f64 {
        let mut a = 0.0;
        for t in m.indices.chunks_exact(3) {
            let p = |i: u32| {
                let v = m.vertices[i as usize].pos;
                v3(v[0] as f64, v[1] as f64, v[2] as f64)
            };
            a += (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])).length() * 0.5;
        }
        a
    }

    fn run(face: &Face, surfaces: &[Surface], curves: &[Curve], tol: f64) -> FaceMesh {
        let mut out = FaceMesh::default();
        tessellate_face(face, surfaces, curves, [1.0; 4], tol, f64::INFINITY, &Xform::IDENTITY, &mut out);
        out
    }

    #[test]
    fn samples_a_circle_within_tolerance() {
        let c = Curve::Circle {
            ax: unit_axis(),
            r: 10.0,
        };
        let mut pts = Vec::new();
        sample_edge(&c, 0.0, TWO_PI, 0.01, f64::INFINITY, &mut pts);
        assert!(pts.len() > 60, "got {} samples", pts.len());
        // Every chord midpoint must be within tolerance of the true arc.
        for w in pts.windows(2) {
            let mid = (w[0] + w[1]) * 0.5;
            assert!((10.0 - mid.length()).abs() <= 0.0101);
        }
    }

    /// A long straight edge carries no chord error at all, so only the segment
    /// cap can subdivide it. Without that, a cylinder's parameter-space outline
    /// is a rectangle with two points along its long sides, which ear clipping
    /// turns into triangles spanning the whole face.
    #[test]
    fn segment_cap_subdivides_a_straight_edge() {
        let line = Curve::Line {
            origin: v3(0.0, 0.0, 0.0),
            dir: v3(0.0, 0.0, 30.0),
        };
        let mut uncapped = Vec::new();
        sample_edge(&line, 0.0, 1.0, 0.01, f64::INFINITY, &mut uncapped);
        assert_eq!(uncapped.len(), 2, "a straight edge needs no chord splitting");

        let mut capped = Vec::new();
        sample_edge(&line, 0.0, 1.0, 0.01, 2.0, &mut capped);
        assert!(capped.len() >= 16, "expected ~15 segments, got {}", capped.len());
        for w in capped.windows(2) {
            assert!(w[0].distance(w[1]) <= 2.0 + 1e-9);
        }
        // Still exactly on the line, and still spanning it end to end.
        assert!(capped[0].distance(v3(0.0, 0.0, 0.0)) < 1e-12);
        assert!(capped.last().unwrap().distance(v3(0.0, 0.0, 30.0)) < 1e-12);
    }

    #[test]
    fn segment_cap_scales_with_tolerance_and_model_size() {
        // Chord of a circle of radius extent/2 whose sagitta is tol.
        let l = max_segment_length(0.015, 50.0);
        assert!((l - (8.0f64 * 0.015 * 25.0).sqrt()).abs() < 1e-12);
        // Bigger models and looser tolerances both allow longer segments.
        assert!(max_segment_length(0.015, 500.0) > l);
        assert!(max_segment_length(0.15, 50.0) > l);
        // Never degenerates to zero.
        assert!(max_segment_length(0.01, 0.0) > 0.0);
    }

    #[test]
    fn uv_step_matches_the_edge_sampler_on_a_cylinder() {
        // The refinement stopping rule and the arc sampler must agree, or a
        // boundary sampled to tolerance still looks too coarse to refinement,
        // which then splits forever against locked boundary edges.
        let tol = 0.01;
        let r = 20.0;
        let s = Surface::Cylinder {
            ax: unit_axis(),
            r,
        };
        let (du, dv) = s.uv_step(tol, v2(0.0, 0.0), v2(TWO_PI, 10.0));
        assert!(dv.is_infinite(), "a cylinder is straight along its axis");
        // The bounding circle of that cylinder, sampled the way an edge is.
        let rim = Curve::Circle {
            ax: unit_axis(),
            r,
        };
        let segments = rim.initial_segments(0.0, TWO_PI, tol);
        let sampled_step = TWO_PI / segments as f64;
        assert!(
            sampled_step <= du + 1e-12,
            "sampler step {sampled_step} exceeds refinement step {du}"
        );
    }

    #[test]
    fn wrap_delta_picks_the_short_way_round() {
        assert!((wrap_delta(0.1, TWO_PI) - 0.1).abs() < 1e-12);
        assert!((wrap_delta(TWO_PI - 0.1, TWO_PI) + 0.1).abs() < 1e-12);
        assert!((wrap_delta(-TWO_PI + 0.1, TWO_PI) - 0.1).abs() < 1e-12);
    }

    /// A square planar face triangulates to exactly its own area.
    #[test]
    fn planar_square_has_exact_area() {
        let surfaces = vec![Surface::Plane(unit_axis())];
        let corners = [
            v3(0.0, 0.0, 0.0),
            v3(4.0, 0.0, 0.0),
            v3(4.0, 4.0, 0.0),
            v3(0.0, 4.0, 0.0),
        ];
        let mut curves = Vec::new();
        let mut edges = Vec::new();
        for i in 0..4 {
            let a = corners[i];
            let b = corners[(i + 1) % 4];
            curves.push(Curve::Line {
                origin: a,
                dir: b - a,
            });
            edges.push(EdgeUse {
                curve: i as u32,
                edge: i as u32,
                t0: 0.0,
                t1: 1.0,
            });
        }
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![Bound { vertex: None, outer: true, edges }],
        };
        let m = run(&face, &surfaces, &curves, 0.01);
        assert_eq!(m.stats.faces_ok, 1);
        assert!((area(&m) - 16.0).abs() < 1e-9, "area {}", area(&m));
        assert_eq!(m.indices.len(), 6, "a square needs exactly two triangles");
        // Plane normal is +Z and same_sense is true.
        for v in &m.vertices {
            assert!((v.normal().z - 1.0).abs() < 1e-6);
        }
    }

    /// A square with a square hole keeps the hole open.
    #[test]
    fn planar_face_with_hole_subtracts_the_hole() {
        let surfaces = vec![Surface::Plane(unit_axis())];
        let mut curves = Vec::new();
        let mut mk_loop = |pts: [Vec3; 4]| {
            let mut edges = Vec::new();
            for i in 0..4 {
                let a = pts[i];
                let b = pts[(i + 1) % 4];
                let idx = curves.len() as u32;
                curves.push(Curve::Line {
                    origin: a,
                    dir: b - a,
                });
                edges.push(EdgeUse {
                    curve: idx,
                    edge: idx,
                    t0: 0.0,
                    t1: 1.0,
                });
            }
            edges
        };
        let outer = mk_loop([
            v3(0.0, 0.0, 0.0),
            v3(10.0, 0.0, 0.0),
            v3(10.0, 10.0, 0.0),
            v3(0.0, 10.0, 0.0),
        ]);
        // Hole wound the opposite way, as a real file would have it.
        let inner = mk_loop([
            v3(3.0, 3.0, 0.0),
            v3(3.0, 7.0, 0.0),
            v3(7.0, 7.0, 0.0),
            v3(7.0, 3.0, 0.0),
        ]);
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![
                Bound { vertex: None,
                    outer: true,
                    edges: outer,
                },
                Bound { vertex: None,
                    outer: false,
                    edges: inner,
                },
            ],
        };
        let m = run(&face, &surfaces, &curves, 0.01);
        assert_eq!(m.stats.faces_ok, 1);
        assert!((area(&m) - (100.0 - 16.0)).abs() < 1e-9, "area {}", area(&m));
    }

    /// Every vertex of a stitched band stays where its parameters say it is,
    /// even when the two rims start at different angles. Shifting one rim's
    /// parameters by anything but whole periods gives vertices the normal of
    /// another angle, which no area or closedness check notices.
    #[test]
    fn a_stitched_band_keeps_its_points_where_their_parameters_put_them() {
        let (r, h) = (5.0, 20.0);
        let surfaces = vec![Surface::Cylinder { ax: unit_axis(), r }];
        let curves = vec![
            Curve::Circle {
                ax: Axis2::new(Vec3::ZERO, None, None),
                r,
            },
            Curve::Circle {
                ax: Axis2::new(v3(0.0, 0.0, h), None, None),
                r,
            },
        ];
        // A third of a turn apart, which is what a rim shared with a neighbour
        // that starts somewhere else looks like.
        let off = TWO_PI / 3.0;
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![
                Bound {
                    vertex: None,
                    outer: true,
                    edges: vec![EdgeUse {
                        curve: 0,
                        edge: 0,
                        t0: 0.0,
                        t1: TWO_PI,
                    }],
                },
                Bound {
                    vertex: None,
                    outer: false,
                    edges: vec![EdgeUse {
                        curve: 1,
                        edge: 1,
                        t0: off + TWO_PI,
                        t1: off,
                    }],
                },
            ],
        };
        let m = run(&face, &surfaces, &curves, 0.01);
        assert_eq!(m.stats.seam_bands, 1);
        assert_eq!(m.stats.faces_ok, 1);
        // Each vertex carries the analytic normal at its own parameters, so a
        // vertex in the wrong place is one whose normal does not point out of
        // the cylinder at the place it actually is.
        for v in &m.vertices {
            let p = v.position();
            let n = v.normal();
            let radial = v3(p.x, p.y, 0.0).normalize();
            assert!(
                n.dot(radial) > 0.999,
                "vertex at {p:.4?} carries the normal {n:.4?} of somewhere else"
            );
        }
        let exact = TWO_PI * r * h;
        let a = area(&m);
        assert!((a - exact).abs() < exact * 0.005, "area {a} vs exact {exact}");
    }

    /// A full cylinder wall, given as two circles with no seam edge, must be
    /// recognised as a seam band and produce the right lateral area.
    #[test]
    fn full_cylinder_wall_is_stitched_across_the_seam() {
        let r = 5.0;
        let h = 20.0;
        let surfaces = vec![Surface::Cylinder {
            ax: unit_axis(),
            r,
        }];
        let bottom = Curve::Circle {
            ax: Axis2::new(Vec3::ZERO, None, None),
            r,
        };
        let top = Curve::Circle {
            ax: Axis2::new(v3(0.0, 0.0, h), None, None),
            r,
        };
        let curves = vec![bottom, top];
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![
                Bound { vertex: None,
                    outer: true,
                    edges: vec![EdgeUse {
                        curve: 0,
                        edge: 0,
                        t0: 0.0,
                        t1: TWO_PI,
                    }],
                },
                Bound { vertex: None,
                    outer: false,
                    edges: vec![EdgeUse {
                        curve: 1,
                        edge: 1,
                        t0: TWO_PI,
                        t1: 0.0,
                    }],
                },
            ],
        };
        let tol = 0.01;
        let m = run(&face, &surfaces, &curves, tol);
        assert_eq!(m.stats.faces_ok, 1, "face should tessellate");
        assert_eq!(m.stats.seam_bands, 1, "should be handled as a seam band");
        let exact = TWO_PI * r * h;
        let got = area(&m);
        assert!(
            (got - exact).abs() < exact * 0.005,
            "area {got} vs exact {exact}"
        );
        // Every vertex must sit exactly on the cylinder.
        for v in &m.vertices {
            let rr = (v.pos[0] as f64).hypot(v.pos[1] as f64);
            assert!((rr - r).abs() < 1e-6, "off-cylinder point r={rr}");
            assert!((-1e-6..=h + 1e-6).contains(&(v.pos[2] as f64)));
        }
        // Normals must point radially outward.
        for v in &m.vertices {
            let n = v.normal();
            let radial = v3(v.pos[0] as f64, v.pos[1] as f64, 0.0).normalize();
            assert!(n.dot(radial) > 0.99, "normal {n:?} not radial");
        }
    }

    /// The seam a face runs off is not always the u one. A sleeve of a torus --
    /// a quarter of the ring, bounded by two cross-section circles and closed
    /// the whole way round the tube -- wraps in v instead, and has to be
    /// stitched along that axis.
    #[test]
    fn torus_sleeve_is_stitched_across_the_v_seam() {
        let (major, minor) = (10.0, 3.0);
        let span = std::f64::consts::FRAC_PI_2;
        let surfaces = vec![Surface::Torus {
            ax: unit_axis(),
            major,
            minor,
            inner: false,
        }];
        // Cross-sections at u = 0 and u = pi/2: circles of the tube's radius
        // standing in the planes that contain the torus axis.
        let curves = vec![
            Curve::Circle {
                ax: Axis2::new(
                    v3(major, 0.0, 0.0),
                    Some(v3(0.0, -1.0, 0.0)),
                    Some(v3(1.0, 0.0, 0.0)),
                ),
                r: minor,
            },
            Curve::Circle {
                ax: Axis2::new(
                    v3(0.0, major, 0.0),
                    Some(v3(1.0, 0.0, 0.0)),
                    Some(v3(0.0, 1.0, 0.0)),
                ),
                r: minor,
            },
        ];
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![
                Bound { vertex: None,
                    outer: true,
                    edges: vec![EdgeUse {
                        curve: 0,
                        edge: 0,
                        t0: 0.0,
                        t1: TWO_PI,
                    }],
                },
                Bound { vertex: None,
                    outer: false,
                    edges: vec![EdgeUse {
                        curve: 1,
                        edge: 1,
                        t0: TWO_PI,
                        t1: 0.0,
                    }],
                },
            ],
        };
        let m = run(&face, &surfaces, &curves, 0.01);
        assert_eq!(m.stats.faces_ok, 1, "face should tessellate");
        assert_eq!(m.stats.seam_bands, 1, "should be handled as a seam band");
        let exact = 2.0 * std::f64::consts::PI * major * minor * span;
        let got = area(&m);
        assert!(
            (got - exact).abs() < exact * 0.005,
            "area {got} vs exact {exact}"
        );
        // Every vertex must sit on the tube, and inside the quarter swept.
        for v in &m.vertices {
            let p = v3(v.pos[0] as f64, v.pos[1] as f64, v.pos[2] as f64);
            let d = (p.x.hypot(p.y) - major).hypot(p.z);
            assert!((d - minor).abs() < 1e-6, "off-torus point, tube radius {d}");
            let u = p.y.atan2(p.x);
            assert!(
                (-1e-6..=span + 1e-6).contains(&u),
                "point at u={u} outside the sleeve"
            );
        }
    }

    /// A lens: two arcs bulging either side of the same chord. At a tolerance
    /// coarser than either bulge both arcs chord to that one line, and the face
    /// between them collapses to a there-and-back ring of three points -- so it
    /// has to be sampled again rather than dropped.
    #[test]
    fn a_sliver_between_two_arcs_survives_a_coarse_tolerance() {
        let surfaces = vec![Surface::Plane(unit_axis())];
        // Two arcs of radius 13 over the same 2.23 chord, bowing opposite ways.
        let arc = |up: bool| {
            let r = 13.0;
            let cy = if up { -r } else { r };
            Curve::Circle {
                ax: Axis2::new(v3(0.0, cy, 0.0), None, Some(v3(0.0, if up { 1.0 } else { -1.0 }, 0.0))),
                r,
            }
        };
        let curves = vec![arc(true), arc(false)];
        let half = (1.1157f64 / 13.0).asin();
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![Bound { vertex: None,
                outer: true,
                edges: vec![
                    EdgeUse {
                        curve: 0,
                        edge: 0,
                        t0: -half,
                        t1: half,
                    },
                    EdgeUse {
                        curve: 1,
                        edge: 1,
                        t0: half,
                        t1: -half,
                    },
                ],
            }],
        };
        // Coarse enough that each arc is within tolerance of the shared chord.
        let m = run(&face, &surfaces, &curves, 0.3);
        assert_eq!(m.stats.faces_ok, 1, "the sliver should still tessellate");
        assert_eq!(m.stats.faces_failed(), 0);
        assert!(area(&m) > 0.0, "and it should enclose something");
    }

    /// A loop that circles a torus once cuts it into two rings rather than
    /// leaving a cap: there is no pole to gather the outline onto, so the face
    /// is the band between the loop and itself one whole turn away in v.
    #[test]
    fn a_lone_loop_circling_a_torus_closes_the_long_way_round() {
        let (major, minor) = (10.0, 3.0);
        let surfaces = vec![Surface::Torus {
            ax: unit_axis(),
            major,
            minor,
            inner: false,
        }];
        // The outer equator: v = 0 all the way round.
        let curves = vec![Curve::Circle {
            ax: Axis2::new(Vec3::ZERO, None, None),
            r: major + minor,
        }];
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![Bound { vertex: None,
                outer: true,
                edges: vec![EdgeUse {
                    curve: 0,
                    edge: 0,
                    t0: 0.0,
                    t1: TWO_PI,
                }],
            }],
        };
        let tol = 0.02;
        let m = run(&face, &surfaces, &curves, tol);
        assert_eq!(m.stats.faces_ok, 1, "face should tessellate");
        assert_eq!(m.stats.seam_bands, 1, "closed against itself in v");
        // Cut at the equator and taken all the way round, that is the whole
        // torus: 4 pi^2 R r.
        let exact = 4.0 * std::f64::consts::PI * std::f64::consts::PI * major * minor;
        let got = area(&m);
        // A chord within `tol` of the tube cuts off about a third of tol / r
        // of the area beside it: allow all of that, and no second skin.
        assert!(
            (got - exact).abs() < exact * tol / minor,
            "area {got} vs whole torus {exact}"
        );
        for v in &m.vertices {
            let p = v3(v.pos[0] as f64, v.pos[1] as f64, v.pos[2] as f64);
            let d = (p.x.hypot(p.y) - major).hypot(p.z);
            assert!((d - minor).abs() < 1e-6, "off-torus point, tube radius {d}");
        }
    }

    /// A sphere has two poles, and which one a cap closes on is read off the
    /// traversal direction -- unless the file says, with a `VERTEX_LOOP`. Here
    /// the two disagree: the traversal points at the north pole and the vertex
    /// names the south. The file wins, and the difference is most of the sphere.
    #[test]
    fn a_vertex_loop_settles_which_pole_a_cap_closes_on() {
        let r = 10.0;
        let lat: f64 = 0.3; // the circle sits above the equator
        let surfaces = vec![Surface::Sphere {
            ax: unit_axis(),
            r,
        }];
        let curves = vec![Curve::Circle {
            ax: Axis2::new(v3(0.0, 0.0, r * lat.sin()), None, None),
            r: r * lat.cos(),
        }];
        let cap = |vertex: Option<Vec3>| {
            let face = Face {
                surface: 0,
                same_sense: true,
                style: None,
                domain: None,
                bounds: vec![
                    Bound {
                        vertex: None,
                        outer: true,
                        edges: vec![EdgeUse {
                            curve: 0,
                            edge: 0,
                            t0: 0.0,
                            t1: TWO_PI,
                        }],
                    },
                    Bound {
                        vertex,
                        outer: false,
                        edges: vec![],
                    },
                ],
            };
            let m = run(&face, &surfaces, &curves, 0.01);
            assert_eq!(m.stats.faces_ok, 1);
            assert_eq!(m.stats.seam_caps, 1);
            area(&m)
        };
        // Zone area is 2*pi*r*h either way; only h differs.
        let north = 2.0 * std::f64::consts::PI * r * (r - r * lat.sin());
        let south = 2.0 * std::f64::consts::PI * r * (r + r * lat.sin());
        // With nothing stated, the traversal decides and gives the north cap.
        let inferred = cap(None);
        assert!(
            (inferred - north).abs() < north * 0.02,
            "inferred {inferred} vs north {north}"
        );
        // Naming the south pole overrides it.
        let stated = cap(Some(v3(0.0, 0.0, -r)));
        assert!(
            (stated - south).abs() < south * 0.02,
            "stated {stated} vs south {south}"
        );
    }

    /// A ball is one spherical face whose only bound runs up the seam meridian
    /// and straight back down it. That encloses nothing in parameter space, but
    /// it is how a complete closed surface is written.
    #[test]
    fn a_ball_bounded_only_by_its_seam_is_a_whole_sphere() {
        let r = 4.0;
        let surfaces = vec![Surface::Sphere {
            ax: unit_axis(),
            r,
        }];
        // The seam meridian at u = pi, pole to pole and back again.
        let curves = vec![Curve::Circle {
            ax: Axis2::new(Vec3::ZERO, Some(v3(0.0, 1.0, 0.0)), Some(v3(-1.0, 0.0, 0.0))),
            r,
        }];
        let half = std::f64::consts::FRAC_PI_2;
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![Bound { vertex: None,
                outer: true,
                edges: vec![
                    EdgeUse {
                        curve: 0,
                        edge: 0,
                        t0: -half,
                        t1: half,
                    },
                    EdgeUse {
                        curve: 0,
                        edge: 0,
                        t0: half,
                        t1: -half,
                    },
                ],
            }],
        };
        let tol = 0.01;
        let m = run(&face, &surfaces, &curves, tol);
        assert_eq!(m.stats.faces_ok, 1, "the ball should tessellate");
        assert_eq!(m.stats.faces_failed(), 0);
        let exact = 4.0 * std::f64::consts::PI * r * r;
        let got = area(&m);
        // What chords within `tol` can cut off and no more: no part of the
        // ball laid down twice.
        assert!(
            (got - exact).abs() < exact * tol / r,
            "area {got} vs whole sphere {exact}"
        );
        for v in &m.vertices {
            let p = v3(v.pos[0] as f64, v.pos[1] as f64, v.pos[2] as f64);
            assert!((p.length() - r).abs() < 1e-6, "off-sphere point {p:?}");
        }
    }

    /// The other spelling of the same thing: no bound at all. A closed surface
    /// needs no boundary curve, so a ball or an O-ring is one face with an empty
    /// bound list.
    #[test]
    fn a_face_with_no_bounds_is_the_whole_surface() {
        // Held to what chords within `tol` can cut off of a surface whose
        // tightest radius is r, which is well under a percent here, and no
        // more, so no part is laid down twice.
        let (r, tol) = (4.0, 0.01);
        let ball = run(
            &Face {
                surface: 0,
                same_sense: true,
                style: None,
                domain: None,
                bounds: vec![],
            },
            &[Surface::Sphere {
                ax: unit_axis(),
                r,
            }],
            &[],
            tol,
        );
        assert_eq!(ball.stats.faces_ok, 1);
        let exact = 4.0 * std::f64::consts::PI * r * r;
        assert!((area(&ball) - exact).abs() < exact * tol / r, "{}", area(&ball));

        let (major, minor, tol) = (10.0, 3.0, 0.02);
        let ring = run(
            &Face {
                surface: 0,
                same_sense: true,
                style: None,
                domain: None,
                bounds: vec![],
            },
            &[Surface::Torus {
                ax: unit_axis(),
                major,
                minor,
                inner: false,
            }],
            &[],
            tol,
        );
        assert_eq!(ring.stats.faces_ok, 1);
        let exact = 4.0 * std::f64::consts::PI * std::f64::consts::PI * major * minor;
        assert!((area(&ring) - exact).abs() < exact * tol / minor, "{}", area(&ring));

        // A plane has no such rectangle, so a bound-less face on one is simply
        // unbounded and must still be reported rather than invented.
        let flat = run(
            &Face {
                surface: 0,
                same_sense: true,
                style: None,
                domain: None,
                bounds: vec![],
            },
            &[Surface::Plane(unit_axis())],
            &[],
            0.01,
        );
        assert_eq!(flat.stats.faces_ok, 0);
        assert_eq!(flat.stats.fail_no_loops, 1);
    }

    /// The same loop, but with an area of 1e-16 instead of an exact zero. An
    /// absolute threshold catches only some of these; the test has to scale
    /// with the loop.
    #[test]
    fn a_loop_encloses_nothing_whether_or_not_the_zero_is_exact() {
        let seam = [
            v2(3.141_592_653_589_793, -1.5),
            v2(3.141_592_653_589_793, 1.5),
            v2(3.141_592_653_589_793, -1.5),
        ];
        assert!(encloses_nothing(&seam));
        assert!(signed_area(&seam).abs() < 1e-15);
        // A real triangle of the same extent is not caught.
        let solid = [v2(0.0, -1.5), v2(0.5, 1.5), v2(-0.5, 0.0)];
        assert!(!encloses_nothing(&solid));
    }

    /// Several separate boundary segments on one line is an ordinary machined
    /// profile, and it has to come out with exactly the area it encloses.
    /// Points on a line are exactly collinear to the triangulator's exact
    /// predicates, so whether such an outline is refused depends on the
    /// coordinates' low digits; these are taken from a real face.
    #[test]
    fn a_comb_shaped_outline_still_triangulates() {
        let surfaces = vec![Surface::Plane(unit_axis())];
        // The exact coordinates are the input under test. A bar along y = 0.21
        // to 0.24 with two teeth reaching down to y = 0 on splayed sides, so
        // four separate segments lie along y = 0.21 and two more along y = 0.
        let corners: Vec<Vec3> = [
            (1.477905, 0.240000),
            (0.000000, 0.240000),
            (0.000000, 0.210000),
            (0.142996, 0.210000),
            (0.038594, 0.000000),
            (0.069666, 0.000000),
            (0.174067, 0.210000),
            (1.303838, 0.210000),
            (1.408239, 0.000000),
            (1.439311, 0.000000),
            (1.334910, 0.210000),
            (1.477905, 0.210000),
        ]
        .iter()
        .map(|&(x, y)| v3(x, y, 0.0))
        .collect();
        let n = corners.len();
        let curves: Vec<Curve> = (0..n)
            .map(|i| {
                let (a, b) = (corners[i], corners[(i + 1) % n]);
                Curve::Line {
                    origin: a,
                    dir: b - a,
                }
            })
            .collect();
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![Bound {
                vertex: None,
                outer: true,
                edges: (0..n as u32)
                    .map(|i| EdgeUse {
                        curve: i,
                        edge: i,
                        t0: 0.0,
                        t1: 1.0,
                    })
                    .collect(),
            }],
        };
        let m = run(&face, &surfaces, &curves, 0.05);
        assert_eq!(m.stats.faces_ok, 1, "face should tessellate");
        assert_eq!(
            m.stats.cdt_failed, 0,
            "a comb is not a reason to fall back to ear clipping"
        );
        // Against the shoelace area of the outline itself, rather than a figure
        // worked out by hand: the outline is straight-sided, so the two agree
        // exactly when the triangulation covers it once.
        let exact: f64 = (0..n)
            .map(|i| {
                let (p, q) = (corners[i], corners[(i + 1) % n]);
                p.x * q.y - q.x * p.y
            })
            .sum::<f64>()
            .abs()
            * 0.5;
        let a = area(&m);
        assert!((a - exact).abs() < exact * 1e-6, "area {a} vs exact {exact}");
    }

    /// A circle's chords must stay close to it relative to its own radius, not
    /// only within the absolute tolerance. At one chord per quarter turn the
    /// polygon sits 29% of the radius inside the circle, and features near the
    /// boundary fall outside the face's own outline.
    #[test]
    fn a_circle_is_never_chorded_into_a_crude_polygon() {
        let r = 0.45;
        // A tolerance far coarser than the circle is small -- which is the
        // normal case in an assembly, where one tolerance covers a metre-long
        // frame and the millimetre features bolted to it.
        let n = Curve::Circle {
            ax: Axis2::new(Vec3::ZERO, None, None),
            r,
        }
        .initial_segments(0.0, TWO_PI, 10.0);
        let sagitta = r * (1.0 - (TWO_PI / (2.0 * n as f64)).cos());
        assert!(
            sagitta < r * 0.08,
            "{n} chords leave the polygon {:.1}% of the radius inside the circle",
            sagitta / r * 100.0
        );
    }

    /// A loop that comes back a hair short of its own start still closes there.
    /// The two edges meeting at the start vertex are evaluated from opposite
    /// ends of their ranges and land a few parts in a million apart; appending
    /// the start beside that leaves a zero-area spike, and the constrained
    /// triangulator refuses the whole outline over it.
    #[test]
    fn a_loop_that_closes_a_hair_short_still_closes() {
        let surfaces = vec![Surface::Plane(unit_axis())];
        // The last edge stops a micron past where the first one began.
        let corners = [
            v3(0.0, 0.0, 0.0),
            v3(10.0, 0.0, 0.0),
            v3(10.0, 10.0, 0.0),
            v3(0.0, 10.0, 0.0),
        ];
        let mut curves: Vec<Curve> = (0..4)
            .map(|i| {
                let (a, b) = (corners[i], corners[(i + 1) % 4]);
                Curve::Line {
                    origin: a,
                    dir: b - a,
                }
            })
            .collect();
        curves[3] = Curve::Line {
            origin: corners[3],
            dir: v3(1e-6, 0.0, 0.0) - corners[3],
        };
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![Bound {
                vertex: None,
                outer: true,
                edges: (0..4)
                    .map(|i| EdgeUse {
                        curve: i,
                        edge: i,
                        t0: 0.0,
                        t1: 1.0,
                    })
                    .collect(),
            }],
        };
        let m = run(&face, &surfaces, &curves, 0.05);
        assert_eq!(m.stats.faces_ok, 1, "face should tessellate");
        assert_eq!(
            m.stats.cdt_failed, 0,
            "a micron of slack is not a reason to fall back to ear clipping"
        );
        let a = area(&m);
        assert!((a - 100.0).abs() < 0.01, "area {a} vs exact 100");
    }

    /// Every segment of the outline has to survive into the triangulation.
    ///
    /// Ear clipping is free to answer with any triangulation of the same region,
    /// and on a run of collinear boundary points it runs a chord straight past
    /// one. The neighbouring face keeps that vertex, so the two stop sharing
    /// edges. Constraining the outline is what makes that impossible.
    #[test]
    fn every_boundary_segment_survives_triangulation() {
        // A square whose bottom edge is split into four collinear pieces, which
        // is what `max_seg` does to any long straight boundary.
        let surfaces = vec![Surface::Plane(unit_axis())];
        let corners = [
            v3(0.0, 0.0, 0.0),
            v3(2.0, 0.0, 0.0),
            v3(4.0, 0.0, 0.0),
            v3(6.0, 0.0, 0.0),
            v3(8.0, 0.0, 0.0),
            v3(8.0, 8.0, 0.0),
            v3(0.0, 8.0, 0.0),
        ];
        let curves: Vec<Curve> = (0..corners.len())
            .map(|i| {
                let (a, b) = (corners[i], corners[(i + 1) % corners.len()]);
                Curve::Line {
                    origin: a,
                    dir: b - a,
                }
            })
            .collect();
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![Bound {
                vertex: None,
                outer: true,
                edges: (0..corners.len() as u32)
                    .map(|i| EdgeUse {
                        curve: i,
                        edge: i,
                        t0: 0.0,
                        t1: 1.0,
                    })
                    .collect(),
            }],
        };
        let m = run(&face, &surfaces, &curves, 0.05);
        assert_eq!(m.stats.faces_ok, 1);

        // No vertex may sit in the middle of somebody else's edge.
        let p = |i: u32| {
            let v = m.vertices[i as usize].pos;
            v3(v[0] as f64, v[1] as f64, v[2] as f64)
        };
        for t in m.indices.chunks_exact(3) {
            for k in 0..3 {
                let (a, b) = (p(t[k]), p(t[(k + 1) % 3]));
                let d = b - a;
                for i in 0..m.vertices.len() as u32 {
                    if i == t[k] || i == t[(k + 1) % 3] {
                        continue;
                    }
                    let v = p(i);
                    let s = (v - a).dot(d) / d.length_sq();
                    if !(1e-6..=1.0 - 1e-6).contains(&s) {
                        continue;
                    }
                    assert!(
                        (v - (a + d * s)).length() > 1e-9,
                        "vertex {v:?} sits inside edge {a:?}-{b:?}"
                    );
                }
            }
        }
    }

    /// And if the triangulator does leave a segment out, what comes back still
    /// has to keep them all. Five nearly collinear rim points along v = 28,
    /// two of them a few parts in 1e15 off the line: the triangulator reports
    /// success with an edge running from the first to the last straight
    /// through the ones between.
    #[test]
    fn a_triangulation_that_leaves_out_a_segment_is_tried_again() {
        let pts = [
            (3.7143788717361437, 1.0625000000000142),
            (2.122400753231653, 28.000000000000014),
            (2.0162688786646905, 28.000000000000007),
            (1.9101370040977026, 28.000000000000007),
            (1.8040051295307415, 28.000000000000004),
            (1.697873254963808, 28.000000000000007),
            (0.0, 36.289658891145315),
        ];
        let contours = vec![(0..pts.len()).chain(std::iter::once(0)).collect::<Vec<_>>()];
        let raw = cdt::triangulate_contours(&pts, &contours).expect("the triangulator takes it");
        assert!(
            !keeps_outline(&raw, &contours),
            "the triangulator kept this outline whole, so it no longer shows the fault"
        );
        let (got, shaken) = triangulate_outline(&pts, &contours);
        let got = got.expect("a triangulation");
        assert!(keeps_outline(&got, &contours), "a segment of the outline is still missing");
        assert!(shaken, "only the second attempt keeps them all");
    }

    /// A cylinder wall is flat along its axis, so one row of quads at the
    /// chord-error step is the whole answer, and anything much past that is a
    /// triangulation fighting itself. A band between two different rims is an
    /// ordinary outline and goes through the constrained triangulator.
    #[test]
    fn a_cylinder_wall_is_one_row_of_quads() {
        let (r, h) = (5.0, 20.0);
        let surfaces = vec![Surface::Cylinder { ax: unit_axis(), r }];
        // The bottom rim in four arcs rather than one circle: a rim shared with
        // a neighbour is normally cut up like this, and each join is a point the
        // neighbour will keep whatever this face decides to do.
        let mut curves: Vec<Curve> = (0..4)
            .map(|_| Curve::Circle {
                ax: Axis2::new(Vec3::ZERO, None, None),
                r,
            })
            .collect();
        curves.push(Curve::Circle {
            ax: Axis2::new(v3(0.0, 0.0, h), None, None),
            r,
        });
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![
                Bound {
                    vertex: None,
                    outer: true,
                    edges: (0..4)
                        .map(|k| EdgeUse {
                            curve: k,
                            edge: k,
                            t0: k as f64 * TWO_PI * 0.25,
                            t1: (k + 1) as f64 * TWO_PI * 0.25,
                        })
                        .collect(),
                },
                Bound {
                    vertex: None,
                    outer: false,
                    edges: vec![EdgeUse {
                        curve: 4,
                        edge: 4,
                        t0: TWO_PI,
                        t1: 0.0,
                    }],
                },
            ],
        };
        let tol = 0.01;
        let m = run(&face, &surfaces, &curves, tol);
        assert_eq!(m.stats.seam_bands, 1, "should be handled as a seam band");

        // Ear clipping fans the band off a handful of vertices, and refinement
        // then has to chase every sliver down to tolerance.
        let step = 2.0 * (1.0 - tol / r).acos();
        let ideal = (TWO_PI / step).ceil() * 2.0;
        let got = m.indices.len() / 3;
        assert!(
            (got as f64) < ideal * 4.0,
            "{got} triangles for a cylinder wall that needs about {ideal}"
        );
        // And it still has to be the right wall.
        let exact = TWO_PI * r * h;
        let a = area(&m);
        assert!((a - exact).abs() < exact * 0.005, "area {a} vs exact {exact}");
        assert_eq!(m.stats.cdt_failed, 0, "and the triangulator should manage it");
    }

    /// A face bounded by nothing but a trim is the rectangle the trim names.
    /// `RECTANGULAR_TRIMMED_SURFACE` does not reparameterise, so the basis
    /// surface is used directly and the rectangle rides on the face; over a
    /// whole period its two sides are the same line.
    #[test]
    fn a_trim_bounds_a_face_that_nothing_else_bounds() {
        let (r, h, tol) = (5.0, 10.0, 0.01);
        let surfaces = vec![Surface::Cylinder { ax: unit_axis(), r }];
        let face = |u1: f64| Face {
            surface: 0,
            same_sense: true,
            style: None,
            bounds: Vec::new(),
            domain: Some((0.0, u1, 0.0, h)),
        };
        // A quarter turn, whose two sides are different lines, and the whole
        // way round, whose two sides are one: the same wall either way.
        for turn in [0.25, 1.0] {
            let m = run(&face(TWO_PI * turn), &surfaces, &[], tol);
            assert_eq!(m.stats.faces_ok, 1);
            assert_eq!(m.stats.cdt_failed, 0, "{turn} of a turn defeated the triangulator");
            let want = TWO_PI * turn * r * h;
            let got = area(&m);
            assert!((got - want).abs() < want * tol / r, "{turn} of a turn: area {got} vs {want}");
            // One row of quads at the chord-error step, as a wall between two
            // rims is.
            let step = 2.0 * (1.0 - tol / r).acos();
            let ideal = (TWO_PI * turn / step).ceil() * 2.0;
            let tris = m.indices.len() / 3;
            assert!((tris as f64) < ideal * 4.0, "{turn} of a turn: {tris} triangles for about {ideal}");
        }
    }

    /// A cone sector that reaches its own apex still comes out the right area.
    /// Closing it onto the apex must not repeat a point the outline already
    /// has: the zero-length segment would make the constrained triangulator
    /// refuse it. Whether a given outline is refused depends on the crate's
    /// internals, so this pins the geometry rather than the refusal.
    #[test]
    fn a_cap_does_not_repeat_an_apex_the_outline_already_has() {
        let (r, h) = (5.0, 10.0);
        // Apex at the top, opening downwards.
        let surfaces = vec![Surface::Cone {
            ax: unit_axis(),
            r,
            // Negative, so the radius shrinks with v and the apex is at the top.
            half_angle: -(r / h).atan(),
        }];
        let apex = v3(0.0, 0.0, h);
        let rim = |u: f64| v3(r * u.cos(), r * u.sin(), 0.0);
        // A sector: apex, out to the rim, round it, and back to the apex.
        let curves = vec![
            Curve::Line {
                origin: apex,
                dir: rim(0.0) - apex,
            },
            Curve::Circle {
                ax: unit_axis(),
                r,
            },
            Curve::Line {
                origin: rim(TWO_PI),
                dir: apex - rim(TWO_PI),
            },
        ];
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![Bound {
                vertex: None,
                outer: true,
                edges: vec![
                    EdgeUse { curve: 0, edge: 0, t0: 0.0, t1: 1.0 },
                    EdgeUse { curve: 1, edge: 1, t0: 0.0, t1: TWO_PI },
                    EdgeUse { curve: 2, edge: 2, t0: 0.0, t1: 1.0 },
                ],
            }],
        };
        let m = run(&face, &surfaces, &curves, 0.01);
        assert_eq!(m.stats.faces_ok, 1, "face should tessellate");
        assert_eq!(
            m.stats.cdt_failed, 0,
            "an apex named twice is not a reason to fall back to ear clipping"
        );
        // Lateral area of the whole cone: pi r * slant.
        let exact = TWO_PI * 0.5 * r * (r * r + h * h).sqrt();
        let a = area(&m);
        assert!((a - exact).abs() < exact * 0.01, "area {a} vs exact {exact}");
    }

    /// An annulus whose rim is thinner than its own chord error is still a ring,
    /// not a disc. Chorded, its two rings cross, so the constrained
    /// triangulator refuses the outline and ear clipping fills the hole;
    /// zipping the two rings into a band cannot make that mistake.
    #[test]
    fn a_thin_annulus_is_not_filled_in() {
        let (ro, ri) = (1.94, 1.88);
        let surfaces = vec![Surface::Plane(unit_axis())];
        let curves = vec![
            Curve::Circle {
                ax: unit_axis(),
                r: ro,
            },
            // Half a step out of phase with the outer ring, which is how two
            // separately-authored circles land and what puts the inner ring's
            // corners across the outer ring's edges.
            Curve::Circle {
                ax: Axis2::new(
                    Vec3::ZERO,
                    None,
                    Some(v3((TWO_PI / 16.0).cos(), (TWO_PI / 16.0).sin(), 0.0)),
                ),
                r: ri,
            },
        ];
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![
                Bound {
                    vertex: None,
                    outer: true,
                    edges: vec![EdgeUse { curve: 0, edge: 0, t0: 0.0, t1: TWO_PI }],
                },
                Bound {
                    vertex: None,
                    outer: false,
                    edges: vec![EdgeUse { curve: 1, edge: 1, t0: TWO_PI, t1: 0.0 }],
                },
            ],
        };
        // Coarse enough that both rings fall back to the eight-segment floor,
        // where the inner ring's corners (1.88) stand outside the outer ring's
        // edges (1.94 cos 22.5 = 1.79).
        let m = run(&face, &surfaces, &curves, 0.5);
        assert_eq!(m.stats.faces_ok, 1, "face should tessellate");
        assert_eq!(m.stats.fail_miscovered, 0, "and not be thrown away");
        let disc = TWO_PI * 0.5 * ro * ro;
        let a = area(&m);
        assert!(
            a < disc * 0.5,
            "area {a} against a disc of {disc}: drawn filled in"
        );
    }

    /// A hole has to be brought into the same period as the ring it sits in.
    /// Each bound is projected and unwrapped on its own, so a hole can land a
    /// whole turn from an outer ring that crosses the seam, and the
    /// triangulator then covers `outer + hole` instead of `outer - hole`.
    #[test]
    fn a_hole_is_brought_into_the_outer_ring_s_period() {
        let r = 5.0;
        let surfaces = vec![Surface::Cylinder { ax: unit_axis(), r }];
        // Projection puts u in (-pi, pi], so an outer ring spanning 2.5 -> 4.0
        // straddles that cut and unwraps past it. A hole wholly beyond pi does
        // not, and comes back a whole turn below.
        let arc = |z: f64| Curve::Circle {
            ax: Axis2::new(v3(0.0, 0.0, z), None, None),
            r,
        };
        let at = |u: f64, z: f64| v3(r * u.cos(), r * u.sin(), z);
        let side = |u: f64, z0: f64, z1: f64| Curve::Line {
            origin: at(u, z0),
            dir: at(u, z1) - at(u, z0),
        };
        let curves = vec![
            arc(0.0),          // 0: outer bottom
            side(4.0, 0.0, 10.0), // 1
            arc(10.0),         // 2: outer top
            side(2.5, 10.0, 0.0), // 3
            arc(4.0),          // 4: hole bottom
            side(3.6, 4.0, 6.0),  // 5
            arc(6.0),          // 6: hole top
            side(3.4, 6.0, 4.0),  // 7
        ];
        let e = |curve: u32, t0: f64, t1: f64| EdgeUse {
            curve,
            edge: curve,
            t0,
            t1,
        };
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![
                Bound {
                    vertex: None,
                    outer: true,
                    edges: vec![e(0, 2.5, 4.0), e(1, 0.0, 1.0), e(2, 4.0, 2.5), e(3, 0.0, 1.0)],
                },
                Bound {
                    vertex: None,
                    outer: false,
                    // Traversed the other way round, as a hole is.
                    edges: vec![e(4, 3.6, 3.4), e(7, 0.0, 1.0), e(6, 3.4, 3.6), e(5, 0.0, 1.0)],
                },
            ],
        };
        let m = run(&face, &surfaces, &curves, 0.01);
        assert_eq!(m.stats.faces_ok, 1, "face should tessellate");
        assert_eq!(
            m.stats.miscovered, 0,
            "the hole was not subtracted from the ring it sits in"
        );
        // A band 1.5 rad by 10 mm on a radius-5 cylinder, less a 0.2 by 2 hole.
        let exact = 1.5 * 10.0 * r - 0.2 * 2.0 * r;
        let a = area(&m);
        assert!((a - exact).abs() < exact * 0.01, "area {a} vs exact {exact}");
    }

    /// How narrow a face is has to be measured across it, not from its extent:
    /// a thin annulus of large radius is enormous by its bounding box and
    /// still thinner than the chord tolerance drawing it.
    #[test]
    fn width_is_measured_across_a_face_not_along_it() {
        let close = |got: Option<f64>, want: f64| {
            let got = got.expect("a width");
            assert!((got - want).abs() < want * 1e-9, "{got} vs {want}");
        };
        // A disc of radius R: area pi R^2 over perimeter 2 pi R.
        let r = 105.5;
        close(outline_width(TWO_PI * 0.5 * r * r, TWO_PI * r), r);
        // A rectangle, which comes out as its short side.
        let (a, b) = (211.0, 0.105);
        close(outline_width(a * b, 2.0 * (a + b)), a * b / (a + b));
        // An annulus of width w, whatever its radius -- the case this exists
        // for. Exact in the limit; at these proportions within a part in 2000.
        let w = 0.105;
        let (ro, ri) = (r, r - w);
        let got = outline_width(TWO_PI * 0.5 * (ro * ro - ri * ri), TWO_PI * (ro + ri)).expect("a width");
        assert!((got - w).abs() < w * 1e-9, "{got} across an annulus {w} wide");
        assert_eq!(outline_width(0.0, 1.0), None, "no area, no width");
        assert_eq!(outline_width(1.0, 0.0), None, "no outline, no width");
    }

    /// Refinement must actually bend a curved face toward the true surface.
    #[test]
    fn refinement_converges_on_a_sphere_patch() {
        let r = 10.0;
        let surfaces = vec![Surface::Sphere {
            ax: unit_axis(),
            r,
        }];
        // A spherical "square" bounded by four circular arcs would be fiddly to
        // build; use the equator band between two latitude circles instead.
        let lat = |z: f64| {
            let rr = (r * r - z * z).sqrt();
            Curve::Circle {
                ax: Axis2::new(v3(0.0, 0.0, z), None, None),
                r: rr,
            }
        };
        let curves = vec![lat(-4.0), lat(4.0)];
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![
                Bound { vertex: None,
                    outer: true,
                    edges: vec![EdgeUse {
                        curve: 0,
                        edge: 0,
                        t0: 0.0,
                        t1: TWO_PI,
                    }],
                },
                Bound { vertex: None,
                    outer: false,
                    edges: vec![EdgeUse {
                        curve: 1,
                        edge: 1,
                        t0: TWO_PI,
                        t1: 0.0,
                    }],
                },
            ],
        };
        let tol = 0.05;
        let m = run(&face, &surfaces, &curves, tol);
        assert_eq!(m.stats.faces_ok, 1);
        // Every emitted vertex must lie on the sphere.
        for v in &m.vertices {
            let p = v3(v.pos[0] as f64, v.pos[1] as f64, v.pos[2] as f64);
            assert!((p.length() - r).abs() < 1e-6, "off-sphere point {p:?}");
        }
        // The band must actually reach the equator, which only happens if
        // refinement pushed interior points onto the sphere.
        let max_r_xy = m
            .vertices
            .iter()
            .map(|v| (v.pos[0] as f64).hypot(v.pos[1] as f64))
            .fold(0.0f64, f64::max);
        assert!(max_r_xy > r - 0.01, "band never reached the equator: {max_r_xy}");
        // The guarantee refinement makes is a chord-error bound: every edge
        // midpoint and every centroid must lie within `tol` of the sphere.
        let p = |i: u32| {
            let v = m.vertices[i as usize].pos;
            v3(v[0] as f64, v[1] as f64, v[2] as f64)
        };
        let mut worst: f64 = 0.0;
        for t in m.indices.chunks_exact(3) {
            let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
            for probe in [
                (a + b) * 0.5,
                (b + c) * 0.5,
                (c + a) * 0.5,
                (a + b + c) * (1.0 / 3.0),
            ] {
                worst = worst.max((probe.length() - r).abs());
            }
        }
        assert!(worst <= tol * 1.05, "chord error {worst} exceeds tol {tol}");

        // Archimedes: the zone area is 2*pi*r*h. An inscribed mesh made of ear
        // clipped slivers overstates area (the Schwarz lantern effect), so this
        // is only a sanity bound, not a convergence claim.
        let exact = TWO_PI * r * 8.0;
        let got = area(&m);
        assert!(
            (got - exact).abs() < exact * 0.10,
            "area {got} vs exact {exact}"
        );
    }

    /// A face on a long helical patch is drawn on its own turn of it.
    ///
    /// A 40-turn screw blade trimmed seven turns in from each end, the way the
    /// faces at the ends of a screw trim its thread. Point inversion that
    /// samples on a fixed grid starts such an outline on whichever turn the
    /// nearest sample sits on. The area is held against the patch's own, the
    /// integral of |Su x Sv| over the trimmed rectangle, which no triangle
    /// reaching across to another turn can keep.
    #[test]
    fn a_face_on_a_long_helix_is_drawn_on_its_own_turn() {
        use crate::geom::{helical_ribbon, BSplineCurve};
        let turns = 40;
        let s = helical_ribbon(turns, 5.0, 10.0, 11.0);
        let (ua, ub) = (7.0 * 8.0 + 2.3, (turns - 7) as f64 * 8.0 - 2.3);
        // The two rims are the patch's own control rows, and each end is
        // straight across it, since the patch is straight in v.
        let rim = |j: usize| {
            let ctrl = (0..s.nu).map(|i| s.ctrl[i * s.nv + j]).collect();
            let w = s.weights.as_ref().map(|w| (0..s.nu).map(|i| w[i * s.nv + j]).collect());
            Curve::BSpline(Box::new(BSplineCurve::new(s.u_degree, ctrl, w, s.u_knots.clone())))
        };
        let across = |u: f64| {
            let a = s.eval(u, 0.0);
            Curve::Line {
                origin: a,
                dir: s.eval(u, 1.0) - a,
            }
        };
        let curves = vec![rim(0), across(ub), rim(1), across(ua)];
        let exact: f64 = {
            let (nu, nv) = (((ub - ua) * 16.0) as usize, 8);
            let (hu, hv) = ((ub - ua) / nu as f64, 1.0 / nv as f64);
            let mut sum = 0.0;
            for i in 0..nu {
                for j in 0..nv {
                    let (_, du, dv) = s.eval_with_derivs(ua + (i as f64 + 0.5) * hu, (j as f64 + 0.5) * hv);
                    sum += du.cross(dv).length() * hu * hv;
                }
            }
            sum
        };
        let surfaces = vec![s.into_surface().unwrap()];
        let use_of = |curve: u32, t0: f64, t1: f64| EdgeUse { curve, edge: curve, t0, t1 };
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![Bound {
                vertex: None,
                outer: true,
                edges: vec![
                    use_of(0, ua, ub),
                    use_of(1, 0.0, 1.0),
                    use_of(2, ub, ua),
                    use_of(3, 1.0, 0.0),
                ],
            }],
        };
        let tol = 0.02;
        let m = run(&face, &surfaces, &curves, tol);
        assert_eq!(m.stats.faces_ok, 1);
        assert_eq!(m.stats.displaced, 0, "outline points a turn from their parameters");
        assert_eq!(m.stats.miscovered, 0, "triangulated to the wrong area");
        let got = area(&m);
        assert!((got - exact).abs() < exact * 0.01, "area {got} against {exact}");
    }

    /// A cone bounded by a single circle closes at its apex.
    #[test]
    fn cone_cap_closes_at_the_apex() {
        // Radius 5 at v = 0, growing with +v, so the apex sits at v = -5.
        let half = std::f64::consts::FRAC_PI_4;
        let surfaces = vec![Surface::Cone {
            ax: unit_axis(),
            r: 5.0,
            half_angle: half,
        }];
        let curves = vec![Curve::Circle {
            ax: unit_axis(),
            r: 5.0,
        }];
        // Traversed with decreasing u, so the material lies toward -v: the apex.
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![Bound { vertex: None,
                outer: true,
                edges: vec![EdgeUse {
                    curve: 0,
                    edge: 0,
                    t0: TWO_PI,
                    t1: 0.0,
                }],
            }],
        };
        let m = run(&face, &surfaces, &curves, 0.01);
        assert_eq!(m.stats.faces_ok, 1);
        assert_eq!(m.stats.seam_caps, 1, "should close at the apex");

        // The apex must actually be in the mesh.
        let apex = v3(0.0, 0.0, -5.0);
        assert!(
            m.vertices.iter().any(|v| {
                v3(v.pos[0] as f64, v.pos[1] as f64, v.pos[2] as f64).distance(apex) < 1e-9
            }),
            "apex missing from the mesh"
        );
        // Every vertex lies on the cone: radius = 5 + z, for z in [-5, 0].
        for v in &m.vertices {
            let (x, y, z) = (v.pos[0] as f64, v.pos[1] as f64, v.pos[2] as f64);
            assert!((x.hypot(y) - (5.0 + z)).abs() < 1e-6, "off-cone {x} {y} {z}");
            assert!((-5.0 - 1e-6..=1e-6).contains(&z));
        }
        // Lateral area is pi * r * slant. As with the sphere, ear-clipped
        // slivers let an inscribed mesh overstate area slightly, so this bound
        // is loose enough for that but tight enough to catch a missing wedge.
        let exact = std::f64::consts::PI * 5.0 * (50.0f64).sqrt();
        let got = area(&m);
        assert!((got - exact).abs() < exact * 0.03, "area {got} vs {exact}");
    }

    /// Two planes meeting along a shared edge: the edge should be drawn when
    /// they form a corner, and suppressed when they are coplanar (the
    /// degenerate case of a tangent join).
    #[test]
    fn tangent_edges_are_suppressed_but_corners_are_kept() {
        use crate::model::Solid;

        // Both faces are unit squares sharing the segment from (0,0,0)-(1,0,0).
        // `tilt` rotates the second one about that shared edge.
        let build = |tilt: f64| {
            let shared = [v3(0.0, 0.0, 0.0), v3(1.0, 0.0, 0.0)];
            let a = Axis2::new(Vec3::ZERO, Some(v3(0.0, 0.0, 1.0)), Some(v3(1.0, 0.0, 0.0)));
            let b = Axis2::new(
                Vec3::ZERO,
                Some(v3(0.0, -tilt.sin(), tilt.cos())),
                Some(v3(1.0, 0.0, 0.0)),
            );
            let surfaces = vec![Surface::Plane(a), Surface::Plane(b)];
            // Only the shared edge matters; give each face a loop containing it.
            let curves = vec![Curve::Line {
                origin: shared[0],
                dir: shared[1] - shared[0],
            }];
            let edge = EdgeUse {
                curve: 0,
                edge: 42,
                t0: 0.0,
                t1: 1.0,
            };
            let face = |surface: u32| Face {
                surface,
                same_sense: true,
                style: None,
                domain: None,
                bounds: vec![Bound { vertex: None,
                    outer: true,
                    edges: vec![edge],
                }],
            };
            Model {
                solids: vec![Solid {
                    source: u32::MAX,
                    name: String::new(),
                    color: [1.0; 3],
                    alpha: 1.0,
                    faces: vec![face(0), face(1)],
                    wires: Vec::new(),
                    placement: Xform::IDENTITY,
                    local_bbox: Aabb::EMPTY,
                    linked: true,
                    node: None,
                    volume: true,
                }],
                nodes: Vec::new(),
                surfaces,
                curves,
                vertex_bbox: Aabb::EMPTY,
                mm_per_unit: 1.0,
                stats: Default::default(),
            }
        };

        // Coplanar: the two faces have identical normals, so no line.
        let e = extract_edges(&build(0.0), 0.01, f64::INFINITY, 0..1);
        assert_eq!(e.stats.edges_tangent, 1, "coplanar edge should be suppressed");
        assert_eq!(e.stats.edges_sharp, 0);
        assert!(e.span[0].1.is_empty(), "nothing should be drawn in sharp mode");

        // A 10 degree bend is still within the tangent threshold.
        let e = extract_edges(&build(10f64.to_radians()), 0.01, f64::INFINITY, 0..1);
        assert_eq!(e.stats.edges_tangent, 1, "10 degrees is below the threshold");

        // A 90 degree corner is a real edge.
        let e = extract_edges(&build(90f64.to_radians()), 0.01, f64::INFINITY, 0..1);
        assert_eq!(e.stats.edges_sharp, 1, "a right-angle corner must be drawn");
        assert_eq!(e.stats.edges_tangent, 0);
        assert_eq!(e.span[0].1, 0..e.lines.len() as u32);
    }

    /// An edge used by a single face bounds an open shell and always shows.
    #[test]
    fn unpaired_edges_are_always_drawn() {
        use crate::model::Solid;
        let surfaces = vec![Surface::Plane(unit_axis())];
        let curves = vec![Curve::Line {
            origin: v3(0.0, 0.0, 0.0),
            dir: v3(1.0, 0.0, 0.0),
        }];
        let model = Model {
            solids: vec![Solid {
                source: u32::MAX,
                name: String::new(),
                color: [1.0; 3],
                alpha: 1.0,
                faces: vec![Face {
                    surface: 0,
                    same_sense: true,
                    style: None,
                    domain: None,
                    bounds: vec![Bound { vertex: None,
                        outer: true,
                        edges: vec![EdgeUse {
                            curve: 0,
                            edge: 7,
                            t0: 0.0,
                            t1: 1.0,
                        }],
                    }],
                }],
                wires: Vec::new(),
                placement: Xform::IDENTITY,
                local_bbox: Aabb::EMPTY,
                linked: true,
                node: None,
                volume: true,
            }],
            nodes: Vec::new(),
            surfaces,
            curves,
            vertex_bbox: Aabb::EMPTY,
            mm_per_unit: 1.0,
            stats: Default::default(),
        };
        let e = extract_edges(&model, 0.01, f64::INFINITY, 0..1);
        assert_eq!(e.stats.edges_sharp, 1);
        assert_eq!(e.span[0].1, 0..2);
    }

    /// A solid's placement must move both its triangles and its wireframe.
    #[test]
    fn assembly_placement_moves_the_geometry() {
        let surfaces = vec![Surface::Plane(unit_axis())];
        let corners = [
            v3(0.0, 0.0, 0.0),
            v3(2.0, 0.0, 0.0),
            v3(2.0, 2.0, 0.0),
            v3(0.0, 2.0, 0.0),
        ];
        let mut curves = Vec::new();
        let mut edges = Vec::new();
        for i in 0..4 {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            curves.push(Curve::Line {
                origin: a,
                dir: b - a,
            });
            edges.push(EdgeUse {
                curve: i as u32,
                edge: 100 + i as u32,
                t0: 0.0,
                t1: 1.0,
            });
        }
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![Bound { vertex: None, outer: true, edges }],
        };

        // Quarter turn about +Z, then shifted along +X.
        let place = Xform::between(
            &Axis2::new(Vec3::ZERO, None, None),
            &Axis2::new(v3(10.0, 0.0, 0.0), Some(v3(0.0, 0.0, 1.0)), Some(v3(0.0, 1.0, 0.0))),
        );

        let mut out = FaceMesh::default();
        tessellate_face(
            &face,
            &surfaces,
            &curves,
            [1.0; 4],
            0.01,
            f64::INFINITY,
            &place,
            &mut out,
        );
        assert_eq!(out.stats.faces_ok, 1);
        // The square's corner at (2,2,0) rotates to (-2,2,0) then shifts to (8,2,0).
        let want = place.point(v3(2.0, 2.0, 0.0));
        assert!(
            out.vertices.iter().any(|v| {
                v3(v.pos[0] as f64, v.pos[1] as f64, v.pos[2] as f64).distance(want) < 1e-9
            }),
            "placed corner {want:?} missing"
        );
        // Area is preserved: the placement is rigid.
        let mut area = 0.0;
        for t in out.indices.chunks_exact(3) {
            let p = |i: u32| {
                let v = out.vertices[i as usize].pos;
                v3(v[0] as f64, v[1] as f64, v[2] as f64)
            };
            area += (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])).length() * 0.5;
        }
        assert!((area - 4.0).abs() < 1e-9, "area {area}");
        // Normals rotate with the solid: +Z is unchanged by a spin about Z.
        for v in &out.vertices {
            assert!((v.normal().z - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn reversed_face_flips_normals_and_winding() {
        let surfaces = vec![Surface::Plane(unit_axis())];
        let corners = [
            v3(0.0, 0.0, 0.0),
            v3(2.0, 0.0, 0.0),
            v3(2.0, 2.0, 0.0),
            v3(0.0, 2.0, 0.0),
        ];
        let mut curves = Vec::new();
        let mut edges = Vec::new();
        for i in 0..4 {
            let a = corners[i];
            let b = corners[(i + 1) % 4];
            curves.push(Curve::Line {
                origin: a,
                dir: b - a,
            });
            edges.push(EdgeUse {
                curve: i as u32,
                edge: i as u32,
                t0: 0.0,
                t1: 1.0,
            });
        }
        let mut face = Face {
            surface: 0,
            same_sense: false,
            style: None,
            domain: None,
            bounds: vec![Bound { vertex: None, outer: true, edges }],
        };
        let m = run(&face, &surfaces, &curves, 0.01);
        for v in &m.vertices {
            assert!((v.normal().z + 1.0).abs() < 1e-6, "normal should be -Z");
        }
        // Winding follows the normal, so the geometric normal flips too.
        let geo = |m: &FaceMesh| {
            let t = &m.indices[0..3];
            let p = |i: u32| {
                let v = m.vertices[i as usize].pos;
                v3(v[0] as f64, v[1] as f64, v[2] as f64)
            };
            (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])).normalize()
        };
        assert!(geo(&m).z < -0.99);
        face.same_sense = true;
        let m2 = run(&face, &surfaces, &curves, 0.01);
        assert!(geo(&m2).z > 0.99);
    }

    /// Occurrences of one component prepare once and emit per placement, and
    /// the result must be indistinguishable from tessellating every
    /// occurrence in full: same triangles at each placement, and -- the part
    /// the baseline depends on -- statistics counted per occurrence, so a
    /// repeated component's dropped face is still reported once per copy.
    #[test]
    fn occurrences_of_one_component_share_their_preparation() {
        const N: usize = 5;
        let surfaces = vec![Surface::Plane(unit_axis())];
        let corners = [
            v3(0.0, 0.0, 0.0),
            v3(2.0, 0.0, 0.0),
            v3(2.0, 2.0, 0.0),
            v3(0.0, 2.0, 0.0),
        ];
        let mut curves = Vec::new();
        let mut edges = Vec::new();
        for i in 0..4 {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            curves.push(Curve::Line { origin: a, dir: b - a });
            edges.push(EdgeUse { curve: i as u32, edge: i as u32, t0: 0.0, t1: 1.0 });
        }
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![Bound { vertex: None, outer: true, edges }],
        };
        let mut local = Aabb::EMPTY;
        for c in corners {
            local.add_point(c);
        }
        // The same source id on every occurrence: one component, five spots.
        let model = Model {
            solids: (0..N)
                .map(|i| Solid {
                    source: 7,
                    name: format!("occ{i}"),
                    color: [1.0; 3],
                    alpha: 1.0,
                    faces: vec![face.clone()],
                    wires: Vec::new(),
                    placement: Xform {
                        origin: v3(i as f64 * 10.0, 0.0, 0.0),
                        ..Xform::IDENTITY
                    },
                    local_bbox: local,
                    linked: true,
                    node: None,
                    volume: false,
                })
                .collect(),
            nodes: Vec::new(),
            surfaces,
            curves,
            vertex_bbox: Aabb::EMPTY,
            mm_per_unit: 1.0,
            stats: Default::default(),
        };

        let mesh = tessellate(&model, 0.01);
        assert_eq!(mesh.parts.len(), N);
        // Statistics are per occurrence, exactly as if nothing were shared.
        assert_eq!(mesh.stats.faces_ok, N);
        // Each occurrence's geometry is the first's, translated -- vertex for
        // vertex, in order.
        let p0 = &mesh.parts[0];
        for (i, p) in mesh.parts.iter().enumerate().skip(1) {
            assert_eq!(p.verts.len(), p0.verts.len(), "occ {i} vertex count");
            assert_eq!(p.tris.len(), p0.tris.len(), "occ {i} triangle count");
            let d = v3(i as f64 * 10.0, 0.0, 0.0);
            for k in 0..p.verts.len() {
                let a = mesh.vertices[p0.verts.start as usize + k].position();
                let b = mesh.vertices[p.verts.start as usize + k].position();
                assert!(
                    (b - a).distance(d) < 1e-6,
                    "occ {i} vertex {k}: {a:?} -> {b:?}, expected shift {d:?}"
                );
            }
        }
    }

    /// Each part must own a contiguous slice of the mesh, and the right one.
    ///
    /// The slices are read off the order the triangles came out in, which
    /// holds only because rayon collects an indexed iterator's contiguous runs
    /// back in order and the jobs are laid out solid by solid. If it stopped
    /// holding, two components would quietly share a range.
    ///
    /// Sixty-four solids, so that the work is actually split.
    #[test]
    fn parts_are_contiguous_slices_of_the_mesh() {
        const N: usize = 64;
        const STEP: f64 = 100.0;
        let surfaces = vec![Surface::Plane(unit_axis())];
        let corners = [
            v3(0.0, 0.0, 0.0),
            v3(2.0, 0.0, 0.0),
            v3(2.0, 2.0, 0.0),
            v3(0.0, 2.0, 0.0),
        ];
        let mut curves = Vec::new();
        let mut edges = Vec::new();
        for i in 0..4 {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            curves.push(Curve::Line { origin: a, dir: b - a });
            edges.push(EdgeUse { curve: i as u32, edge: i as u32, t0: 0.0, t1: 1.0 });
        }
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![Bound { vertex: None, outer: true, edges }],
        };
        let mut local = Aabb::EMPTY;
        for c in corners {
            local.add_point(c);
        }
        let model = Model {
            solids: (0..N)
                .map(|i| Solid {
                    source: u32::MAX,
                    name: format!("part{i}"),
                    color: [1.0; 3],
                    alpha: 1.0,
                    faces: vec![face.clone()],
                    wires: Vec::new(),
                    // Each one a long way from the next, so a triangle that
                    // landed in the wrong part is nowhere near its bounds.
                    placement: Xform {
                        origin: v3(i as f64 * STEP, 0.0, 0.0),
                        ..Xform::IDENTITY
                    },
                    local_bbox: local,
                    linked: true,
                    node: None,
                    volume: true,
                })
                .collect(),
            nodes: Vec::new(),
            surfaces,
            curves,
            vertex_bbox: Aabb::EMPTY,
            mm_per_unit: 1.0,
            stats: Default::default(),
        };

        let mesh = tessellate(&model, 0.01);
        assert_eq!(mesh.parts.len(), N);
        let (mut covered, mut covered_v) = (0usize, 0usize);
        for (i, p) in mesh.parts.iter().enumerate() {
            assert!(!p.tris.is_empty(), "part {i} got no triangles");
            covered += p.tris.len();
            covered_v += p.verts.len();
            for &idx in &mesh.indices[p.tris.start as usize..p.tris.end as usize] {
                let v = mesh.vertices[idx as usize].pos;
                let x = v[0] as f64;
                assert!(
                    x >= p.bbox.min.x - 1e-6 && x <= p.bbox.max.x + 1e-6,
                    "part {i} owns a vertex at x={x}, outside {:?}",
                    p.bbox
                );
                // The vertex slices have to line up with the triangle slices,
                // because a pick names its component by looking up one vertex
                // index and nothing checks the answer afterwards.
                assert_eq!(
                    mesh.part_of_vertex(idx),
                    Some(i),
                    "part {i}'s triangles reference vertex {idx}, which is not in {:?}",
                    p.verts
                );
            }
        }
        // And between them they account for the whole buffer, with nothing
        // counted twice.
        assert_eq!(covered, mesh.indices.len());
        assert_eq!(covered_v, mesh.vertices.len());
    }

    /// The packed normal has to survive the round trip to a hair.
    ///
    /// Eight bits a component is worth 0.4 degrees, which bands visibly along
    /// a specular highlight; sixteen-bit octahedral is 0.004, below anything
    /// the tone curve can show.
    ///
    /// Swept over the sphere, with the poles and the fold of the octahedron
    /// included by hand: the encoding folds the lower hemisphere into the
    /// corners of a square, so its worst case lands exactly on a fold.
    #[test]
    fn a_packed_normal_comes_back_within_a_hundredth_of_a_degree() {
        let mut worst: f64 = 0.0;
        let mut check = |n: Vec3| {
            let n = n.normalize();
            let got = oct_decode(oct_encode(n));
            assert!((got.length() - 1.0).abs() < 1e-9, "{got:?} is not a unit vector");
            worst = worst.max(n.dot(got).clamp(-1.0, 1.0).acos().to_degrees());
        };
        for i in 0..64 {
            for j in 0..128 {
                let (t, p) = (
                    std::f64::consts::PI * (i as f64 + 0.5) / 64.0,
                    TWO_PI * j as f64 / 128.0,
                );
                check(v3(t.sin() * p.cos(), t.sin() * p.sin(), t.cos()));
            }
        }
        for n in [
            v3(0.0, 0.0, 1.0),
            v3(0.0, 0.0, -1.0),
            v3(1.0, 0.0, 0.0),
            v3(0.0, -1.0, 0.0),
            // On the fold, where the two hemispheres meet.
            v3(1.0, 1.0, 0.0),
            v3(-1.0, 1.0, 0.0),
            v3(1.0, -1.0, -1e-12),
        ] {
            check(n);
        }
        assert!(worst < 0.01, "worst error {worst} degrees");
        // A degenerate normal must still come back as *a* direction rather
        // than as nothing: every consumer normalises, and NaN would spread.
        assert!(oct_decode(oct_encode(Vec3::ZERO)).is_finite());
    }

    /// Releasing the uploaded buffers must actually give the memory back, so
    /// capacity is asserted, not length. The vertex size is pinned here too;
    /// a field added without thinking would be invisible everywhere else.
    #[test]
    fn releasing_the_uploaded_buffers_frees_them() {
        let mut m = Mesh::default();
        m.vertices = vec![Vertex::new(Vec3::ZERO, v3(0.0, 0.0, 1.0), [0.5; 3], 1.0); 1000];
        m.indices = vec![0u32; 3000];
        assert_eq!(std::mem::size_of::<Vertex>(), 20);
        let before = m.bytes();
        let freed = m.release_uploaded();
        assert_eq!(freed, 1000 * 20 + 3000 * 4);
        assert_eq!(m.bytes(), before - freed);
        assert_eq!(m.vertices.capacity(), 0, "the allocation was kept");
        assert_eq!(m.indices.capacity(), 0, "the allocation was kept");
    }

    /// A model of `n` one-square solids, each a step further along x.
    fn squares(n: usize) -> Model {
        use crate::model::Solid;
        let corners = [
            v3(0.0, 0.0, 0.0),
            v3(2.0, 0.0, 0.0),
            v3(2.0, 2.0, 0.0),
            v3(0.0, 2.0, 0.0),
        ];
        let mut curves = Vec::new();
        let mut edges = Vec::new();
        for i in 0..4 {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            curves.push(Curve::Line { origin: a, dir: b - a });
            edges.push(EdgeUse { curve: i as u32, edge: i as u32, t0: 0.0, t1: 1.0 });
        }
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![Bound { vertex: None, outer: true, edges }],
        };
        let mut local = Aabb::EMPTY;
        for c in corners {
            local.add_point(c);
        }
        Model {
            solids: (0..n)
                .map(|i| Solid {
                    source: u32::MAX,
                    name: format!("part{i}"),
                    color: [1.0; 3],
                    alpha: 1.0,
                    faces: vec![face.clone()],
                    wires: Vec::new(),
                    placement: Xform {
                        origin: v3(i as f64 * 100.0, 0.0, 0.0),
                        ..Xform::IDENTITY
                    },
                    local_bbox: local,
                    linked: true,
                    node: None,
                    volume: true,
                })
                .collect(),
            nodes: Vec::new(),
            surfaces: vec![Surface::Plane(unit_axis())],
            curves,
            vertex_bbox: Aabb::EMPTY,
            mm_per_unit: 1.0,
            stats: Default::default(),
        }
    }

    /// Streaming has to give exactly the mesh that doing it all at once gives.
    /// The viewer takes the mesh in batches of whole solids, each tessellated
    /// as though it were the whole model and shifted into place on arrival.
    ///
    /// Enough solids to cross `BATCH_FACES`, so at least one seam is tested.
    #[test]
    fn streaming_and_whole_agree() {
        // Enough faces for the batch size to double several times over.
        let model = squares(FIRST_BATCH_FACES * 12);
        let tol = 0.01;
        let mut whole = Mesh::default();
        whole.absorb(tessellate_range(&model, tol, None, None, 0..model.solids.len()));
        let streamed = tessellate(&model, tol);

        assert_eq!(streamed.vertices.len(), whole.vertices.len());
        assert_eq!(streamed.indices, whole.indices);
        assert_eq!(streamed.parts.len(), whole.parts.len());
        assert_eq!(streamed.edges.len(), whole.edges.len());
        for (i, (a, b)) in streamed.parts.iter().zip(&whole.parts).enumerate() {
            assert_eq!(a.tris, b.tris, "part {i} triangles");
            assert_eq!(a.verts, b.verts, "part {i} vertices");
            // The edge buffer is grouped by class *within* a batch, so the two
            // lay it out differently on purpose. What has to match is the
            // geometry each part's runs point at.
            let at = |m: &Mesh, r: &std::ops::Range<u32>| {
                m.edges[r.start as usize..r.end as usize].to_vec()
            };
            assert_eq!(at(&streamed, &a.sharp), at(&whole, &b.sharp), "part {i} edges");
            assert_eq!(at(&streamed, &a.wire), at(&whole, &b.wire), "part {i} curves");
        }
        assert_eq!(streamed.stats.edges_sharp, whole.stats.edges_sharp);
        assert_eq!(streamed.stats.faces_ok, whole.stats.faces_ok);
    }

    /// Streaming makes every batch after the first side by side, and has to
    /// hand them over in order, each to the byte what its solids make on their
    /// own -- every buffer, the edge buffer included, and every part range.
    #[test]
    fn batches_arrive_in_order_as_if_made_in_turn() {
        let model = squares(FIRST_BATCH_FACES * 12);
        let tol = 0.01;
        let want: Vec<Mesh> = batches(&model)
            .into_iter()
            .map(|sel| tessellate_range(&model, tol, None, None, sel))
            .collect();
        assert!(want.len() > 3, "enough batches to be made side by side");
        let mut got = Vec::new();
        // A sink that waits on rayon itself, which is the one that could take
        // up another batch's work while it waits.
        tessellate_streaming(&model, tol, |b| {
            let n: u64 = (0..1000u64).into_par_iter().sum();
            assert_eq!(n, 499_500);
            got.push(b);
        });
        assert_eq!(got.len(), want.len());

        let bytes = |m: &Mesh| {
            let v: &[u8] = bytemuck::cast_slice(&m.vertices);
            let e: &[u8] = bytemuck::cast_slice(&m.edges);
            (v.to_vec(), m.indices.clone(), e.to_vec())
        };
        let ranges = |p: &Part| [&p.tris, &p.verts, &p.wire, &p.sharp, &p.tangent].map(|r| r.clone());
        for (k, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(bytes(a) == bytes(b), "batch {k}: the buffers differ");
            assert_eq!(a.parts.len(), b.parts.len(), "batch {k}");
            for (i, (pa, pb)) in a.parts.iter().zip(&b.parts).enumerate() {
                assert_eq!(pa.name, pb.name, "batch {k}, part {i}");
                assert_eq!(ranges(pa), ranges(pb), "batch {k}, part {i}");
            }
        }
    }

    /// A mirrored placement must not leave the mesh inside out.
    ///
    /// `same_sense` flips the normal and the vertex order together, which is
    /// enough only while the placement preserves handedness. A mirror reverses
    /// the winding while the normal, carried through the same basis, does not
    /// flip. Shading is two-sided and the mesh stays closed, so only the
    /// section cap, which tells entries from exits, notices.
    #[test]
    fn a_mirrored_placement_still_winds_outward() {
        let surfaces = vec![Surface::Plane(unit_axis())];
        let corners = [
            v3(0.0, 0.0, 0.0),
            v3(2.0, 0.0, 0.0),
            v3(2.0, 2.0, 0.0),
            v3(0.0, 2.0, 0.0),
        ];
        let mut curves = Vec::new();
        let mut edges = Vec::new();
        for i in 0..4 {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            curves.push(Curve::Line { origin: a, dir: b - a });
            edges.push(EdgeUse { curve: i as u32, edge: i as u32, t0: 0.0, t1: 1.0 });
        }
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![Bound { vertex: None, outer: true, edges }],
        };
        // A placement with a negative determinant: x negated, nothing else.
        let mirror = Xform {
            x: v3(-1.0, 0.0, 0.0),
            ..Xform::IDENTITY
        };
        let mut out = FaceMesh::default();
        tessellate_face(&face, &surfaces, &curves, [1.0; 4], 0.01, f64::INFINITY, &mirror, &mut out);
        assert!(!out.indices.is_empty(), "the face produced no triangles");

        let mesh = Mesh {
            vertices: out.vertices,
            indices: out.indices,
            ..Mesh::default()
        };
        let w = winding(&mesh);
        assert_eq!(w.inward, 0, "{} of {} triangles are inside out", w.inward, w.outward + w.inward);
        assert!(w.outward > 0, "nothing was decided either way");
    }

    /// A capped cylinder: a closed solid of three faces meeting along two
    /// circles, with its axis on z from the origin.
    fn closed_cylinder(r: f64, h: f64) -> Model {
        use crate::model::Solid;
        let circle = |z: f64| Curve::Circle {
            ax: Axis2::new(v3(0.0, 0.0, z), None, None),
            r,
        };
        let turn = |curve: u32, forward: bool| EdgeUse {
            curve,
            edge: curve,
            t0: if forward { 0.0 } else { TWO_PI },
            t1: if forward { TWO_PI } else { 0.0 },
        };
        let face = |surface: u32, same_sense: bool, bounds: Vec<Bound>| Face {
            surface,
            same_sense,
            style: None,
            domain: None,
            bounds,
        };
        let ring = |outer: bool, edge: EdgeUse| Bound { vertex: None, outer, edges: vec![edge] };
        let faces = vec![
            face(0, true, vec![ring(true, turn(0, true)), ring(false, turn(1, false))]),
            // The floor faces down and the lid up, each bounded by its circle
            // run the other way from the wall's use of it.
            face(1, false, vec![ring(true, turn(0, false))]),
            face(2, true, vec![ring(true, turn(1, true))]),
        ];
        let mut local = Aabb::EMPTY;
        local.add_point(v3(-r, -r, 0.0));
        local.add_point(v3(r, r, h));
        Model {
            solids: vec![Solid {
                source: 0,
                name: "can".into(),
                color: [1.0; 3],
                alpha: 1.0,
                faces,
                wires: Vec::new(),
                placement: Xform::IDENTITY,
                local_bbox: local,
                linked: true,
                node: None,
                volume: true,
            }],
            nodes: Vec::new(),
            surfaces: vec![
                Surface::Cylinder { ax: unit_axis(), r },
                Surface::Plane(unit_axis()),
                Surface::Plane(Axis2::new(v3(0.0, 0.0, h), None, None)),
            ],
            curves: vec![circle(0.0), circle(h)],
            vertex_bbox: local,
            mm_per_unit: 1.0,
            stats: Default::default(),
        }
    }

    /// A camera at `eye` looking along +y, with z up and a 60 degree view.
    fn looking_along_y(eye: Vec3, per_depth: f64, floor: f64) -> ViewTolerance {
        let t = 30f64.to_radians().tan();
        ViewTolerance {
            eye,
            right: v3(1.0, 0.0, 0.0),
            up: v3(0.0, 0.0, 1.0),
            forward: v3(0.0, 1.0, 0.0),
            tan_x: t,
            tan_y: t,
            per_depth,
            near: 0.1,
            floor,
        }
    }

    /// A view asks for nothing where it cannot see, and for a coarser
    /// tolerance the further away it looks -- in proportion, since a pixel's
    /// size is.
    #[test]
    fn a_view_asks_less_the_further_it_looks_and_nothing_it_cannot_see() {
        let v = looking_along_y(v3(0.0, 0.0, 0.0), 1e-3, 1e-6);
        let near = v.at(v3(0.0, 10.0, 0.0)).expect("straight ahead is in view");
        let far = v.at(v3(0.0, 40.0, 0.0)).expect("further ahead is in view");
        assert!((far / near - 4.0).abs() < 1e-9, "four times as far is {} times as coarse", far / near);
        assert!(v.at(v3(0.0, -10.0, 0.0)).is_none(), "behind the eye");
        assert!(v.at(v3(20.0, 10.0, 0.0)).is_none(), "off to the side");
        assert!(v.at(v3(0.0, 10.0, 20.0)).is_none(), "overhead");
        // Nearer than the near plane is the near plane, and nothing is finer
        // than the floor.
        assert_eq!(v.at(v3(0.0, 1e-4, 0.0)), Some(1e-4));
        let floored = looking_along_y(v3(0.0, 0.0, 0.0), 1e-3, 5e-3);
        assert_eq!(floored.at(v3(0.0, 1.0, 0.0)), Some(5e-3));

        // A box answers for the nearest it could come, and is out of view only
        // when all of it is behind one side of the frustum.
        let boxed = |lo: Vec3, hi: Vec3| {
            let mut b = Aabb::EMPTY;
            b.add_point(lo);
            b.add_point(hi);
            b
        };
        assert_eq!(
            v.finest_in(&boxed(v3(-1.0, 10.0, -1.0), v3(1.0, 30.0, 1.0))),
            Some(1e-2),
            "nearest corner at depth ten"
        );
        assert!(v.finest_in(&boxed(v3(-1.0, -30.0, -1.0), v3(1.0, -10.0, 1.0))).is_none());
        assert!(v.finest_in(&boxed(v3(50.0, 10.0, -1.0), v3(60.0, 20.0, 1.0))).is_none());
        // Straddling the edge of the view counts as in it.
        assert!(v.finest_in(&boxed(v3(5.0, 10.0, -1.0), v3(60.0, 20.0, 1.0))).is_some());
    }

    /// A solid tessellated for a view is finer where the view looks, as fine
    /// as the view asks there, and exactly as closed as it was. Faces along an
    /// edge sample it independently and agree only because the field depends
    /// on position, not on which face asks.
    #[test]
    fn a_solid_tessellated_for_a_view_is_finer_where_it_looks_and_still_closed() {
        let (r, h, tol) = (5.0, 20.0, 0.05);
        let model = closed_cylinder(r, h);
        let base = tessellate(&model, tol);
        assert_eq!(seams(&base, tol * 1e-3).unpaired, 0, "the can is not closed to start with");

        // Close to the wall and low down, looking at its lower half: half a
        // millimetre a thousand millimetres off, so ten times finer than the
        // model at the wall's nearest -- and nothing above z = 16 in view.
        let view = looking_along_y(v3(0.0, -15.0, 4.0), 5e-4, tol / 1024.0);
        let want = parts_to_refine(&model, &base.parts, &[], &view, tol);
        assert_eq!(want, vec![0], "the view is close enough to want the can finer");
        let fine = tessellate_for_view(&model, tol, &view, &base.parts, &want);

        // What is drawn is the faces drawn again, and the first tessellation of
        // every face that was not -- and that, together, is what has to be
        // closed.
        let redone: HashSet<u32> = fine.parts[0].faces.iter().map(|f| f.face).collect();
        let mut drawn = Mesh {
            vertices: base.vertices.clone(),
            ..Default::default()
        };
        for f in base.parts[0].faces.iter().filter(|f| !redone.contains(&f.face)) {
            drawn.indices.extend_from_slice(&base.indices[f.tris.start as usize..f.tris.end as usize]);
        }
        let at = drawn.vertices.len() as u32;
        drawn.vertices.extend_from_slice(&fine.vertices);
        drawn.indices.extend(fine.indices.iter().map(|i| i + at));
        let s = seams(&drawn, tol * 1e-3);
        assert_eq!(s.unpaired, 0, "refining opened cracks at {:?}", s.cracks);
        assert_eq!(s.overshared, 0, "refining laid faces over each other");
        assert!(
            fine.indices.len() > base.indices.len() * 2,
            "{} triangles against {}",
            fine.indices.len() / 3,
            base.indices.len() / 3
        );

        // Every chord of the wall in view is within what the view asked for
        // there, measured the way the refinement measures it: at the chord's
        // own midpoint. The wall's triangles are the ones standing upright; a
        // chord of the lid can join two points of the rim and lie on the lid
        // exactly, and measured from the axis it would look like an error.
        let mut checked = 0;
        for t in fine.indices.chunks_exact(3) {
            let p = |k: usize| fine.vertices[t[k] as usize].position();
            if (p(1) - p(0)).cross(p(2) - p(0)).normalize().z.abs() > 0.5 {
                continue;
            }
            for k in 0..3 {
                let (a, b) = (p(k), p((k + 1) % 3));
                let m = (a + b) * 0.5;
                let Some(allowed) = view.at(m) else { continue };
                let err = r - m.x.hypot(m.y);
                assert!(
                    err <= allowed.min(tol) * 1.001,
                    "a chord at {m:?} ({a:?} to {b:?}) strays {err} where the view allows {allowed}"
                );
                checked += 1;
            }
        }
        assert!(checked > 100, "only {checked} chords of the wall were in view");

        // And the rim at the top, out of view, keeps the model's tolerance: the
        // very points it had. Out of the frustum, that is -- not merely out of
        // sight: the back of the can is behind its front but in the frustum,
        // and is refined, because a section plane can take the front away.
        assert!(view.at(v3(0.0, -r, h - 1.0)).is_none(), "the top of the can is in view");
        // Welded at a micron: the seam point is the circle's start to one face
        // and its end to the other, zero and a sine of two pi apart.
        let rim = |m: &Mesh| {
            let mut pts: Vec<[i64; 3]> = m
                .vertices
                .iter()
                .filter(|v| v.pos[2] == h as f32)
                .map(|v| v.pos.map(|c| (c as f64 * 1e6).round() as i64))
                .collect();
            pts.sort_unstable();
            pts.dedup();
            pts
        };
        assert!(!rim(&base).is_empty());
        assert_eq!(rim(&fine), rim(&base), "the rim out of view was sampled again differently");

        // The lid is wholly out of view, so it is not drawn again at all: the
        // first tessellation stands for it.
        let redone: Vec<u32> = fine.parts[0].faces.iter().map(|f| f.face).collect();
        assert!(redone.contains(&0), "the wall was not drawn again");
        assert!(!redone.contains(&2), "the lid, out of view, was drawn again");
    }

    /// A face the view cannot reach comes out of the view's field exactly as it
    /// did the first time, to the last bit, so the faces a view leaves alone
    /// share edges with the ones it redraws. Only a segment the view wants
    /// finer than the model's tolerance is given extra depth.
    #[test]
    fn a_face_out_of_view_is_tessellated_as_it_was() {
        let (r, h, tol) = (5.0, 20.0, 0.05);
        let model = closed_cylinder(r, h);
        let view = looking_along_y(v3(0.0, -15.0, 4.0), 5e-4, tol / 1024.0);
        assert!(view.at(v3(0.0, 0.0, h)).is_none(), "the lid is in view");
        let lid = |si: usize, k: usize| si == 0 && k == 2;
        let first = tessellate_range(&model, tol, None, Some(&lid), 0..1);
        let again = tessellate_range(&model, tol, Some(&view), Some(&lid), 0..1);
        let pos = |m: &Mesh| m.vertices.iter().map(|v| v.pos).collect::<Vec<_>>();
        assert!(!first.indices.is_empty(), "the lid drew nothing");
        assert_eq!(pos(&first), pos(&again), "the lid moved under a view that cannot see it");
        assert_eq!(first.indices, again.indices, "the lid was triangulated differently");
    }

    /// The overlay's lines for a solid tessellated for a view are sampled from
    /// the same field as its faces, so every point of every line is a vertex
    /// of the faces it runs between; otherwise the line breaks into dashes.
    #[test]
    fn lines_refined_for_a_view_lie_on_the_faces_own_boundary() {
        let (r, h, tol) = (5.0, 20.0, 0.05);
        let model = closed_cylinder(r, h);
        let view = looking_along_y(v3(0.0, -15.0, 10.0), 5e-4, tol / 1024.0);
        let base = tessellate(&model, tol);
        let fine = tessellate_for_view(&model, tol, &view, &base.parts, &[0]);
        let key = |p: [f32; 3]| (p[0].to_bits(), p[1].to_bits(), p[2].to_bits());
        let vertices: HashSet<_> = fine.vertices.iter().map(|v| key(v.pos)).collect();
        let p = &fine.parts[0];
        let lines = &fine.edges[p.sharp.start as usize..p.sharp.end as usize];
        assert!(lines.len() > 100, "only {} line points", lines.len());
        let off = lines.iter().filter(|q| !vertices.contains(&key(**q))).count();
        assert_eq!(off, 0, "{off} of {} line points are no vertex of any face", lines.len());
    }


    /// A torus of R = 20, r = 5 about z, written the way a kernel writes one:
    /// one vertex where the outer equator meets the meridian at u = 0, and
    /// every edge a whole circle through it. `sweep` of `None` is the whole
    /// ring -- one face bounded by both of its seams, equator out and back and
    /// meridian up and down -- and `Some(a)` is an elbow through `a` radians,
    /// its two open ends capped by planes.
    fn torus_step(sweep: Option<f64>) -> String {
        let mut s = String::from(
            "DATA;\n\
             #1=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));\n\
             #2=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.));\n\
             #3=(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNIT_ASSIGNED_CONTEXT((#1,#2))\
             REPRESENTATION_CONTEXT('',''));\n\
             #10=CARTESIAN_POINT('',(0.,0.,0.));\n\
             #11=DIRECTION('',(0.,0.,1.));\n\
             #12=DIRECTION('',(1.,0.,0.));\n\
             #13=AXIS2_PLACEMENT_3D('',#10,#11,#12);\n\
             #14=TOROIDAL_SURFACE('',#13,20.,5.);\n\
             #20=CARTESIAN_POINT('',(25.,0.,0.));\n\
             #21=VERTEX_POINT('',#20);\n\
             #22=CIRCLE('',#13,25.);\n\
             #30=CARTESIAN_POINT('',(20.,0.,0.));\n\
             #31=DIRECTION('',(0.,-1.,0.));\n\
             #32=AXIS2_PLACEMENT_3D('',#30,#31,#12);\n\
             #33=CIRCLE('',#32,5.);\n\
             #34=EDGE_CURVE('',#21,#21,#33,.T.);\n",
        );
        match sweep {
            None => {
                s += "#40=EDGE_CURVE('',#21,#21,#22,.T.);\n\
                      #50=ORIENTED_EDGE('',*,*,#40,.T.);\n\
                      #51=ORIENTED_EDGE('',*,*,#34,.T.);\n\
                      #52=ORIENTED_EDGE('',*,*,#40,.F.);\n\
                      #53=ORIENTED_EDGE('',*,*,#34,.F.);\n\
                      #54=EDGE_LOOP('',(#50,#51,#52,#53));\n\
                      #55=FACE_OUTER_BOUND('',#54,.T.);\n\
                      #56=ADVANCED_FACE('',(#55),#14,.T.);\n\
                      #90=CLOSED_SHELL('',(#56));\n";
            }
            Some(a) => {
                let (c, n) = (a.cos(), a.sin());
                // The far end: its vertex, the meridian through it -- axis
                // against the sweep, reference outward, so its parameter is
                // the torus's own v -- and the plane that caps it.
                s += &format!(
                    "#60=CARTESIAN_POINT('',({:.17},{:.17},0.));\n\
                     #61=VERTEX_POINT('',#60);\n\
                     #40=EDGE_CURVE('',#21,#61,#22,.T.);\n\
                     #62=CARTESIAN_POINT('',({:.17},{:.17},0.));\n\
                     #63=DIRECTION('',({n:.17},{:.17},0.));\n\
                     #64=DIRECTION('',({c:.17},{n:.17},0.));\n\
                     #65=AXIS2_PLACEMENT_3D('',#62,#63,#64);\n\
                     #66=CIRCLE('',#65,5.);\n\
                     #67=EDGE_CURVE('',#61,#61,#66,.T.);\n\
                     #68=AXIS2_PLACEMENT_3D('',#60,#63,#64);\n\
                     #69=PLANE('',#68);\n\
                     #70=ORIENTED_EDGE('',*,*,#67,.F.);\n\
                     #71=EDGE_LOOP('',(#70));\n\
                     #72=FACE_OUTER_BOUND('',#71,.T.);\n\
                     #73=ADVANCED_FACE('',(#72),#69,.F.);\n",
                    25.0 * c,
                    25.0 * n,
                    20.0 * c,
                    20.0 * n,
                    -c
                );
                // The near end, at u = 0, faces back along -y.
                s += "#74=AXIS2_PLACEMENT_3D('',#20,#31,#12);\n\
                      #75=PLANE('',#74);\n\
                      #76=ORIENTED_EDGE('',*,*,#34,.T.);\n\
                      #77=EDGE_LOOP('',(#76));\n\
                      #78=FACE_OUTER_BOUND('',#77,.T.);\n\
                      #79=ADVANCED_FACE('',(#78),#75,.T.);\n\
                      #50=ORIENTED_EDGE('',*,*,#40,.T.);\n\
                      #51=ORIENTED_EDGE('',*,*,#67,.T.);\n\
                      #52=ORIENTED_EDGE('',*,*,#40,.F.);\n\
                      #53=ORIENTED_EDGE('',*,*,#34,.F.);\n\
                      #54=EDGE_LOOP('',(#50,#51,#52,#53));\n\
                      #55=FACE_OUTER_BOUND('',#54,.T.);\n\
                      #56=ADVANCED_FACE('',(#55),#14,.T.);\n\
                      #90=CLOSED_SHELL('',(#56,#73,#79));\n";
            }
        }
        s += "#91=MANIFOLD_SOLID_BREP('torus',#90);\n\
              #92=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#91,#13),#3);\n\
              ENDSEC;\n";
        s
    }

    /// Total triangle area of a whole model's mesh.
    fn mesh_area(m: &Mesh) -> f64 {
        m.indices
            .chunks_exact(3)
            .map(|t| {
                let p = |i: u32| m.vertices[i as usize].position();
                (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])).length() * 0.5
            })
            .sum()
    }

    /// The whole ring as a kernel writes it: bounded by both of its seams, so
    /// its outline is the whole parameter square. Its one vertex must not be
    /// taken for the model's extent, and an edge across the square, which
    /// starts and ends at one point in space, must not be taken for a
    /// degenerate one and left unsplit.
    #[test]
    fn a_torus_bounded_by_both_its_seams_is_one_closed_ring() {
        let model = crate::model::build(&crate::step::parse(torus_step(None).as_bytes()).unwrap());
        let size = model.vertex_bbox.size();
        assert!(size.distance(v3(50.0, 50.0, 10.0)) < 1e-9, "the ring measures {size:?}");
        let tol = model.vertex_bbox.extent() * 3e-4;
        let mesh = tessellate(&model, tol);
        assert_eq!(mesh.stats.faces_failed(), 0);
        let s = seams(&mesh, tol * 1e-3);
        assert_eq!((s.unpaired, s.overshared), (0, 0), "cracks at {:?}, folds at {:?}", s.cracks, s.folds);
        assert_eq!(winding(&mesh).inward, 0, "the ring is wound inside out in places");
        // 4 pi^2 R r.
        let exact = 4.0 * std::f64::consts::PI * std::f64::consts::PI * 20.0 * 5.0;
        let got = mesh_area(&mesh);
        assert!((got - exact).abs() < exact * 0.005, "area {got} against {exact}");
    }

    /// The same ring bent through four radians and capped with two planes: a
    /// pipe elbow. Its outline is a rectangle whose top and bottom are one
    /// seam, so the same unsplit edges across it would fill the tube.
    #[test]
    fn a_torus_elbow_is_a_closed_bent_tube() {
        let a = 4.0f64;
        let model = crate::model::build(&crate::step::parse(torus_step(Some(a)).as_bytes()).unwrap());
        // The sweep runs through 90 and 180 degrees, where no vertex is.
        let (lo, hi) = (model.vertex_bbox.min, model.vertex_bbox.max);
        assert!(
            lo.distance(v3(-25.0, 25.0 * a.sin(), -5.0)) < 1e-9 && hi.distance(v3(25.0, 25.0, 5.0)) < 1e-9,
            "the elbow measures {lo:?} .. {hi:?}"
        );
        let tol = model.vertex_bbox.extent() * 3e-4;
        let mesh = tessellate(&model, tol);
        assert_eq!(mesh.stats.faces_failed(), 0);
        let s = seams(&mesh, tol * 1e-3);
        assert_eq!((s.unpaired, s.overshared), (0, 0), "cracks at {:?}, folds at {:?}", s.cracks, s.folds);
        assert_eq!(winding(&mesh).inward, 0, "the elbow is wound inside out in places");
        // R r for each radian of sweep and each of the tube's, and two discs.
        let exact = a * TWO_PI * 20.0 * 5.0 + 2.0 * std::f64::consts::PI * 25.0;
        let got = mesh_area(&mesh);
        assert!((got - exact).abs() < exact * 0.005, "area {got} against {exact}");
    }

    /// A cylinder bounded by its seam line up and back down, a common way to
    /// write a pipe. The outline is then the whole rectangle, and a point
    /// partway up the seam stands on both of its sides: joined straight across,
    /// that is an edge of no length which goes the whole way round.
    #[test]
    fn a_cylinder_bounded_by_its_seam_line_is_covered_once() {
        let (r, h) = (4.65, 72.61);
        let circle = |z: f64| Curve::Circle {
            ax: Axis2::new(v3(0.0, 0.0, z), None, None),
            r,
        };
        let curves = vec![
            circle(0.0),
            circle(h),
            Curve::Line {
                origin: v3(r, 0.0, 0.0),
                dir: v3(0.0, 0.0, 1.0),
            },
        ];
        let edge = |curve: u32, t0: f64, t1: f64| EdgeUse { curve, edge: curve, t0, t1 };
        let face = Face {
            surface: 0,
            same_sense: true,
            style: None,
            domain: None,
            bounds: vec![Bound {
                vertex: None,
                outer: true,
                edges: vec![edge(0, 0.0, TWO_PI), edge(2, 0.0, h), edge(1, TWO_PI, 0.0), edge(2, h, 0.0)],
            }],
        };
        let surfaces = vec![Surface::Cylinder { ax: unit_axis(), r }];
        let tol = 0.5;
        let mut m = FaceMesh::default();
        // Long enough chords that the seam line takes one point partway up.
        tessellate_face(&face, &surfaces, &curves, [1.0; 4], tol, 40.0, &Xform::IDENTITY, &mut m);
        assert_eq!(m.stats.faces_ok, 1);
        // Every vertex is on the cylinder and every chord cuts inside it, so
        // the mesh can come short of the true area but never past it.
        let exact = TWO_PI * r * h;
        let got = area(&m);
        assert!(got <= exact * (1.0 + 1e-9) && got > exact * 0.95, "area {got} against {exact}");
    }

    /// `--check`'s counts belong to the mesh, not to where its weld grid falls.
    /// Faces meet at copies of one vertex rounded to f32 independently, so
    /// they may differ by an ulp; a grid would split such a pair whenever a
    /// cell boundary ran between them. So: a closed elbow with every vertex
    /// nudged by up to an ulp in each coordinate, moved bodily by sixteen
    /// fractions of the weld. It has to be closed every time, and to count
    /// the same every time.
    #[test]
    fn the_weld_does_not_move_with_its_grid() {
        let model = crate::model::build(&crate::step::parse(torus_step(Some(4.0)).as_bytes()).unwrap());
        let tol = model.vertex_bbox.extent() * 3e-4;
        let mut mesh = tessellate(&model, tol);
        let weld = tol * 1e-3;
        // An ulp here is about an eighth of the weld, so the nudged copies all
        // lie well within reach of one another.
        let ulp = mesh.bbox.max.x.abs().max(mesh.bbox.min.x.abs()) * f32::EPSILON as f64;
        assert!(ulp < weld / 4.0, "the test wants an ulp well under the weld: {ulp} against {weld}");
        for (i, v) in mesh.vertices.iter_mut().enumerate() {
            for (k, c) in v.pos.iter_mut().enumerate() {
                // A step of one in the bits is an ulp either way for anything
                // but zero, where stepping down would make a NaN.
                let h = ((i * 3 + k) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 62;
                if *c != 0.0 {
                    *c = match h {
                        0 => f32::from_bits(c.to_bits() + 1),
                        1 => f32::from_bits(c.to_bits() - 1),
                        _ => *c,
                    };
                }
            }
        }
        let counts = |m: &Mesh| {
            let s = seams(m, weld);
            (s.unpaired, s.overshared, closed_parts(m, weld))
        };
        let want = counts(&mesh);
        assert_eq!((want.0, want.1), (0, 0), "the nudged elbow is not closed");
        assert!(want.2.iter().all(|&c| c), "and cannot be capped");
        for k in 1..=16 {
            let f = |p: u32| (k as f64 * (p as f64).sqrt()).fract() * weld;
            let shift = [f(2), f(3), f(5)];
            let mut moved = Mesh {
                vertices: mesh.vertices.clone(),
                indices: mesh.indices.clone(),
                parts: mesh.parts.clone(),
                ..Default::default()
            };
            for v in &mut moved.vertices {
                for (c, s) in v.pos.iter_mut().zip(shift) {
                    *c = (*c as f64 + s) as f32;
                }
            }
            assert_eq!(counts(&moved), want, "moved by {shift:?}");
        }
    }

    /// The overlay's grouping belongs to the model, not to where its weld grid
    /// falls. When every face writes its edges as its own entities, the two
    /// sides of one edge agree only to a hair, and splitting them makes a
    /// tangent edge look sharp or open. So: a cube whose top is two coplanar
    /// halves, every face with its own edges and its own copy of each corner
    /// nudged by up to a tenth of the weld, moved bodily by sixteen fractions
    /// of the weld. Fourteen sharp edges and one tangent one, every time.
    #[test]
    fn the_overlay_does_not_move_with_its_grid() {
        use crate::model::Solid;
        let tol = 0.01;
        let weld = tol * 1e-3;
        let corner = |i: usize| {
            let p = [
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
                (0.5, 0.0, 1.0), (0.5, 1.0, 1.0),
            ][i];
            v3(p.0, p.1, p.2)
        };
        // Each face as its corners in order, and its outward normal. The top
        // is two halves meeting along x = 0.5.
        let faces: [(&[usize], Vec3); 7] = [
            (&[0, 3, 2, 1], v3(0.0, 0.0, -1.0)),
            (&[4, 8, 9, 7], v3(0.0, 0.0, 1.0)),
            (&[8, 5, 6, 9], v3(0.0, 0.0, 1.0)),
            (&[0, 1, 5, 8, 4], v3(0.0, -1.0, 0.0)),
            (&[3, 7, 9, 6, 2], v3(0.0, 1.0, 0.0)),
            (&[0, 4, 7, 3], v3(-1.0, 0.0, 0.0)),
            (&[1, 2, 6, 5], v3(1.0, 0.0, 0.0)),
        ];
        let mut surfaces = Vec::new();
        let mut curves = Vec::new();
        let mut model_faces = Vec::new();
        for (f, (ring, normal)) in faces.iter().enumerate() {
            // This face's own copy of a corner, a hair from every other face's.
            let nudge = |i: usize| {
                let h = ((f * 16 + i) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let c = |s: u32| (((h >> s) & 0xffff) as f64 / 65535.0 - 0.5) * 0.2 * weld;
                corner(i) + v3(c(0), c(21), c(42))
            };
            surfaces.push(Surface::Plane(Axis2::new(corner(ring[0]), Some(*normal), None)));
            let mut edges = Vec::new();
            for k in 0..ring.len() {
                let (a, b) = (nudge(ring[k]), nudge(ring[(k + 1) % ring.len()]));
                edges.push(EdgeUse {
                    curve: curves.len() as u32,
                    edge: curves.len() as u32,
                    t0: 0.0,
                    t1: 1.0,
                });
                curves.push(Curve::Line { origin: a, dir: b - a });
            }
            model_faces.push(Face {
                surface: f as u32,
                same_sense: true,
                style: None,
                domain: None,
                bounds: vec![Bound { vertex: None, outer: true, edges }],
            });
        }
        let mut cube = Aabb::EMPTY;
        cube.add_point(corner(0));
        cube.add_point(corner(6));
        let mut model = Model {
            solids: vec![Solid {
                source: 0,
                name: String::new(),
                color: [1.0; 3],
                alpha: 1.0,
                faces: model_faces,
                wires: Vec::new(),
                placement: Xform::IDENTITY,
                local_bbox: cube,
                linked: true,
                node: None,
                volume: true,
            }],
            nodes: Vec::new(),
            surfaces,
            curves,
            vertex_bbox: cube,
            mm_per_unit: 1.0,
            stats: Default::default(),
        };
        for k in 0..16 {
            let f = |p: u32| (k as f64 * (p as f64).sqrt()).fract() * weld;
            let shift = v3(f(2), f(3), f(5));
            model.solids[0].placement = Xform::translate(shift);
            let e = extract_edges(&model, tol, f64::INFINITY, 0..1);
            let counts = (e.stats.edges_sharp, e.stats.edges_tangent, e.stats.edges_single_use);
            assert_eq!(counts, (14, 1, 0), "moved by {shift:?}: sharp, tangent, used by one face");
        }
    }
