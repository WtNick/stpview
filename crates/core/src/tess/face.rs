//! Per-face tessellation: constrained triangulation of the trimmed domain,
//! prepared once per unique component and emitted once per occurrence.
use super::*;

/// The region between two rings is a ribbon whether or not their chords cross,
/// so it is zipped rather than triangulated: walk both in step, always advancing
/// whichever is behind in angle about the common centre. Every boundary segment
/// survives and no point moves. Both rings must run once round that centre
/// without doubling back (a washer, not a kidney-shaped slot), hence the
/// `None`. The caller keeps whichever candidate lands nearer the enclosed area.
pub(crate) fn ribbon(pts: &[(f64, f64)], a: &[usize], b: &[usize]) -> Option<Vec<[usize; 3]>> {
    // Closed contours repeat their first index; the cycle is everything before.
    let (a, b) = (&a[..a.len() - 1], &b[..b.len() - 1]);
    if a.len() < 3 || b.len() < 3 {
        return None;
    }
    let n = (a.len() + b.len()) as f64;
    let cx = a.iter().chain(b).map(|&i| pts[i].0).sum::<f64>() / n;
    let cy = a.iter().chain(b).map(|&i| pts[i].1).sum::<f64>() / n;

    // Each ring as a cycle running anticlockwise about that centre, with the
    // angle it reaches at each step. A ring that does not make exactly one turn
    // is not ordered by angle and cannot be zipped this way.
    let walk = |ring: &[usize]| -> Option<(Vec<usize>, Vec<f64>)> {
        let raw: Vec<f64> = ring
            .iter()
            .map(|&i| (pts[i].1 - cy).atan2(pts[i].0 - cx))
            .collect();
        let mut acc = Vec::with_capacity(raw.len() + 1);
        acc.push(raw[0]);
        for k in 1..raw.len() {
            acc.push(acc[k - 1] + wrap_delta(raw[k] - raw[k - 1], TWO_PI));
        }
        let turn = acc[raw.len() - 1] - acc[0] + wrap_delta(raw[0] - raw[raw.len() - 1], TWO_PI);
        if !(TWO_PI * 0.99..TWO_PI * 1.01).contains(&turn.abs()) {
            return None;
        }
        if turn > 0.0 {
            Some((ring.to_vec(), acc))
        } else {
            // Walking the ring backwards makes its angles ascend, so the
            // accumulated values simply run the other way; negating them as
            // well would undo that.
            let mut r: Vec<usize> = ring.to_vec();
            r.reverse();
            Some((r, acc.iter().rev().copied().collect()))
        }
    };
    let (ra, wa) = walk(a)?;
    let (rb, wb) = walk(b)?;

    // Start both at the same angle, so step k of one faces step k of the other.
    let begin = |w: &[f64]| {
        (0..w.len())
            .min_by(|&x, &y| {
                wrap_delta(w[x], TWO_PI)
                    .rem_euclid(TWO_PI)
                    .total_cmp(&wrap_delta(w[y], TWO_PI).rem_euclid(TWO_PI))
            })
            .unwrap()
    };
    let (sa, sb) = (begin(&wa), begin(&wb));
    // Angle of step k, measured from each ring's own start so the two are
    // directly comparable.
    let ahead = |w: &[f64], s: usize, k: usize| {
        let i = (s + k) % w.len();
        let mut d = w[i] - w[s];
        if k > 0 && d <= 0.0 {
            d += TWO_PI;
        }
        d
    };

    let (mut i, mut j) = (0usize, 0usize);
    let mut out: Vec<[usize; 3]> = Vec::with_capacity(ra.len() + rb.len());
    while i < ra.len() || j < rb.len() {
        let step_a = if i >= ra.len() {
            false
        } else if j >= rb.len() {
            true
        } else {
            ahead(&wa, sa, i + 1) <= ahead(&wb, sb, j + 1)
        };
        let (ca, cb) = (ra[(sa + i) % ra.len()], rb[(sb + j) % rb.len()]);
        if step_a {
            out.push([ca, ra[(sa + i + 1) % ra.len()], cb]);
            i += 1;
        } else {
            out.push([ca, rb[(sb + j + 1) % rb.len()], cb]);
            j += 1;
        }
    }

    // Emit anticlockwise, like the constrained triangulator does, since the
    // caller reads winding as the surface normal's direction.
    let signed: f64 = out
        .iter()
        .map(|&[x, y, z]| {
            let (p, q, r) = (pts[x], pts[y], pts[z]);
            (q.0 - p.0) * (r.1 - p.1) - (q.1 - p.1) * (r.0 - p.0)
        })
        .sum();
    if signed < 0.0 {
        for t in &mut out {
            t.swap(1, 2);
        }
    }
    Some(out)
}

/// How wide a face is at its narrowest: twice the enclosed area over the
/// perimeter. A disc of radius R gives R, a thin a x b rectangle gives b, and
/// an annulus of width w gives w whatever its radius -- a thin ring is wide in
/// every extent-based measure.
pub(crate) fn outline_width(area: f64, perimeter: f64) -> Option<f64> {
    (area > 0.0 && perimeter > 0.0).then(|| 2.0 * area / perimeter)
}

/// Says *how* a set of contours is malformed, for faces the constrained
/// triangulator refuses: self-crossing (a folded projection), crossing another
/// ring (a hole in the wrong period), or touching at a point (legitimate CAD).
pub(crate) fn diagnose_contours(pts: &[(f64, f64)], contours: &[Vec<usize>]) -> String {
    let ring_of = |i: usize| {
        contours
            .iter()
            .position(|c| c[..c.len() - 1].contains(&i))
            .unwrap_or(usize::MAX)
    };
    // Every constraint segment, with the ring it came from.
    let segs: Vec<(usize, usize, usize)> = contours
        .iter()
        .enumerate()
        .flat_map(|(r, c)| c.windows(2).map(move |w| (w[0], w[1], r)))
        .collect();
    let side = |a: (f64, f64), b: (f64, f64), p: (f64, f64)| {
        (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
    };
    let span = {
        let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
        for p in pts {
            lo = (lo.0.min(p.0), lo.1.min(p.1));
            hi = (hi.0.max(p.0), hi.1.max(p.1));
        }
        (hi.0 - lo.0).max(hi.1 - lo.1).max(1e-30)
    };
    // Relative to the outline's own size, so this reads the same on a 1 mm bore
    // and a 1 m panel.
    let eps = span * 1e-9;

    let (mut cross_same, mut cross_other, mut touch_same, mut touch_other) = (0, 0, 0, 0);
    let mut worst = String::new();
    for (i, &(a0, a1, ra)) in segs.iter().enumerate() {
        for &(b0, b1, rb) in &segs[i + 1..] {
            if a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1 {
                continue; // shares an endpoint: adjacent, not a crossing
            }
            let (p, q, r, s) = (pts[a0], pts[a1], pts[b0], pts[b1]);
            // `side` is twice a triangle area; dividing by the base turns it
            // into the distance from the line, which is what `eps` measures.
            let la = ((q.0 - p.0).hypot(q.1 - p.1)).max(1e-300);
            let lb = ((s.0 - r.0).hypot(s.1 - r.1)).max(1e-300);
            let (d1, d2) = (side(p, q, r) / la, side(p, q, s) / la);
            let (d3, d4) = (side(r, s, p) / lb, side(r, s, q) / lb);
            // A proper crossing puts each segment's ends on opposite sides of
            // the other; within eps is a touch. Rings meeting at a point is
            // ordinary CAD, rings overlapping is a bug upstream.
            let touching = d1.abs() <= eps || d2.abs() <= eps || d3.abs() <= eps || d4.abs() <= eps;
            let crossing = (d1 > 0.0) != (d2 > 0.0) && (d3 > 0.0) != (d4 > 0.0);
            if !touching && !crossing {
                continue;
            }
            match (touching, ra == rb) {
                (true, true) => touch_same += 1,
                (true, false) => touch_other += 1,
                (false, true) => cross_same += 1,
                (false, false) => cross_other += 1,
            }
            if worst.is_empty() && crossing {
                worst = format!(
                    " first at ({:.5}, {:.5})-({:.5}, {:.5}) vs ({:.5}, {:.5})-({:.5}, {:.5})",
                    p.0, p.1, q.0, q.1, r.0, r.1, s.0, s.1
                );
            }
        }
    }
    let _ = ring_of;
    // Small outlines get printed in full: at this size the shape is the
    // answer.
    if pts.len() <= 48 && std::env::var_os("STPVIEW_CDT").is_some_and(|v| v == "2") {
        let rings: Vec<String> = contours
            .iter()
            .map(|c| {
                let r: Vec<String> = c
                    .iter()
                    .map(|&i| format!("({:.6}, {:.6})", pts[i].0, pts[i].1))
                    .collect();
                r.join(" ")
            })
            .collect();
        return format!("{} pts\n    {}", pts.len(), rings.join("\n    "));
    }
    format!(
        "{} rings, {} pts, crossings {} self {} between, touches {} self {} between,{}",
        contours.len(),
        pts.len(),
        cross_same,
        cross_other,
        touch_same,
        touch_other,
        if worst.is_empty() { " -".into() } else { worst }
    )
}


/// Dumps every face whose sampled outline passes near a given point.
///
/// `STPVIEW_PROBE="x,y,z"` (optionally `x,y,z,radius`) reports, for each such
/// face, the surface it lies on and the edges of every bound with their curve
/// and parameter range, then the boundary points themselves. Two faces sharing
/// an edge should print the same points for it; where they do not is where the
/// mesh comes apart.
pub(crate) fn probe(surface: &Surface, face: &Face, curves: &[Curve], loops: &[LoopUv], place: &Xform) {
    let Some(spec) = std::env::var_os("STPVIEW_PROBE") else {
        return;
    };
    let n: Vec<f64> = spec
        .to_string_lossy()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if n.len() < 3 {
        return;
    }
    let target = crate::math::v3(n[0], n[1], n[2]);
    let radius = n.get(3).copied().unwrap_or(0.5);
    let near = |p: Vec3| place.point(p).distance(target) <= radius;
    if !loops.iter().any(|l| l.pts.iter().any(|p| near(*p))) {
        return;
    }

    let mut s = format!("PROBE {} face:", Surface::KINDS[surface.kind_index()]);
    for (bi, b) in face.bounds.iter().enumerate() {
        s += &format!("\n  bound {bi} outer={} vertex={}", b.outer, b.vertex.is_some());
        for e in &b.edges {
            let c = &curves[e.curve as usize];
            s += &format!(
                "\n    edge #{} curve[{}] {} t[{:.9},{:.9}]",
                e.edge,
                e.curve,
                match c {
                    Curve::Line { .. } => "line",
                    Curve::Circle { .. } => "circle",
                    Curve::Ellipse { .. } => "ellipse",
                    Curve::BSpline(_) => "bspline",
                },
                e.t0,
                e.t1
            );
        }
    }
    for (li, l) in loops.iter().enumerate() {
        let hits: Vec<String> = l
            .pts
            .iter()
            .enumerate()
            .filter(|(_, p)| near(**p))
            .map(|(k, p)| format!("[{k}] {:.6?}", place.point(*p)))
            .collect();
        if !hits.is_empty() {
            s += &format!("\n  loop {li} ({} pts) near probe:", l.pts.len());
            for h in &hits {
                s += &format!("\n    {h}");
            }
        }
    }
    eprintln!("{s}");
}

/// The same probe, run on the finished triangles: which mesh edges this face
/// actually contributes near the point. Two faces sharing a boundary must
/// contribute the same edges along it.
pub(crate) fn probe_mesh(surface: &Surface, patch: &Patch, place: &Xform, locked: usize) {
    let Some(spec) = std::env::var_os("STPVIEW_PROBE") else {
        return;
    };
    let n: Vec<f64> = spec
        .to_string_lossy()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if n.len() < 3 {
        return;
    }
    let target = crate::math::v3(n[0], n[1], n[2]);
    let radius = n.get(3).copied().unwrap_or(0.5);
    let at = |i: u32| place.point(patch.p3[i as usize]);
    let mut lines: Vec<String> = Vec::new();
    for t in &patch.tris {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if a >= b {
                continue; // each undirected edge once
            }
            let (pa, pb) = (at(a), at(b));
            if ((pa + pb) * 0.5).distance(target) <= radius {
                lines.push(format!(
                    "    {:.4?} -> {:.4?}  len {:.4}",
                    pa,
                    pb,
                    pa.distance(pb)
                ));
            }
        }
    }
    if !lines.is_empty() {
        lines.sort();
        lines.dedup();
        eprintln!(
            "MESH {} face, {} pts, {locked} locked edges:\n{}",
            Surface::KINDS[surface.kind_index()],
            patch.p3.len(),
            lines.join("\n")
        );
    }
}

// Per-face tessellation
// ---------------------------------------------------------------------------

#[derive(Default)]
pub(crate) struct FaceMesh {
    pub(crate) vertices: Vec<Vertex>,
    pub(crate) indices: Vec<u32>,
    /// Which solid produced each run, and how many indices and vertices it
    /// grew them by. Vertices too, so a picked vertex maps to its component.
    pub(crate) runs: Vec<(u32, u32, u32)>,
    /// Each face drawn, in order: its solid, which of the solid's faces it
    /// is, how many indices it added, and where it is.
    pub(crate) faces: Vec<(u32, u32, u32, Aabb)>,
    pub(crate) bbox: Aabb,
    pub(crate) stats: TessStats,
}

pub(crate) struct PreparedFace {
    /// Sampled boundary points, in the solid's own space. Added to the mesh
    /// bounds per occurrence so a face that failed to triangulate still
    /// frames.
    frame_pts: Vec<Vec3>,
    /// The triangulation, absent when the face failed.
    body: Option<PreparedBody>,
    color: [f32; 4],
}

pub(crate) struct PreparedBody {
    /// Positions in f64. Emit places and *then* rounds to f32, exactly once;
    /// re-placing cached f32 vertices would round twice and move the mesh.
    p3: Vec<Vec3>,
    /// Surface normals with pole fallbacks resolved, before the `same_sense`
    /// flip, in f64 for the same reason.
    normals: Vec<Vec3>,
    tris: Vec<[u32; 3]>,
    flip: bool,
}

/// Places one prepared face: positions and normals through the occurrence's
/// placement, triangles wound against their own placed normals.
pub(crate) fn emit_face(prep: &PreparedFace, place: &Xform, out: &mut FaceMesh) -> Aabb {
    // The face's own bounds as well as the mesh's: the outline it sampled,
    // which is where a face that failed still is, and every vertex it drew.
    let mut own = Aabb::EMPTY;
    for p in &prep.frame_pts {
        let p = place.point(*p);
        out.bbox.add_point(p);
        own.add_point(p);
    }
    let Some(body) = &prep.body else { return own };
    let color = prep.color;
    let flip = body.flip;
    let base = out.vertices.len() as u32;
    // Each stored vertex read back once rather than once per corner:
    // unpacking the normal is a square root.
    let mut stored: Vec<(Vec3, Vec3)> = Vec::with_capacity(body.p3.len());
    for i in 0..body.p3.len() {
        let n = if flip { -body.normals[i] } else { body.normals[i] };
        // Tessellation happens in the solid's own space; the assembly placement
        // is applied here, on the way out.
        let p = place.point(body.p3[i]);
        let v = Vertex::new(
            p,
            // Normalised on the way out: a placement may carry a scale, which
            // lives in the basis vectors and would otherwise stretch the normal
            // along with the geometry.
            place.dir(n).normalize_or(Vec3::ZERO),
            [color[0], color[1], color[2]],
            color[3],
        );
        stored.push((v.position(), v.normal()));
        out.vertices.push(v);
        out.bbox.add_point(p);
        own.add_point(p);
    }
    for t in &body.tris {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        // Drop slivers that contribute nothing but z-fighting.
        let area2 = (body.p3[b] - body.p3[a])
            .cross(body.p3[c] - body.p3[a])
            .length();
        if !(area2 > 0.0) {
            continue;
        }
        // Wind each triangle to agree with the normal its own vertices carry.
        //
        // `same_sense` flips the normal and the vertex order together, which
        // is right only when the parameter directions run right-handed about
        // the surface normal, and they do not always. Two-sided shading hides
        // a reversed triangle; a section cap, which counts entries against
        // exits, does not. `winding()` measures it for `--check`.
        let ((pa, na), (pb, nb), (pc, nc)) = (stored[a], stored[b], stored[c]);
        let (g, s) = ((pb - pa).cross(pc - pa), na + nb + nc);
        let scale = g.length() * s.length();
        // Only where the two plainly disagree. A triangle nearly edge-on to
        // its vertex normals (at poles, apexes, long thin refinements) has no
        // reliable sign, so it keeps the order `same_sense` gave it and
        // `--check` reports it as undecided.
        let reversed = match g.dot(s) / scale {
            c if c > 0.1 => false,
            c if c < -0.1 => true,
            _ => flip,
        };
        if reversed {
            out.indices.extend_from_slice(&[base + t[0], base + t[2], base + t[1]]);
        } else {
            out.indices.extend_from_slice(&[base + t[0], base + t[1], base + t[2]]);
        }
    }
    own
}

/// Prepare and emit in one go, for the tests: pins that the two halves
/// compose.
#[cfg(test)]
pub(crate) fn tessellate_face(
    face: &Face,
    surfaces: &[Surface],
    curves: &[Curve],
    color: [f32; 4],
    tol: f64,
    max_seg: f64,
    place: &Xform,
    out: &mut FaceMesh,
) {
    let prep = prepare_face(face, surfaces, curves, color, &Tol::fixed(tol, max_seg), place, &mut out.stats);
    emit_face(&prep, place, out);
}

// The preparation half: sampling, seam resolution, constrained triangulation
// and refinement, none of which depends on where the solid stands.
// `diag_place` only positions the diagnostics (failure records, stray-face
// check), unless `tol` carries a view, which is placed with its solid.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_face(
    face: &Face,
    surfaces: &[Surface],
    curves: &[Curve],
    color: [f32; 4],
    tol: &Tol,
    diag_place: &Xform,
    stats: &mut TessStats,
) -> PreparedFace {
    let surface = &surfaces[face.surface as usize];
    let mut frame_pts: Vec<Vec3> = Vec::new();
    // The face without a body, for every failing path below.
    let dropped = |frame_pts: Vec<Vec3>| PreparedFace { frame_pts, body: None, color };

    // -- 1. sample the bounding loops ------------------------------------
    //
    // A face with nothing to sample (no bounds, or only a vertex loop) is the
    // whole surface, so the outline is the parameter rectangle.
    let mut loops: Vec<LoopUv> = Vec::with_capacity(face.bounds.len().max(1));
    if face.bounds.iter().all(|b| b.edges.is_empty()) {
        // A `RECTANGULAR_TRIMMED_SURFACE` says where the face stops; without
        // one the face is the whole closed surface. A plane or cylinder has no
        // whole, so needs the trim.
        let Some((u0, u1, v0, v1)) = face.domain.or_else(|| surface.full_domain()) else {
            stats.fail(Fail::NoLoops, surface, &loops, diag_place);
            return dropped(frame_pts);
        };
        let (u0, u1) = (u0.min(u1), u0.max(u1));
        let (v0, v1) = (v0.min(v1), v0.max(v1));
        if !(u1 > u0 && v1 > v0) {
            stats.fail(Fail::Degenerate, surface, &loops, diag_place);
            return dropped(frame_pts);
        }
        let l = whole_surface_loop(surface, tol.base, v2(u0, v0), v1, u1 - u0);
        for p in &l.pts {
            frame_pts.push(*p);
        }
        stats.seam_caps += 1;
        loops.push(l);
    }
    // A vertex bound carries no curve to sample; it is remembered so the seam
    // cap below can close onto the point the file actually named.
    let vertex_bound = face.bounds.iter().find_map(|b| b.vertex);
    for bound in &face.bounds {
        if bound.edges.is_empty() {
            continue;
        }
        let mut pts = sample_bound(bound, curves, tol, 1);
        // A ring of three points or fewer encloses nothing and the face is
        // lost. That happens when every edge is within tolerance of its chord
        // and the chords coincide (two shallow arcs either side of one line).
        // Sampling again with every edge halved gives it an interior; a truly
        // degenerate face is still counted as such later.
        if pts.len() < 4 {
            pts = sample_bound(bound, curves, tol, 2);
        }
        if pts.len() < 4 {
            stats.bounds_unsampled += 1;
            if std::env::var_os("STPVIEW_CDT").is_some() {
                let mut b = Aabb::EMPTY;
                for p in &pts {
                    b.add_point(*p);
                }
                eprintln!(
                    "LOST bound: {} edges, {} pts, extent {:.6?}",
                    bound.edges.len(),
                    pts.len(),
                    b.size()
                );
            }
            continue;
        }
        // Keep failed faces inside the bounds so the camera still frames them.
        for p in &pts {
            frame_pts.push(*p);
        }
        if let Some(l) = project_loop(surface, &pts, bound.outer, tol) {
            loops.push(l);
        } else {
            stats.bounds_unprojected += 1;
        }
    }
    if loops.is_empty() {
        stats.fail(Fail::NoLoops, surface, &loops, diag_place);
        return dropped(frame_pts);
    }

    probe(surface, face, curves, &loops, diag_place);

    // -- 2. resolve the seam ---------------------------------------------
    //
    // A face can circle either parameter. A torus is periodic both ways, and a
    // fillet ring that goes all the way round the tube wraps in v while a plain
    // revolved band wraps in u, so both have to be stitched.
    let u_period = surface.u_period().unwrap_or(f64::INFINITY);
    let v_period = surface.v_period().unwrap_or(f64::INFINITY);
    let wrapping: Vec<usize> = (0..loops.len())
        .filter(|&i| loops[i].winding_u.abs() > u_period * 0.5)
        .collect();
    let wrapping_v: Vec<usize> = (0..loops.len())
        .filter(|&i| loops[i].winding_v.abs() > v_period * 0.5)
        .collect();

    if wrapping.is_empty() && wrapping_v.len() == 1 {
        // The cap case, along the other axis: one loop circling v closes onto a
        // u at which the surface collapses. Only a NURBS patch has one -- a
        // dome written with its top control row repeated -- and the material
        // lies to the left of the traversal as always, which across v means
        // toward *decreasing* u.
        let li = wrapping_v[0];
        let positive = (loops[li].winding_v > 0.0) != face.same_sense;
        let Some(u_deg) = surface.degenerate_u(positive) else {
            stats.fail(Fail::Seam, surface, &loops, diag_place);
            return dropped(frame_pts);
        };
        let l = &mut loops[li];
        let n = l.uv.len();
        let (v_end, v_start) = (l.uv[n - 1].y, l.uv[0].y);
        for v in [v_end, v_start] {
            l.uv.push(v2(u_deg, v));
            l.pts.push(surface.eval(u_deg, v));
        }
        l.open_at = vec![n - 1, n, n + 1];
        l.winding_v = 0.0;
        l.outer = true;
        stats.seam_caps += 1;
    } else if wrapping.is_empty() && !wrapping_v.is_empty() {
        // Same stitch, along the other axis.
        if wrapping_v.len() != 2
            || loops[wrapping_v[0]].winding_v * loops[wrapping_v[1]].winding_v > 0.0
        {
            stats.fail(Fail::Seam, surface, &loops, diag_place);
            return dropped(frame_pts);
        }
        loops = stitch_band(loops, wrapping_v[0], wrapping_v[1], v_period, false, surface, tol.base);
        stats.seam_bands += 1;
    } else if !wrapping.is_empty() {
        // A face that circles a periodic surface has no seam edge, so each of
        // its loops is an open band in parameter space. Two loops with opposite
        // winding (the two ends of a full cylinder, say) join into one closed
        // polygon: the first runs u0 -> u0+P and the second comes back to u0.
        // The two joins are seam segments, not real edges, so they are recorded
        // in `open_at` and stay refinable.
        if wrapping.len() == 1 {
            // Only one loop circles the surface, so the face must close on a
            // degenerate parameter line: a cone cap or a spherical cap. Extend
            // the outline to that line and the region becomes an ordinary
            // parameter-space rectangle again.
            //
            // Which side holds the material follows from the B-rep convention
            // that a bound runs counter-clockwise seen from outside the face:
            // in (u, v) that puts the material to the left of the traversal,
            // mirrored when the face opposes its surface.
            let li = wrapping[0];
            let positive = (loops[li].winding_u > 0.0) == face.same_sense;
            // A `VERTEX_LOOP` names the closing point outright; only fall back
            // to reading it off the traversal direction when the file did not.
            let stated = vertex_bound.map(|p| surface.project(p, None).uv.y);
            if let Some(v_deg) = stated.or_else(|| surface.degenerate_v(positive)) {
                let l = &mut loops[li];
                let n = l.uv.len();
                let (u_end, u_start) = (l.uv[n - 1].x, l.uv[0].x);
                // Not a point the loop already ends or begins on: a repeated
                // point leaves a zero-length segment the constrained
                // triangulator refuses.
                let (lo, hi) = uv_bounds(&l.uv);
                let eps = (hi.x - lo.x).max(hi.y - lo.y) * 1e-9;
                let same = |a: Vec2, b: Vec2| (a.x - b.x).hypot(a.y - b.y) <= eps;
                for u in [u_end, u_start] {
                    let p = v2(u, v_deg);
                    if l.uv.last().is_some_and(|q| same(*q, p)) || same(l.uv[0], p) {
                        continue;
                    }
                    l.uv.push(p);
                    l.pts.push(surface.eval(u, v_deg));
                }
                // The added segments are synthetic, so they stay refinable --
                // the meridians running to a sphere's pole genuinely are curved.
                l.open_at = (n - 1..l.uv.len()).collect();
                l.winding_u = 0.0;
                l.outer = true;
                stats.seam_caps += 1;
            } else if v_period.is_finite() {
                // A surface closed both ways has no pole; a loop circling it
                // once cuts it into two annular halves. The face is the band
                // between the loop and itself one turn away in v, so the
                // stitcher's partner is the loop's own reflection. The turn
                // holding material follows the same left-of-traversal rule.
                let src = &loops[li];
                let n = src.uv.len();
                let shift = if positive { v_period } else { -v_period };
                let twin = LoopUv {
                    uv: src.uv.iter().rev().map(|p| v2(p.x, p.y + shift)).collect(),
                    pts: src.pts.iter().rev().copied().collect(),
                    winding_u: -src.winding_u,
                    winding_v: 0.0,
                    outer: false,
                    // Segment k runs backwards as segment n - 2 - k. The twin
                    // is the same 3D curve, so its segments stay as locked as
                    // the original's, or the join to the neighbour cracks.
                    open_at: src.open_at.iter().map(|k| n - 2 - k).collect(),
                };
                loops.push(twin);
                let j = loops.len() - 1;
                loops = stitch_band(loops, li, j, u_period, true, surface, tol.base);
                stats.seam_bands += 1;
            } else {
                // Nothing to close against: on a surface closed neither way a
                // lone circling loop (a cross-section of a cylinder) encloses
                // no area, so it is degenerate rather than a seam failure.
                stats.fail(Fail::Degenerate, surface, &loops, diag_place);
                return dropped(frame_pts);
            }
        } else if wrapping.len() != 2
            || loops[wrapping[0]].winding_u * loops[wrapping[1]].winding_u > 0.0
        {
            stats.fail(Fail::Seam, surface, &loops, diag_place);
            return dropped(frame_pts);
        } else {
            loops = stitch_band(loops, wrapping[0], wrapping[1], u_period, true, surface, tol.base);
            stats.seam_bands += 1;
        }
    }

    // -- 3. pick the outer loop and orient everything --------------------
    //
    // A bound that runs up a seam and straight back down it encloses nothing in
    // parameter space, yet it is exactly how a *complete* closed surface is
    // bounded: a ball is one spherical face whose only outer bound is the
    // meridian from pole to pole and back. Give that face the whole parameter
    // rectangle rather than dropping it as degenerate.
    if let Some(period) = surface.u_period()
        && let Some(i) =
            (0..loops.len()).find(|&i| loops[i].outer && encloses_nothing(&loops[i].uv))
    {
        let (lo, hi) = uv_bounds(&loops[i].uv);
        // A seam runs along the surface, not around it, and has to cover
        // real ground in v for there to be anything inside.
        if hi.y - lo.y > 0.0 && hi.x - lo.x < period * 0.5 {
            loops[i] = whole_surface_loop(surface, tol.base, lo, hi.y, period);
            // The holes were projected onto whichever period the seam did
            // not land in, so bring them into the rectangle's window.
            for (k, l) in loops.iter_mut().enumerate() {
                if k == i {
                    continue;
                }
                let mid = l.uv.iter().map(|p| p.x).sum::<f64>() / l.uv.len() as f64;
                let shift = ((lo.x + period * 0.5 - mid) / period).round() * period;
                if shift != 0.0 {
                    for p in &mut l.uv {
                        p.x += shift;
                    }
                }
            }
            stats.seam_caps += 1;
        }
    }

    // Every outline point must still be where its own parameters put it. Only
    // these can be wrong: refinement builds each new vertex as `eval` of its
    // own midpoint. See `TessStats::displaced`.
    for l in &loops {
        for i in 0..l.uv.len().min(l.pts.len()) {
            let at = surface.eval(l.uv[i].x, l.uv[i].y);
            if at.is_finite() && at.distance(l.pts[i]) > tol.base {
                stats.displaced += 1;
                if stats.displacements.len() < MAX_RECORDED_FAILURES {
                    stats.displacements.push(FaceFailure {
                        kind: Surface::KINDS[surface.kind_index()],
                        why: "not where its parameters put it",
                        center: diag_place.point(l.pts[i]),
                        size: diag_place.point(at),
                        uv_size: l.uv[i],
                    });
                }
            }
        }
    }

    let areas: Vec<f64> = loops.iter().map(|l| signed_area(&l.uv)).collect();
    let outer_idx = (0..loops.len())
        .filter(|&i| loops[i].outer)
        .max_by(|&a, &b| areas[a].abs().total_cmp(&areas[b].abs()))
        .or_else(|| (0..loops.len()).max_by(|&a, &b| areas[a].abs().total_cmp(&areas[b].abs())))
        .unwrap();
    if encloses_nothing(&loops[outer_idx].uv) {
        stats.fail(Fail::Degenerate, surface, &loops, diag_place);
        return dropped(frame_pts);
    }

    // Bring every hole into the same period as the outer ring.
    //
    // Each bound is projected alone, so on a periodic surface a hole can land
    // a whole turn away from its ring; the triangulator would then cover
    // `outer + hole` instead of `outer - hole`. Only whole periods are added,
    // so no point moves on the surface.
    let (o_lo, o_hi) = uv_bounds(&loops[outer_idx].uv);
    let o_mid = (o_lo + o_hi) * 0.5;
    for i in 0..loops.len() {
        if i == outer_idx {
            continue;
        }
        let (lo, hi) = uv_bounds(&loops[i].uv);
        let mid = (lo + hi) * 0.5;
        let du = surface
            .u_period()
            .map_or(0.0, |p| ((o_mid.x - mid.x) / p).round() * p);
        let dv = surface
            .v_period()
            .map_or(0.0, |p| ((o_mid.y - mid.y) / p).round() * p);
        if du != 0.0 || dv != 0.0 {
            for p in &mut loops[i].uv {
                *p = v2(p.x + du, p.y + dv);
            }
        }
    }

    let mut order: Vec<usize> = vec![outer_idx];
    order.extend((0..loops.len()).filter(|&i| i != outer_idx));

    // earcut wants the outer ring counter-clockwise and holes clockwise.
    let mut coords: Vec<f64> = Vec::new();
    let mut hole_starts: Vec<usize> = Vec::new();
    let mut uv: Vec<Vec2> = Vec::new();
    let mut p3: Vec<Vec3> = Vec::new();
    let mut locked = EdgeSet::default();
    // Index ranges of each closed ring, which are the constraints the
    // triangulator has to honour.
    let mut rings: Vec<(usize, usize)> = Vec::new();

    for (rank, &li) in order.iter().enumerate() {
        let l = &loops[li];
        let want_ccw = rank == 0;
        let reverse = (areas[li] > 0.0) != want_ccw;
        let base = uv.len() as u32;
        let n = l.uv.len();
        if rank > 0 {
            if encloses_nothing(&l.uv) {
                continue; // degenerate hole
            }
            hole_starts.push(uv.len());
        }
        rings.push((base as usize, n));
        for k in 0..n {
            let src = if reverse { n - 1 - k } else { k };
            uv.push(l.uv[src]);
            p3.push(l.pts[src]);
        }
        // Lock the segments that came from real boundary curves. Reversing a
        // loop permutes indices but keeps each segment's endpoint pair, so the
        // mapping below works either way.
        let at = |orig: usize| base + if reverse { (n - 1 - orig) as u32 } else { orig as u32 };
        for k in 0..n {
            if l.open_at.contains(&k) {
                continue;
            }
            locked.insert(ekey(at(k), at((k + 1) % n)));
        }
    }

    // Give the triangulator an outline shaped like the actual surface: u is
    // often radians while v is millimetres, and ear clipping is not affine
    // invariant, so each axis is scaled by the 3D length it covers.
    let (mut lo, mut hi) = (v2(f64::MAX, f64::MAX), v2(f64::MIN, f64::MIN));
    for p in &uv {
        lo = v2(lo.x.min(p.x), lo.y.min(p.y));
        hi = v2(hi.x.max(p.x), hi.y.max(p.y));
    }
    let mid = (lo + hi) * 0.5;
    let (mut sx, mut sy) = surface.param_scale(mid.x, mid.y);
    if !(sx.is_finite() && sx > 1e-12) {
        sx = 1.0;
    }
    if !(sy.is_finite() && sy > 1e-12) {
        sy = 1.0;
    }
    coords.reserve(uv.len() * 2);
    for p in &uv {
        coords.push((p.x - lo.x) * sx);
        coords.push((p.y - lo.y) * sy);
    }

    // Triangulate with the outline as a *constraint*. Ear clipping, on a run of
    // collinear boundary points, may clip a zero-area ear and run a chord past
    // a vertex the neighbouring face keeps, opening a hairline crack. A
    // constrained triangulation keeps every boundary segment by construction,
    // and being Delaunay it hands refinement fewer slivers.
    let pts: Vec<(f64, f64)> = coords.chunks_exact(2).map(|c| (c[0], c[1])).collect();
    // Each contour has to come back to where it started; the loops are stored
    // without that repeat because everything else treats them as cyclic.
    let contours: Vec<Vec<usize>> = rings
        .iter()
        .filter(|&&(_, n)| n >= 3)
        .map(|&(s, n)| (s..s + n).chain(std::iter::once(s)).collect())
        .collect();
    // An outline closed by repeating one of its sides a whole period away (a
    // whole closed surface, or a lone loop circling a torus) goes this way
    // too. Its opposite rims are the same curve, so an edge straight across
    // starts and ends at one point and goes all the way round; refinement
    // checks a short chord's midpoint for that.
    let (cdt_res, shaken) = triangulate_outline(&pts, &contours);
    if shaken {
        stats.cdt_shaken += 1;
    }
    let look = std::env::var_os("STPVIEW_CDT").is_some();
    match &cdt_res {
        Err(e) => {
            stats.cdt_failed += 1;
            // Which way the outline defeated it; usually it touches or
            // crosses itself in parameter space.
            if look {
                eprintln!(
                    "CDT {e:?} {} at {:.2?}: {}",
                    Surface::KINDS[surface.kind_index()],
                    diag_place.point(loops[0].pts[0]),
                    diagnose_contours(&pts, &contours)
                );
            }
        }
        // The faces kept with a segment missing, which are cracks, for the
        // same look.
        Ok(t) if look && !keeps_outline(t, &contours) => {
            eprintln!(
                "CDT kept with a segment missing: {} at {:.2?}: {}",
                Surface::KINDS[surface.kind_index()],
                diag_place.point(loops[0].pts[0]),
                diagnose_contours(&pts, &contours)
            );
        }
        Ok(_) => {}
    }
    // What the outline says the face covers: the outer ring less its holes.
    let want_area: f64 = order
        .iter()
        .enumerate()
        .map(|(rank, &li)| {
            let a = areas[li].abs();
            if rank == 0 {
                a
            } else {
                -a
            }
        })
        .sum::<f64>()
        .abs()
        * sx
        * sy;
    let mut cdt_area_rejected = false;
    let cdt_ok = cdt_res.is_ok();
    let tri3 = cdt_res
        .ok()
        .filter(|t| !t.is_empty())
        // A constrained triangulation should cover exactly the region the
        // outline encloses. Otherwise it read the contours differently (a band
        // wrapping a seam can confuse it), so hand it to ear clipping.
        .filter(|t| {
            let got: f64 = t
                .iter()
                .map(|&(a, b, c)| {
                    let (p, q, r) = (pts[a], pts[b], pts[c]);
                    ((q.0 - p.0) * (r.1 - p.1) - (q.1 - p.1) * (r.0 - p.0)).abs() * 0.5
                })
                .sum();
            let ok = want_area <= 0.0 || (got - want_area).abs() <= want_area * 1e-6;
            cdt_area_rejected = !ok;
            ok
        })
        .map(|t| t.into_iter().map(|(a, b, c)| [a, b, c]).collect::<Vec<_>>())
        // Ear clipping is the fallback: less careful but more forgiving.
        //
        // A face of exactly two rings (an annulus) also gets a zipped ribbon:
        // a rim thinner than its chord error has rings that cross, and ear
        // clipping then covers the whole disc. Whichever lands nearer the
        // enclosed area wins.
        .or_else(|| {
            let ear: Option<Vec<[usize; 3]>> = earcutr::earcut(&coords, &hole_starts, 2)
                .ok()
                .map(|f| f.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect());
            let zip = (contours.len() == 2)
                .then(|| ribbon(&pts, &contours[0], &contours[1]))
                .flatten();
            let miss = |t: &Vec<[usize; 3]>| {
                let got: f64 = t
                    .iter()
                    .map(|&[a, b, c]| {
                        let (p, q, r) = (pts[a], pts[b], pts[c]);
                        ((q.0 - p.0) * (r.1 - p.1) - (q.1 - p.1) * (r.0 - p.0)).abs() * 0.5
                    })
                    .sum();
                (got - want_area).abs()
            };
            match (ear, zip) {
                (Some(e), Some(z)) => Some(if miss(&z) < miss(&e) { z } else { e }),
                (e, z) => e.or(z),
            }
        });
    // How wide the face is at its narrowest, which decides whether the
    // tolerance can represent it at all; see `outline_width`.
    let perimeter: f64 = loops
        .iter()
        .flat_map(|l| l.pts.windows(2).map(|w| w[0].distance(w[1])))
        .sum();
    let width = outline_width(want_area, perimeter);
    let note_width = |stats: &mut TessStats| {
        if let Some(w) = width
            && stats.narrowest.is_none_or(|n| w < n)
        {
            stats.narrowest = Some(w);
        }
    };

    let Some(flat) = tri3.filter(|t: &Vec<[usize; 3]>| !t.is_empty()) else {
        stats.fail(Fail::Earcut, surface, &loops, diag_place);
        return dropped(frame_pts);
    };

    // The same area test, applied to whatever was finally emitted. Covering
    // more than the outline encloses lays two skins over part of the face,
    // which nothing downstream can detect.
    if want_area > 0.0 {
        let got: f64 = flat
            .iter()
            .map(|&[a, b, c]| {
                let (p, q, r) = (pts[a], pts[b], pts[c]);
                ((q.0 - p.0) * (r.1 - p.1) - (q.1 - p.1) * (r.0 - p.0)).abs() * 0.5
            })
            .sum();
        if (got - want_area).abs() > want_area * 1e-3 {
            stats.miscovered += 1;
            note_width(stats);
            if std::env::var_os("STPVIEW_CDT").is_some() {
                eprintln!(
                    "MISCOVER {} {} rings {} pts, want {want_area:.4} got {got:.4} ({:.3}x), \
                     {} bounds, {}",
                    Surface::KINDS[surface.kind_index()],
                    rings.len(),
                    uv.len(),
                    got / want_area,
                    face.bounds.len(),
                    if cdt_area_rejected {
                        "cdt area rejected"
                    } else if cdt_ok {
                        "cdt"
                    } else {
                        "ear clipped"
                    }
                );
            }
            if stats.miscovers.len() < MAX_RECORDED_FAILURES {
                let bbox = loop_bbox(&loops, diag_place);
                stats.miscovers.push(FaceFailure {
                    kind: Surface::KINDS[surface.kind_index()],
                    why: if got > want_area { "covered twice" } else { "left uncovered" },
                    center: bbox.center(),
                    size: bbox.size(),
                    uv_size: v2(want_area, got),
                });
            }
        }
        // Far past the outline's area it is a different face, not an
        // imperfect one: a thin annulus whose chorded rings cross gets the
        // whole disc from ear clipping. Better a hole the size of the true
        // face than a sheet many times larger; `--stats` reports it.
        if got > want_area * 10.0 {
            stats.fail(Fail::Miscovered, surface, &loops, diag_place);
            return dropped(frame_pts);
        }
    }

    let mut patch = Patch {
        uv,
        p3,
        tris: flat
            .iter()
            .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32])
            .collect(),
        locked,
    };

    // -- 4. refine to tolerance ------------------------------------------
    patch.refine(surface, tol, stats);

    probe_mesh(surface, &patch, diag_place, patch.locked.len());

    // -- 5. emit ----------------------------------------------------------
    //
    // Triangles reaching well past the outline mean it was mis-projected and
    // the face is the wrong shape, which nothing else detects. A closed
    // surface legitimately exceeds its seam, so those are exempt.
    let closed = surface.u_period().is_some() || surface.v_period().is_some();
    if !closed {
        let bnd = loop_bbox(&loops, diag_place);
        let mut got = Aabb::EMPTY;
        for p in &patch.p3 {
            got.add_point(diag_place.point(*p));
        }
        if got.extent() > bnd.extent() * 1.5 + 1e-9 {
            stats.stray_faces += 1;
            if stats.strays.len() < MAX_RECORDED_FAILURES {
                stats.strays.push(FaceFailure {
                    kind: Surface::KINDS[surface.kind_index()],
                    why: "drawn outside its outline",
                    center: bnd.center(),
                    size: got.size(),
                    uv_size: v2(bnd.extent(), got.extent()),
                });
            }
        }
    }

    // The normals with their pole fallbacks, resolved here since they do not
    // depend on placement; the same_sense flip and the placement wait for
    // emit.
    let mut normals: Vec<Vec3> = patch
        .uv
        .iter()
        .map(|p| surface.normal(p.x, p.y).unwrap_or(Vec3::ZERO))
        .collect();
    // Singular points (cone apex, sphere pole) get their normal from the
    // triangles that touch them.
    let mut fallback: Vec<Vec3> = vec![Vec3::ZERO; patch.uv.len()];
    for t in &patch.tris {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        let fnrm = (patch.p3[b] - patch.p3[a]).cross(patch.p3[c] - patch.p3[a]);
        for &i in &[a, b, c] {
            fallback[i] += fnrm;
        }
    }
    for i in 0..normals.len() {
        if normals[i].length_sq() < 0.5 {
            normals[i] = fallback[i].normalize_or(Vec3::ZERO);
        }
    }

    stats.faces_ok += 1;
    PreparedFace {
        frame_pts,
        body: Some(PreparedBody {
            p3: patch.p3,
            normals,
            tris: patch.tris,
            flip: !face.same_sense,
        }),
        color,
    }
}

/// The constrained triangulation of an outline, and whether it took a second
/// attempt: `pts` are the outline's points as the triangulator sees them and
/// `contours` its rings, each closed by repeating its first index.
///
/// The triangulator panics on some outlines, so it is called inside a catch.
/// It is a pure computation over two slices, so unwinding leaves nothing
/// behind; the face falls back to ear clipping.
pub(crate) fn triangulate_outline(
    pts: &[(f64, f64)],
    contours: &[Vec<usize>],
) -> (Result<Vec<(usize, usize, usize)>, cdt::Error>, bool) {
    let run = |p: &[(f64, f64)]| {
        quietly(|| cdt::triangulate_contours(p, contours)).unwrap_or(Err(cdt::Error::EmptyInput))
    };
    let whole = |r: &Result<Vec<(usize, usize, usize)>, cdt::Error>| {
        r.as_ref().is_ok_and(|t| keeps_outline(t, contours))
    };
    let r = run(pts);
    // A second attempt with the input shaken by a distance nothing can see.
    //
    // The predicates are exact, and several boundary segments lying exactly
    // along one line (a strip with teeth cut in it) is a case the triangulator
    // refuses. These coordinates decide connectivity only; vertices keep their
    // positions in `uv` and `p3`. So shaking by a billionth of the outline's
    // size moves no geometry, only which valid triangulation comes back. The
    // offsets come from the index, so a file always gives the same mesh.
    //
    // A result missing an outline segment also gets a second attempt, but
    // yields only to one that keeps them all; an incomplete triangulation is
    // still a better fallback than ear clipping, which steps over boundary
    // points.
    if whole(&r) {
        return (r, false);
    }
    let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
    for p in pts {
        lo = (lo.0.min(p.0), lo.1.min(p.1));
        hi = (hi.0.max(p.0), hi.1.max(p.1));
    }
    let e = (hi.0 - lo.0).max(hi.1 - lo.1) * 1e-9;
    let shaken: Vec<(f64, f64)> = pts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let jitter = |s: u32| ((h >> s) & 0xffff) as f64 / 32768.0 - 1.0;
            (p.0 + jitter(11) * e, p.1 + jitter(37) * e)
        })
        .collect();
    let second = run(&shaken);
    if (r.is_err() && second.is_ok()) || whole(&second) {
        return (second, true);
    }
    (r, false)
}

/// Whether a triangulation has every segment of every contour among its
/// edges, which is the whole of what the triangulator promises.
///
/// It can break that and still report success, e.g. on nearly collinear rim
/// points a hair off a line of constant v. The missing piece has no area in
/// (u, v), so the area test cannot see it, but it is a crack against the
/// neighbouring face.
pub(crate) fn keeps_outline(t: &[(usize, usize, usize)], contours: &[Vec<usize>]) -> bool {
    // Every segment joins a point to the next one round its ring, so a point
    // and a flag identify it; cheaper than hashing every edge of every face.
    let n = contours.iter().flatten().max().map_or(0, |&i| i + 1);
    let mut next = vec![usize::MAX; n];
    for w in contours.iter().flat_map(|c| c.windows(2)) {
        next[w[0]] = w[1];
    }
    let mut kept = vec![false; n];
    for &(a, b, c) in t {
        for (x, y) in [(a, b), (b, c), (c, a), (b, a), (c, b), (a, c)] {
            if next.get(x) == Some(&y) {
                kept[x] = true;
            }
        }
    }
    (0..n).all(|i| next[i] == usize::MAX || kept[i])
}
