//! The edge overlay: B-rep edges grouped by where they are, classified
//! sharp or tangent by the faces that meet there.
use super::*;

// Edge overlay
// ---------------------------------------------------------------------------

/// Faces meeting at less than this angle are treated as tangent, and the edge
/// between them is not drawn. A fillet blending into a wall meets it at 0
/// degrees; a chamfer or a bore breaking a surface meets it at a real angle.
pub(crate) const TANGENT_ANGLE: f64 = 20.0;

/// Outward normal of `face` at a point lying on one of its edges, and where on
/// the surface that is, to start from at the next point along the edge. See
/// `Surface::project_near` for `hint` and `close`.
pub(crate) fn face_normal_at(
    surfaces: &[Surface],
    surface: u32,
    same_sense: bool,
    p: Vec3,
    hint: Option<Vec2>,
    close: f64,
) -> (Option<Vec3>, Vec2) {
    let s = &surfaces[surface as usize];
    let uv = s.project_near(p, hint, close).uv;
    let n = s.normal(uv.x, uv.y).map(|n| if same_sense { n } else { -n });
    (n, uv)
}

/// Everything the overlay needs from the model's edges.
pub(crate) struct Edges {
    /// Line-list positions, loose curves first, then sharp, then tangent,
    /// within this batch; every pass slices the buffer by part.
    pub(crate) lines: Vec<[f32; 3]>,
    pub(crate) span: Vec<EdgeSpan>,
    pub(crate) bbox: Aabb,
    pub(crate) stats: TessStats,
}

/// `extract_edges_in` at one tolerance everywhere, for the tests.
#[cfg(test)]
pub(crate) fn extract_edges(model: &Model, tol: f64, max_seg: f64, sel: std::ops::Range<usize>) -> Edges {
    extract_edges_in(model, &Tol::fixed(tol, max_seg), None, sel)
}

/// Collects the wireframe.
///
/// Each edge is sampled once however many faces use it. An edge is tangent
/// when the faces sharing it have nearly the same outward normal along it:
/// a fillet meeting a wall, or the seam of a closed surface meeting itself.
///
/// Sampled from the same tolerance field as the faces, so a line along a face
/// refined for a view lies on that face's own boundary points.
///
/// Returns the line-list vertices with sharp edges packed first.
pub(crate) fn extract_edges_in(
    model: &Model,
    tol: &Tol,
    view: Option<&ViewTolerance>,
    sel: std::ops::Range<usize>,
) -> Edges {
    // Group every use of an edge by *where it is*, not by the entity it came
    // from. Some files give each face its own edge entities, so grouping by id
    // would leave every edge single-use, never compared for tangency, and
    // drawn twice.
    //
    // The key is the two endpoints plus the midpoint, so two closed edges
    // meeting at one point stay apart. The solid index is part of it because
    // two occurrences of one component must not pool, and each solid is
    // welded on its own so a chain through another solid's points cannot join
    // two of this one's.
    //
    // Welded through `Weld`, not by rounding to a grid: copies of one point
    // computed from different edges land a hair apart, and a grid splits any
    // pair a cell boundary runs between.
    let weld = (tol.base * 1e-3).max(model.vertex_bbox.extent() * 1e-9);
    /// One face's use of an edge: the traversal, the surface that face lies
    /// on, and whether the face runs with that surface's normal.
    type FaceUse = (EdgeUse, u32, bool);
    // Every group, in the order its first use was met. No group spans two
    // solids, so each solid is grouped on its own thread and the lists joined
    // in solid order, giving the same list as a serial pass.
    let per_solid: Vec<Vec<Vec<FaceUse>>> = model.solids[sel.clone()]
        .par_iter()
        .map(|solid| {
            let place = &solid.placement;
            let mut w = Weld::new(weld);
            let mut found: Vec<(FaceUse, [u32; 3])> = Vec::new();
            for face in &solid.faces {
                for bound in &face.bounds {
                    for e in &bound.edges {
                        let c = &model.curves[e.curve as usize];
                        let mut at = |t: f64| w.add(place.point(c.eval(t)));
                        let ends = [at(e.t0), at(e.t1), at((e.t0 + e.t1) * 0.5)];
                        found.push(((*e, face.surface, face.same_sense), ends));
                    }
                }
            }
            w.settle();
            let mut group_of: HashMap<(u32, u32, u32), usize> = HashMap::new();
            let mut groups: Vec<Vec<FaceUse>> = Vec::new();
            for (face_use, [a, b, mid]) in found {
                let (a, b, mid) = (w.point(a), w.point(b), w.point(mid));
                // Order-independent: the same edge traced either way is the
                // same edge.
                let key = (a.min(b), a.max(b), mid);
                let next = groups.len();
                let g = *group_of.entry(key).or_insert(next);
                if g == next {
                    groups.push(Vec::new());
                }
                groups[g].push(face_use);
            }
            groups
        })
        .collect();
    let mut items: Vec<(usize, Vec<FaceUse>)> = per_solid
        .into_iter()
        .enumerate()
        .flat_map(|(si, groups)| groups.into_iter().map(move |g| (si, g)))
        .collect();

    let cos_limit = TANGENT_ANGLE.to_radians().cos();
    // A loose curve in a `GEOMETRIC_SET` bounds nothing, so it has no faces
    // to measure an angle between. It is always drawn.
    let mut loose: Vec<(usize, EdgeUse)> = Vec::new();
    for (si, solid) in model.solids[sel.clone()].iter().enumerate() {
        loose.extend(solid.wires.iter().map(|e| (si, *e)));
    }

    // By solid, so the buffer can be sliced per part, and within a solid by the
    // first use's edge id. The sort is stable and groups arrive in the order
    // met, so nothing depends on a hash seed: edge quads blend, and drawing
    // order can move a pixel.
    items.sort_by_key(|(si, g)| (*si, g.first().map(|(e, _, _)| e.edge).unwrap_or(0)));

    let single_use = items.iter().filter(|(_, g)| g.len() == 1).count();
    // Class 0 is a loose curve, 1 a sharp edge, 2 a tangent one. Sorting by it
    // puts each kind in its own run of the buffer.
    let mut lines: Vec<(u8, usize, Vec<Vec3>)> = items
        .par_iter()
        .filter_map(|(si, group)| {
            let place = &model.solids[sel.start + *si].placement;
            let (e, _, _) = *group.first()?;
            let curve = &model.curves[e.curve as usize];
            let placed = tol.placed(view, place);
            let mut pts = Vec::new();
            sample_edge_in(curve, e.t0, e.t1, &placed, &mut pts);
            if pts.len() < 2 {
                return None;
            }
            for p in &mut pts {
                *p = place.point(*p);
            }

            // An edge used by only one face bounds an open shell, so it is
            // always a real outline. Anything unmeasurable stays visible.
            //
            // The second sample point starts its projection from where the
            // first landed, and a thousandth of the tolerance is close enough
            // for a normal.
            let sharp = match (group.first(), group.get(1)) {
                (Some(&(_, sa, ta)), Some(&(_, sb, tb))) => {
                    let (mut from_a, mut from_b) = (None, None);
                    [0.3, 0.7].iter().any(|f| {
                        let p = curve.eval(e.t0 + (e.t1 - e.t0) * f);
                        let close = placed.at(p) * 1e-3;
                        let (na, at_a) = face_normal_at(&model.surfaces, sa, ta, p, from_a, close);
                        let (nb, at_b) = face_normal_at(&model.surfaces, sb, tb, p, from_b, close);
                        (from_a, from_b) = (Some(at_a), Some(at_b));
                        match (na, nb) {
                            (Some(na), Some(nb)) => na.dot(nb) < cos_limit,
                            _ => true,
                        }
                    })
                }
                _ => true,
            };
            Some((if sharp { 1u8 } else { 2u8 }, *si, pts))
        })
        .collect();

    lines.extend(loose.par_iter().filter_map(|(si, e)| {
        let place = &model.solids[sel.start + *si].placement;
        let mut pts = Vec::new();
        sample_edge_in(&model.curves[e.curve as usize], e.t0, e.t1, &tol.placed(view, place), &mut pts);
        if pts.len() < 2 {
            return None;
        }
        for p in &mut pts {
            *p = place.point(*p);
        }
        Some((0u8, *si, pts))
    }).collect::<Vec<_>>());

    // Loose curves first, then sharp, then tangent, and by solid within each,
    // so one part is one run of each. This lets the renderer show a subset
    // from one buffer and draw loose curves without the eye-ward nudge.
    lines.sort_by_key(|(class, si, _)| (*class, *si));

    let mut bbox = Aabb::EMPTY;
    let mut stats = TessStats::default();
    let total: usize = lines.iter().map(|(_, _, p)| (p.len() - 1) * 2).sum();
    let mut out = Vec::with_capacity(total);
    // Three runs per solid: its loose curves, its sharp edges, its tangent ones.
    let mut span: Vec<EdgeSpan> = vec![(0..0, 0..0, 0..0); sel.len()];
    for (class, si, pts) in &lines {
        match class {
            0 => {}
            1 => stats.edges_sharp += 1,
            _ => stats.edges_tangent += 1,
        }
        let from = out.len() as u32;
        for w in pts.windows(2) {
            out.push(w[0].as_f32());
            out.push(w[1].as_f32());
        }
        for p in pts {
            bbox.add_point(*p);
        }
        let to = out.len() as u32;
        let e = &mut span[*si];
        let slot = match class {
            0 => &mut e.0,
            1 => &mut e.1,
            _ => &mut e.2,
        };
        if slot.start == slot.end {
            *slot = from..to;
        } else {
            slot.end = to;
        }
    }
    stats.edges_single_use = single_use;

    Edges {
        lines: out,
        span,
        bbox,
        stats,
    }
}
