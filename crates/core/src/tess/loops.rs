//! Parameter-space loops: projection, unwrapping, seam stitching -- the
//! machinery that turns sampled 3D bounds into outlines a triangulator can
//! hold, without ever separating a point from its parameters.
use super::*;

// Parameter-space loops
// ---------------------------------------------------------------------------

/// Shifts `d` into (-period/2, period/2].
pub(crate) fn wrap_delta(d: f64, period: f64) -> f64 {
    let mut d = d % period;
    if d > period * 0.5 {
        d -= period;
    } else if d < -period * 0.5 {
        d += period;
    }
    d
}

#[derive(Clone)]
pub(crate) struct LoopUv {
    pub(crate) uv: Vec<Vec2>,
    pub(crate) pts: Vec<Vec3>,
    /// Net change in u around the closed loop: ~0, or ~+/-period at a seam.
    pub(crate) winding_u: f64,
    pub(crate) winding_v: f64,
    pub(crate) outer: bool,
    /// Indices `i` where the segment `i -> i+1` is *not* a real boundary curve
    /// (it was synthesised when stitching a seam), so it may be refined.
    pub(crate) open_at: Vec<usize>,
}

/// Projects a closed 3D polyline into parameter space, repairing points where
/// `u` is undefined and unwrapping across the seam.
///
/// `pts` must be closed: the last point repeats the first. That repeat is what
/// makes the seam visible: after unwrapping, a loop that circles a periodic
/// surface ends one full period away from where it started, and the duplicate
/// is kept so the band has a real edge there. Loops that do not wrap get the
/// duplicate dropped again.
pub(crate) fn project_loop(surface: &Surface, pts: &[Vec3], outer: bool, tol: &Tol) -> Option<LoopUv> {
    let n = pts.len();
    if n < 4 {
        return None;
    }
    // Walk the polyline in order so each projection seeds the next, stopping
    // within a thousandth of the tolerance. The first point has no seed and
    // decides which side of a seam the loop starts on, so it is searched fully.
    let mut hint: Option<Vec2> = None;
    let projs: Vec<_> = pts
        .iter()
        .map(|&p| {
            let close = if hint.is_some() { tol.at(p) * 1e-3 } else { 0.0 };
            let pr = surface.project_near(p, hint, close);
            hint = Some(pr.uv);
            pr
        })
        .collect();

    // At a cone apex or a sphere pole every u is equally valid; borrow the
    // value from the nearest neighbour that has a real one.
    let good: Vec<usize> = (0..n).filter(|&i| !projs[i].u_degenerate).collect();
    if good.is_empty() {
        return None;
    }
    let mut uv: Vec<Vec2> = projs.iter().map(|p| p.uv).collect();
    if good.len() < n {
        for i in 0..n {
            if projs[i].u_degenerate {
                // Nearest good index by cyclic distance.
                let best = good
                    .iter()
                    .copied()
                    .min_by_key(|&g| {
                        let d = (g as isize - i as isize).unsigned_abs();
                        d.min(n - d)
                    })
                    .unwrap();
                uv[i].x = projs[best].uv.x;
            }
        }
    }

    let mut winding_u = 0.0;
    let mut winding_v = 0.0;
    if let Some(period) = surface.u_period() {
        let raw: Vec<f64> = uv.iter().map(|p| p.x).collect();
        let mut acc = raw[0];
        for i in 1..n {
            acc += wrap_delta(raw[i] - raw[i - 1], period);
            uv[i].x = acc;
        }
        winding_u = uv[n - 1].x - uv[0].x;
    }
    if let Some(period) = surface.v_period() {
        let raw: Vec<f64> = uv.iter().map(|p| p.y).collect();
        let mut acc = raw[0];
        for i in 1..n {
            acc += wrap_delta(raw[i] - raw[i - 1], period);
            uv[i].y = acc;
        }
        winding_v = uv[n - 1].y - uv[0].y;
    }

    let mut pts = pts.to_vec();
    let wraps = winding_u.abs() > 1e-9 || winding_v.abs() > 1e-9;
    if !wraps {
        // The closing point coincides with the first in parameter space too.
        uv.pop();
        pts.pop();
    }
    if uv.len() < 3 {
        return None;
    }

    Some(LoopUv {
        uv,
        pts,
        winding_u,
        winding_v,
        outer,
        open_at: Vec::new(),
    })
}

/// Walks from the end of `uv` across to `to`, recording each segment as open.
///
/// Normally one straight segment, which refinement later splits where the
/// surface bends. The exception is a join whose two ends are the *same place*
/// (a crossing spanning a whole period): refinement will not touch a
/// zero-length edge, so that one is laid out along the surface up front.
pub(crate) fn bridge(
    surface: &Surface,
    tol: f64,
    uv: &mut Vec<Vec2>,
    pts: &mut Vec<Vec3>,
    open_at: &mut Vec<usize>,
    to: Vec2,
) {
    let from = *uv.last().unwrap();
    let degenerate = pts
        .last()
        .is_some_and(|p| p.distance(surface.eval(to.x, to.y)) <= tol);
    let steps = if !degenerate {
        1
    } else {
        let (du, dv) = surface.uv_step(
            tol,
            v2(from.x.min(to.x), from.y.min(to.y)),
            v2(from.x.max(to.x), from.y.max(to.y)),
        );
        (((to.x - from.x).abs() / du)
            .max((to.y - from.y).abs() / dv)
            .ceil() as usize)
            .clamp(1, 256)
    };
    for k in 1..steps {
        let p = from + (to - from) * (k as f64 / steps as f64);
        open_at.push(uv.len() - 1);
        uv.push(p);
        pts.push(surface.eval(p.x, p.y));
    }
    open_at.push(uv.len() - 1);
}

/// Re-starts a loop that circles the surface at whichever of its own samples
/// sits nearest `target`, without moving any of them.
///
/// A band's second rim has to be traced over the same window as the first.
/// Choosing where to begin walking the cycle keeps every point with the
/// parameters it was projected at; translating the parameters instead would
/// slide points round the surface while leaving them where they were in space.
pub(crate) fn rotate_loop(l: &LoopUv, target: f64, period: f64, on_u: bool) -> LoopUv {
    let coord = |p: Vec2| if on_u { p.x } else { p.y };
    let shifted = |p: Vec2, d: f64| if on_u { v2(p.x + d, p.y) } else { v2(p.x, p.y + d) };
    // The last point repeats the first one period along, so the distinct cycle
    // is everything before it.
    let n = l.uv.len();
    if n < 3 {
        return l.clone();
    }
    let cycle = n - 1;
    let turn = coord(l.uv[n - 1]) - coord(l.uv[0]);
    let start = (0..cycle)
        .min_by(|&a, &b| {
            let d = |i: usize| wrap_delta(coord(l.uv[i]) - target, period).abs();
            d(a).total_cmp(&d(b))
        })
        .unwrap();

    let mut uv = Vec::with_capacity(n);
    let mut pts = Vec::with_capacity(n);
    for j in 0..cycle {
        let i = (start + j) % cycle;
        // Points that came before the new start are one turn behind it.
        let d = if i < start { turn } else { 0.0 };
        uv.push(shifted(l.uv[i], d));
        pts.push(l.pts[i]);
    }
    uv.push(shifted(l.uv[start], turn));
    pts.push(l.pts[start]);

    LoopUv {
        uv,
        pts,
        winding_u: l.winding_u,
        winding_v: l.winding_v,
        outer: l.outer,
        // Segment k ran from point k; it is now segment k - start round the
        // cycle. The closing segment has no original to inherit from.
        open_at: l
            .open_at
            .iter()
            .filter(|&&k| k < cycle)
            .map(|&k| (k + cycle - start) % cycle)
            .collect(),
    }
}

/// Joins two loops that each circle the surface into one closed outline.
///
/// The first runs from its start to one period further on and the second
/// comes back, closing the band. `on_u` selects the periodic parameter; a
/// torus needs both. The two joins are seam segments, so they go in `open_at`
/// and stay refinable.
pub(crate) fn stitch_band(
    loops: Vec<LoopUv>,
    i: usize,
    j: usize,
    period: f64,
    on_u: bool,
    surface: &Surface,
    tol: f64,
) -> Vec<LoopUv> {
    let winding = |l: &LoopUv| if on_u { l.winding_u } else { l.winding_v };
    let coord = |p: Vec2| if on_u { p.x } else { p.y };
    let shifted = |p: Vec2, d: f64| if on_u { v2(p.x + d, p.y) } else { v2(p.x, p.y + d) };

    let (first, second) = if winding(&loops[i]) > 0.0 { (i, j) } else { (j, i) };
    // Begin the second rim beside the end of the first by choosing where to
    // start walking it; the bridge spans what is left (under one sample).
    let end = coord(*loops[first].uv.last().unwrap());
    let turned = rotate_loop(&loops[second], end, period, on_u);
    // Then whole periods only, which are a no-op on the surface.
    let shift = ((end - coord(turned.uv[0])) / period).round() * period;

    let mut uv = loops[first].uv.clone();
    let mut pts = loops[first].pts.clone();
    let mut open_at = Vec::new();

    let second_uv: Vec<Vec2> = turned.uv.iter().map(|p| shifted(*p, shift)).collect();
    bridge(surface, tol, &mut uv, &mut pts, &mut open_at, second_uv[0]);
    for (k, p) in second_uv.iter().enumerate() {
        uv.push(*p);
        pts.push(turned.pts[k]);
    }
    // And back to where the first loop started, closing the ring.
    let start = uv[0];
    bridge(surface, tol, &mut uv, &mut pts, &mut open_at, start);

    let others: Vec<LoopUv> = loops
        .into_iter()
        .enumerate()
        .filter(|(k, _)| *k != i && *k != j)
        .map(|(_, l)| l)
        .collect();
    let mut out = vec![LoopUv {
        uv,
        pts,
        winding_u: 0.0,
        winding_v: 0.0,
        outer: true,
        open_at,
    }];
    out.extend(others);
    out
}

/// True when a loop encloses no area worth speaking of, measured against its
/// own size. An exactly degenerate loop comes back as 1e-16 as often as 0, so
/// an absolute threshold would not do.
pub(crate) fn encloses_nothing(uv: &[Vec2]) -> bool {
    let (lo, hi) = uv_bounds(uv);
    let scale = (hi.x - lo.x).abs().max((hi.y - lo.y).abs());
    signed_area(uv).abs() <= (scale * scale * 1e-12).max(1e-18)
}

pub(crate) fn loop_bbox(loops: &[LoopUv], place: &Xform) -> Aabb {
    let mut bb = Aabb::EMPTY;
    for l in loops {
        for p in &l.pts {
            bb.add_point(place.point(*p));
        }
    }
    bb
}

pub(crate) fn uv_bounds(uv: &[Vec2]) -> (Vec2, Vec2) {
    let (mut lo, mut hi) = (uv[0], uv[0]);
    for p in uv {
        lo = v2(lo.x.min(p.x), lo.y.min(p.y));
        hi = v2(hi.x.max(p.x), hi.y.max(p.y));
    }
    (lo, hi)
}

/// The outline of a whole closed surface: one full period of `u` against the
/// span of `v` the seam covered, walked anticlockwise.
///
/// The edges are laid out at `uv_step` rather than left as four corners: along
/// a pole every point is the same point, so the chord test sees nothing to
/// split there.
pub(crate) fn whole_surface_loop(surface: &Surface, tol: f64, lo: Vec2, v_hi: f64, period: f64) -> LoopUv {
    let (du, dv) = surface.uv_step(tol, lo, v2(lo.x + period, v_hi));
    let nu = ((period / du).ceil() as usize).clamp(4, 256);
    let nv = (((v_hi - lo.y) / dv).ceil() as usize).clamp(1, 256);
    let (u0, u1, v0, v1) = (lo.x, lo.x + period, lo.y, v_hi);
    let mut uv = Vec::with_capacity(2 * (nu + nv));
    let lerp = |a: f64, b: f64, k: usize, n: usize| a + (b - a) * k as f64 / n as f64;
    for k in 0..nu {
        uv.push(v2(lerp(u0, u1, k, nu), v0));
    }
    for k in 0..nv {
        uv.push(v2(u1, lerp(v0, v1, k, nv)));
    }
    for k in 0..nu {
        uv.push(v2(lerp(u1, u0, k, nu), v1));
    }
    for k in 0..nv {
        uv.push(v2(u0, lerp(v1, v0, k, nv)));
    }
    let pts = uv.iter().map(|p| surface.eval(p.x, p.y)).collect();
    let open_at = (0..uv.len()).collect();
    LoopUv {
        uv,
        pts,
        winding_u: 0.0,
        winding_v: 0.0,
        outer: true,
        open_at,
    }
}

pub(crate) fn signed_area(poly: &[Vec2]) -> f64 {
    let n = poly.len();
    let mut a = 0.0;
    for i in 0..n {
        let p = poly[i];
        let q = poly[(i + 1) % n];
        a += p.x * q.y - q.x * p.y;
    }
    a * 0.5
}
