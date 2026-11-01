//! Curve sampling: edges and bounds into chorded polylines.
use super::*;

// Curve sampling
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) fn subdivide(
    c: &Curve,
    ta: f64,
    pa: Vec3,
    tb: f64,
    pb: Vec3,
    tol: &Tol,
    depth: i32,
    out: &mut Vec<Vec3>,
) {
    // Past the model tolerance's depth, only a segment a view wants finer may
    // go on down, so faces a view leaves alone keep exactly the same points.
    if depth > 0 || (!tol.is_uniform() && depth > -VIEW_CURVE_DEPTH) {
        let tm = 0.5 * (ta + tb);
        let pm = c.eval(tm);
        // Asked of all three points so the segment gets the same answer
        // whichever face samples it and in whichever direction; the two
        // faces along an edge must agree on every point or a crack opens.
        let (t, max_seg) = tol.segment(pa, pm, pb);
        let deep_enough = depth <= 0 && t >= tol.base;
        if !deep_enough && (pa.distance(pb) > max_seg || pm.distance((pa + pb) * 0.5) > t) {
            subdivide(c, ta, pa, tm, pm, tol, depth - 1, out);
            subdivide(c, tm, pm, tb, pb, tol, depth - 1, out);
            return;
        }
    }
    out.push(pb);
}

/// `sample_edge_in` at one tolerance and one longest chord, for the tests.
#[cfg(test)]
pub(crate) fn sample_edge(c: &Curve, t0: f64, t1: f64, tol: f64, max_seg: f64, out: &mut Vec<Vec3>) {
    sample_edge_in(c, t0, t1, &Tol::fixed(tol, max_seg), out);
}

/// Appends the polyline for `t0 -> t1`, including both endpoints.
///
/// The tolerance's `max_seg` caps a segment's 3D length however straight the
/// curve: a cylinder's bounding circles are straight lines in parameter space,
/// and an outline with many collinear points on two sides and two on the
/// others degenerates the triangulator.
///
/// It is a single global length, not derived from the surface, so two faces
/// sharing an edge sample it identically. Refined for a view it shrinks as a
/// function of position, so the faces still agree.
///
/// The first cut is made at the model's own tolerance, the coarsest the field
/// asks for; subdivision goes finer where a view wants it.
pub(crate) fn sample_edge_in(c: &Curve, t0: f64, t1: f64, tol: &Tol, out: &mut Vec<Vec3>) {
    let n = c.initial_segments(t0, t1, tol.base).max(1);
    let depth = MAX_CURVE_DEPTH as i32;
    let mut prev_t = t0;
    let mut prev_p = c.eval(t0);
    out.push(prev_p);
    for i in 1..=n {
        let t = t0 + (t1 - t0) * i as f64 / n as f64;
        let p = c.eval(t);
        subdivide(c, prev_t, prev_p, t, p, tol, depth, out);
        prev_t = t;
        prev_p = p;
    }
}

/// How many halvings a view may take a curve past `MAX_CURVE_DEPTH`: twelve
/// below each first cut are enough for the model's own tolerance, and a view
/// asks for at most a thousand times finer, which is five more.
const VIEW_CURVE_DEPTH: i32 = 5;

/// Chord length appropriate for the largest feature the model could contain at
/// this tolerance: the chord of a circle of radius `extent / 2` whose sagitta
/// is `tol`.
pub(crate) fn max_segment_length(tol: f64, extent: f64) -> f64 {
    (8.0 * tol * (extent * 0.5)).sqrt().max(tol * 4.0)
}

// ---------------------------------------------------------------------------

/// Samples a bound's edges into one closed 3D ring, splitting each edge's
/// parameter range into `pieces` before sampling so a caller can force a
/// minimum resolution the chord tolerance alone would not ask for.
pub(crate) fn sample_bound(bound: &Bound, curves: &[Curve], tol: &Tol, pieces: usize) -> Vec<Vec3> {
    let mut pts: Vec<Vec3> = Vec::new();
    for e in &bound.edges {
        let c = &curves[e.curve as usize];
        for k in 0..pieces {
            let a = e.t0 + (e.t1 - e.t0) * k as f64 / pieces as f64;
            let b = e.t0 + (e.t1 - e.t0) * (k + 1) as f64 / pieces as f64;
            let mut seg = Vec::new();
            sample_edge_in(c, a, b, tol, &mut seg);
            // Drop the joint duplicated with the previous piece's end.
            let start = if pts.is_empty() { 0 } else { 1 };
            pts.extend_from_slice(&seg[start.min(seg.len())..]);
        }
    }
    // Collapse repeated samples, then close the ring explicitly. The closing
    // duplicate is what `project_loop` uses to detect a seam.
    //
    // Two edges meeting at a vertex arrive a few parts in 1e6 apart, leaving
    // a hair-thin spike the constrained triangulator refuses, so "repeated"
    // needs a real distance. A ten-thousandth of the chord tolerance (tighter
    // than `--check`'s weld): too tight and spikes survive, too loose and
    // genuinely tiny bounds vanish with their faces.
    let weld = (tol.base * 1e-4).max(1e-12);
    pts.dedup_by(|a, b| a.distance(*b) <= weld);
    if pts.len() >= 2 {
        let first = pts[0];
        if first.distance(*pts.last().unwrap()) <= weld {
            // Already back where it started: land it exactly there rather
            // than leave the gap as a spike.
            *pts.last_mut().unwrap() = first;
        } else {
            pts.push(first);
        }
    }
    pts
}

