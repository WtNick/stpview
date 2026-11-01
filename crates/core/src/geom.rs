//! Analytic and NURBS geometry: the curve and surface types that ISO 10303-42
//! defines, with evaluation, derivatives and point inversion.
//!
//! Parameterisations follow ISO 10303-42 exactly, so `project` is the true
//! inverse of `eval` for the analytic surfaces.

use crate::math::{v2, v3, Aabb, Vec2, Vec3};

pub const TWO_PI: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;

/// A parameter rectangle, `(u0, u1, v0, v1)`.
///
/// The one tuple that crosses every layer: surfaces report their domain in it,
/// `RECTANGULAR_TRIMMED_SURFACE` narrows it, a `Face` carries the narrowed one,
/// and the tessellator seeds unbounded faces from it.
pub type Domain = (f64, f64, f64, f64);

/// Right-handed orthonormal frame (a STEP `AXIS2_PLACEMENT_3D`).
#[derive(Debug, Clone, Copy)]
pub struct Axis2 {
    pub origin: Vec3,
    pub x: Vec3,
    pub y: Vec3,
    pub z: Vec3,
}

impl Axis2 {
    /// The world frame: origin at zero, axes along X, Y and Z.
    pub const IDENTITY: Axis2 = Axis2 {
        origin: Vec3::ZERO,
        x: v3(1.0, 0.0, 0.0),
        y: v3(0.0, 1.0, 0.0),
        z: v3(0.0, 0.0, 1.0),
    };

    /// Builds a frame from an origin, an axis (local Z) and a reference
    /// direction (local X), applying the STEP defaults and re-orthogonalising.
    pub fn new(origin: Vec3, axis: Option<Vec3>, ref_dir: Option<Vec3>) -> Axis2 {
        let z = axis.unwrap_or(v3(0.0, 0.0, 1.0)).normalize_or(v3(0.0, 0.0, 1.0));
        // STEP leaves the default ref_direction implementation-defined; +X is
        // the conventional choice and the fallback below covers z parallel to it.
        let raw_x = ref_dir.unwrap_or(v3(1.0, 0.0, 0.0));
        // Gram-Schmidt: STEP only requires ref_direction to be non-parallel to
        // the axis, not perpendicular to it.
        let mut x = raw_x - z * raw_x.dot(z);
        if x.length_sq() < 1e-24 {
            x = z.any_perpendicular();
        }
        let x = x.normalize();
        Axis2 {
            origin,
            x,
            y: z.cross(x),
            z,
        }
    }

    #[inline]
    pub fn to_local(self, p: Vec3) -> Vec3 {
        let d = p - self.origin;
        v3(d.dot(self.x), d.dot(self.y), d.dot(self.z))
    }

    #[inline]
    pub fn to_world(self, l: Vec3) -> Vec3 {
        self.origin + self.x * l.x + self.y * l.y + self.z * l.z
    }
}

/// A rigid placement: three rotation columns plus a translation.
///
/// Assemblies express a component's position as a pair of frames, one in the
/// component's own space and one in its parent's, so this is built by mapping
/// the first onto the second.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Xform {
    pub x: Vec3,
    pub y: Vec3,
    pub z: Vec3,
    pub origin: Vec3,
}

impl Default for Xform {
    fn default() -> Self {
        Xform::IDENTITY
    }
}

impl Xform {
    pub const IDENTITY: Xform = Xform {
        x: v3(1.0, 0.0, 0.0),
        y: v3(0.0, 1.0, 0.0),
        z: v3(0.0, 0.0, 1.0),
        origin: Vec3::ZERO,
    };

    /// The placement that carries frame `from` onto frame `to`: a point with
    /// given coordinates in `from` ends up at the same coordinates in `to`.
    pub fn between(from: &Axis2, to: &Axis2) -> Xform {
        // R = [to.x to.y to.z] * [from.x from.y from.z]^T, column by column.
        let col = |i: usize| {
            let pick = |v: Vec3| match i {
                0 => v.x,
                1 => v.y,
                _ => v.z,
            };
            to.x * pick(from.x) + to.y * pick(from.y) + to.z * pick(from.z)
        };
        let (x, y, z) = (col(0), col(1), col(2));
        // Translation chosen so `from.origin` lands on `to.origin`.
        let o = from.origin;
        Xform {
            x,
            y,
            z,
            origin: to.origin - (x * o.x + y * o.y + z * o.z),
        }
    }

    #[inline]
    pub fn point(&self, p: Vec3) -> Vec3 {
        self.origin + self.x * p.x + self.y * p.y + self.z * p.z
    }

    /// Directions ignore the translation. The rotation is orthonormal for a
    /// well-formed placement, so normals need no inverse-transpose.
    #[inline]
    pub fn dir(&self, d: Vec3) -> Vec3 {
        self.x * d.x + self.y * d.y + self.z * d.z
    }

    /// `self` applied after `inner`.
    pub fn compose(&self, inner: &Xform) -> Xform {
        Xform {
            x: self.dir(inner.x),
            y: self.dir(inner.y),
            z: self.dir(inner.z),
            origin: self.point(inner.origin),
        }
    }

    pub fn is_identity(&self) -> bool {
        *self == Xform::IDENTITY
    }

    /// Determinant of the rotation-and-scale block.
    ///
    /// The sign is what matters: a mirrored placement (negative) reverses the
    /// orientation of every triangle it carries, and whoever applies the
    /// transform to a mesh must swap the winding to match -- exactly as the
    /// tessellator does when it emits through a mirrored assembly placement.
    pub fn det(&self) -> f64 {
        self.x.cross(self.y).dot(self.z)
    }

    /// Pure translation.
    pub fn translate(by: Vec3) -> Xform {
        Xform {
            origin: by,
            ..Xform::IDENTITY
        }
    }

    /// Uniform scale about the origin. This is how a scene brings models of
    /// different units into one space: the instance transform bakes each
    /// model's millimetres-per-unit.
    pub fn scale(s: f64) -> Xform {
        Xform {
            x: v3(s, 0.0, 0.0),
            y: v3(0.0, s, 0.0),
            z: v3(0.0, 0.0, s),
            origin: Vec3::ZERO,
        }
    }

    /// Rotation about one of the world axes, in radians. Axis 0, 1, 2 = X, Y,
    /// Z; anything else is a bug and panics in tests rather than guessing.
    pub fn rotate_axis(axis: usize, angle: f64) -> Xform {
        let (s, c) = angle.sin_cos();
        let mut m = Xform::IDENTITY;
        match axis {
            0 => {
                m.y = v3(0.0, c, s);
                m.z = v3(0.0, -s, c);
            }
            1 => {
                m.x = v3(c, 0.0, -s);
                m.z = v3(s, 0.0, c);
            }
            2 => {
                m.x = v3(c, s, 0.0);
                m.y = v3(-s, c, 0.0);
            }
            _ => panic!("axis {axis} is not X, Y or Z"),
        }
        m
    }
}

// ---------------------------------------------------------------------------
// B-spline / NURBS basis
// ---------------------------------------------------------------------------

/// Expands STEP's (distinct knots, multiplicities) into a full knot vector.
pub fn expand_knots(knots: &[f64], mults: &[i64]) -> Vec<f64> {
    let total: usize = mults.iter().map(|&m| m.max(0) as usize).sum();
    let mut out = Vec::with_capacity(total);
    for (&k, &m) in knots.iter().zip(mults) {
        for _ in 0..m.max(0) {
            out.push(k);
        }
    }
    out
}

/// Index of the knot span containing `u`. `n` is the last control point index.
fn find_span(n: usize, p: usize, u: f64, knots: &[f64]) -> usize {
    if u >= knots[n + 1] {
        return n;
    }
    if u <= knots[p] {
        return p;
    }
    let (mut lo, mut hi) = (p, n + 1);
    let mut mid = (lo + hi) / 2;
    while u < knots[mid] || u >= knots[mid + 1] {
        if u < knots[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
        mid = (lo + hi) / 2;
        if mid == lo && hi <= lo + 1 {
            break;
        }
    }
    mid
}

/// Highest B-spline degree supported. Real CAD data rarely exceeds 7; the cap
/// exists so basis evaluation can use stack arrays and never allocate, which
/// matters because it runs millions of times during tessellation.
pub const MAX_DEGREE: usize = 15;

/// Non-zero basis functions at `u`, without their derivatives: the first half
/// of [`basis_and_deriv`], to the bit, for the evaluations that only want a
/// point. Writes `p + 1` values into `n`.
fn basis(span: usize, u: f64, p: usize, knots: &[f64], n: &mut [f64; MAX_DEGREE + 1]) {
    // Size the table to the degree, which is almost always cubic or less:
    // clearing the full-size table on every call is a measurable share of
    // tessellation, since this runs many times per projected point.
    if p < 4 {
        let ndu = basis_table::<4>(span, u, p, knots);
        for j in 0..=p {
            n[j] = ndu[j][p];
        }
    } else {
        let ndu = basis_table::<{ MAX_DEGREE + 1 }>(span, u, p, knots);
        for j in 0..=p {
            n[j] = ndu[j][p];
        }
    }
}

/// The triangular table of algorithm A2.3 for degree `p < N`: basis values
/// above the diagonal and knot differences below it.
#[inline]
fn basis_table<const N: usize>(span: usize, u: f64, p: usize, knots: &[f64]) -> [[f64; N]; N] {
    let mut ndu = [[0.0f64; N]; N];
    let mut left = [0.0f64; N];
    let mut right = [0.0f64; N];
    ndu[0][0] = 1.0;
    for j in 1..=p {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            let den = right[r + 1] + left[j - r];
            ndu[j][r] = den;
            let temp = if den.abs() > 1e-300 {
                ndu[r][j - 1] / den
            } else {
                0.0
            };
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    ndu
}

/// Non-zero basis functions and their first derivatives at `u`
/// (The NURBS Book, algorithm A2.3 with `nders = 1`).
///
/// Writes `p + 1` values into each of `n` and `dn`.
fn basis_and_deriv(
    span: usize,
    u: f64,
    p: usize,
    knots: &[f64],
    n: &mut [f64; MAX_DEGREE + 1],
    ders: &mut [f64; MAX_DEGREE + 1],
) {
    if p < 4 {
        derivs_from(&basis_table::<4>(span, u, p, knots), p, n, ders);
    } else {
        derivs_from(&basis_table::<{ MAX_DEGREE + 1 }>(span, u, p, knots), p, n, ders);
    }
}

/// The values and first derivatives out of a table from [`basis_table`].
#[inline]
fn derivs_from<const N: usize>(
    ndu: &[[f64; N]; N],
    p: usize,
    n: &mut [f64; MAX_DEGREE + 1],
    ders: &mut [f64; MAX_DEGREE + 1],
) {
    for j in 0..=p {
        n[j] = ndu[j][p];
        ders[j] = 0.0;
    }

    // First derivatives. A degree-0 basis is piecewise constant, so its
    // derivative is identically zero and the general formula below (which
    // indexes `p - 1`) does not apply.
    if p == 0 {
        return;
    }
    // Specialisation of A2.3 to nders == 1: the algorithm's inner `j1..j2` loop
    // is always empty for k == 1, leaving only the first and last terms.
    let pk = p - 1;
    for r in 0..=p {
        let mut d = 0.0;
        if r >= 1 {
            d += ndu[r - 1][pk] / ndu[pk + 1][r - 1];
        }
        if r <= pk {
            d -= ndu[r][pk] / ndu[pk + 1][r];
        }
        ders[r] = d * p as f64;
    }
}

/// A NURBS curve. Weights are `None` for the (common) non-rational case.
#[derive(Debug, Clone)]
pub struct BSplineCurve {
    pub degree: usize,
    pub ctrl: Vec<Vec3>,
    pub weights: Option<Vec<f64>>,
    pub knots: Vec<f64>,
    /// Domain length if the curve closes on itself, cached because `eval` needs
    /// the answer and working it out means evaluating the curve.
    period: Option<f64>,
    /// Length of the control polygon: the curve's own scale, against which
    /// "the same point" is measured.
    reach: f64,
}

impl BSplineCurve {
    pub fn new(
        degree: usize,
        ctrl: Vec<Vec3>,
        weights: Option<Vec<f64>>,
        knots: Vec<f64>,
    ) -> Self {
        let reach = ctrl.windows(2).map(|w| w[0].distance(w[1])).sum();
        let mut c = Self {
            degree,
            ctrl,
            weights,
            knots,
            period: None,
            reach,
        };
        c.period = c.closure();
        c
    }

    /// Whether the curve returns to where it started, and over what span.
    ///
    /// An oval or slot written as one closed B-spline bounds a face as a loop,
    /// not the zero-length segment its coincident vertices suggest. The
    /// `closed_curve` flag is routinely `.U.`, so the geometry is asked.
    fn closure(&self) -> Option<f64> {
        if !self.is_valid() {
            return None;
        }
        let (t0, t1) = self.domain();
        if !(t1 > t0) {
            return None;
        }
        // Compared against the control net rather than an absolute distance, so
        // the test means the same thing on a 2 mm slot and a 2 m hull plate.
        let gap = self.eval_clamped(t0).distance(self.eval_clamped(t1));
        (self.reach > 0.0 && gap <= self.reach * 1e-9).then_some(t1 - t0)
    }

    pub fn period(&self) -> Option<f64> {
        self.period
    }

    pub fn is_valid(&self) -> bool {
        let n = self.ctrl.len();
        self.degree <= MAX_DEGREE && n > self.degree && self.knots.len() == n + self.degree + 1
    }

    pub fn domain(&self) -> (f64, f64) {
        let n = self.ctrl.len() - 1;
        (self.knots[self.degree], self.knots[n + 1])
    }

    /// A closed curve is evaluated by wrapping rather than clamping. Seam
    /// unwrapping deliberately carries a parameter past the join, and clamping
    /// there collapses the far end of the loop onto a single point.
    pub fn eval(&self, u: f64) -> Vec3 {
        let (lo, _) = self.domain();
        match self.period {
            Some(p) => self.eval_clamped(lo + (u - lo).rem_euclid(p)),
            None => self.eval_clamped(u),
        }
    }

    /// `eval` with the first derivative, wrapping where the curve closes just
    /// as `eval` does.
    fn eval_with_deriv_wrapped(&self, u: f64) -> (Vec3, Vec3) {
        let (lo, _) = self.domain();
        match self.period {
            Some(p) => self.eval_with_deriv(lo + (u - lo).rem_euclid(p)),
            None => self.eval_with_deriv(u),
        }
    }

    fn eval_clamped(&self, u: f64) -> Vec3 {
        let p = self.degree;
        let n = self.ctrl.len() - 1;
        let (lo, hi) = self.domain();
        let u = u.clamp(lo, hi);
        let span = find_span(n, p, u, &self.knots);
        let mut nb = [0.0; MAX_DEGREE + 1];
        basis(span, u, p, &self.knots, &mut nb);
        let mut num = Vec3::ZERO;
        let mut den = 0.0;
        for i in 0..=p {
            let idx = span - p + i;
            let w = self.weights.as_ref().map_or(1.0, |w| w[idx]);
            num += self.ctrl[idx] * (nb[i] * w);
            den += nb[i] * w;
        }
        if den.abs() > 1e-300 {
            num / den
        } else {
            num
        }
    }

    /// Grows `b` by the control points that can pull the curve anywhere in
    /// `lo..hi`: a point's basis is non-zero between knots `i` and `i + p + 1`,
    /// and the curve over a range lies in the hull of the points live there.
    /// A closed curve whose range runs past its own seam takes them all.
    fn extend_bounds(&self, lo: f64, hi: f64, b: &mut Aabb) {
        if !self.is_valid() {
            return;
        }
        let (d0, d1) = self.domain();
        let (lo, hi) = match self.period {
            Some(p) => {
                let s = d0 + (lo - d0).rem_euclid(p);
                (s, s + (hi - lo))
            }
            None => (lo.max(d0), hi.min(d1)),
        };
        let p = self.degree;
        for (i, c) in self.ctrl.iter().enumerate() {
            if hi > d1 || (self.knots[i] < hi && self.knots[i + p + 1] > lo) {
                b.add_point(*c);
            }
        }
    }

    /// Point and first derivative, clamped to the domain.
    fn eval_with_deriv(&self, u: f64) -> (Vec3, Vec3) {
        let p = self.degree;
        let n = self.ctrl.len() - 1;
        let (lo, hi) = self.domain();
        let u = u.clamp(lo, hi);
        let span = find_span(n, p, u, &self.knots);
        let mut nb = [0.0; MAX_DEGREE + 1];
        let mut nd = [0.0; MAX_DEGREE + 1];
        basis_and_deriv(span, u, p, &self.knots, &mut nb, &mut nd);
        // Homogeneous, so the rational case follows by the quotient rule.
        let (mut a, mut ad, mut w, mut wd) = (Vec3::ZERO, Vec3::ZERO, 0.0, 0.0);
        for i in 0..=p {
            let idx = span - p + i;
            let wt = self.weights.as_ref().map_or(1.0, |w| w[idx]);
            a += self.ctrl[idx] * (nb[i] * wt);
            ad += self.ctrl[idx] * (nd[i] * wt);
            w += nb[i] * wt;
            wd += nd[i] * wt;
        }
        if w.abs() < 1e-300 {
            return (a, ad);
        }
        let pt = a / w;
        (pt, (ad - pt * wd) / w)
    }

    /// Gauss-Newton towards the point nearest `target`, from `start`.
    fn newton(&self, target: Vec3, start: f64) -> f64 {
        let (lo, hi) = self.domain();
        let mut t = start.clamp(lo, hi);
        for _ in 0..32 {
            let (c, d) = self.eval_with_deriv(t);
            let dd = d.length_sq();
            if !(dd > 1e-300) {
                break;
            }
            let next = (t + (target - c).dot(d) / dd).clamp(lo, hi);
            let moved = (next - t).abs();
            t = next;
            if moved <= (hi - lo) * 1e-15 || moved <= t.abs() * 1e-15 {
                break;
            }
        }
        t
    }

    /// The parameter of the point on the curve nearest `target`, the ends
    /// first: an end within `slack` of the target is that end.
    ///
    /// A vertex off the end of its curve may be the sloppy one (several curves
    /// end together and the vertex sits beside them) or the curve may be
    /// (the curves end at the vertex). Moving one edge onto the vertex's foot
    /// point in the first case opens a crack, so the caller passes the slack
    /// below which the end is kept regardless; a billionth of the control
    /// polygon from an end is always that end.
    ///
    /// Past the slack the nearest point wins, the end breaking ties, found as
    /// `BSplineSurface::nearest` does a dimension down: with positive weights
    /// a span lies in its control points' hull, so only spans whose box could
    /// beat the best so far get a Newton solve, from the nearest of five
    /// samples. A span with a non-positive weight has no hull and is always
    /// solved.
    pub fn param_of(&self, target: Vec3, slack: f64) -> f64 {
        let (lo, hi) = self.domain();
        let (d_lo, d_hi) = (self.eval(lo).distance(target), self.eval(hi).distance(target));
        let (end, d_end) = if d_hi < d_lo { (hi, d_hi) } else { (lo, d_lo) };
        if !(d_end > slack.max(self.reach * 1e-9)) {
            return end;
        }
        let p = self.degree;
        // Closer than the end by more than rounding, squared like the box
        // distances it is compared with.
        const CLOSER: f64 = 1.0 - 1e-9;
        let mut beat = d_end * d_end * CLOSER;
        let mut spans: Vec<(f64, usize)> = (p..self.ctrl.len())
            .filter(|&i| self.knots[i] < self.knots[i + 1])
            .map(|i| {
                let hull = &self.ctrl[i - p..=i];
                let bounded = self
                    .weights
                    .as_ref()
                    .is_none_or(|w| w[i - p..=i].iter().all(|&w| w > 0.0));
                if !bounded {
                    return (0.0, i);
                }
                let mut b = (hull[0], hull[0]);
                for q in hull {
                    b = (b.0.min(*q), b.1.max(*q));
                }
                (box_distance_sq(&b, target), i)
            })
            .filter(|&(d, _)| d < beat)
            .collect();
        spans.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let mut best = end;
        for (d, i) in spans {
            if d >= beat {
                break;
            }
            let (a, b) = (self.knots[i], self.knots[i + 1]);
            let (mut start, mut nearest) = (a, f64::INFINITY);
            for k in 0..5 {
                let t = a + (b - a) * (k as f64 + 0.5) / 5.0;
                let e = self.eval_clamped(t).distance_sq(target);
                if e < nearest {
                    (start, nearest) = (t, e);
                }
            }
            let t = self.newton(target, start);
            let e = self.eval_clamped(t).distance_sq(target);
            if e < beat {
                (best, beat) = (t, e);
            }
        }
        best
    }
}

/// A NURBS surface (tensor product).
#[derive(Debug, Clone, Default)]
pub struct BSplineSurface {
    pub u_degree: usize,
    pub v_degree: usize,
    /// Row-major: `ctrl[i * nv + j]` with `i` along u.
    pub ctrl: Vec<Vec3>,
    pub weights: Option<Vec<f64>>,
    pub nu: usize,
    pub nv: usize,
    pub u_knots: Vec<f64>,
    pub v_knots: Vec<f64>,
    /// Domain lengths in each direction if the surface closes on itself there.
    /// Filled in by [`BSplineSurface::into_surface`]; build through that rather
    /// than reading these, since `eval` depends on them being right.
    pub(crate) u_period: Option<f64>,
    pub(crate) v_period: Option<f64>,
    /// Whether the patch collapses to a point at the low and high ends of `u`.
    pub(crate) u_poles: (bool, bool),
    /// Where [`BSplineSurface::project`] looks, built the first time it is
    /// asked. Crate-visible only so the reader's `..Default::default()` can
    /// still build the struct.
    pub(crate) blocks: std::sync::OnceLock<SpanBlocks>,
}

/// Boxes around blocks of a patch's knot spans, so point inversion finds the
/// right part of a patch whose shape no fixed sample grid can see.
///
/// A NURBS patch with positive weights lies in the convex hull of its control
/// points, span by span, so a block whose box is further from the target than
/// the best answer so far is never opened.
///
/// Only the blocks are kept -- about the square root of the span count -- and
/// a span's own box is worked out when its block is opened, since a box per
/// span costs too much memory on large models.
#[derive(Debug, Clone, Default)]
pub(crate) struct SpanBlocks {
    /// Knot index of every non-empty span, along u and along v. Repeated knots
    /// leave empty intervals behind them, and those hold no surface.
    us: Vec<u32>,
    vs: Vec<u32>,
    /// Spans per block along each direction, and blocks along v.
    bu: usize,
    bv: usize,
    nbv: usize,
    /// Row-major over blocks: `boxes[a * nbv + b]` covers spans
    /// `us[a * bu..][..bu]` by `vs[b * bv..][..bv]`.
    boxes: Vec<(Vec3, Vec3)>,
    /// False when a weight is not positive. The convex hull property is then
    /// gone, no box bounds anything, and inversion falls back to sampling.
    sound: bool,
}

impl SpanBlocks {
    fn new(s: &BSplineSurface) -> SpanBlocks {
        let spans = |knots: &[f64], degree: usize, n: usize| -> Vec<u32> {
            (degree..n)
                .filter(|&i| knots[i] < knots[i + 1])
                .map(|i| i as u32)
                .collect()
        };
        let us = spans(&s.u_knots, s.u_degree, s.nu);
        let vs = spans(&s.v_knots, s.v_degree, s.nv);
        let sound = s.weights.as_ref().is_none_or(|w| w.iter().all(|&w| w > 0.0))
            && s.ctrl.iter().all(|p| p.is_finite());
        if !sound || us.is_empty() || vs.is_empty() {
            return SpanBlocks::default();
        }
        // About the square root of the span count per block, so scanning the
        // blocks and scanning inside one cost about the same; capped, since
        // opening a block builds every span box in it. Squarish in spans,
        // until one direction runs out.
        let per = ((us.len() * vs.len()) as f64).sqrt().ceil().clamp(1.0, 64.0) as usize;
        let bv = vs.len().min((per as f64).sqrt().ceil() as usize).max(1);
        let bu = us.len().min(per.div_ceil(bv)).max(1);
        let bv = vs.len().min(per.div_ceil(bu)).max(1);
        let (nbu, nbv) = (us.len().div_ceil(bu), vs.len().div_ceil(bv));
        let mut boxes = Vec::with_capacity(nbu * nbv);
        for a in 0..nbu {
            let (i0, i1) = (us[a * bu] as usize, us[((a + 1) * bu).min(us.len()) - 1] as usize);
            for b in 0..nbv {
                let (j0, j1) = (vs[b * bv] as usize, vs[((b + 1) * bv).min(vs.len()) - 1] as usize);
                boxes.push(s.hull_box(i0 - s.u_degree..=i1, j0 - s.v_degree..=j1));
            }
        }
        SpanBlocks {
            us,
            vs,
            bu,
            bv,
            nbv,
            boxes,
            sound: true,
        }
    }
}

/// Squared distance from `p` to the box `(lo, hi)`; zero inside it.
fn box_distance_sq(b: &(Vec3, Vec3), p: Vec3) -> f64 {
    let gap = |x: f64, lo: f64, hi: f64| (lo - x).max(x - hi).max(0.0);
    let (dx, dy, dz) = (gap(p.x, b.0.x, b.1.x), gap(p.y, b.0.y, b.1.y), gap(p.z, b.0.z, b.1.z));
    dx * dx + dy * dy + dz * dz
}

impl BSplineSurface {
    /// Validates the surface and works out which way, if either, it closes.
    ///
    /// A tube written as a NURBS patch is as periodic as a
    /// `CYLINDRICAL_SURFACE`; a face wrapping one has two end curves and no
    /// seam edge, and without the period the seam stitcher never runs.
    pub fn into_surface(mut self) -> Option<Surface> {
        if !self.is_valid() {
            return None;
        }
        let (u0, u1, v0, v1) = self.domain();
        // Relative to the control net, so the test reads the same on a 2 mm
        // boss and a 2 m panel.
        let scale: f64 = self.ctrl.windows(2).map(|w| w[0].distance(w[1])).sum();
        if scale > 0.0 {
            const N: usize = 4;
            let closes = |along_u: bool| {
                (0..=N).all(|k| {
                    let t = k as f64 / N as f64;
                    let (a, b) = if along_u {
                        let v = v0 + (v1 - v0) * t;
                        (self.eval_raw(u0, v), self.eval_raw(u1, v))
                    } else {
                        let u = u0 + (u1 - u0) * t;
                        (self.eval_raw(u, v0), self.eval_raw(u, v1))
                    };
                    a.distance(b) <= scale * 1e-9
                })
            };
            let (cu, cv) = (u1 > u0 && closes(true), v1 > v0 && closes(false));
            self.u_period = cu.then_some(u1 - u0);
            self.v_period = cv.then_some(v1 - v0);

            // A patch edge that collapses to a point is a pole, like the top
            // of a sphere; a face wrapping such a patch closes onto the pole
            // rather than against a second boundary.
            let pole = |u: f64| {
                let p0 = self.eval_raw(u, v0);
                (0..=N).all(|k| {
                    let v = v0 + (v1 - v0) * k as f64 / N as f64;
                    self.eval_raw(u, v).distance(p0) <= scale * 1e-9
                })
            };
            self.u_poles = (v1 > v0 && pole(u0), v1 > v0 && pole(u1));
        }
        Some(Surface::BSpline(Box::new(self)))
    }

    pub fn u_period(&self) -> Option<f64> {
        self.u_period
    }

    pub fn v_period(&self) -> Option<f64> {
        self.v_period
    }

    pub fn is_valid(&self) -> bool {
        self.u_degree <= MAX_DEGREE
            && self.v_degree <= MAX_DEGREE
            && self.nu > self.u_degree
            && self.nv > self.v_degree
            && self.ctrl.len() == self.nu * self.nv
            && self.u_knots.len() == self.nu + self.u_degree + 1
            && self.v_knots.len() == self.nv + self.v_degree + 1
    }

    pub fn domain(&self) -> Domain {
        (
            self.u_knots[self.u_degree],
            self.u_knots[self.nu],
            self.v_knots[self.v_degree],
            self.v_knots[self.nv],
        )
    }

    /// Point plus both first partial derivatives.
    ///
    /// A direction the surface closes in wraps instead of clamping. Seam
    /// unwrapping carries parameters a whole period past the join, and clamping
    /// there folds that side of the band back onto the surface's edge.
    pub fn eval_with_derivs(&self, u: f64, v: f64) -> (Vec3, Vec3, Vec3) {
        let (u, v) = self.wrap(u, v);
        self.derivs_raw(u, v)
    }

    /// Brings a closing direction's parameter back into the domain.
    fn wrap(&self, u: f64, v: f64) -> (f64, f64) {
        let (u0, _, v0, _) = self.domain();
        let u = match self.u_period {
            Some(p) => u0 + (u - u0).rem_euclid(p),
            None => u,
        };
        let v = match self.v_period {
            Some(p) => v0 + (v - v0).rem_euclid(p),
            None => v,
        };
        (u, v)
    }

    /// `derivs_raw` without the derivatives: the same sums in the same order,
    /// so the same bits, and cheaper for callers that only want points.
    fn eval_raw(&self, u: f64, v: f64) -> Vec3 {
        let (u0, u1, v0, v1) = self.domain();
        let u = u.clamp(u0, u1);
        let v = v.clamp(v0, v1);
        let p = self.u_degree;
        let q = self.v_degree;
        let uspan = find_span(self.nu - 1, p, u, &self.u_knots);
        let vspan = find_span(self.nv - 1, q, v, &self.v_knots);
        let mut nu_b = [0.0; MAX_DEGREE + 1];
        let mut nv_b = [0.0; MAX_DEGREE + 1];
        basis(uspan, u, p, &self.u_knots, &mut nu_b);
        basis(vspan, v, q, &self.v_knots, &mut nv_b);
        let (mut s, mut w) = (Vec3::ZERO, 0.0);
        for i in 0..=p {
            let ui = uspan - p + i;
            for j in 0..=q {
                let idx = ui * self.nv + vspan - q + j;
                let wt = self.weights.as_ref().map_or(1.0, |ws| ws[idx]);
                let b = nu_b[i] * nv_b[j];
                s += self.ctrl[idx] * wt * b;
                w += wt * b;
            }
        }
        if w.abs() < 1e-300 {
            return s;
        }
        s / w
    }

    fn derivs_raw(&self, u: f64, v: f64) -> (Vec3, Vec3, Vec3) {
        let (u0, u1, v0, v1) = self.domain();
        let u = u.clamp(u0, u1);
        let v = v.clamp(v0, v1);
        let p = self.u_degree;
        let q = self.v_degree;
        let uspan = find_span(self.nu - 1, p, u, &self.u_knots);
        let vspan = find_span(self.nv - 1, q, v, &self.v_knots);
        let mut nu_b = [0.0; MAX_DEGREE + 1];
        let mut nu_d = [0.0; MAX_DEGREE + 1];
        let mut nv_b = [0.0; MAX_DEGREE + 1];
        let mut nv_d = [0.0; MAX_DEGREE + 1];
        basis_and_deriv(uspan, u, p, &self.u_knots, &mut nu_b, &mut nu_d);
        basis_and_deriv(vspan, v, q, &self.v_knots, &mut nv_b, &mut nv_d);

        // Accumulate in homogeneous coordinates so the rational case works via
        // the quotient rule.
        let (mut s, mut su, mut sv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        let (mut w, mut wu, mut wv) = (0.0, 0.0, 0.0);
        for i in 0..=p {
            let ui = uspan - p + i;
            for j in 0..=q {
                let vj = vspan - q + j;
                let idx = ui * self.nv + vj;
                let wt = self.weights.as_ref().map_or(1.0, |ws| ws[idx]);
                let pw = self.ctrl[idx] * wt;
                let (b, bu, bv) = (
                    nu_b[i] * nv_b[j],
                    nu_d[i] * nv_b[j],
                    nu_b[i] * nv_d[j],
                );
                s += pw * b;
                su += pw * bu;
                sv += pw * bv;
                w += wt * b;
                wu += wt * bu;
                wv += wt * bv;
            }
        }
        if w.abs() < 1e-300 {
            return (s, su, sv);
        }
        let pt = s / w;
        // d/du (S/w) = (Su - pt * wu) / w
        let du = (su - pt * wu) / w;
        let dv = (sv - pt * wv) / w;
        (pt, du, dv)
    }

    pub fn eval(&self, u: f64, v: f64) -> Vec3 {
        let (u, v) = self.wrap(u, v);
        self.eval_raw(u, v)
    }

    /// The box around control rows `rows` by columns `cols`, which bounds the
    /// surface wherever only those points have a say. See [`SpanBlocks`].
    fn hull_box(
        &self,
        rows: std::ops::RangeInclusive<usize>,
        cols: std::ops::RangeInclusive<usize>,
    ) -> (Vec3, Vec3) {
        let mut lo = v3(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut hi = v3(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for i in rows {
            for p in &self.ctrl[i * self.nv + cols.start()..=i * self.nv + cols.end()] {
                lo = lo.min(*p);
                hi = hi.max(*p);
            }
        }
        (lo, hi)
    }

    /// Nearest point of a coarse sample grid. Only for a patch no box can
    /// bound -- a weight that is not positive, which the standard does not
    /// allow and a file can still contain -- where there is nothing better to
    /// go on.
    fn grid_seed(&self, target: Vec3) -> Vec2 {
        let (u0, u1, v0, v1) = self.domain();
        let su = (self.nu * 2).clamp(6, 20);
        let sv = (self.nv * 2).clamp(6, 20);
        let (mut bu, mut bv, mut best) = (u0, v0, f64::INFINITY);
        for i in 0..=su {
            let u = u0 + (u1 - u0) * i as f64 / su as f64;
            for j in 0..=sv {
                let v = v0 + (v1 - v0) * j as f64 / sv as f64;
                let d = self.eval(u, v).distance_sq(target);
                if d < best {
                    best = d;
                    bu = u;
                    bv = v;
                }
            }
        }
        v2(bu, bv)
    }

    /// Gauss-Newton point inversion, over the whole patch.
    ///
    /// `hint` should be the parameters of a nearby point when one is known.
    /// Boundary polylines are walked in order, so the previous point's answer
    /// is nearly always right, and when Newton from it lands on the surface
    /// nothing else is tried. When it does not -- the point is off the surface
    /// by the file's own tolerance, or the hint was on the wrong side of a
    /// fold -- the rest of the patch is searched for anything closer.
    ///
    /// A search, not a sample grid: a fixed grid is blind on a long patch
    /// (a helical groove of many turns gets several turns per sample), and
    /// Newton then settles a pitch away from the point.
    pub fn project(&self, target: Vec3, hint: Option<Vec2>) -> Vec2 {
        self.project_near(target, hint, 0.0)
    }

    /// `project`, told how near is near enough: an answer within `close` of
    /// `target` ends the search, whether Newton found it from `hint` or in the
    /// first span the search opened.
    ///
    /// Boundary curves often sit a micron or so off their surfaces, which
    /// sends many points to the search; the search is still needed, since a
    /// hint can lead onto the wrong stretch of the patch. `close` lets callers
    /// that know what nearer would be worth (an outline drawn to a tolerance
    /// passes a thousandth of it) stop early.
    pub fn project_near(&self, target: Vec3, hint: Option<Vec2>, close: f64) -> Vec2 {
        let mut best = v2(f64::NAN, f64::NAN);
        let mut best_d2 = f64::INFINITY;
        if let Some(h) = hint.filter(|h| h.x.is_finite() && h.y.is_finite()) {
            let uv = self.newton(target, h);
            let d2 = self.eval(uv.x, uv.y).distance_sq(target);
            // On the surface to a millionth of the coordinates' own size. Any
            // other part of the patch that close is below what a tolerance
            // could tell apart from this one.
            if d2 <= (1e-12 * (1.0 + target.length_sq())).max(close * close) {
                return uv;
            }
            if d2.is_finite() {
                (best, best_d2) = (uv, d2);
            }
        }
        self.nearest(target, best, best_d2, close)
    }

    /// The parameters of the point of the patch nearest `target`, given the
    /// best answer so far and its squared distance -- or of the first point
    /// found within `close` of it, which is as near as the caller cares.
    ///
    /// Every span that could hold a closer point gets a Newton solve of its
    /// own, started from the nearest of a few samples across it; every other
    /// span is ruled out by its box without evaluating anything. Blocks and
    /// spans are opened nearest first, so the first solve usually sets a bound
    /// that closes everything else.
    ///
    /// The answer in hand wins a tie. A point on the seam of a closed patch is
    /// as near one side as the other, and the side the loop was already on is
    /// the one that keeps its outline in one piece.
    fn nearest(&self, target: Vec3, mut best: Vec2, mut best_d2: f64, close: f64) -> Vec2 {
        // Closer by more than rounding, in squared distance.
        const CLOSER: f64 = 1.0 - 1e-9;
        let ix = self.blocks.get_or_init(|| SpanBlocks::new(self));
        if !ix.sound {
            let uv = self.newton(target, self.grid_seed(target));
            let d2 = self.eval(uv.x, uv.y).distance_sq(target);
            return if d2 < best_d2 * CLOSER || !best.x.is_finite() { uv } else { best };
        }
        let (p, q) = (self.u_degree, self.v_degree);
        let mut blocks: Vec<(f64, usize)> = ix
            .boxes
            .iter()
            .enumerate()
            .map(|(k, b)| (box_distance_sq(b, target), k))
            .filter(|&(d, _)| d < best_d2)
            .collect();
        blocks.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let mut spans: Vec<(f64, usize, usize)> = Vec::new();
        for &(d, k) in &blocks {
            if d >= best_d2 {
                break;
            }
            let (a, b) = (k / ix.nbv, k % ix.nbv);
            spans.clear();
            for &i in &ix.us[a * ix.bu..((a + 1) * ix.bu).min(ix.us.len())] {
                for &j in &ix.vs[b * ix.bv..((b + 1) * ix.bv).min(ix.vs.len())] {
                    let (i, j) = (i as usize, j as usize);
                    let d = box_distance_sq(&self.hull_box(i - p..=i, j - q..=j), target);
                    if d < best_d2 {
                        spans.push((d, i, j));
                    }
                }
            }
            spans.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            for &(d, i, j) in &spans {
                if d >= best_d2 {
                    break;
                }
                let (ua, ub) = (self.u_knots[i], self.u_knots[i + 1]);
                let (va, vb) = (self.v_knots[j], self.v_knots[j + 1]);
                // Newton has already settled inside this span once, and would
                // only settle there again.
                if ua < best.x && best.x < ub && va < best.y && best.y < vb {
                    continue;
                }
                let mut start = v2(ua, va);
                let mut nearest = f64::INFINITY;
                for fu in [1.0 / 6.0, 0.5, 5.0 / 6.0] {
                    for fv in [1.0 / 6.0, 0.5, 5.0 / 6.0] {
                        let t = v2(ua + (ub - ua) * fu, va + (vb - va) * fv);
                        let e = self.eval_raw(t.x, t.y).distance_sq(target);
                        if e < nearest {
                            nearest = e;
                            start = t;
                        }
                    }
                }
                let uv = self.newton(target, start);
                let e = self.eval(uv.x, uv.y).distance_sq(target);
                if e < best_d2 * CLOSER || !best.x.is_finite() {
                    best = uv;
                    best_d2 = e;
                    if best_d2 <= close * close {
                        return best;
                    }
                }
            }
        }
        if !(best.x.is_finite() && best.y.is_finite()) {
            // A target nothing could be measured against. Any answer on the
            // patch beats one that is not a number.
            return self.newton(target, self.grid_seed(target));
        }
        best
    }

    fn newton(&self, target: Vec3, start: Vec2) -> Vec2 {
        let (u0, u1, v0, v1) = self.domain();
        let (mut u, mut v) = (start.x.clamp(u0, u1), start.y.clamp(v0, v1));
        for _ in 0..24 {
            let (s, du, dv) = self.eval_with_derivs(u, v);
            let r = target - s;
            if r.length_sq() < 1e-24 {
                break;
            }
            // Solve [a b; b c] [x y]' = [d e]', the normal equations of
            // r ~= Su*x + Sv*y.
            let (a, b, c) = (du.dot(du), du.dot(dv), dv.dot(dv));
            let (d, e) = (r.dot(du), r.dot(dv));
            let det = a * c - b * b;
            if det.abs() < 1e-300 {
                break;
            }
            let x = (d * c - e * b) / det;
            let y = (a * e - b * d) / det;
            let nu_ = (u + x).clamp(u0, u1);
            let nv_ = (v + y).clamp(v0, v1);
            let moved = (nu_ - u).abs() + (nv_ - v).abs();
            u = nu_;
            v = nv_;
            if moved < 1e-14 * ((u1 - u0).abs() + (v1 - v0).abs() + 1.0) {
                break;
            }
        }
        v2(u, v)
    }
}

// ---------------------------------------------------------------------------
// Curves
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Curve {
    Line { origin: Vec3, dir: Vec3 },
    Circle { ax: Axis2, r: f64 },
    Ellipse { ax: Axis2, a: f64, b: f64 },
    BSpline(Box<BSplineCurve>),
}

impl Curve {
    /// What the curve holds beyond its own enum, in bytes: a B-spline's
    /// control polygon, weights and knots.
    pub fn heap_bytes(&self) -> usize {
        match self {
            Curve::BSpline(c) => {
                std::mem::size_of::<BSplineCurve>()
                    + c.ctrl.capacity() * std::mem::size_of::<Vec3>()
                    + c.weights.as_ref().map_or(0, |w| w.capacity() * 8)
                    + c.knots.capacity() * 8
            }
            _ => 0,
        }
    }

    pub fn eval(&self, t: f64) -> Vec3 {
        match self {
            Curve::Line { origin, dir } => *origin + *dir * t,
            Curve::Circle { ax, r } => {
                ax.origin + ax.x * (r * t.cos()) + ax.y * (r * t.sin())
            }
            Curve::Ellipse { ax, a, b } => {
                ax.origin + ax.x * (a * t.cos()) + ax.y * (b * t.sin())
            }
            Curve::BSpline(c) => c.eval(t),
        }
    }

    /// The point at `t` and the first derivative there.
    pub fn eval_with_deriv(&self, t: f64) -> (Vec3, Vec3) {
        match self {
            Curve::Line { origin, dir } => (*origin + *dir * t, *dir),
            Curve::Circle { ax, r } => (
                ax.origin + ax.x * (r * t.cos()) + ax.y * (r * t.sin()),
                ax.y * (r * t.cos()) - ax.x * (r * t.sin()),
            ),
            Curve::Ellipse { ax, a, b } => (
                ax.origin + ax.x * (a * t.cos()) + ax.y * (b * t.sin()),
                ax.y * (b * t.cos()) - ax.x * (a * t.sin()),
            ),
            Curve::BSpline(c) => c.eval_with_deriv_wrapped(t),
        }
    }

    /// Grows `b` to take in the curve between `t0` and `t1`, given either way
    /// round.
    ///
    /// The vertices alone can miss a closed edge entirely (a torus is often
    /// two whole circles through one vertex). Exact for the conics, which are
    /// extreme along a world axis where their tangent is square to it. A
    /// B-spline is bounded, slightly generously, by the hull of the control
    /// points whose basis reaches into the range.
    pub fn extend_bounds(&self, t0: f64, t1: f64, b: &mut Aabb) {
        let (lo, hi) = (t0.min(t1), t0.max(t1));
        if !(lo.is_finite() && hi.is_finite()) {
            return;
        }
        b.add_point(self.eval(lo));
        b.add_point(self.eval(hi));
        let conic = |ax: &Axis2, a: f64, bb: f64, b: &mut Aabb| {
            let (xs, ys) = ([ax.x.x, ax.x.y, ax.x.z], [ax.y.x, ax.y.y, ax.y.z]);
            for k in 0..3 {
                let at = (ys[k] * bb).atan2(xs[k] * a);
                for t in [at, at + PI] {
                    // The first turn of `t` at or after `lo`.
                    let t = lo + (t - lo).rem_euclid(TWO_PI);
                    if t <= hi {
                        b.add_point(self.eval(t));
                    }
                }
            }
        };
        match self {
            Curve::Line { .. } => {}
            Curve::Circle { ax, r } => conic(ax, *r, *r, b),
            Curve::Ellipse { ax, a, b: minor } => conic(ax, *a, *minor, b),
            Curve::BSpline(c) => c.extend_bounds(lo, hi, b),
        }
    }

    /// Period in the curve parameter, for closed analytic curves.
    pub fn period(&self) -> Option<f64> {
        match self {
            Curve::Circle { .. } | Curve::Ellipse { .. } => Some(TWO_PI),
            Curve::BSpline(c) => c.period(),
            _ => None,
        }
    }

    /// Parameter of the point on the curve nearest `p`.
    pub fn param_of(&self, p: Vec3) -> f64 {
        self.param_of_vertex(p, 0.0)
    }

    /// The same, for a vertex bounding an edge on this curve: one within
    /// `slack` of an end of a B-spline is taken to be that end. See
    /// [`BSplineCurve::param_of`] for why, and for the rest of it.
    pub fn param_of_vertex(&self, p: Vec3, slack: f64) -> f64 {
        match self {
            Curve::Line { origin, dir } => {
                let dd = dir.length_sq();
                if dd < 1e-300 {
                    0.0
                } else {
                    (p - *origin).dot(*dir) / dd
                }
            }
            Curve::Circle { ax, .. } => {
                let l = ax.to_local(p);
                l.y.atan2(l.x)
            }
            Curve::Ellipse { ax, a, b } => {
                let l = ax.to_local(p);
                (l.y / b.max(1e-300)).atan2(l.x / a.max(1e-300))
            }
            Curve::BSpline(c) => c.param_of(p, slack),
        }
    }

    /// A starting subdivision count that is already close for analytic curves;
    /// adaptive refinement handles the rest.
    pub fn initial_segments(&self, t0: f64, t1: f64, tol: f64) -> usize {
        let span = (t1 - t0).abs();
        match self {
            Curve::Line { .. } => 1,
            Curve::Circle { r, .. } => arc_segments(*r, span, tol),
            Curve::Ellipse { a, b, .. } => arc_segments(a.max(*b), span, tol),
            Curve::BSpline(c) => {
                let (lo, hi) = c.domain();
                let frac = if hi > lo { span / (hi - lo) } else { 1.0 };
                ((c.ctrl.len() as f64 * 2.0 * frac).ceil() as usize).clamp(4, 200)
            }
        }
    }
}

/// Widest angle a single chord may subtend, whatever the tolerance says. The
/// sagitta is then `r(1 - cos(angle/2))`, about 7.6% of the radius.
const MAX_CHORD_ANGLE: f64 = PI / 4.0;

/// Segments needed so a polyline chord stays within `tol` of an arc of radius
/// `r` spanning `span` radians.
fn arc_segments(r: f64, span: f64, tol: f64) -> usize {
    let span = span.abs();
    if span <= 0.0 {
        return 1;
    }
    // However coarse the tolerance, no chord may subtend more than
    // `MAX_CHORD_ANGLE`. A sub-tolerance feature must still come out as a
    // polygon with area, or the face it bounds is lost; small fillets in
    // large parts make this the normal case. The polygon must also stay close
    // to the arc relative to its radius, or features near the boundary (small
    // holes in a small disc) poke through the outline.
    let floor = (span / MAX_CHORD_ANGLE).ceil().max(1.0) as usize;
    if r <= tol {
        return floor;
    }
    // Sagitta of a chord subtending theta is r(1 - cos(theta/2)).
    let ratio = 1.0 - (tol / r).min(1.0);
    let theta = 2.0 * ratio.clamp(-1.0, 1.0).acos();
    if theta <= 1e-9 {
        return 512;
    }
    ((span / theta).ceil() as usize).clamp(floor, 512.max(floor))
}

// ---------------------------------------------------------------------------
// Surfaces
// ---------------------------------------------------------------------------

/// Where along a swept surface's curve a point lies: the parameter in `lo..=hi`
/// at which the curve comes nearest it. `res(t)` gives what is left to cover
/// at `t` -- the point less the curve's, in whatever plane the sweep leaves the
/// curve in -- and the curve's tangent in that plane.
///
/// From `hint`, the answer for the previous point along an outline, by
/// Gauss-Newton, taken when it lands on the surface to a millionth of the
/// point's own size (the test `BSplineSurface::project` makes). Otherwise the
/// whole range is sampled and descended from too, and the nearer answer kept:
/// the hint's on a tie, since staying on the loop's side of a seam keeps its
/// outline in one piece.
fn invert_swept(
    lo: f64,
    hi: f64,
    reach: f64,
    hint: Option<f64>,
    res: impl Fn(f64) -> (Vec3, Vec3),
) -> f64 {
    // Closer by more than rounding, in squared distance.
    const CLOSER: f64 = 1.0 - 1e-9;
    const COARSE: usize = 48;
    let (lo, hi) = (lo.min(hi), lo.max(hi));
    let mut best = f64::NAN;
    let mut best_e = f64::INFINITY;
    if let Some(h) = hint.filter(|h| h.is_finite()) {
        let (t, e) = descend(h.clamp(lo, hi), lo, hi, &res);
        if e <= 1e-12 * (1.0 + reach * reach) {
            return t;
        }
        (best, best_e) = (t, e);
    }
    let (mut start, mut start_e) = (lo, f64::INFINITY);
    for i in 0..=COARSE {
        let t = lo + (hi - lo) * i as f64 / COARSE as f64;
        let e = res(t).0.length_sq();
        if e < start_e {
            (start, start_e) = (t, e);
        }
    }
    let (t, e) = descend(start, lo, hi, &res);
    if e < best_e * CLOSER || !best.is_finite() { t } else { best }
}

/// Gauss-Newton from `t` to the parameter in `lo..=hi` nearest the point
/// `res` measures from, and its distance squared: each step is what is left to
/// cover, along the tangent, over the tangent's length squared. A step that
/// does not bring the point nearer is halved until it does, so this settles
/// into a nearest point rather than circling one.
fn descend(mut t: f64, lo: f64, hi: f64, res: impl Fn(f64) -> (Vec3, Vec3)) -> (f64, f64) {
    let (mut r, mut d) = res(t);
    let mut e = r.length_sq();
    for _ in 0..32 {
        let dd = d.length_sq();
        if !(dd > 1e-300 && e > 0.0) {
            break;
        }
        let mut step = r.dot(d) / dd;
        let mut took = false;
        for _ in 0..24 {
            let next = (t + step).clamp(lo, hi);
            if next == t {
                break;
            }
            let (nr, nd) = res(next);
            let ne = nr.length_sq();
            if ne < e {
                let moved = (next - t).abs();
                (t, r, d, e) = (next, nr, nd, ne);
                took = true;
                if moved <= 1e-14 * ((hi - lo) + t.abs()) {
                    return (t, e);
                }
                break;
            }
            step *= 0.5;
        }
        if !took {
            break;
        }
    }
    (t, e)
}

/// A curve swept around an axis. `u` is the angle turned through and `v` is the
/// meridian's own parameter, following ISO 10303-42. Translators often use it
/// for ordinary cones and cylinders.
#[derive(Debug, Clone)]
pub struct RevolutionSurface {
    pub meridian: Curve,
    /// Parameter range of the meridian, after any trimming.
    pub v0: f64,
    pub v1: f64,
    pub origin: Vec3,
    /// Unit axis direction.
    pub dir: Vec3,
}

impl RevolutionSurface {
    /// Turns `p` about the axis by `u`.
    fn turn(&self, p: Vec3, u: f64) -> Vec3 {
        let w = p - self.origin;
        let z = w.dot(self.dir);
        let r = w - self.dir * z;
        self.origin + self.dir * z + r * u.cos() + self.dir.cross(r) * u.sin()
    }

    /// Rotates a free vector about the axis (Rodrigues).
    fn turn_dir(&self, t: Vec3, u: f64) -> Vec3 {
        let (c, s) = (u.cos(), u.sin());
        t * c + self.dir.cross(t) * s + self.dir * (self.dir.dot(t) * (1.0 - c))
    }

    /// Axial and radial coordinates of a point, in the meridian half-plane.
    fn meridian_coords(&self, p: Vec3) -> (f64, f64) {
        let w = p - self.origin;
        let z = w.dot(self.dir);
        (z, (w - self.dir * z).length())
    }

    /// As with an extrusion, `v` is not clamped: the meridian evaluates
    /// sensibly outside its stated range, and clamping would fold the surface.
    pub fn eval(&self, u: f64, v: f64) -> Vec3 {
        self.turn(self.meridian.eval(v), u)
    }

    fn deriv_v(&self, v: f64) -> Vec3 {
        let h = (self.v1 - self.v0).abs().max(1e-12) * 1e-6;
        (self.meridian.eval(v + h) - self.meridian.eval(v - h)) / (2.0 * h)
    }

    /// Finds the meridian parameter whose axial and radial coordinates match,
    /// which is what inverting this surface comes down to: the nearest point of
    /// the meridian, seen in its own half-plane.
    fn invert_meridian(&self, p: Vec3, z: f64, radius: f64, hint: Option<f64>) -> f64 {
        invert_swept(self.v0, self.v1, p.length(), hint, |v| {
            let (c, dc) = self.meridian.eval_with_deriv(v);
            let w = c - self.origin;
            let cz = w.dot(self.dir);
            let out = w - self.dir * cz;
            let cr = out.length();
            // How fast the meridian leaves the axis. Where it touches the axis
            // that has no one value, and the axial part of the step has to do.
            let dr = if cr > 0.0 {
                out.dot(dc - self.dir * dc.dot(self.dir)) / cr
            } else {
                0.0
            };
            (v3(z - cz, radius - cr, 0.0), v3(dc.dot(self.dir), dr, 0.0))
        })
    }

    pub fn project(&self, p: Vec3, hint: Option<Vec2>) -> Projection {
        let (z, radius) = self.meridian_coords(p);
        let v = self.invert_meridian(p, z, radius, hint.map(|h| h.y));
        let c = self.meridian.eval(v);
        let cw = c - self.origin;
        let cr = cw - self.dir * cw.dot(self.dir);
        let w = p - self.origin;
        let r = w - self.dir * w.dot(self.dir);
        // Signed angle from the meridian's radial direction round to the point's.
        let u = self.dir.dot(cr.cross(r)).atan2(cr.dot(r));
        Projection {
            uv: v2(u, v),
            // On the axis every angle is the same point, exactly like a pole.
            u_degenerate: radius <= 1e-9 * (radius + cr.length() + 1.0),
        }
    }

    pub fn normal(&self, u: f64, v: f64) -> Option<Vec3> {
        let s = self.eval(u, v);
        let su = self.dir.cross(s - self.origin);
        let sv = self.turn_dir(self.deriv_v(v), u);
        let n = su.cross(sv);
        (n.length_sq() > 1e-30 && n.is_finite()).then(|| n.normalize())
    }

    /// Largest distance from the axis over the meridian.
    fn max_radius(&self) -> f64 {
        let mut r: f64 = 0.0;
        for i in 0..=32 {
            let v = self.v0 + (self.v1 - self.v0) * i as f64 / 32.0;
            r = r.max(self.meridian_coords(self.meridian.eval(v)).1);
        }
        r
    }
}

/// A curve translated along a vector. `u` is the profile's own parameter and
/// `v` scales the extrusion vector, following ISO 10303-42.
#[derive(Debug, Clone)]
pub struct ExtrusionSurface {
    pub profile: Curve,
    /// Parameter range of the profile, after any trimming.
    pub u0: f64,
    pub u1: f64,
    /// Extrusion vector, magnitude included: `v = 1` is one full extrusion.
    pub dir: Vec3,
}

impl ExtrusionSurface {
    /// `u` is deliberately not clamped to the profile's range. Seam unwrapping
    /// carries parameters past the join of a closed profile, and clamping there
    /// would fold the surface back on itself. Every curve evaluates sensibly
    /// outside its stated range: the periodic ones wrap, and a B-spline clamps
    /// on its own.
    pub fn eval(&self, u: f64, v: f64) -> Vec3 {
        self.profile.eval(u) + self.dir * v
    }

    fn deriv_u(&self, u: f64) -> Vec3 {
        let h = (self.u1 - self.u0).abs().max(1e-12) * 1e-6;
        (self.profile.eval(u + h) - self.profile.eval(u - h)) / (2.0 * h)
    }

    pub fn project(&self, p: Vec3, hint: Option<Vec2>) -> Projection {
        let axis = self.dir.normalize_or(v3(0.0, 0.0, 1.0));
        // The profile as seen along the extrusion, where the ruling through p
        // is a point: the parameter we want is where the profile passes it.
        let flat = |w: Vec3| w - axis * w.dot(axis);
        let u = invert_swept(self.u0, self.u1, p.length(), hint.map(|h| h.x), |u| {
            let (c, dc) = self.profile.eval_with_deriv(u);
            (flat(p - c), flat(dc))
        });
        // How far along the extrusion, in units of the vector itself.
        let len_sq = self.dir.length_sq();
        let v = if len_sq > 1e-300 {
            (p - self.profile.eval(u)).dot(self.dir) / len_sq
        } else {
            0.0
        };
        Projection {
            uv: v2(u, v),
            u_degenerate: false,
        }
    }

    pub fn normal(&self, u: f64, _v: f64) -> Option<Vec3> {
        let n = self.deriv_u(u).cross(self.dir);
        (n.length_sq() > 1e-30 && n.is_finite()).then(|| n.normalize())
    }
}

#[derive(Debug, Clone)]
pub enum Surface {
    Plane(Axis2),
    Cylinder { ax: Axis2, r: f64 },
    /// `r` is the radius in the z = 0 plane; the radius grows by `tan(half_angle)` per unit v.
    Cone { ax: Axis2, r: f64, half_angle: f64 },
    Sphere { ax: Axis2, r: f64 },
    /// `inner` selects the vortex sheet of a *degenerate* torus -- one whose
    /// tube is fatter than the ring it follows, so it reaches past the axis and
    /// cuts through itself. The two sheets share this one parametrisation and
    /// differ only in which half of `v` they occupy, but the halves overlap
    /// when seen from outside, so inverting a point needs to be told which.
    Torus {
        ax: Axis2,
        major: f64,
        minor: f64,
        inner: bool,
    },
    BSpline(Box<BSplineSurface>),
    Revolution(Box<RevolutionSurface>),
    Extrusion(Box<ExtrusionSurface>),
}

/// Result of inverting a surface: the parameters, plus a flag marking points
/// where `u` is undefined (cone apex, sphere pole).
#[derive(Debug, Clone, Copy)]
pub struct Projection {
    pub uv: Vec2,
    pub u_degenerate: bool,
}

impl Surface {
    /// What the surface holds beyond its own enum, in bytes: a B-spline's
    /// control net, weights and knots, or the curve a swept surface sweeps.
    pub fn heap_bytes(&self) -> usize {
        match self {
            Surface::BSpline(s) => {
                std::mem::size_of::<BSplineSurface>()
                    + s.ctrl.capacity() * std::mem::size_of::<Vec3>()
                    + s.weights.as_ref().map_or(0, |w| w.capacity() * 8)
                    + (s.u_knots.capacity() + s.v_knots.capacity()) * 8
            }
            Surface::Revolution(s) => std::mem::size_of::<RevolutionSurface>() + s.meridian.heap_bytes(),
            Surface::Extrusion(s) => std::mem::size_of::<ExtrusionSurface>() + s.profile.heap_bytes(),
            _ => 0,
        }
    }

    pub fn eval(&self, u: f64, v: f64) -> Vec3 {
        match self {
            Surface::Plane(ax) => ax.to_world(v3(u, v, 0.0)),
            Surface::Cylinder { ax, r } => {
                ax.origin + ax.x * (r * u.cos()) + ax.y * (r * u.sin()) + ax.z * v
            }
            Surface::Cone { ax, r, half_angle } => {
                let rad = r + v * half_angle.tan();
                ax.origin + ax.x * (rad * u.cos()) + ax.y * (rad * u.sin()) + ax.z * v
            }
            Surface::Sphere { ax, r } => {
                let (cv, sv) = (v.cos(), v.sin());
                ax.origin + ax.x * (r * cv * u.cos()) + ax.y * (r * cv * u.sin()) + ax.z * (r * sv)
            }
            Surface::Torus {
                ax, major, minor, ..
            } => {
                let rad = major + minor * v.cos();
                ax.origin
                    + ax.x * (rad * u.cos())
                    + ax.y * (rad * u.sin())
                    + ax.z * (minor * v.sin())
            }
            Surface::BSpline(s) => s.eval(u, v),
            Surface::Revolution(s) => s.eval(u, v),
            Surface::Extrusion(s) => s.eval(u, v),
        }
    }

    /// Unit normal, oriented along `du x dv`. Returns `None` at singular points.
    pub fn normal(&self, u: f64, v: f64) -> Option<Vec3> {
        let n = match self {
            Surface::Plane(ax) => ax.z,
            Surface::Cylinder { ax, .. } => ax.x * u.cos() + ax.y * u.sin(),
            Surface::Cone { ax, r, half_angle } => {
                let rad = r + v * half_angle.tan();
                if rad.abs() < 1e-12 {
                    return None; // apex
                }
                let radial = ax.x * u.cos() + ax.y * u.sin();
                // Outward normal tilts away from the axis by the half angle.
                (radial * half_angle.cos() - ax.z * half_angle.sin()).normalize()
                    * rad.signum()
            }
            Surface::Sphere { ax, .. } => {
                let cv = v.cos();
                ax.x * (cv * u.cos()) + ax.y * (cv * u.sin()) + ax.z * v.sin()
            }
            Surface::Torus {
                ax, minor, major, ..
            } => {
                let rad = major + minor * v.cos();
                if rad.abs() < 1e-12 {
                    return None;
                }
                (ax.x * u.cos() + ax.y * u.sin()) * v.cos() + ax.z * v.sin()
            }
            Surface::BSpline(s) => {
                let (_, du, dv) = s.eval_with_derivs(u, v);
                let n = du.cross(dv);
                // Derivatives within about half a degree of parallel cross to
                // noise: the patch is folded nearly flat there and its normal
                // swings wildly. Return none; the edge overlay draws an edge it
                // cannot measure, and a vertex takes its normal from the
                // triangles around it, as at a pole.
                if n.length_sq() < 1e-30 || n.length_sq() < 1e-4 * du.length_sq() * dv.length_sq() {
                    return None;
                }
                n
            }
            Surface::Revolution(s) => return s.normal(u, v),
            Surface::Extrusion(s) => return s.normal(u, v),
        };
        if n.length_sq() < 1e-30 || !n.is_finite() {
            None
        } else {
            Some(n.normalize())
        }
    }

    /// Inverts `eval`: maps a 3D point on (or near) the surface back to (u, v).
    ///
    /// `hint` is the answer for a nearby point, if one is known. The NURBS and
    /// swept surfaces start from it; a NURBS patch still searches the rest of
    /// itself whenever the answer it leads to is off the surface.
    pub fn project(&self, p: Vec3, hint: Option<Vec2>) -> Projection {
        self.project_near(p, hint, 0.0)
    }

    /// `project`, taking a NURBS patch's answer from `hint` whenever it is
    /// within `close` of `p`; see [`BSplineSurface::project_near`]. Nothing
    /// else searches, so nothing else has anything to skip.
    pub fn project_near(&self, p: Vec3, hint: Option<Vec2>, close: f64) -> Projection {
        let plain = |uv| Projection {
            uv,
            u_degenerate: false,
        };
        match self {
            Surface::Plane(ax) => {
                let l = ax.to_local(p);
                plain(v2(l.x, l.y))
            }
            Surface::Cylinder { ax, r } => {
                let l = ax.to_local(p);
                Projection {
                    uv: v2(l.y.atan2(l.x), l.z),
                    u_degenerate: (l.x * l.x + l.y * l.y).sqrt() < 1e-9 * r.max(1.0),
                }
            }
            Surface::Cone { ax, r, half_angle } => {
                let l = ax.to_local(p);
                let rad = (l.x * l.x + l.y * l.y).sqrt();
                // v is the height along the axis; recovering it from z is exact.
                let v = l.z;
                let expected = r + v * half_angle.tan();
                Projection {
                    uv: v2(l.y.atan2(l.x), v),
                    u_degenerate: rad < 1e-7 * (expected.abs() + r.abs() + 1.0),
                }
            }
            Surface::Sphere { ax, r } => {
                let l = ax.to_local(p);
                let rr = r.max(1e-300);
                let v = (l.z / rr).clamp(-1.0, 1.0).asin();
                let rad = (l.x * l.x + l.y * l.y).sqrt();
                Projection {
                    uv: v2(l.y.atan2(l.x), v),
                    u_degenerate: rad < 1e-7 * rr,
                }
            }
            Surface::Torus {
                ax,
                major,
                minor,
                inner,
            } => {
                let l = ax.to_local(p);
                let rad = (l.x * l.x + l.y * l.y).sqrt();
                let mut u = l.y.atan2(l.x);
                let mut v = if *inner {
                    // Past the axis the tube comes out the far side, so the
                    // point stands half a turn round from its own `u` and its
                    // distance from the axis counts as negative.
                    u += if u > 0.0 { -PI } else { PI };
                    l.z.atan2(-rad - major)
                } else {
                    l.z.atan2(rad - major)
                };
                // The vortex sheet straddles v = pi, so measure it over 0..2pi
                // and keep the branch cut out where the surface is not.
                if *inner && v < 0.0 {
                    v += TWO_PI;
                }
                Projection {
                    uv: v2(u, v),
                    u_degenerate: rad < 1e-9 * (major.abs() + minor.abs() + 1.0),
                }
            }
            Surface::BSpline(s) => plain(s.project_near(p, hint, close)),
            Surface::Revolution(s) => s.project(p, hint),
            Surface::Extrusion(s) => s.project(p, hint),
        }
    }

    /// Period of `u`, for surfaces that close on themselves.
    ///
    /// This is what drives seam handling, so a swept surface has to report it
    /// too: a revolution always closes in `u`, and an extrusion closes exactly
    /// when the profile it sweeps does.
    pub fn u_period(&self) -> Option<f64> {
        match self {
            Surface::Cylinder { .. }
            | Surface::Cone { .. }
            | Surface::Sphere { .. }
            | Surface::Torus { .. }
            | Surface::Revolution(_) => Some(TWO_PI),
            Surface::BSpline(s) => s.u_period(),
            Surface::Extrusion(s) => {
                let period = s.profile.period()?;
                // Only if the profile is swept through a whole turn; a trimmed
                // arc leaves the surface open.
                ((s.u1 - s.u0).abs() >= period - 1e-9).then_some(period)
            }
            _ => None,
        }
    }

    pub fn v_period(&self) -> Option<f64> {
        match self {
            // A degenerate torus is pinched shut where the tube crosses the
            // axis, so v runs pole to pole as on a sphere and never wraps.
            Surface::Torus { major, minor, .. } => {
                (minor.abs() <= major.abs()).then_some(TWO_PI)
            }
            Surface::BSpline(s) => s.v_period(),
            // A revolution closes in `v` exactly when its meridian does, the
            // mirror of the extrusion rule in `u_period`. A whole circle
            // revolved is a torus written the general way; without the period
            // a loop round its tube encloses no area and the face is dropped.
            Surface::Revolution(s) => {
                let period = s.meridian.period()?;
                // Only if the whole period is swept. A domed end cap is a
                // trimmed arc revolved, and leaves the surface open in `v`.
                ((s.v1 - s.v0).abs() >= period - 1e-9).then_some(period)
            }
            _ => None,
        }
    }

    /// True for surfaces that are exactly reproduced by any triangulation of
    /// their boundary, so refinement can be skipped entirely.
    pub fn is_flat(&self) -> bool {
        matches!(self, Surface::Plane(_))
    }

    /// Short names for diagnostics, indexed by [`Surface::kind_index`].
    pub const KINDS: [&'static str; 8] = [
        "plane",
        "cylinder",
        "cone",
        "sphere",
        "torus",
        "bspline",
        "revolution",
        "extrusion",
    ];

    /// Which of [`Surface::KINDS`] this is. Reporting failures per surface kind
    /// is the quickest way to tell a broken reader from broken geometry: one
    /// kind failing wholesale is a bug here, a scatter across every kind is the
    /// file.
    pub fn kind_index(&self) -> usize {
        match self {
            Surface::Plane(_) => 0,
            Surface::Cylinder { .. } => 1,
            Surface::Cone { .. } => 2,
            Surface::Sphere { .. } => 3,
            Surface::Torus { .. } => 4,
            Surface::BSpline(_) => 5,
            Surface::Revolution(_) => 6,
            Surface::Extrusion(_) => 7,
        }
    }

    /// Largest step in each parameter that still keeps the surface within `tol`
    /// of the chords across it, over the parameter box `lo..hi`. `INFINITY`
    /// means the surface is straight in that direction, so any step is exact.
    ///
    /// This tells refinement when to stop. On a trimmed face most boundary
    /// edges are locked (splitting them would crack the join to the
    /// neighbour), and splitting only interior diagonals does not reduce the
    /// measured error, so without a step bound refinement never converges.
    pub fn uv_step(&self, tol: f64, lo: Vec2, hi: Vec2) -> (f64, f64) {
        // Angular step whose sagitta on a circle of radius r stays within tol.
        // Deliberately the same formula the edge sampler uses, so a boundary
        // sampled to tolerance already satisfies this bound.
        let ang = |r: f64| {
            let r = r.abs();
            if r <= tol {
                f64::INFINITY
            } else {
                2.0 * (1.0 - (tol / r).min(1.0)).clamp(-1.0, 1.0).acos()
            }
        };
        match self {
            Surface::Plane(_) => (f64::INFINITY, f64::INFINITY),
            // A cylinder and a cone are both ruled: exact along v.
            Surface::Cylinder { r, .. } => (ang(*r), f64::INFINITY),
            Surface::Cone { r, half_angle, .. } => {
                let t = half_angle.tan();
                let widest = (r + lo.y * t).abs().max((r + hi.y * t).abs());
                (ang(widest), f64::INFINITY)
            }
            // Tightest at the equator, where a parallel has the full radius.
            Surface::Sphere { r, .. } => (ang(*r), ang(*r)),
            Surface::Torus {
                major,
                minor,
                inner,
                ..
            } => {
                // Widest parallel: the outside of the ring, or -- on the vortex
                // sheet, which lies inside the axis -- the far wall of the tube.
                let widest = if *inner {
                    (minor.abs() - major.abs()).abs()
                } else {
                    major.abs() + minor.abs()
                };
                (ang(widest), ang(*minor))
            }
            Surface::BSpline(s) => {
                // No closed form. Fall back to a fraction of the control net
                // spacing and let the deviation test refine from there.
                let (u0, u1, v0, v1) = s.domain();
                (
                    (u1 - u0) / (4 * s.nu.max(2)) as f64,
                    (v1 - v0) / (4 * s.nv.max(2)) as f64,
                )
            }
            Surface::Revolution(s) => {
                // Around the axis this is a circle of the widest radius. Along
                // the meridian it is exactly the curve's own sampling problem,
                // so reuse the answer the edge sampler would give.
                let span = s.v1 - s.v0;
                let segments = s.meridian.initial_segments(s.v0, s.v1, tol).max(1);
                (ang(s.max_radius()), (span / segments as f64).abs())
            }
            Surface::Extrusion(s) => {
                // Straight along the extrusion; across it, the profile's own
                // sampling requirement.
                let span = s.u1 - s.u0;
                let segments = s.profile.initial_segments(s.u0, s.u1, tol).max(1);
                ((span / segments as f64).abs(), f64::INFINITY)
            }
        }
    }

    /// How much 3D length one unit of each parameter covers at `(u, v)`, i.e.
    /// `|dS/du|` and `|dS/dv|`. Used to give parameter space the same shape as
    /// the surface before ear clipping, which otherwise sees a badly distorted
    /// outline and triangulates it badly.
    pub fn param_scale(&self, _u: f64, v: f64) -> (f64, f64) {
        match self {
            Surface::Plane(_) => (1.0, 1.0),
            Surface::Cylinder { r, .. } => (r.abs(), 1.0),
            Surface::Cone { r, half_angle, .. } => {
                let radius = (r + v * half_angle.tan()).abs();
                (radius, 1.0 / half_angle.cos().abs().max(1e-12))
            }
            Surface::Sphere { r, .. } => (r.abs() * v.cos().abs(), r.abs()),
            Surface::Torus { major, minor, .. } => {
                ((major + minor * v.cos()).abs(), minor.abs())
            }
            Surface::BSpline(s) => {
                let (u0, u1, v0, v1) = s.domain();
                let (_, du, dv) =
                    s.eval_with_derivs((u0 + u1) * 0.5, (v0 + v1) * 0.5);
                (du.length(), dv.length())
            }
            Surface::Revolution(s) => {
                let c = s.meridian.eval(v.clamp(s.v0, s.v1));
                (s.meridian_coords(c).1, s.deriv_v(v).length().max(1e-12))
            }
            Surface::Extrusion(s) => {
                (s.deriv_u(_u).length().max(1e-12), s.dir.length().max(1e-12))
            }
        }
    }

    /// The `v` at which the whole `u` line collapses to a single point: a cone
    /// apex or a sphere pole. Faces bounded by only one edge loop (a cone cap,
    /// say) close along that line rather than along a second loop.
    ///
    /// `positive_side` selects which pole to return for a sphere; a cone has
    /// only one apex, so the flag is ignored there.
    pub fn degenerate_v(&self, positive_side: bool) -> Option<f64> {
        match self {
            Surface::Cone { r, half_angle, .. } => {
                let t = half_angle.tan();
                (t.abs() > 1e-12).then(|| -r / t)
            }
            Surface::Sphere { .. } => Some(if positive_side {
                std::f64::consts::FRAC_PI_2
            } else {
                -std::f64::consts::FRAC_PI_2
            }),
            // Where a degenerate torus's tube crosses the axis it is pinched to
            // a point, and the whole u circle collapses there -- a degenerate
            // parameter line exactly like a cone's apex. An ordinary torus has
            // none, and says so by leaving `cos v` out of range.
            Surface::Torus {
                major,
                minor,
                inner,
                ..
            } => {
                let c = -major / minor;
                if !(-1.0..1.0).contains(&c) {
                    return None;
                }
                let vp = c.acos();
                Some(match (*inner, positive_side) {
                    // The vortex sheet lives on v in (vp, 2pi - vp).
                    (true, true) => TWO_PI - vp,
                    (true, false) => vp,
                    (false, true) => vp,
                    (false, false) => -vp,
                })
            }
            _ => None,
        }
    }

    /// Whether each parameter direction is an angle.
    ///
    /// A `RECTANGULAR_TRIMMED_SURFACE`'s trim values carry the file's
    /// plane-angle unit (usually degrees) wherever the direction is angular.
    /// An extrusion's `u` is the profile's own parameter, an angle when the
    /// profile is a conic.
    pub fn angular_uv(&self) -> (bool, bool) {
        match self {
            Surface::Plane(_) | Surface::BSpline(_) => (false, false),
            Surface::Cylinder { .. } | Surface::Cone { .. } | Surface::Revolution(_) => {
                (true, false)
            }
            Surface::Sphere { .. } | Surface::Torus { .. } => (true, true),
            Surface::Extrusion(s) => (
                matches!(s.profile, Curve::Circle { .. } | Curve::Ellipse { .. }),
                false,
            ),
        }
    }

    /// The parameter rectangle covering the surface in full, for surfaces that
    /// a bounded rectangle covers wholly.
    ///
    /// A face with no bounds at all (a ball, an O-ring) is closed already. A
    /// plane or cylinder runs to infinity and returns `None`.
    pub fn full_domain(&self) -> Option<Domain> {
        match self {
            Surface::Sphere { .. } => Some((-PI, PI, -PI / 2.0, PI / 2.0)),
            Surface::Torus {
                major,
                minor,
                inner,
                ..
            } => {
                let c = -major / minor;
                if !(-1.0..1.0).contains(&c) {
                    return Some((-PI, PI, -PI, PI));
                }
                // A degenerate torus covers only the sheet the flag selected,
                // between the pinches.
                let vp = c.acos();
                Some(if *inner {
                    (-PI, PI, vp, TWO_PI - vp)
                } else {
                    (-PI, PI, -vp, vp)
                })
            }
            Surface::Revolution(s) => Some((-PI, PI, s.v0, s.v1)),
            Surface::BSpline(s) => {
                let (u0, u1, v0, v1) = s.domain();
                Some((u0, u1, v0, v1))
            }
            _ => None,
        }
    }

    /// The mirror of [`Surface::degenerate_v`]: a `u` at which the surface
    /// collapses to a point, so a loop circling `v` can be closed onto it.
    ///
    /// The analytic surfaces are all parameterised with their singularities in
    /// `v` -- a cone's apex, a sphere's poles -- so only a NURBS patch can have
    /// one here, and only by having a control row collapse.
    pub fn degenerate_u(&self, positive_side: bool) -> Option<f64> {
        let Surface::BSpline(s) = self else {
            return None;
        };
        let (u0, u1, _, _) = s.domain();
        match s.u_poles {
            (true, true) => Some(if positive_side { u1 } else { u0 }),
            (true, false) => Some(u0),
            (false, true) => Some(u1),
            (false, false) => None,
        }
    }
}

/// A screw-blade ribbon written the way a translator writes a thread: one
/// rational patch of 45 degree arcs round the z axis, rising `pitch` a turn,
/// and straight across from radius `r0` at v = 0 to `r1` at v = 1. Its `u`
/// counts arcs, so a turn is 8 of it.
#[cfg(test)]
pub(crate) fn helical_ribbon(turns: usize, pitch: f64, r0: f64, r1: f64) -> BSplineSurface {
    let arcs = turns * 8;
    // The middle point of a rational quadratic arc sits out at r over the
    // cosine of the half angle, with that cosine for its weight: an exact
    // circle in plan.
    let w = (PI / 8.0).cos();
    let mut ctrl = Vec::with_capacity((2 * arcs + 1) * 2);
    let mut weights = Vec::with_capacity((2 * arcs + 1) * 2);
    for k in 0..=2 * arcs {
        let a = k as f64 * PI / 8.0;
        let (out, weight) = if k % 2 == 1 { (1.0 / w, w) } else { (1.0, 1.0) };
        for r in [r0, r1] {
            ctrl.push(v3(r * out * a.cos(), r * out * a.sin(), pitch * a / TWO_PI));
            weights.push(weight);
        }
    }
    let mut u_knots = vec![0.0; 3];
    for k in 1..arcs {
        u_knots.extend([k as f64; 2]);
    }
    u_knots.extend([arcs as f64; 3]);
    BSplineSurface {
        u_degree: 2,
        v_degree: 1,
        nu: 2 * arcs + 1,
        nv: 2,
        ctrl,
        weights: Some(weights),
        u_knots,
        v_knots: vec![0.0, 0.0, 1.0, 1.0],
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: Vec3, b: Vec3, tol: f64) {
        assert!(a.distance(b) < tol, "{a:?} vs {b:?}");
    }

    fn unit_axis() -> Axis2 {
        Axis2::new(v3(0.0, 0.0, 0.0), None, None)
    }

    /// The bounds of a curve over a range have to hold every point of it, and
    /// for a conic they are exact: checked against the box of ten thousand
    /// samples, over ranges run either way, across the seam and past a turn.
    #[test]
    fn curve_bounds_hold_the_whole_curve_and_the_conics_exactly() {
        let ax = Axis2::new(v3(3.0, -2.0, 1.0), Some(v3(0.3, 1.0, 1.0)), Some(v3(1.0, 0.2, 0.0)));
        // Two rational quadratic arcs, a quarter circle and then a conic
        // climbing out of its plane, so the spline has weights and control
        // points off the curve to answer for, and a range that stops partway
        // along it.
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let spline = Curve::BSpline(Box::new(BSplineCurve::new(
            2,
            vec![v3(1.0, 0.0, 0.0), v3(1.0, 1.0, 0.0), v3(0.0, 1.0, 0.0), v3(-1.0, 1.0, 2.0), v3(-1.0, 0.0, 2.0)],
            Some(vec![1.0, w, 1.0, w, 1.0]),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        )));
        let cases: Vec<(Curve, f64, f64, bool)> = vec![
            (Curve::Circle { ax, r: 4.0 }, 0.3, 2.9, true),
            (Curve::Circle { ax, r: 4.0 }, 2.9, 0.3, true),
            (Curve::Circle { ax, r: 4.0 }, -1.0, 0.5, true),
            (Curve::Circle { ax, r: 4.0 }, 0.0, TWO_PI, true),
            (Curve::Circle { ax, r: 4.0 }, 5.0, 12.0, true),
            (Curve::Ellipse { ax, a: 5.0, b: 2.0 }, 1.0, 4.0, true),
            (Curve::Ellipse { ax, a: 5.0, b: 2.0 }, 0.0, TWO_PI, true),
            (Curve::Line { origin: v3(1.0, 2.0, 3.0), dir: v3(-2.0, 0.5, 1.0) }, 0.2, 1.7, true),
            (spline.clone(), 0.0, 2.0, false),
            (spline, 1.6, 0.4, false),
        ];
        for (c, t0, t1, exact) in cases {
            let mut got = Aabb::EMPTY;
            c.extend_bounds(t0, t1, &mut got);
            let mut sampled = Aabb::EMPTY;
            let n = 10_000;
            for k in 0..=n {
                sampled.add_point(c.eval(t0 + (t1 - t0) * k as f64 / n as f64));
            }
            let slack = 1e-9;
            let holds = |a: f64, b: f64| a <= b + slack;
            for (lo, hi, s_lo, s_hi) in [
                (got.min.x, got.max.x, sampled.min.x, sampled.max.x),
                (got.min.y, got.max.y, sampled.min.y, sampled.max.y),
                (got.min.z, got.max.z, sampled.min.z, sampled.max.z),
            ] {
                assert!(holds(lo, s_lo) && holds(s_hi, hi), "{c:?} over {t0}..{t1}: {got:?} misses {sampled:?}");
                if exact {
                    // Ten thousand samples of a turn fall at most a
                    // sagitta short of an extreme, well under a millionth.
                    assert!(s_lo - lo < 1e-6 && hi - s_hi < 1e-6, "{c:?} over {t0}..{t1}: {got:?} is not {sampled:?}");
                }
            }
        }
    }

    #[test]
    fn axis_orthogonalises_reference_direction() {
        // ref_direction not perpendicular to the axis must be projected.
        let ax = Axis2::new(v3(1.0, 2.0, 3.0), Some(v3(0.0, 0.0, 2.0)), Some(v3(1.0, 0.0, 5.0)));
        assert!(ax.x.dot(ax.z).abs() < 1e-12);
        assert_close(ax.x, v3(1.0, 0.0, 0.0), 1e-12);
        assert_close(ax.y, v3(0.0, 1.0, 0.0), 1e-12);
        // Round-trip through the frame.
        let p = v3(4.0, -1.0, 0.5);
        assert_close(ax.to_world(ax.to_local(p)), p, 1e-12);
    }

    /// Every analytic surface must satisfy project(eval(u,v)) == (u,v).
    #[test]
    fn analytic_surfaces_invert_exactly() {
        let ax = Axis2::new(v3(3.0, -2.0, 1.0), Some(v3(0.0, 1.0, 1.0)), Some(v3(1.0, 0.0, 0.0)));
        let cases: Vec<(Surface, f64, f64)> = vec![
            (Surface::Plane(ax), 2.5, -1.5),
            (Surface::Cylinder { ax, r: 4.0 }, 1.1, 2.0),
            (
                Surface::Cone {
                    ax,
                    r: 3.0,
                    half_angle: 0.3,
                },
                -2.0,
                1.5,
            ),
            (Surface::Sphere { ax, r: 5.0 }, 0.7, 0.4),
            (
                Surface::Torus {
                    ax,
                    major: 6.0,
                    minor: 1.5,
                    inner: false,
                },
                -1.2,
                0.9,
            ),
        ];
        for (s, u, v) in cases {
            let p = s.eval(u, v);
            let pr = s.project(p, None);
            assert!(!pr.u_degenerate, "{s:?} unexpectedly degenerate");
            assert!(
                (pr.uv.x - u).abs() < 1e-9 && (pr.uv.y - v).abs() < 1e-9,
                "{s:?}: got {:?} want ({u}, {v})",
                pr.uv
            );
        }
    }

    #[test]
    fn surface_normals_point_outward() {
        let ax = unit_axis();
        let cyl = Surface::Cylinder { ax, r: 2.0 };
        assert_close(cyl.normal(0.0, 5.0).unwrap(), v3(1.0, 0.0, 0.0), 1e-12);
        let sph = Surface::Sphere { ax, r: 3.0 };
        assert_close(sph.normal(0.0, 0.0).unwrap(), v3(1.0, 0.0, 0.0), 1e-12);
        assert_close(
            sph.normal(0.0, std::f64::consts::FRAC_PI_2).unwrap(),
            v3(0.0, 0.0, 1.0),
            1e-12,
        );
        let tor = Surface::Torus {
            ax,
            major: 5.0,
            minor: 1.0,
            inner: false,
        };
        assert_close(tor.normal(0.0, 0.0).unwrap(), v3(1.0, 0.0, 0.0), 1e-12);
        // Cone normal must be perpendicular to the ruling line.
        let cone = Surface::Cone {
            ax,
            r: 2.0,
            half_angle: 0.4,
        };
        let n = cone.normal(0.5, 1.0).unwrap();
        let ruling = cone.eval(0.5, 2.0) - cone.eval(0.5, 1.0);
        assert!(n.dot(ruling.normalize()).abs() < 1e-12);
        assert!(n.dot(v3(0.5f64.cos(), 0.5f64.sin(), 0.0)) > 0.0);
    }

    #[test]
    fn cone_apex_is_flagged_degenerate() {
        let cone = Surface::Cone {
            ax: unit_axis(),
            r: 2.0,
            half_angle: -std::f64::consts::FRAC_PI_4,
        };
        // radius 2 shrinks to 0 at v = 2 for a -45 degree half angle.
        let apex = cone.eval(0.0, 2.0);
        assert_close(apex, v3(0.0, 0.0, 2.0), 1e-12);
        assert!(cone.project(apex, None).u_degenerate);
        assert!(!cone.project(cone.eval(1.0, 0.0), None).u_degenerate);
    }

    #[test]
    fn bspline_curve_reproduces_a_straight_line() {
        // Degree 1 spline through three collinear points is exactly the line.
        let c = BSplineCurve::new(
            1,
            vec![v3(0.0, 0.0, 0.0), v3(1.0, 2.0, 3.0), v3(2.0, 4.0, 6.0)],
            None,
            vec![0.0, 0.0, 0.5, 1.0, 1.0],
        );
        assert!(c.is_valid());
        assert_eq!(c.domain(), (0.0, 1.0));
        assert_close(c.eval(0.0), v3(0.0, 0.0, 0.0), 1e-12);
        assert_close(c.eval(0.25), v3(0.5, 1.0, 1.5), 1e-12);
        assert_close(c.eval(1.0), v3(2.0, 4.0, 6.0), 1e-12);
    }

    #[test]
    fn rational_bspline_represents_a_quarter_circle() {
        // Standard degree-2 rational representation of a 90 degree arc.
        let w = 1.0 / 2.0f64.sqrt();
        let c = BSplineCurve::new(
            2,
            vec![v3(1.0, 0.0, 0.0), v3(1.0, 1.0, 0.0), v3(0.0, 1.0, 0.0)],
            Some(vec![1.0, w, 1.0]),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        );
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let p = c.eval(t);
            assert!((p.length() - 1.0).abs() < 1e-12, "t={t} r={}", p.length());
        }
    }

    #[test]
    fn bspline_surface_evaluates_and_inverts() {
        // Bilinear degree-1 patch over the unit square, tilted in z.
        let s = BSplineSurface {
            u_degree: 1,
            v_degree: 1,
            nu: 2,
            nv: 2,
            ctrl: vec![
                v3(0.0, 0.0, 0.0),
                v3(0.0, 1.0, 1.0),
                v3(1.0, 0.0, 0.0),
                v3(1.0, 1.0, 1.0),
            ],
            weights: None,
            u_knots: vec![0.0, 0.0, 1.0, 1.0],
            v_knots: vec![0.0, 0.0, 1.0, 1.0],
            ..Default::default()
        };
        assert!(s.is_valid());
        let (p, du, dv) = s.eval_with_derivs(0.5, 0.5);
        assert_close(p, v3(0.5, 0.5, 0.5), 1e-12);
        assert_close(du, v3(1.0, 0.0, 0.0), 1e-12);
        assert_close(dv, v3(0.0, 1.0, 1.0), 1e-12);
        let uv = s.project(v3(0.25, 0.75, 0.75), None);
        assert!((uv.x - 0.25).abs() < 1e-6 && (uv.y - 0.75).abs() < 1e-6, "{uv:?}");
        // A hint must reach the same answer as no hint.
        let hinted = s.project(v3(0.25, 0.75, 0.75), Some(v2(0.9, 0.1)));
        assert!(
            (hinted.x - 0.25).abs() < 1e-6 && (hinted.y - 0.75).abs() < 1e-6,
            "{hinted:?}"
        );
    }

    /// A patch many spans long has to invert onto the span the point is on:
    /// a long helix, approached cold, from a hint a whole turn out, and from
    /// off the surface by more than a file's tolerance.
    #[test]
    fn a_long_helical_patch_inverts_onto_the_right_turn() {
        let s = helical_ribbon(61, 5.0, 10.0, 11.0);
        let (u0, u1, _, _) = s.domain();
        let turn = 8.0;
        let mut checked = 0;
        let mut u = u0 + 0.37;
        while u < u1 {
            for v in [0.05, 0.5, 0.95] {
                let (p, du, dv) = s.eval_with_derivs(u, v);
                let n = du.cross(dv).normalize();
                let wrong = v2(if u + turn < u1 { u + turn } else { u - turn }, v);
                for (target, hint) in [
                    (p, None),
                    (p, Some(wrong)),
                    (p + n * 0.02, None),
                    (p - n * 0.02, Some(wrong)),
                ] {
                    let got = s.project(target, hint);
                    assert!(
                        (got.x - u).abs() < 1e-6 && (got.y - v).abs() < 1e-6,
                        "({u}, {v}) came back as {got:?} from hint {hint:?}"
                    );
                }
                checked += 1;
            }
            u += 3.71;
        }
        assert!(checked > 300, "only {checked} points checked");
    }

    /// A patch folded nearly flat has no normal: at one corner the derivatives
    /// are a ten-thousandth of a radian apart; at the far corner they are well
    /// apart and the normal is the plain answer.
    #[test]
    fn a_patch_folded_nearly_flat_has_no_normal_there() {
        let patch = BSplineSurface {
            u_degree: 1,
            v_degree: 1,
            ctrl: vec![
                v3(0.0, 0.0, 0.0),
                v3(0.0, 1.0, 0.0),
                v3(0.0, 1.0, 1e-4),
                v3(1.0, 2.0, 0.0),
            ],
            weights: None,
            nu: 2,
            nv: 2,
            u_knots: vec![0.0, 0.0, 1.0, 1.0],
            v_knots: vec![0.0, 0.0, 1.0, 1.0],
            ..Default::default()
        }
        .into_surface()
        .unwrap();
        assert_eq!(patch.normal(0.0, 0.0), None, "a normal from derivatives 1e-4 apart");
        let n = patch.normal(0.0, 1.0).expect("derivatives well apart");
        assert!((n.z.abs() - 1.0).abs() < 1e-12, "{n:?}");
    }

    /// An answer from the hint within `close` is taken as it stands, and one
    /// further off still sends the patch to be searched. With `close` too
    /// generous (ten units), an answer a turn away is accepted.
    #[test]
    fn a_close_enough_answer_is_taken_and_a_far_one_is_searched_past() {
        let s = helical_ribbon(61, 5.0, 10.0, 11.0);
        let (u0, u1, _, _) = s.domain();
        let turn = 8.0;
        let mut u = u0 + 0.37;
        while u < u1 {
            for v in [0.05, 0.5, 0.95] {
                let (p, du, dv) = s.eval_with_derivs(u, v);
                let off = p + du.cross(dv).normalize() * 0.02;
                let own = v2(u + 0.01, v);
                let wrong = v2(if u + turn < u1 { u + turn } else { u - turn }, v);
                for (hint, close) in [(own, 0.05), (own, 0.001), (wrong, 0.001)] {
                    let got = s.project_near(off, Some(hint), close);
                    assert!(
                        (got.x - u).abs() < 1e-6 && (got.y - v).abs() < 1e-6,
                        "({u}, {v}) came back as {got:?} from {hint:?} within {close}"
                    );
                }
                let promised = s.project_near(off, Some(wrong), 10.0);
                assert!((promised.x - u).abs() > turn * 0.9, "{promised:?} for ({u}, {v})");
            }
            u += 3.71;
        }
    }

    /// A vertex partway along a long curve has to invert onto its own point,
    /// not onto another turn of a helix; ends still win within the slack.
    #[test]
    fn a_vertex_along_a_long_helix_inverts_onto_its_own_turn() {
        let s = helical_ribbon(61, 5.0, 10.0, 11.0);
        let ctrl: Vec<Vec3> = (0..s.nu).map(|i| s.ctrl[i * s.nv]).collect();
        let w: Vec<f64> = (0..s.nu).map(|i| s.weights.as_ref().unwrap()[i * s.nv]).collect();
        let c = Curve::BSpline(Box::new(BSplineCurve::new(2, ctrl, Some(w), s.u_knots.clone())));
        let (lo, hi) = (s.u_knots[0], *s.u_knots.last().unwrap());
        let mut checked = 0;
        let mut t = lo + 0.37;
        while t < hi {
            let p = c.eval(t);
            let got = c.param_of(p);
            assert!((got - t).abs() < 1e-7, "{t} came back as {got}");
            // Off the curve square to it, as a vertex within a file's tolerance
            // of its edge is: the nearest point is still the one at t.
            let tangent = c.eval(t + 1e-6) - c.eval(t - 1e-6);
            let off = tangent.cross(v3(0.3, -0.2, 1.0)).normalize() * 0.01;
            let got = c.param_of(p + off);
            assert!((got - t).abs() < 1e-7, "{t} came back as {got} from 0.01 off");
            checked += 1;
            t += 3.71;
        }
        assert!(checked > 100, "only {checked} points checked");

        // The ends still come first. A vertex on one, or a hair off it, names
        // that end exactly, and so does one off the curve square to the end:
        // no point along it is any nearer.
        for end in [lo, hi] {
            assert_eq!(c.param_of(c.eval(end)), end);
            assert_eq!(c.param_of(c.eval(end) + v3(1e-9, -1e-9, 1e-9)), end);
            let into = if end == lo { 1e-6 } else { -1e-6 };
            let tangent = c.eval(end + into) - c.eval(end);
            let off = tangent.cross(v3(0.3, -0.2, 1.0)).normalize() * 0.05;
            assert_eq!(c.param_of(c.eval(end) + off), end);
        }

        // Within the slack a caller allows, a vertex a little way in from an
        // end is still that end, even sitting on the curve; past it, it is the
        // point it is at.
        let near = c.eval(lo + 0.01);
        let gap = near.distance(c.eval(lo));
        assert_eq!(c.param_of_vertex(near, gap * 2.0), lo);
        let got = c.param_of_vertex(near, gap * 0.5);
        assert!((got - (lo + 0.01)).abs() < 1e-9, "came back as {got}");
    }

    #[test]
    fn arc_segment_count_meets_tolerance() {
        let r = 10.0;
        let tol = 0.01;
        let n = arc_segments(r, TWO_PI, tol);
        let theta = TWO_PI / n as f64;
        let sagitta = r * (1.0 - (theta / 2.0).cos());
        assert!(sagitta <= tol * 1.0001, "sagitta {sagitta} > tol {tol}");
    }

    /// A circle finer than the tolerance still has to come out as a polygon
    /// with an interior; a zero-area loop loses the face inside it.
    #[test]
    fn arcs_below_tolerance_still_enclose_an_area() {
        for r in [1e-6, 0.01, 0.05, 0.106] {
            let n = arc_segments(r, TWO_PI, 0.106);
            assert!(n >= 4, "radius {r} gave {n} segments, cannot bound an area");
        }
        // `MAX_CHORD_ANGLE` is the coarsest any single chord may be, so
        // partial spans scale down with the span rather than collapsing to one.
        assert!(arc_segments(1e-6, std::f64::consts::PI, 1.0) >= 2);
        assert_eq!(arc_segments(1e-6, 0.001, 1.0), 1);
        // Tolerance still governs once the arc is big enough to need it.
        assert!(arc_segments(10.0, TWO_PI, 0.01) > 4);
    }

    /// Both sheets of a self-intersecting torus have to invert onto themselves.
    /// They overlap in space -- a point on the vortex sheet also lies at some
    /// (u, v) of the apple -- so `project` can only tell them apart by the flag,
    /// and getting that wrong lands the face on the wrong side of the axis.
    #[test]
    fn degenerate_torus_inverts_on_both_sheets() {
        let (major, minor) = (2.0, 5.0);
        let pinch = (-major / minor as f64).acos(); // where the tube crosses the axis
        // Angles are only ever equal up to a whole turn.
        let same_angle = |a: f64, b: f64| {
            let d = (a - b).rem_euclid(TWO_PI);
            d < 1e-9 || (TWO_PI - d) < 1e-9
        };
        for inner in [false, true] {
            let s = Surface::Torus {
                ax: Axis2::new(v3(1.0, -2.0, 0.5), None, None),
                major,
                minor,
                inner,
            };
            // v runs pole to pole across the sheet this flag selects.
            let (v0, v1) = if inner {
                (pinch, TWO_PI - pinch)
            } else {
                (-pinch, pinch)
            };
            for i in 1..8 {
                let v = v0 + (v1 - v0) * i as f64 / 8.0;
                for &u in &[-2.9, -0.7, 0.0, 1.3, 3.0] {
                    let p = s.eval(u, v);
                    let got = s.project(p, None);
                    assert!(!got.u_degenerate);
                    assert_close(s.eval(got.uv.x, got.uv.y), p, 1e-9);
                    assert!(
                        same_angle(got.uv.x, u) && (got.uv.y - v).abs() < 1e-9,
                        "inner={inner} (u,v)=({u},{v}) came back as {:?}",
                        got.uv
                    );
                }
            }
            // The pinch is a pole: the whole u circle collapses onto the axis.
            let pole = s.project(s.eval(0.0, v0), None);
            assert!(pole.u_degenerate, "inner={inner} pinch not flagged");
            // And it is a degenerate parameter line, so a face wrapping the
            // sheet has something to close its outline onto.
            for positive in [true, false] {
                let v = s
                    .degenerate_v(positive)
                    .expect("degenerate torus must report its pinch");
                assert!(
                    (v - v0).abs() < 1e-9 || (v - v1).abs() < 1e-9,
                    "inner={inner} positive={positive}: pinch {v} is not an end of {v0}..{v1}"
                );
            }
        }
        // An ordinary torus has no such line: it closes in v instead.
        let plain = Surface::Torus {
            ax: unit_axis(),
            major: 5.0,
            minor: 2.0,
            inner: false,
        };
        assert_eq!(plain.degenerate_v(true), None);
        // Only the ordinary torus closes on itself in v.
        assert_eq!(
            Surface::Torus {
                ax: unit_axis(),
                major,
                minor,
                inner: false
            }
            .v_period(),
            None
        );
        assert_eq!(
            Surface::Torus {
                ax: unit_axis(),
                major: 5.0,
                minor: 2.0,
                inner: false
            }
            .v_period(),
            Some(TWO_PI)
        );
    }

    /// A closed B-spline has to say so: an edge running the whole way round
    /// one is a loop, not the zero-length segment its coincident vertices
    /// suggest, and misreading it drops the whole bound.
    #[test]
    fn closed_bspline_curves_report_a_period() {
        // Degree 1 through four corners and back to the first: a closed square.
        let square = BSplineCurve::new(
            1,
            vec![
                v3(0.0, 0.0, 0.0),
                v3(1.0, 0.0, 0.0),
                v3(1.0, 1.0, 0.0),
                v3(0.0, 1.0, 0.0),
                v3(0.0, 0.0, 0.0),
            ],
            None,
            vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0],
        );
        assert_eq!(square.period(), Some(4.0));
        assert_eq!(Curve::BSpline(Box::new(square.clone())).period(), Some(4.0));
        // Past the join it wraps rather than clamping, so a seam-unwrapped
        // parameter keeps travelling instead of piling up on the last point.
        assert_close(square.eval(4.5), square.eval(0.5), 1e-12);
        assert_close(square.eval(-0.5), square.eval(3.5), 1e-12);

        // The same net without the closing point stays open.
        let open = BSplineCurve::new(
            1,
            vec![v3(0.0, 0.0, 0.0), v3(1.0, 0.0, 0.0), v3(1.0, 1.0, 0.0)],
            None,
            vec![0.0, 0.0, 1.0, 2.0, 2.0],
        );
        assert_eq!(open.period(), None);
        assert_close(open.eval(5.0), open.eval(2.0), 1e-12); // still clamps
    }

    /// A tube written as a NURBS patch is as periodic as a cylinder, and a dome
    /// written with its top control row repeated has a pole just like a sphere.
    /// Both have to be reported, or a face wrapping one is bounded by two end
    /// curves that enclose nothing and the face is lost.
    #[test]
    fn closed_bspline_surfaces_report_periods_and_poles() {
        // A square tube: four sides plus the repeated first, straight along v.
        let ring = [
            v3(1.0, 0.0, 0.0),
            v3(0.0, 1.0, 0.0),
            v3(-1.0, 0.0, 0.0),
            v3(0.0, -1.0, 0.0),
            v3(1.0, 0.0, 0.0),
        ];
        let mut ctrl = Vec::new();
        for p in ring {
            for h in [0.0, 5.0] {
                ctrl.push(v3(p.x, p.y, h));
            }
        }
        let tube = BSplineSurface {
            u_degree: 1,
            v_degree: 1,
            ctrl,
            weights: None,
            nu: 5,
            nv: 2,
            u_knots: vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0],
            v_knots: vec![0.0, 0.0, 1.0, 1.0],
            ..Default::default()
        }
        .into_surface()
        .unwrap();
        assert_eq!(tube.u_period(), Some(4.0));
        assert_eq!(tube.v_period(), None);
        assert_eq!(tube.degenerate_u(true), None);
        // Wrapping, not clamping: a point one period on is the same point.
        assert_close(tube.eval(4.5, 0.5), tube.eval(0.5, 0.5), 1e-12);

        // A cone-like patch whose u = 1 row collapses onto the axis.
        let mut ctrl = Vec::new();
        for p in ring {
            ctrl.push(v3(p.x, p.y, 0.0)); // u = 0: the open rim
            ctrl.push(v3(0.0, 0.0, 3.0)); // u = 1: all one point
        }
        let dome = BSplineSurface {
            u_degree: 1,
            v_degree: 1,
            ctrl,
            weights: None,
            nu: 5,
            nv: 2,
            u_knots: vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0],
            v_knots: vec![0.0, 0.0, 1.0, 1.0],
            ..Default::default()
        }
        .into_surface()
        .unwrap();
        // Here u indexes the ring and v runs rim-to-tip, so the pole is in v.
        assert_eq!(dome.u_period(), Some(4.0));
        assert_close(dome.eval(1.7, 1.0), v3(0.0, 0.0, 3.0), 1e-12);
    }

    fn revolve(meridian: Curve, v0: f64, v1: f64) -> Surface {
        Surface::Revolution(Box::new(RevolutionSurface {
            meridian,
            v0,
            v1,
            origin: Vec3::ZERO,
            dir: v3(0.0, 0.0, 1.0),
        }))
    }

    /// A line parallel to the axis sweeps an exact cylinder, which gives a
    /// closed form to check the general machinery against.
    #[test]
    fn revolving_a_parallel_line_gives_a_cylinder() {
        let s = revolve(
            Curve::Line {
                origin: v3(5.0, 0.0, 0.0),
                dir: v3(0.0, 0.0, 1.0),
            },
            0.0,
            10.0,
        );
        let reference = Surface::Cylinder {
            ax: unit_axis(),
            r: 5.0,
        };
        for &(u, v) in &[(0.0, 0.0), (1.1, 3.0), (-2.4, 7.5), (TWO_PI - 0.2, 10.0)] {
            assert_close(s.eval(u, v), reference.eval(u, v), 1e-9);
            // Normals must agree with the cylinder's, i.e. point outward.
            assert_close(s.normal(u, v).unwrap(), reference.normal(u, v).unwrap(), 1e-6);
            // And inversion must return the parameters we started from. The
            // angle comes back from atan2, so it is only equal modulo a turn --
            // the same convention the analytic periodic surfaces use.
            let pr = s.project(s.eval(u, v), None);
            assert!(!pr.u_degenerate);
            let du = (pr.uv.x - u).rem_euclid(TWO_PI);
            assert!(
                du.min(TWO_PI - du) < 1e-6 && (pr.uv.y - v).abs() < 1e-6,
                "got {:?} want ({u}, {v})",
                pr.uv
            );
        }
    }

    /// A slanted line sweeps a cone; the normal must stay perpendicular to the
    /// ruling it lies on.
    #[test]
    fn revolving_a_slanted_line_gives_a_cone() {
        let s = revolve(
            Curve::Line {
                origin: v3(5.0, 0.0, 0.0),
                dir: v3(1.0, 0.0, 1.0),
            },
            0.0,
            5.0,
        );
        // Radius grows one-for-one with height.
        let p = s.eval(0.0, 3.0);
        assert_close(p, v3(8.0, 0.0, 3.0), 1e-9);

        let u = 0.7;
        let n = s.normal(u, 2.0).unwrap();
        let ruling = (s.eval(u, 4.0) - s.eval(u, 1.0)).normalize();
        assert!(n.dot(ruling).abs() < 1e-6, "normal not perpendicular to ruling");
        // Outward, i.e. away from the axis.
        let radial = v3(u.cos(), u.sin(), 0.0);
        assert!(n.dot(radial) > 0.0);
    }

    /// Revolving a circle offset from the axis is a torus, so the analytic
    /// torus is available as an independent answer.
    #[test]
    fn revolving_a_circle_gives_a_torus() {
        let major = 6.0;
        let minor = 1.5;
        // Meridian circle in the xz plane, centred at radius `major`.
        let ax = Axis2::new(
            v3(major, 0.0, 0.0),
            Some(v3(0.0, -1.0, 0.0)),
            Some(v3(1.0, 0.0, 0.0)),
        );
        let s = revolve(Curve::Circle { ax, r: minor }, 0.0, TWO_PI);
        let reference = Surface::Torus {
            ax: unit_axis(),
            major,
            minor,
            inner: false,
        };
        for &(u, v) in &[(0.0, 0.0), (0.9, 1.2), (-1.5, 2.7)] {
            assert_close(s.eval(u, v), reference.eval(u, v), 1e-9);
        }
        // Inversion round-trips through the general search.
        let pr = s.project(s.eval(0.9, 1.2), None);
        assert!((pr.uv.x - 0.9).abs() < 1e-5 && (pr.uv.y - 1.2).abs() < 1e-5, "{:?}", pr.uv);
    }

    /// A revolution that reproduces a torus has to close in the same two
    /// directions the torus does; a trimmed meridian or an open one does not
    /// close in `v`.
    #[test]
    fn a_revolution_closes_wherever_the_surface_it_reproduces_does() {
        let (major, minor) = (6.0, 1.5);
        let ax = Axis2::new(
            v3(major, 0.0, 0.0),
            Some(v3(0.0, -1.0, 0.0)),
            Some(v3(1.0, 0.0, 0.0)),
        );
        let torus = Surface::Torus {
            ax: unit_axis(),
            major,
            minor,
            inner: false,
        };
        let tube = revolve(Curve::Circle { ax, r: minor }, 0.0, TWO_PI);
        assert_eq!(tube.u_period(), torus.u_period());
        assert_eq!(tube.v_period(), torus.v_period());

        // Half a meridian is a domed end cap, not a tube. It still closes
        // round the axis and must stay open along the meridian, whatever
        // period the untrimmed circle would report.
        let dome = revolve(Curve::Circle { ax, r: minor }, 0.0, PI);
        assert_eq!(dome.u_period(), Some(TWO_PI));
        assert_eq!(dome.v_period(), None);

        // And a meridian that does not close cannot close the surface.
        let cylinder = revolve(
            Curve::Line {
                origin: v3(5.0, 0.0, 0.0),
                dir: v3(0.0, 0.0, 1.0),
            },
            0.0,
            10.0,
        );
        assert_eq!(cylinder.v_period(), None);
    }

    #[test]
    fn revolution_marks_the_axis_as_degenerate() {
        // Meridian running into the axis: the end point is a pole.
        let s = revolve(
            Curve::Line {
                origin: v3(4.0, 0.0, 0.0),
                dir: v3(-4.0, 0.0, 2.0),
            },
            0.0,
            1.0,
        );
        assert!(s.project(s.eval(0.0, 1.0), None).u_degenerate, "apex not flagged");
        assert!(!s.project(s.eval(0.0, 0.0), None).u_degenerate);
    }

    #[test]
    fn revolution_uv_step_is_bounded_in_both_directions() {
        let s = revolve(
            Curve::Line {
                origin: v3(5.0, 0.0, 0.0),
                dir: v3(0.0, 0.0, 1.0),
            },
            0.0,
            10.0,
        );
        let (du, dv) = s.uv_step(0.01, v2(0.0, 0.0), v2(TWO_PI, 10.0));
        // Around the axis the bound is the circle's, as for a cylinder.
        let circle = Surface::Cylinder {
            ax: unit_axis(),
            r: 5.0,
        };
        let (want_du, _) = circle.uv_step(0.01, v2(0.0, 0.0), v2(TWO_PI, 10.0));
        assert!((du - want_du).abs() < 1e-9, "{du} vs {want_du}");
        // A straight meridian needs no subdivision beyond one segment, but the
        // step must still be finite so refinement has a stopping rule.
        assert!(dv.is_finite() && dv > 0.0, "dv = {dv}");
    }

    fn extrude(profile: Curve, u0: f64, u1: f64, dir: Vec3) -> Surface {
        Surface::Extrusion(Box::new(ExtrusionSurface {
            profile,
            u0,
            u1,
            dir,
        }))
    }

    /// Extruding a circle gives a cylinder, so the analytic cylinder is an
    /// independent answer to check against.
    #[test]
    fn extruding_a_circle_gives_a_cylinder() {
        let r = 4.0;
        let s = extrude(
            Curve::Circle {
                ax: unit_axis(),
                r,
            },
            0.0,
            TWO_PI,
            v3(0.0, 0.0, 1.0),
        );
        let reference = Surface::Cylinder {
            ax: unit_axis(),
            r,
        };
        for &(u, v) in &[(0.0, 0.0), (1.1, 3.0), (-2.4, 7.5)] {
            assert_close(s.eval(u, v), reference.eval(u, v), 1e-9);
            assert_close(s.normal(u, v).unwrap(), reference.normal(u, v).unwrap(), 1e-5);
            let pr = s.project(s.eval(u, v), None);
            let du = (pr.uv.x - u).rem_euclid(TWO_PI);
            assert!(
                du.min(TWO_PI - du) < 1e-5 && (pr.uv.y - v).abs() < 1e-6,
                "got {:?} want ({u}, {v})",
                pr.uv
            );
        }
        // A closed profile makes the surface periodic, which is what drives
        // seam handling; without it a face wrapping the join renders wrong.
        assert_eq!(s.u_period(), Some(TWO_PI));
        assert_eq!(s.v_period(), None);
    }

    #[test]
    fn extruding_a_trimmed_arc_is_not_periodic() {
        let s = extrude(
            Curve::Circle {
                ax: unit_axis(),
                r: 4.0,
            },
            0.0,
            1.0, // a short arc, not a whole turn
            v3(0.0, 0.0, 1.0),
        );
        assert_eq!(s.u_period(), None, "an open profile leaves the surface open");
    }

    #[test]
    fn extruding_a_line_gives_a_plane() {
        // A line along +X extruded along +Z spans the xz plane.
        let s = extrude(
            Curve::Line {
                origin: v3(0.0, 5.0, 0.0),
                dir: v3(1.0, 0.0, 0.0),
            },
            0.0,
            10.0,
            v3(0.0, 0.0, 2.0),
        );
        assert_close(s.eval(3.0, 0.5), v3(3.0, 5.0, 1.0), 1e-12);
        // Normal is perpendicular to both the profile and the extrusion.
        let n = s.normal(3.0, 0.5).unwrap();
        assert!(n.dot(v3(1.0, 0.0, 0.0)).abs() < 1e-9);
        assert!(n.dot(v3(0.0, 0.0, 1.0)).abs() < 1e-9);
        // v is measured in units of the extrusion vector, not in millimetres.
        let pr = s.project(v3(3.0, 5.0, 1.0), None);
        assert!((pr.uv.x - 3.0).abs() < 1e-6 && (pr.uv.y - 0.5).abs() < 1e-6, "{:?}", pr.uv);
        assert!(s.u_period().is_none());
    }

    /// A wavy open cubic in the xz plane, clear of the z axis: somewhere
    /// different at every parameter, so a parameter off by anything shows.
    fn wavy() -> Curve {
        Curve::BSpline(Box::new(BSplineCurve::new(
            3,
            vec![
                v3(3.0, 0.0, 0.0),
                v3(5.0, 0.0, 1.0),
                v3(4.0, 0.0, 3.0),
                v3(6.0, 0.0, 4.0),
                v3(5.0, 0.0, 6.0),
                v3(7.0, 0.0, 7.5),
            ],
            None,
            vec![0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0, 1.0],
        )))
    }

    /// A swept surface inverts to the parameters it was evaluated at, to the
    /// precision of the arithmetic, with or without a hint from the previous
    /// point; a point a little off the surface comes back as the foot of the
    /// normal it was moved along.
    #[test]
    fn a_swept_surface_inverts_to_the_parameters_it_was_evaluated_at() {
        let swept = [
            (extrude(wavy(), 0.0, 1.0, v3(0.0, 2.0, 0.0)), true),
            (revolve(wavy(), 0.0, 1.0), false),
        ];
        for (s, along_u) in &swept {
            let mut behind = None;
            for i in 0..=64 {
                let t = i as f64 / 64.0;
                // The curve's own parameter is u on an extrusion and v on a
                // revolution; the other is the sweep's.
                let want = if *along_u { v2(t, 0.3 + 0.2 * t) } else { v2(0.4 + 2.0 * t, t) };
                let p = s.eval(want.x, want.y);
                let gap = |a: Vec2| (a.x - want.x).hypot(a.y - want.y);
                for hint in [behind, None] {
                    let got = s.project(p, hint).uv;
                    assert!(gap(got) < 1e-13, "{got:?} for {want:?} from {hint:?}");
                }
                behind = Some(want);
                let off = p + s.normal(want.x, want.y).unwrap() * 1e-3;
                let foot = s.project(off, behind).uv;
                assert!(gap(foot) < 1e-9, "{foot:?} for {want:?}, off the surface");
            }
        }
    }

    /// A hint on the wrong stretch of curve must not keep the answer there:
    /// from the far arm of a hairpin profile, the near arm is still found.
    #[test]
    fn a_hint_on_the_far_arm_of_a_hairpin_still_finds_the_near_one() {
        let hairpin = Curve::BSpline(Box::new(BSplineCurve::new(
            1,
            vec![
                v3(0.0, 0.0, 0.0),
                v3(10.0, 0.0, 0.0),
                v3(10.0, 1.0, 0.0),
                v3(0.0, 1.0, 0.0),
            ],
            None,
            vec![0.0, 0.0, 1.0, 2.0, 3.0, 3.0],
        )));
        let s = extrude(hairpin, 0.0, 3.0, v3(0.0, 0.0, 1.0));
        // On the near arm at u = 0.37; the far arm passes it at u = 2.63.
        let p = v3(3.7, 0.0, 0.5);
        let got = s.project(p, Some(v2(2.63, 0.5))).uv;
        assert!((got.x - 0.37).hypot(got.y - 0.5) < 1e-12, "{got:?}");
    }

    /// A revolution closes on itself, and saying so is what lets the seam
    /// stitcher handle a face that wraps the join.
    #[test]
    fn revolution_reports_its_period() {
        let s = revolve(
            Curve::Line {
                origin: v3(5.0, 0.0, 0.0),
                dir: v3(0.0, 0.0, 1.0),
            },
            0.0,
            10.0,
        );
        assert_eq!(s.u_period(), Some(TWO_PI));
    }

    #[test]
    fn xform_between_frames_maps_one_onto_the_other() {
        let from = Axis2::new(v3(1.0, 2.0, 3.0), Some(v3(0.0, 0.0, 1.0)), Some(v3(1.0, 0.0, 0.0)));
        let to = Axis2::new(v3(-5.0, 7.0, 0.5), Some(v3(1.0, 1.0, 0.0)), Some(v3(0.0, 0.0, 1.0)));
        let m = Xform::between(&from, &to);

        // The frame origin and each axis must land on their counterparts.
        assert_close(m.point(from.origin), to.origin, 1e-12);
        assert_close(m.dir(from.x), to.x, 1e-12);
        assert_close(m.dir(from.y), to.y, 1e-12);
        assert_close(m.dir(from.z), to.z, 1e-12);

        // A point keeps its coordinates in the new frame.
        let local = v3(0.3, -1.7, 4.2);
        assert_close(m.point(from.to_world(local)), to.to_world(local), 1e-12);

        // Rigid: lengths and angles survive.
        let (a, b) = (v3(1.0, -2.0, 0.5), v3(-0.25, 0.75, 2.0));
        assert!((m.dir(a).length() - a.length()).abs() < 1e-12);
        assert!((m.dir(a).dot(m.dir(b)) - a.dot(b)).abs() < 1e-12);
    }

    #[test]
    fn xform_identity_and_composition() {
        let ax = Axis2::new(Vec3::ZERO, None, None);
        assert!(Xform::between(&ax, &ax).is_identity());

        let a = Xform::between(
            &Axis2::new(Vec3::ZERO, None, None),
            &Axis2::new(v3(3.0, 0.0, 0.0), Some(v3(0.0, 0.0, 1.0)), Some(v3(0.0, 1.0, 0.0))),
        );
        let b = Xform::between(
            &Axis2::new(Vec3::ZERO, None, None),
            &Axis2::new(v3(0.0, 5.0, 0.0), Some(v3(1.0, 0.0, 0.0)), Some(v3(0.0, 1.0, 0.0))),
        );
        // Composition must match applying them in order.
        let p = v3(1.0, 2.0, 3.0);
        assert_close(b.compose(&a).point(p), b.point(a.point(p)), 1e-12);
        assert_close(Xform::IDENTITY.compose(&a).point(p), a.point(p), 1e-12);
    }

    #[test]
    fn expand_knots_applies_multiplicities() {
        assert_eq!(
            expand_knots(&[0.0, 1.0, 2.0], &[3, 1, 3]),
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]
        );
    }
}
