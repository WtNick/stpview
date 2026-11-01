//! Turns the raw STEP entity graph into a compact, self-contained B-rep:
//! solids -> faces -> bounds -> parameterised edge uses.
//!
//! Everything the tessellator needs is resolved here, so tessellation itself
//! touches no `StepFile` and can run in parallel without synchronisation.

use std::collections::{HashMap, HashSet};

use crate::geom::{expand_knots, Axis2, BSplineCurve, BSplineSurface, Curve, Domain, ExtrusionSurface, RevolutionSurface, Surface, Xform, TWO_PI};
use crate::math::{v3, Aabb, Vec3};
use crate::step::{StepFile, Value};

/// One use of an edge by a face loop, already resolved to a parameter range.
/// Walking `t0 -> t1` traverses the edge in the loop's direction.
#[derive(Debug, Clone, Copy)]
pub struct EdgeUse {
    pub curve: u32,
    /// Entity id of the `EDGE_CURVE`. Two faces of a closed solid share one,
    /// which lets the edge overlay pair them and measure the angle between.
    pub edge: u32,
    pub t0: f64,
    pub t1: f64,
}

#[derive(Debug, Clone)]
pub struct Bound {
    pub outer: bool,
    pub edges: Vec<EdgeUse>,
    /// A `VERTEX_LOOP` bounds a face at a single point rather than along a
    /// curve -- the surface closes to nothing there, as at a cone's apex or a
    /// sphere's pole. It carries no edges, so the point itself is the bound.
    pub vertex: Option<Vec3>,
}

#[derive(Debug, Clone)]
pub struct Face {
    pub surface: u32,
    /// False when the face normal opposes the surface normal.
    pub same_sense: bool,
    pub bounds: Vec<Bound>,
    /// `(u0, u1, v0, v1)` where a `RECTANGULAR_TRIMMED_SURFACE` cut the basis
    /// down, in the basis surface's own parameters. Only consulted for a face
    /// with no bounds of its own; bounds are more precise.
    pub domain: Option<Domain>,
    /// Colour and opacity the file gave this face in particular, which beats
    /// whatever its body carries.
    ///
    /// This is the common way to write a multi-material part: a
    /// `MANIFOLD_SOLID_BREP` has a single appearance slot, so a part with
    /// black plastic and gold pins hangs its colours off the faces.
    pub style: Option<([f32; 3], f32)>,
}

#[derive(Debug, Clone)]
pub struct Solid {
    pub name: String,
    pub color: [f32; 3],
    /// 1.0 for an opaque part. STEP states transparency, where 0 is opaque, so
    /// this is one minus what the file says.
    pub alpha: f32,
    pub faces: Vec<Face>,
    /// Curves the file put in a `GEOMETRIC_SET` on their own, with no surface:
    /// construction geometry, centrelines, drawing outlines. Drawn in the edge
    /// overlay rather than tessellated. A file may consist of nothing else.
    pub wires: Vec<EdgeUse>,
    /// Where this occurrence sits in the assembly. Geometry is tessellated in
    /// the solid's own space and placed on emit, so one component instanced
    /// several times shares its surfaces and curves.
    pub placement: Xform,
    /// Bounds of the solid's vertices and edges before placement.
    pub local_bbox: Aabb,
    /// False when no assembly relationship named this solid, so `placement` is
    /// the reader's fallback rather than something the file asked for.
    pub linked: bool,
    /// Which node of the assembly tree this occurrence belongs to.
    pub node: Option<usize>,
    /// Whether the file describes this body as a closed volume rather than a
    /// sheet. Only a volume has an inside, and only a volume can be capped.
    pub volume: bool,
    /// The entity this occurrence's geometry was built from. Occurrences that
    /// share it carry identical face lists and differ only in placement, so
    /// the tessellator meshes a repeated component once. `u32::MAX` for the
    /// loose-face pseudo-solid.
    pub source: u32,
}

/// One placement of one representation while the assembly is being walked.
struct Occurrence {
    rep: u32,
    world: Xform,
    parent: Option<usize>,
    name: String,
}

/// A node of the assembly tree: one occurrence of one component.
#[derive(Debug, Clone)]
pub struct AsmNode {
    /// The product's name, or empty where the file gives none.
    pub name: String,
    /// The occurrence that instanced this one, or `None` at a root.
    pub parent: Option<usize>,
}

pub struct Model {
    pub solids: Vec<Solid>,
    /// The assembly tree, in depth-first order, so a node always follows its
    /// parent. Empty for a file with no assembly structure at all.
    pub nodes: Vec<AsmNode>,
    pub surfaces: Vec<Surface>,
    pub curves: Vec<Curve>,
    /// Bounding box of the placed B-rep boundary (vertices and edges, since a
    /// closed edge has only one vertex): the length scale for tolerances, and
    /// what the viewer frames before a triangle exists.
    pub vertex_bbox: Aabb,
    /// Millimetres per unit of the coordinates above. Geometry stays in the
    /// file's own units, since rescaling would touch every radius and knot
    /// too; this converts user lengths in and display lengths out.
    pub mm_per_unit: f64,
    pub stats: BuildStats,
}

impl Model {
    /// Approximate memory the model holds, in bytes: the cost of keeping it to
    /// re-tessellate a close view. Counted from the dominant buffers (face
    /// lists, copied per occurrence, and B-spline control nets), so it slightly
    /// underestimates.
    pub fn heap_bytes(&self) -> usize {
        use std::mem::size_of;
        let mut n = self.solids.capacity() * size_of::<Solid>()
            + self.nodes.capacity() * size_of::<AsmNode>()
            + self.surfaces.capacity() * size_of::<Surface>()
            + self.curves.capacity() * size_of::<Curve>();
        for s in &self.solids {
            n += s.name.capacity()
                + s.faces.capacity() * size_of::<Face>()
                + s.wires.capacity() * size_of::<EdgeUse>();
            for f in &s.faces {
                n += f.bounds.capacity() * size_of::<Bound>();
                for b in &f.bounds {
                    n += b.edges.capacity() * size_of::<EdgeUse>();
                }
            }
        }
        n += self.nodes.iter().map(|a| a.name.capacity()).sum::<usize>();
        n += self.surfaces.iter().map(Surface::heap_bytes).sum::<usize>();
        n += self.curves.iter().map(Curve::heap_bytes).sum::<usize>();
        n
    }
}

#[derive(Debug, Default, Clone)]
pub struct BuildStats {
    pub faces: usize,
    pub unsupported_surfaces: HashMap<String, usize>,
    pub unsupported_curves: HashMap<String, usize>,
    pub skipped_faces: usize,
    /// Of those, the ones lost because the reader does not know their surface
    /// type. The type itself is in `unsupported_surfaces`.
    pub faces_unsupported: usize,
    /// Edges in a loop that could not be resolved and were dropped.
    pub dropped_edges: usize,
    /// Bodies whose shell yielded no faces at all, so the part is not drawn.
    pub dropped_bodies: usize,
    /// Bounds whose edges all resolved away, leaving the face the wrong shape.
    pub dropped_bounds: usize,
    /// Solid instances emitted, counting assembly repeats separately.
    pub instances: usize,
    /// Of those, the ones the file gave a transparency of its own.
    pub translucent: usize,
    /// Faces carrying a `STYLED_ITEM` of their own rather than inheriting the
    /// body's: the difference between a part in its own colours and one in
    /// default grey.
    pub styled_faces: usize,
    /// Items given a second, different colour by a later `STYLED_ITEM`. The
    /// last one wins, which is arbitrary.
    pub restyled: usize,
}

pub(crate) const DEFAULT_COLOR: [f32; 3] = [0.62, 0.65, 0.70];

/// Backstops for malformed or maliciously nested assembly graphs.
const MAX_OCCURRENCES: usize = 20_000;
const MAX_ASSEMBLY_DEPTH: u32 = 32;

/// True for the representation types that can hold shape items. Excludes
/// `CONTEXT_DEPENDENT_SHAPE_REPRESENTATION`, which despite the name is an
/// assembly *link* rather than a representation.
fn is_shape_rep(ty: &str) -> bool {
    ty != "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION" && ty.ends_with("SHAPE_REPRESENTATION")
}

/// Radians per unit of the file's plane-angle measure.
///
/// STEP records angles in whatever the representation context declares, and
/// many translators declare degrees. Read as radians, a 45 degree cone angle
/// becomes 45 radians.
fn plane_angle_scale(f: &StepFile) -> f64 {
    unit_scale(f, "PLANE_ANGLE_UNIT").unwrap_or(1.0)
}

/// Millimetres per unit of length in the file.
///
/// A uniform scale changes nothing visible, but it decides what the numbers
/// mean: `--tol 0.1` on a file in inches would otherwise be 2.5 mm.
fn length_scale(f: &StepFile) -> f64 {
    for id in f.ids() {
        let Some(ctx) = f.complex_part(id, "GLOBAL_UNIT_ASSIGNED_CONTEXT") else {
            continue;
        };
        let Some(&list) = ctx.first() else { continue };
        for u in f.list(list).iter().filter_map(|v| v.as_ref_id()) {
            if f.complex_part(u, "LENGTH_UNIT").is_none() {
                continue;
            }
            // An inch is written as a conversion whose factor is quoted in the
            // SI unit the measure names -- 2.54 against centimetres, not
            // 0.0254 against metres -- so the base unit must be read too.
            if let Some(conv) = f.complex_part(u, "CONVERSION_BASED_UNIT") {
                let Some(m) = conv.get(1).and_then(|v| v.as_ref_id()) else {
                    continue;
                };
                let p = f.params(m);
                let factor = p
                    .first()
                    .and_then(|&v| v.as_f64().or_else(|| f.typed_params(v).first()?.as_f64()));
                let base = p
                    .get(1)
                    .and_then(|v| v.as_ref_id())
                    .and_then(|b| si_length_mm(f, b))
                    .unwrap_or(1000.0);
                if let Some(x) = factor.filter(|x| *x > 0.0 && x.is_finite()) {
                    return x * base;
                }
                continue;
            }
            if let Some(mm) = si_length_mm(f, u) {
                return mm;
            }
        }
    }
    1.0
}

/// Millimetres per unit for an SI length unit, read off its prefix.
fn si_length_mm(f: &StepFile, unit: u32) -> Option<f64> {
    let si = f.complex_part(unit, "SI_UNIT")?;
    Some(match f.str_of(*si.first().unwrap_or(&Value::Null)) {
        "MILLI" => 1.0,
        "CENTI" => 10.0,
        "DECI" => 100.0,
        "KILO" => 1e6,
        "MICRO" => 1e-3,
        // No prefix at all: the metre itself.
        _ => 1000.0,
    })
}

/// The conversion factor a `GLOBAL_UNIT_ASSIGNED_CONTEXT` states for one kind of
/// unit, when it states one as a conversion from the SI unit.
fn unit_scale(f: &StepFile, kind: &str) -> Option<f64> {
    for id in f.ids() {
        let Some(ctx) = f.complex_part(id, "GLOBAL_UNIT_ASSIGNED_CONTEXT") else {
            continue;
        };
        let Some(&list) = ctx.first() else { continue };
        for u in f.list(list).iter().filter_map(|v| v.as_ref_id()) {
            if f.complex_part(u, kind).is_none() {
                continue;
            }
            let Some(conv) = f.complex_part(u, "CONVERSION_BASED_UNIT") else {
                continue;
            };
            let Some(m) = conv.get(1).and_then(|v| v.as_ref_id()) else {
                continue;
            };
            let scale = f
                .params(m)
                .first()
                .and_then(|&v| v.as_f64().or_else(|| f.typed_params(v).first()?.as_f64()));
            if let Some(s) = scale.filter(|s| *s > 0.0 && s.is_finite()) {
                return Some(s);
            }
        }
    }
    None
}

/// Entity types that carry a body's worth of geometry, and so become one solid.
///
/// `GEOMETRIC_SET` is the odd one out: it holds faces directly rather than
/// shells, and may also hold loose curves.
fn is_solid_root(ty: &str) -> bool {
    matches!(
        ty,
        "MANIFOLD_SOLID_BREP"
            | "BREP_WITH_VOIDS"
            | "FACETED_BREP"
            | "SHELL_BASED_SURFACE_MODEL"
            | "GEOMETRIC_SET"
    )
}

/// Of those, the ones that describe a closed volume rather than a sheet.
///
/// A `SHELL_BASED_SURFACE_MODEL` is a set of surfaces and a `GEOMETRIC_SET` is
/// loose geometry; neither has an inside. The section cap counts entries
/// against exits, so a sheet would get its whole silhouette filled.
fn is_volume(ty: &str) -> bool {
    matches!(ty, "MANIFOLD_SOLID_BREP" | "BREP_WITH_VOIDS" | "FACETED_BREP")
}

/// Types that can stand as a face in their own right.
fn is_face_item(ty: &str) -> bool {
    matches!(
        ty,
        "CURVE_BOUNDED_SURFACE" | "ADVANCED_FACE" | "FACE_SURFACE" | "ORIENTED_FACE"
    )
}

/// Full knot vector for the B-spline subtypes that leave it implicit.
///
/// `n` is the number of control points, and the result always has `n + degree
/// + 1` entries.
fn implicit_knots(form: &str, degree: usize, n: usize) -> Option<Vec<f64>> {
    if degree == 0 || n <= degree {
        return None;
    }
    let mut knots = Vec::with_capacity(n + degree + 1);
    match form {
        // Clamped at both ends, single interior knots: the curve runs from the
        // first control point to the last.
        "QUASI_UNIFORM_CURVE" | "QUASI_UNIFORM_SURFACE" => {
            knots.extend(std::iter::repeat_n(0.0, degree + 1));
            knots.extend((1..=n - degree - 1).map(|i| i as f64));
            knots.extend(std::iter::repeat_n((n - degree) as f64, degree + 1));
        }
        // Every knot simple, so the curve stops short of its end points.
        "UNIFORM_CURVE" | "UNIFORM_SURFACE" => {
            knots.extend((0..=n + degree).map(|i| i as f64));
        }
        // A single Bezier span, which only exists when degree == n - 1.
        "BEZIER_CURVE" | "BEZIER_SURFACE" => {
            if n != degree + 1 {
                return None;
            }
            knots.extend(std::iter::repeat_n(0.0, degree + 1));
            knots.extend(std::iter::repeat_n(1.0, degree + 1));
        }
        _ => return None,
    }
    (knots.len() == n + degree + 1).then_some(knots)
}

struct Builder<'a> {
    f: &'a StepFile,
    surfaces: Vec<Surface>,
    curves: Vec<Curve>,
    /// entity id -> index into `surfaces` / `curves`, or -1.
    surf_of: Vec<i32>,
    curve_of: Vec<i32>,
    stats: BuildStats,
    vertex_bbox: Aabb,
    /// Bounds of the loose curves alone, kept apart so annotation cannot set the
    /// chord tolerance for the solids it annotates.
    wire_bbox: Aabb,
    /// Counter for edge ids invented for faceted loops, which have none.
    synthetic_edges: u32,
    /// Bounds of the solid currently being built, before placement.
    solid_bbox: Aabb,
    /// Radians per unit of the file's declared plane-angle measure.
    angle_scale: f64,
    /// How far a vertex may sit from an end of its curve and still name that
    /// end, in the file's length unit. See `BSplineCurve::param_of`.
    end_slack: f64,
}

impl<'a> Builder<'a> {
    // -- primitive readers -------------------------------------------------

    fn coords(&self, id: u32) -> Option<Vec3> {
        let p = self.f.params(id);
        let list = self.f.list(*p.get(1)?);
        let x = list.first()?.as_f64()?;
        let y = list.get(1)?.as_f64()?;
        // DIRECTION may be 2D; CARTESIAN_POINT may legally be 1D or 2D too.
        let z = list.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
        Some(v3(x, y, z))
    }

    fn point(&self, v: Value) -> Option<Vec3> {
        self.coords(v.as_ref_id()?)
    }

    fn direction(&self, v: Value) -> Option<Vec3> {
        if v.is_null() {
            return None;
        }
        self.coords(v.as_ref_id()?)
    }

    /// `VECTOR(name, orientation, magnitude)` collapsed to a single vector.
    fn vector(&self, v: Value) -> Option<Vec3> {
        let id = v.as_ref_id()?;
        let p = self.f.params(id);
        let dir = self.direction(*p.get(1)?)?;
        let mag = p.get(2).and_then(|m| m.as_f64()).unwrap_or(1.0);
        Some(dir.normalize_or(dir) * mag)
    }

    fn axis2(&self, v: Value) -> Option<Axis2> {
        let id = v.as_ref_id()?;
        let p = self.f.params(id);
        let origin = self.point(*p.get(1)?)?;
        let axis = p.get(2).and_then(|&a| self.direction(a));
        let refd = p.get(3).and_then(|&a| self.direction(a));
        Some(Axis2::new(origin, axis, refd))
    }

    /// A placement written either way round.
    ///
    /// A `MAPPED_ITEM` may target an `AXIS2_PLACEMENT_3D` or a
    /// `CARTESIAN_TRANSFORMATION_OPERATOR_3D`; missing either silently drops
    /// the component.
    ///
    /// The operator can also carry a uniform scale. It rides in the length of
    /// the basis vectors, so only normals need re-normalising on the way out.
    fn placement(&self, v: Value) -> Option<Xform> {
        let id = v.as_ref_id()?;
        if self.f.type_of(id) != "CARTESIAN_TRANSFORMATION_OPERATOR_3D" {
            return self.axis2(v).map(|a| Xform::between(&Axis2::IDENTITY, &a));
        }
        // CTO_3D(name, axis1, axis2, local_origin, scale, axis3): axis1 is the
        // x direction and axis3 the z, either of which may be absent.
        let p = self.f.params(id);
        let origin = self.point(*p.get(3)?)?;
        let refd = p.get(1).and_then(|&a| self.direction(a));
        let axis = p.get(5).and_then(|&a| self.direction(a));
        let scale = p.get(4).and_then(|v| v.as_f64()).filter(|s| *s > 0.0).unwrap_or(1.0);
        let frame = Axis2::new(origin, axis, refd);
        let mut x = Xform::between(&Axis2::IDENTITY, &frame);
        x.x = x.x * scale;
        x.y = x.y * scale;
        x.z = x.z * scale;
        Some(x)
    }

    fn vertex_point(&mut self, v: Value) -> Option<Vec3> {
        let id = v.as_ref_id()?;
        let p = self.f.params(id);
        let g = self.point(*p.get(1)?)?;
        self.vertex_bbox.add_point(g);
        self.solid_bbox.add_point(g);
        Some(g)
    }

    fn reals(&self, v: Value) -> Vec<f64> {
        self.f
            .list(v)
            .iter()
            .filter_map(|x| x.as_f64())
            .collect()
    }

    fn ints(&self, v: Value) -> Vec<i64> {
        self.f
            .list(v)
            .iter()
            .filter_map(|x| x.as_i64())
            .collect()
    }

    fn points(&self, v: Value) -> Option<Vec<Vec3>> {
        self.f.list(v).iter().map(|&p| self.point(p)).collect()
    }

    // -- curves ------------------------------------------------------------

    fn curve_index(&mut self, id: u32) -> Option<u32> {
        match self.curve_of.get(id as usize).copied() {
            Some(i) if i >= 0 => return Some(i as u32),
            Some(-2) => return None, // already known-unsupported
            _ => {}
        }
        let c = self.build_curve(id, 0);
        match c {
            Some(c) => {
                let idx = self.curves.len() as u32;
                self.curves.push(c);
                self.curve_of[id as usize] = idx as i32;
                Some(idx)
            }
            None => {
                let name = self.f.type_of(id).to_string();
                *self.stats.unsupported_curves.entry(name).or_default() += 1;
                self.curve_of[id as usize] = -2;
                None
            }
        }
    }

    fn build_curve(&mut self, id: u32, depth: u32) -> Option<Curve> {
        if depth > 8 {
            return None; // a cycle in the file, or nesting beyond reason
        }
        let p = self.f.params(id);
        match self.f.type_of(id) {
            // These carry surface-space representations *and* a genuine 3D
            // curve; attribute 2 is the latter, which is all we need.
            "SURFACE_CURVE" | "INTERSECTION_CURVE" | "SEAM_CURVE" => {
                let inner = p.get(1)?.as_ref_id()?;
                self.build_curve(inner, depth + 1)
            }
            // A trimmed curve is its basis curve restricted to a parameter
            // range. Used as an edge, the range comes from the edge's own
            // vertices instead, so only the basis matters here; the trim is
            // read separately when the curve is swept into a surface.
            "TRIMMED_CURVE" => {
                let inner = p.get(1)?.as_ref_id()?;
                self.build_curve(inner, depth + 1)
            }
            // B-splines whose knot vector is implied by the form rather than
            // written out.
            "UNIFORM_CURVE" | "QUASI_UNIFORM_CURVE" | "BEZIER_CURVE" => {
                let form = self.f.type_of(id);
                let degree = p.get(1)?.as_i64()? as usize;
                let ctrl = self.points(*p.get(2)?)?;
                let knots = implicit_knots(form, degree, ctrl.len())?;
                let c = BSplineCurve::new(degree, ctrl, None, knots);
                c.is_valid().then(|| Curve::BSpline(Box::new(c)))
            }
            "LINE" => {
                let origin = self.point(*p.get(1)?)?;
                let dir = self.vector(*p.get(2)?)?;
                if dir.length_sq() < 1e-300 {
                    return None;
                }
                Some(Curve::Line { origin, dir })
            }
            "CIRCLE" => {
                let ax = self.axis2(*p.get(1)?)?;
                let r = p.get(2)?.as_f64()?;
                Some(Curve::Circle { ax, r })
            }
            "ELLIPSE" => {
                let ax = self.axis2(*p.get(1)?)?;
                let a = p.get(2)?.as_f64()?;
                let b = p.get(3)?.as_f64()?;
                Some(Curve::Ellipse { ax, a, b })
            }
            "B_SPLINE_CURVE_WITH_KNOTS" => {
                let degree = p.get(1)?.as_i64()? as usize;
                let ctrl = self.points(*p.get(2)?)?;
                let mults = self.ints(*p.get(6)?);
                let knots = self.reals(*p.get(7)?);
                self.finish_bspline_curve(degree, ctrl, None, &knots, &mults)
            }
            crate::step::COMPLEX_TYPE => {
                // Rational B-spline: attributes are split across the parts, and
                // each part carries only its own.
                let base = self.f.complex_part(id, "B_SPLINE_CURVE")?;
                let degree = base.first()?.as_i64()? as usize;
                let ctrl = self.points(*base.get(1)?)?;
                let weights = self
                    .f
                    .complex_part(id, "RATIONAL_B_SPLINE_CURVE")
                    .and_then(|r| r.first().copied())
                    .map(|w| self.reals(w));

                // The knots are either written out in a `..._WITH_KNOTS` part
                // or implied by whichever uniform form appears instead. A
                // rational curve can be spelled either way.
                match self.f.complex_part(id, "B_SPLINE_CURVE_WITH_KNOTS") {
                    Some(wk) => {
                        let mults = self.ints(*wk.first()?);
                        let knots = self.reals(*wk.get(1)?);
                        self.finish_bspline_curve(degree, ctrl, weights, &knots, &mults)
                    }
                    None => {
                        let form = ["QUASI_UNIFORM_CURVE", "UNIFORM_CURVE", "BEZIER_CURVE"]
                            .into_iter()
                            .find(|f| self.f.complex_part(id, f).is_some())?;
                        let knots = implicit_knots(form, degree, ctrl.len())?;
                        self.make_bspline_curve(degree, ctrl, weights, knots)
                    }
                }
            }
            _ => None,
        }
    }

    /// Builds a curve from an already-expanded knot vector.
    fn make_bspline_curve(
        &self,
        degree: usize,
        ctrl: Vec<Vec3>,
        weights: Option<Vec<f64>>,
        knots: Vec<f64>,
    ) -> Option<Curve> {
        if let Some(w) = &weights
            && w.len() != ctrl.len()
        {
            return None;
        }
        let c = BSplineCurve::new(degree, ctrl, weights, knots);
        c.is_valid().then(|| Curve::BSpline(Box::new(c)))
    }

    fn finish_bspline_curve(
        &self,
        degree: usize,
        ctrl: Vec<Vec3>,
        weights: Option<Vec<f64>>,
        knots: &[f64],
        mults: &[i64],
    ) -> Option<Curve> {
        self.make_bspline_curve(degree, ctrl, weights, expand_knots(knots, mults))
    }

    // -- surfaces ----------------------------------------------------------

    fn surface_index(&mut self, id: u32) -> Option<u32> {
        match self.surf_of.get(id as usize).copied() {
            Some(i) if i >= 0 => return Some(i as u32),
            Some(-2) => return None,
            _ => {}
        }
        match self.build_surface(id) {
            Some(s) => {
                let idx = self.surfaces.len() as u32;
                self.surfaces.push(s);
                self.surf_of[id as usize] = idx as i32;
                Some(idx)
            }
            None => {
                let name = self.f.type_of(id).to_string();
                *self.stats.unsupported_surfaces.entry(name).or_default() += 1;
                self.surf_of[id as usize] = -2;
                None
            }
        }
    }

    /// `AXIS1_PLACEMENT(name, location, axis)`: a point and a direction, with
    /// no reference direction -- unlike `AXIS2_PLACEMENT_3D`.
    fn axis1(&self, v: Value) -> Option<(Vec3, Vec3)> {
        let id = v.as_ref_id()?;
        let p = self.f.params(id);
        let origin = self.point(*p.get(1)?)?;
        let dir = p
            .get(2)
            .and_then(|&a| self.direction(a))
            .unwrap_or(v3(0.0, 0.0, 1.0));
        Some((origin, dir.normalize_or(v3(0.0, 0.0, 1.0))))
    }

    /// Parameter range a curve entity covers, honouring `TRIMMED_CURVE`.
    ///
    /// Trim values may be given as parameters or as points on the curve, and
    /// for a conic the parameter is an angle whose unit the file chooses. A
    /// value past a full turn can only be degrees, which is the one ambiguity
    /// worth resolving by inspection.
    fn curve_range(&self, id: u32, curve: &Curve) -> (f64, f64) {
        let default = match curve {
            Curve::Circle { .. } | Curve::Ellipse { .. } => (0.0, TWO_PI),
            Curve::BSpline(c) => c.domain(),
            Curve::Line { .. } => (0.0, 1.0),
        };
        if self.f.type_of(id) != "TRIMMED_CURVE" {
            return default;
        }
        let p = self.f.params(id);
        let angular = matches!(curve, Curve::Circle { .. } | Curve::Ellipse { .. });
        let trim = |slot: usize| -> Option<f64> {
            let list = self.f.list(*p.get(slot)?);
            // Either a bare parameter value or a point on the curve.
            for v in list {
                if let Some(x) = v.as_f64() {
                    return Some(x);
                }
            }
            for v in list {
                if let Some(pt) = self.point(*v) {
                    return Some(curve.param_of_vertex(pt, self.end_slack));
                }
            }
            None
        };
        let (Some(mut a), Some(mut b)) = (trim(2), trim(3)) else {
            return default;
        };
        // A conic's parameter *is* an angle, so it carries the file's declared
        // plane-angle unit. Where nothing was declared, fall back to reading a
        // parameter past a whole turn as degrees, which is the only thing it
        // can sensibly be.
        if angular {
            if self.angle_scale != 1.0 {
                a *= self.angle_scale;
                b *= self.angle_scale;
            } else if a.abs() > TWO_PI + 1e-9 || b.abs() > TWO_PI + 1e-9 {
                a = a.to_radians();
                b = b.to_radians();
            }
        }
        if (b - a).abs() < 1e-12 {
            return default;
        }
        (a, b)
    }

    fn build_surface(&mut self, id: u32) -> Option<Surface> {
        let p = self.f.params(id);
        match self.f.type_of(id) {
            "PLANE" => Some(Surface::Plane(self.axis2(*p.get(1)?)?)),
            // SURFACE_OF_REVOLUTION(name, swept_curve, axis_position)
            "SURFACE_OF_REVOLUTION" => {
                let curve_id = p.get(1)?.as_ref_id()?;
                let meridian = self.build_curve(curve_id, 0)?;
                let (v0, v1) = self.curve_range(curve_id, &meridian);
                let (origin, dir) = self.axis1(*p.get(2)?)?;
                let s = RevolutionSurface {
                    meridian,
                    v0: v0.min(v1),
                    v1: v0.max(v1),
                    origin,
                    dir,
                };
                (s.v1 > s.v0).then(|| Surface::Revolution(Box::new(s)))
            }
            // SURFACE_OF_LINEAR_EXTRUSION(name, swept_curve, extrusion_axis)
            "SURFACE_OF_LINEAR_EXTRUSION" => {
                let curve_id = p.get(1)?.as_ref_id()?;
                let profile = self.build_curve(curve_id, 0)?;
                let (u0, u1) = self.curve_range(curve_id, &profile);
                let dir = self.vector(*p.get(2)?)?;
                if dir.length_sq() < 1e-24 {
                    return None;
                }
                let s = ExtrusionSurface {
                    profile,
                    u0: u0.min(u1),
                    u1: u0.max(u1),
                    dir,
                };
                (s.u1 > s.u0).then(|| Surface::Extrusion(Box::new(s)))
            }
            "CYLINDRICAL_SURFACE" => Some(Surface::Cylinder {
                ax: self.axis2(*p.get(1)?)?,
                r: p.get(2)?.as_f64()?,
            }),
            "CONICAL_SURFACE" => Some(Surface::Cone {
                ax: self.axis2(*p.get(1)?)?,
                r: p.get(2)?.as_f64()?,
                half_angle: p.get(3)?.as_f64()? * self.angle_scale,
            }),
            "SPHERICAL_SURFACE" => Some(Surface::Sphere {
                ax: self.axis2(*p.get(1)?)?,
                r: p.get(2)?.as_f64()?,
            }),
            "TOROIDAL_SURFACE" => Some(Surface::Torus {
                ax: self.axis2(*p.get(1)?)?,
                major: p.get(2)?.as_f64()?,
                minor: p.get(3)?.as_f64()?,
                inner: false,
            }),
            // Same surface, but fat enough to cut through its own axis; the
            // flag says which of the two sheets that leaves is wanted.
            "DEGENERATE_TOROIDAL_SURFACE" => Some(Surface::Torus {
                ax: self.axis2(*p.get(1)?)?,
                major: p.get(2)?.as_f64()?,
                minor: p.get(3)?.as_f64()?,
                inner: self.f.str_of(*p.get(4)?) == "F",
            }),
            form @ ("UNIFORM_SURFACE" | "QUASI_UNIFORM_SURFACE" | "BEZIER_SURFACE") => {
                let u_degree = p.get(1)?.as_i64()? as usize;
                let v_degree = p.get(2)?.as_i64()? as usize;
                let (ctrl, nu, nv) = self.control_grid(*p.get(3)?)?;
                let s = BSplineSurface {
                    u_degree,
                    v_degree,
                    ctrl,
                    weights: None,
                    nu,
                    nv,
                    u_knots: implicit_knots(form, u_degree, nu)?,
                    v_knots: implicit_knots(form, v_degree, nv)?,
                    ..Default::default()
                };
                s.into_surface()
            }
            "B_SPLINE_SURFACE_WITH_KNOTS" => {
                let u_degree = p.get(1)?.as_i64()? as usize;
                let v_degree = p.get(2)?.as_i64()? as usize;
                let grid = self.control_grid(*p.get(3)?)?;
                let u_mults = self.ints(*p.get(8)?);
                let v_mults = self.ints(*p.get(9)?);
                let u_knots = self.reals(*p.get(10)?);
                let v_knots = self.reals(*p.get(11)?);
                self.finish_bspline_surface(
                    u_degree, v_degree, grid, None, &u_knots, &u_mults, &v_knots, &v_mults,
                )
            }
            crate::step::COMPLEX_TYPE => {
                let base = self.f.complex_part(id, "B_SPLINE_SURFACE")?;
                let u_degree = base.first()?.as_i64()? as usize;
                let v_degree = base.get(1)?.as_i64()? as usize;
                let grid = self.control_grid(*base.get(2)?)?;
                let weights = self
                    .f
                    .complex_part(id, "RATIONAL_B_SPLINE_SURFACE")
                    .and_then(|r| r.first().copied())
                    .map(|w| {
                        self.f
                            .list(w)
                            .iter()
                            .flat_map(|&row| self.reals(row))
                            .collect::<Vec<f64>>()
                    });

                // As for curves, the knots are either written out or implied.
                match self.f.complex_part(id, "B_SPLINE_SURFACE_WITH_KNOTS") {
                    Some(wk) => {
                        let u_mults = self.ints(*wk.first()?);
                        let v_mults = self.ints(*wk.get(1)?);
                        let u_knots = self.reals(*wk.get(2)?);
                        let v_knots = self.reals(*wk.get(3)?);
                        self.finish_bspline_surface(
                            u_degree, v_degree, grid, weights, &u_knots, &u_mults, &v_knots,
                            &v_mults,
                        )
                    }
                    None => {
                        let form =
                            ["QUASI_UNIFORM_SURFACE", "UNIFORM_SURFACE", "BEZIER_SURFACE"]
                                .into_iter()
                                .find(|f| self.f.complex_part(id, f).is_some())?;
                        let (ctrl, nu, nv) = grid;
                        let s = BSplineSurface {
                            u_degree,
                            v_degree,
                            weights: weights.filter(|w| w.len() == ctrl.len()),
                            ctrl,
                            nu,
                            nv,
                            u_knots: implicit_knots(form, u_degree, nu)?,
                            v_knots: implicit_knots(form, v_degree, nv)?,
                            ..Default::default()
                        };
                        s.into_surface()
                    }
                }
            }
            _ => None,
        }
    }

    /// Reads `LIST OF LIST OF CARTESIAN_POINT` as (rows along u, cols along v).
    fn control_grid(&self, v: Value) -> Option<(Vec<Vec3>, usize, usize)> {
        let rows = self.f.list(v);
        let nu = rows.len();
        if nu == 0 {
            return None;
        }
        let nv = self.f.list(rows[0]).len();
        if nv == 0 {
            return None;
        }
        let mut ctrl = Vec::with_capacity(nu * nv);
        for &row in rows {
            let cols = self.f.list(row);
            if cols.len() != nv {
                return None;
            }
            for &c in cols {
                ctrl.push(self.point(c)?);
            }
        }
        Some((ctrl, nu, nv))
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_bspline_surface(
        &self,
        u_degree: usize,
        v_degree: usize,
        grid: (Vec<Vec3>, usize, usize),
        weights: Option<Vec<f64>>,
        u_knots: &[f64],
        u_mults: &[i64],
        v_knots: &[f64],
        v_mults: &[i64],
    ) -> Option<Surface> {
        let (ctrl, nu, nv) = grid;
        if let Some(w) = &weights
            && w.len() != ctrl.len()
        {
            return None;
        }
        let s = BSplineSurface {
            u_degree,
            v_degree,
            ctrl,
            weights,
            nu,
            nv,
            u_knots: expand_knots(u_knots, u_mults),
            v_knots: expand_knots(v_knots, v_mults),
            ..Default::default()
        };
        s.into_surface()
    }

    // -- topology ----------------------------------------------------------

    /// Resolves an `ORIENTED_EDGE` to a parameter range on its curve.
    ///
    /// `lone_in_loop` says this is the loop's only edge, which is what settles
    /// whether a periodic curve was meant to close on itself.
    fn edge_use(&mut self, oriented_edge: u32, lone_in_loop: bool) -> Option<EdgeUse> {
        let p = self.f.params(oriented_edge);
        let (edge_id, forward) = if self.f.type_of(oriented_edge) == "ORIENTED_EDGE" {
            let e = p.get(3)?.as_ref_id()?;
            (e, self.f.str_of(*p.get(4)?) != "F")
        } else {
            (oriented_edge, true)
        };

        if self.f.type_of(edge_id) != "EDGE_CURVE" {
            return None;
        }
        let ep = self.f.params(edge_id);
        let v_start = *ep.get(1)?;
        let v_end = *ep.get(2)?;
        let curve_id = ep.get(3)?.as_ref_id()?;
        let same_sense = self.f.str_of(*ep.get(4)?) != "F";

        // Does this edge close on itself and cover the whole curve?
        //
        // One vertex entity at both ends says so, but not every translator
        // shares the vertex. The other certain case is an edge alone in its
        // loop: an arc on its own bounds nothing. Guessing from a near-zero
        // parameter span instead would expand a legitimate hairline edge into
        // a full circle and add a spurious disc.
        let closed_edge =
            lone_in_loop || (v_start.as_ref_id().is_some() && v_start.as_ref_id() == v_end.as_ref_id());

        let p_start = self.vertex_point(v_start)?;
        let p_end = self.vertex_point(v_end)?;

        let curve = self.curve_index(curve_id)?;
        let c = &self.curves[curve as usize];
        let mut t_a = c.param_of_vertex(p_start, self.end_slack);
        let mut t_b = c.param_of_vertex(p_end, self.end_slack);

        if let Some(period) = c.period() {
            // The vertices only fix the parameters modulo the period, so the
            // edge's sense picks which of the two arcs is meant; closure was
            // decided above from the vertices, not the parameters.
            if closed_edge {
                t_b = if same_sense { t_a + period } else { t_a - period };
            } else {
                if same_sense {
                    while t_b < t_a {
                        t_b += period;
                    }
                } else {
                    while t_b > t_a {
                        t_b -= period;
                    }
                }
                // Distinct vertices that land on the same point: a hairline
                // edge with nothing to contribute.
                if (t_b - t_a).abs() < period * 1e-9 {
                    return None;
                }
            }
        } else if (t_b - t_a).abs() < 1e-12 {
            return None; // degenerate edge
        }

        // The vertices alone are not the extent of the edge: a solid of whole
        // circles through one vertex would otherwise have a point for a box,
        // and a tolerance to match.
        let c = &self.curves[curve as usize];
        c.extend_bounds(t_a, t_b, &mut self.vertex_bbox);
        c.extend_bounds(t_a, t_b, &mut self.solid_bbox);

        if !forward {
            std::mem::swap(&mut t_a, &mut t_b);
        }
        Some(EdgeUse {
            curve,
            edge: edge_id,
            t0: t_a,
            t1: t_b,
        })
    }

    fn bound(&mut self, bound_id: u32) -> Option<Bound> {
        let ty = self.f.type_of(bound_id);
        let outer = ty == "FACE_OUTER_BOUND";
        if !outer && ty != "FACE_BOUND" {
            return None;
        }
        let p = self.f.params(bound_id);
        let loop_id = p.get(1)?.as_ref_id()?;
        let orientation = self.f.str_of(*p.get(2)?) != "F";

        // A loop that is one vertex: the face closes to a point there, and the
        // point says which degenerate parameter line it is -- which the
        // traversal direction cannot tell for a sphere's two poles.
        if self.f.type_of(loop_id) == "VERTEX_LOOP" {
            let lp = self.f.params(loop_id);
            let p = self.vertex_point(*lp.get(1)?)?;
            return Some(Bound {
                outer,
                edges: Vec::new(),
                vertex: Some(p),
            });
        }

        // A faceted brep bounds its faces with a plain list of points instead
        // of an edge loop. Turn each span into a line segment so the rest of
        // the pipeline sees an ordinary loop.
        if self.f.type_of(loop_id) == "POLY_LOOP" {
            let lp = self.f.params(loop_id);
            let pts: Vec<Vec3> = self
                .f
                .list(*lp.get(1)?)
                .iter()
                .filter_map(|&v| self.point(v))
                .collect();
            if pts.len() < 3 {
                return None;
            }
            for p in &pts {
                self.vertex_bbox.add_point(*p);
                self.solid_bbox.add_point(*p);
            }
            let mut edges = Vec::with_capacity(pts.len());
            for i in 0..pts.len() {
                let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
                let dir = b - a;
                if dir.length_sq() < 1e-24 {
                    continue;
                }
                let curve = self.curves.len() as u32;
                self.curves.push(Curve::Line { origin: a, dir });
                // Synthetic ids, above every real one, so the edge overlay
                // never confuses them with entities from the file.
                self.synthetic_edges += 1;
                edges.push(EdgeUse {
                    curve,
                    edge: self.f.id_bound() + self.synthetic_edges,
                    t0: 0.0,
                    t1: 1.0,
                });
            }
            if edges.is_empty() {
                return None;
            }
            if !orientation {
                edges.reverse();
                for e in &mut edges {
                    std::mem::swap(&mut e.t0, &mut e.t1);
                }
            }
            return Some(Bound { outer, edges, vertex: None });
        }

        if self.f.type_of(loop_id) != "EDGE_LOOP" {
            return None; // VERTEX_LOOP carries no area
        }
        let lp = self.f.params(loop_id);
        let items: Vec<u32> = self
            .f
            .list(*lp.get(1)?)
            .iter()
            .filter_map(|v| v.as_ref_id())
            .collect();

        // A loop written as a single edge on a periodic curve has to close that
        // curve, whatever its vertices say. Anywhere else a periodic edge is an
        // arc, so a hairline -- distinct vertices landing on the same point --
        // simply drops out, leaving the loop's other edges to bound the face.
        let lone = items.len() == 1;
        let mut edges: Vec<EdgeUse> =
            items.iter().filter_map(|&e| self.edge_use(e, lone)).collect();
        self.stats.dropped_edges += items.len() - edges.len();
        if edges.is_empty() {
            return None;
        }
        if !orientation {
            edges.reverse();
            for e in &mut edges {
                std::mem::swap(&mut e.t0, &mut e.t1);
            }
        }
        Some(Bound { outer, edges, vertex: None })
    }

    /// A geometrically bounded face: a surface trimmed by composite curves
    /// instead of by an edge loop.
    ///
    /// There are no vertices anywhere in this model, so a segment's parameter
    /// range has to come from its own trimming rather than from the points at
    /// its ends. Otherwise it maps onto an ordinary face exactly.
    fn curve_bounded_face(&mut self, id: u32) -> Option<Face> {
        let p = self.f.params(id);
        // CURVE_BOUNDED_SURFACE(basis_surface, boundaries, implicit_outer)
        let surface = self.surface_index(p.get(1)?.as_ref_id()?)?;
        let boundary_ids: Vec<u32> = self
            .f
            .list(*p.get(2)?)
            .iter()
            .filter_map(|v| v.as_ref_id())
            .collect();

        let mut bounds = Vec::new();
        for b in boundary_ids {
            // OUTER_BOUNDARY_CURVE and BOUNDARY_CURVE are both composite
            // curves; only the type says which one encloses the face.
            let outer = self.f.type_of(b) == "OUTER_BOUNDARY_CURVE";
            let Some(&segments) = self.f.params(b).get(1) else {
                continue;
            };
            let seg_ids: Vec<u32> = self
                .f
                .list(segments)
                .iter()
                .filter_map(|v| v.as_ref_id())
                .collect();

            let mut edges = Vec::new();
            for s in seg_ids {
                // COMPOSITE_CURVE_SEGMENT(transition, same_sense, parent_curve)
                let sp = self.f.params(s);
                let (Some(parent), Some(sense)) =
                    (sp.get(2).and_then(|v| v.as_ref_id()), sp.get(1))
                else {
                    continue;
                };
                let forward = self.f.str_of(*sense) != "F";
                let Some(curve) = self.curve_index(parent) else {
                    continue;
                };
                let (mut t0, mut t1) =
                    self.curve_range(parent, &self.curves[curve as usize]);
                if !forward {
                    std::mem::swap(&mut t0, &mut t1);
                }
                // No vertices in this model, so the bounds come from the curve
                // itself (a whole circle begins and ends at one point).
                let c = &self.curves[curve as usize];
                c.extend_bounds(t0, t1, &mut self.vertex_bbox);
                c.extend_bounds(t0, t1, &mut self.solid_bbox);

                self.synthetic_edges += 1;
                edges.push(EdgeUse {
                    curve,
                    edge: self.f.id_bound() + self.synthetic_edges,
                    t0,
                    t1,
                });
            }
            if !edges.is_empty() {
                bounds.push(Bound { outer, edges, vertex: None });
            }
        }
        (!bounds.is_empty()).then_some(Face {
            surface,
            same_sense: true,
            bounds,
            // This model always states its boundaries.
            domain: None,
            // Filled in by the caller, which knows the entity id a
            // `STYLED_ITEM` would name.
            style: None,
        })
    }

    fn face(&mut self, face_id: u32) -> Option<Face> {
        if self.f.type_of(face_id) == "CURVE_BOUNDED_SURFACE" {
            return self.curve_bounded_face(face_id);
        }
        // ORIENTED_FACE wraps another face and flips it. Its inherited `bounds`
        // attribute is derived, written `*`, so the wrapped face is attribute 2,
        // not the surface a positional read would expect.
        if self.f.type_of(face_id) == "ORIENTED_FACE" {
            let p = self.f.params(face_id);
            let inner = p.get(2)?.as_ref_id()?;
            let forward = self.f.str_of(*p.get(3)?) != "F";
            let mut face = self.face(inner)?;
            if !forward {
                face.same_sense = !face.same_sense;
            }
            return Some(face);
        }
        let p = self.f.params(face_id);
        let surface_id = p.get(2)?.as_ref_id()?;
        let mut same_sense = self.f.str_of(*p.get(3)?) != "F";
        let (surface_id, domain, reversed) = self.untrim(surface_id);
        // Reversing one parameter direction turns the surface normal round,
        // and the face's sense is stated against the trimmed surface, not the
        // basis now used.
        if reversed {
            same_sense = !same_sense;
        }
        // Counted apart from faces lost for want of a boundary: this one has a
        // surface type the reader does not know.
        let Some(surface) = self.surface_index(surface_id) else {
            self.stats.faces_unsupported += 1;
            return None;
        };

        let bound_ids: Vec<u32> = self
            .f
            .list(*p.get(1)?)
            .iter()
            .filter_map(|v| v.as_ref_id())
            .collect();
        let bounds: Vec<Bound> = bound_ids.iter().filter_map(|&b| self.bound(b)).collect();
        // A face that loses some of its bounds still tessellates, as the wrong
        // shape, so count it.
        self.stats.dropped_bounds += bound_ids.len() - bounds.len();
        // An empty bound list is not a broken face: a closed surface (a ball,
        // an O-ring) needs no boundary. Losing every bound a face did have
        // still fails.
        if bounds.is_empty() && !bound_ids.is_empty() {
            return None;
        }
        // A face with no bounds contributes no vertices, yet the bounding box
        // sets the chord tolerance and the camera; size it from the surface's
        // own corners and centre.
        if bounds.is_empty()
            && let Some((u0, u1, v0, v1)) =
                domain.or_else(|| self.surfaces[surface as usize].full_domain())
        {
            let s = &self.surfaces[surface as usize];
            let at = |a: f64, b: f64, t: f64| a + (b - a) * t;
            let mut pts = Vec::with_capacity(9);
            for i in 0..3 {
                for j in 0..3 {
                    let (fu, fv) = (i as f64 * 0.5, j as f64 * 0.5);
                    pts.push(s.eval(at(u0, u1, fu), at(v0, v1, fv)));
                }
            }
            for p in pts {
                if p.is_finite() {
                    self.vertex_bbox.add_point(p);
                    self.solid_bbox.add_point(p);
                }
            }
        }
        Some(Face {
            surface,
            same_sense,
            bounds,
            domain,
            style: None,
        })
    }

    /// Sees through a `RECTANGULAR_TRIMMED_SURFACE` to the surface it trims.
    ///
    /// The trim does not reparameterise, so evaluation goes straight to the
    /// basis. The trim adds where the face stops when it has no bounds of its
    /// own, and whether one direction is turned round.
    ///
    /// Returns the entity to build the surface from, that domain, and whether
    /// the normal is reversed. Anything else is passed through untouched.
    fn untrim(&mut self, id: u32) -> (u32, Option<Domain>, bool) {
        if self.f.type_of(id) != "RECTANGULAR_TRIMMED_SURFACE" {
            return (id, None, false);
        }
        // RECTANGULAR_TRIMMED_SURFACE(name, basis, u1, u2, v1, v2, usense, vsense)
        let p = self.f.params(id);
        let Some(basis) = p.get(1).and_then(|v| v.as_ref_id()) else {
            return (id, None, false);
        };
        let num = |slot: usize| p.get(slot).and_then(|v| v.as_f64());
        let flag = |slot: usize| p.get(slot).map(|v| self.f.str_of(*v) != "F");
        let (Some(u0), Some(u1), Some(v0), Some(v1)) = (num(2), num(3), num(4), num(5)) else {
            return (basis, None, false);
        };
        // A trim on an angular direction is quoted in the file's plane-angle
        // unit, often degrees, as a conic's trim is.
        let scale = plane_angle_scale(self.f);
        let (au, av) = match self.surface_index(basis).map(|i| &self.surfaces[i as usize]) {
            Some(s) => s.angular_uv(),
            None => (false, false),
        };
        let s = |x: f64, angular: bool| if angular { x * scale } else { x };
        let reversed = flag(6).unwrap_or(true) != flag(7).unwrap_or(true);
        (
            basis,
            Some((s(u0, au), s(u1, au), s(v0, av), s(v1, av))),
            reversed,
        )
    }

    // -- assembly structure ------------------------------------------------

    /// True when a representation directly carries body geometry, as opposed to
    /// being a placeholder that only points at other representations.
    fn rep_has_solids(&self, rep: u32) -> bool {
        self.rep_items(rep)
            .iter()
            .filter_map(|v| v.as_ref_id())
            .any(|id| is_solid_root(self.f.type_of(id)))
    }

    /// Items of a representation, or empty if `rep` is not one.
    fn rep_items(&self, rep: u32) -> &'a [Value] {
        let p = self.f.params(rep);
        match p.get(1) {
            Some(&items) => self.f.list(items),
            None => &[],
        }
    }

    /// `PRODUCT_DEFINITION` -> the shape representation that describes it.
    fn product_shapes(&self) -> HashMap<u32, u32> {
        let mut out = HashMap::new();
        for id in self.f.ids() {
            if self.f.type_of(id) != "SHAPE_DEFINITION_REPRESENTATION" {
                continue;
            }
            // PROPERTY_DEFINITION_REPRESENTATION(definition, used_representation)
            let p = self.f.params(id);
            let (Some(pds), Some(rep)) = (
                p.first().and_then(|v| v.as_ref_id()),
                p.get(1).and_then(|v| v.as_ref_id()),
            ) else {
                continue;
            };
            // PRODUCT_DEFINITION_SHAPE(name, description, definition)
            if let Some(pd) = self.f.params(pds).get(2).and_then(|v| v.as_ref_id()) {
                out.insert(pd, rep);
            }
        }
        out
    }

    /// Parent representation -> its children and where each one sits.
    ///
    /// STEP has two ways of saying this and files use both. A `MAPPED_ITEM`
    /// instances one representation inside another directly. A
    /// `CONTEXT_DEPENDENT_SHAPE_REPRESENTATION` instead ties a transform to an
    /// assembly link in the *product* structure, and which of the two related
    /// representations is the child is only implied, so the product structure
    /// is consulted rather than assuming an attribute order that translators
    /// disagree about.
    fn assembly_links(&self) -> HashMap<u32, Vec<(u32, Xform)>> {
        let mut links: HashMap<u32, Vec<(u32, Xform)>> = HashMap::new();
        let shapes = self.product_shapes();

        for id in self.f.ids() {
            match self.f.type_of(id) {
                "MAPPED_ITEM" => {
                    // MAPPED_ITEM(name, mapping_source, mapping_target)
                    let p = self.f.params(id);
                    let Some(map_id) = p.get(1).and_then(|v| v.as_ref_id()) else {
                        continue;
                    };
                    // REPRESENTATION_MAP(mapping_origin, mapped_representation)
                    let mp = self.f.params(map_id);
                    let (Some(origin), Some(child)) = (
                        mp.first().and_then(|&v| self.axis2(v)),
                        mp.get(1).and_then(|v| v.as_ref_id()),
                    ) else {
                        continue;
                    };
                    let Some(target) = p.get(2).and_then(|&v| self.placement(v)) else {
                        continue;
                    };
                    // The mapped item belongs to whichever representation lists it.
                    for parent in self.f.ids().filter(|&r| is_shape_rep(self.f.type_of(r))) {
                        if self
                            .rep_items(parent)
                            .iter()
                            .any(|v| v.as_ref_id() == Some(id))
                        {
                            links
                                .entry(parent)
                                .or_default()
                                // The target is already a world-from-frame
                            // transform (and may carry a scale), so bring the
                            // map's own origin frame to world first.
                            .push((
                                child,
                                target.compose(&Xform::between(&origin, &Axis2::IDENTITY)),
                            ));
                        }
                    }
                }

                "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION" => {
                    // CDSR(representation_relation, represented_product_relation)
                    let p = self.f.params(id);
                    let (Some(rel), Some(pds)) = (
                        p.first().and_then(|v| v.as_ref_id()),
                        p.get(1).and_then(|v| v.as_ref_id()),
                    ) else {
                        continue;
                    };

                    // The relationship is a complex instance carrying both the
                    // pair of representations and the transform between them.
                    let Some(rr) = self.f.complex_part(rel, "REPRESENTATION_RELATIONSHIP") else {
                        continue;
                    };
                    let (Some(rep_1), Some(rep_2)) = (
                        rr.get(2).and_then(|v| v.as_ref_id()),
                        rr.get(3).and_then(|v| v.as_ref_id()),
                    ) else {
                        continue;
                    };
                    let Some(idt) = self
                        .f
                        .complex_part(rel, "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION")
                        .and_then(|t| t.first())
                        .and_then(|v| v.as_ref_id())
                    else {
                        continue;
                    };
                    // ITEM_DEFINED_TRANSFORMATION(name, description, item_1, item_2)
                    let ip = self.f.params(idt);
                    let (Some(item_1), Some(item_2)) = (
                        ip.get(2).and_then(|&v| self.axis2(v)),
                        ip.get(3).and_then(|&v| self.axis2(v)),
                    ) else {
                        continue;
                    };

                    // Which side is the component? Ask the product structure:
                    // NAUO(id, name, description, relating = parent, related = child).
                    let child_rep = self
                        .f
                        .params(pds)
                        .get(2)
                        .and_then(|v| v.as_ref_id())
                        .filter(|&n| self.f.type_of(n) == "NEXT_ASSEMBLY_USAGE_OCCURRENCE")
                        .and_then(|n| self.f.params(n).get(4).and_then(|v| v.as_ref_id()))
                        .and_then(|pd| shapes.get(&pd).copied());

                    // Attribute 3 being the component is the usual convention;
                    // it is only overridden when the product structure clearly
                    // says otherwise.
                    let (parent, child, from, to) = if child_rep == Some(rep_2) {
                        (rep_1, rep_2, item_2, item_1)
                    } else {
                        (rep_2, rep_1, item_1, item_2)
                    };
                    if is_shape_rep(self.f.type_of(parent)) && is_shape_rep(self.f.type_of(child)) {
                        links
                            .entry(parent)
                            .or_default()
                            .push((child, Xform::between(&from, &to)));
                    }
                }

                // A relationship with no transform still matters: it links a
                // placeholder representation to the one holding the geometry,
                // which would otherwise be a root and miss its placement.
                "SHAPE_REPRESENTATION_RELATIONSHIP" | "REPRESENTATION_RELATIONSHIP" => {
                    let p = self.f.params(id);
                    let (Some(a), Some(b)) = (
                        p.get(2).and_then(|v| v.as_ref_id()),
                        p.get(3).and_then(|v| v.as_ref_id()),
                    ) else {
                        continue;
                    };
                    if !is_shape_rep(self.f.type_of(a)) || !is_shape_rep(self.f.type_of(b)) {
                        continue;
                    }
                    // Whichever side carries bodies is the one being pointed at.
                    let (parent, child) = match (self.rep_has_solids(a), self.rep_has_solids(b)) {
                        (false, true) => (a, b),
                        (true, false) => (b, a),
                        _ => (a, b),
                    };
                    links
                        .entry(parent)
                        .or_default()
                        .push((child, Xform::IDENTITY));
                }
                _ => {}
            }
        }
        links
    }

    /// Every placed instance of a representation, in tree order.
    ///
    /// Each occurrence keeps the index of the one that instanced it, which is
    /// the assembly tree: a component in two subassemblies is two occurrences
    /// with different parents.
    fn assembly_occurrences(&self) -> Vec<Occurrence> {
        let links = self.assembly_links();
        let children: HashSet<u32> = links
            .values()
            .flat_map(|v| v.iter().map(|(c, _)| *c))
            .collect();

        let roots: Vec<u32> = self
            .f
            .ids()
            .filter(|&r| is_shape_rep(self.f.type_of(r)) && !children.contains(&r))
            .collect();

        let names = self.product_names();
        let mut out: Vec<Occurrence> = Vec::new();
        // Depth-first with an explicit stack, bounded on both depth and total
        // instances so a malformed file cannot spin here.
        let mut stack: Vec<(u32, Xform, u32, Option<usize>)> = roots
            .into_iter()
            .rev()
            .map(|r| (r, Xform::IDENTITY, 0, None))
            .collect();
        while let Some((rep, world, depth, parent)) = stack.pop() {
            if out.len() >= MAX_OCCURRENCES {
                break;
            }
            let me = out.len();
            out.push(Occurrence {
                rep,
                world,
                parent,
                name: names.get(&rep).cloned().unwrap_or_default(),
            });
            if depth >= MAX_ASSEMBLY_DEPTH {
                continue;
            }
            // Reversed, so that popping walks the children in the order the file
            // lists them and the printed tree matches the file.
            for (child, local) in links.get(&rep).into_iter().flatten().rev() {
                stack.push((*child, world.compose(local), depth + 1, Some(me)));
            }
        }
        out
    }

    /// Representation -> the name of the product it describes.
    ///
    /// The chain is `SHAPE_DEFINITION_REPRESENTATION` -> `PRODUCT_DEFINITION` ->
    /// `PRODUCT_DEFINITION_FORMATION` -> `PRODUCT`, and the name wanted is the
    /// last one's. A solid's own name is usually blank, or names the body
    /// rather than the component ("Solid1").
    fn product_names(&self) -> HashMap<u32, String> {
        let mut out = HashMap::new();
        for (pd, rep) in self.product_shapes() {
            // PRODUCT_DEFINITION(id, description, formation, frame)
            let name = self
                .f
                .params(pd)
                .get(2)
                .and_then(|v| v.as_ref_id())
                // PRODUCT_DEFINITION_FORMATION(id, description, of_product)
                .and_then(|form| self.f.params(form).get(2).and_then(|v| v.as_ref_id()))
                // PRODUCT(id, name, description, frame_of_reference)
                .map(|prod| {
                    let p = self.f.params(prod);
                    let name = self.f.str_of(*p.get(1).unwrap_or(&Value::Null));
                    if name.is_empty() {
                        self.f.str_of(*p.first().unwrap_or(&Value::Null))
                    } else {
                        name
                    }
                    .to_string()
                });
            if let Some(name) = name.filter(|n| !n.is_empty()) {
                out.insert(rep, name);
            }
        }
        out
    }

    /// Collects the `ADVANCED_FACE` ids of a shell, honouring orientation.
    fn shell_faces(&self, shell_id: u32, flip: bool, out: &mut Vec<(u32, bool)>) {
        match self.f.type_of(shell_id) {
            "ORIENTED_CLOSED_SHELL" | "ORIENTED_OPEN_SHELL" => {
                // The inherited face set is derived in these subtypes, written
                // `*`, so the wrapped shell lands one attribute later than a
                // plain reading suggests (as for ORIENTED_FACE). Missing it
                // drops the shell silently, and a void's faces escape their
                // solid. Take the first reference, so either spelling works.
                let p = self.f.params(shell_id);
                let found = (1..=2).find_map(|i| {
                    p.get(i).and_then(|v| v.as_ref_id()).map(|r| (i, r))
                });
                if let Some((i, inner)) = found {
                    let forward = p
                        .get(i + 1)
                        .map(|&v| self.f.str_of(v) != "F")
                        .unwrap_or(true);
                    self.shell_faces(inner, flip ^ !forward, out);
                }
            }
            "CLOSED_SHELL" | "OPEN_SHELL" | "CONNECTED_FACE_SET" => {
                let p = self.f.params(shell_id);
                if let Some(&faces) = p.get(1) {
                    for v in self.f.list(faces) {
                        if let Some(fid) = v.as_ref_id() {
                            out.push((fid, flip));
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// Follows references from a styled item until a colour is found.
fn resolve_color(f: &StepFile, root: Value, depth: usize, seen: &mut HashSet<u32>) -> Option<[f32; 3]> {
    if depth == 0 {
        return None;
    }
    match root {
        Value::Ref(id) => {
            if !seen.insert(id) {
                return None;
            }
            let p = f.params(id);
            match f.type_of(id) {
                "COLOUR_RGB" => {
                    let r = p.get(1)?.as_f64()? as f32;
                    let g = p.get(2)?.as_f64()? as f32;
                    let b = p.get(3)?.as_f64()? as f32;
                    Some([r, g, b])
                }
                "DRAUGHTING_PRE_DEFINED_COLOUR" => predefined_color(f.str_of(*p.first()?)),
                _ => p
                    .iter()
                    .find_map(|&sub| resolve_color(f, sub, depth - 1, seen)),
            }
        }
        Value::List { .. } => f
            .list(root)
            .iter()
            .find_map(|&sub| resolve_color(f, sub, depth - 1, seen)),
        _ => None,
    }
}

fn predefined_color(name: &str) -> Option<[f32; 3]> {
    Some(match name.to_ascii_lowercase().as_str() {
        "red" => [1.0, 0.0, 0.0],
        "green" => [0.0, 1.0, 0.0],
        "blue" => [0.0, 0.0, 1.0],
        "yellow" => [1.0, 1.0, 0.0],
        "magenta" => [1.0, 0.0, 1.0],
        "cyan" => [0.0, 1.0, 1.0],
        "black" => [0.0, 0.0, 0.0],
        "white" => [1.0, 1.0, 1.0],
        _ => return None,
    })
}

/// Follows the same references looking for `SURFACE_STYLE_TRANSPARENT`.
///
/// STEP states transparency, where 0 is opaque, so alpha is one minus it. It
/// hangs off `SURFACE_STYLE_RENDERING_WITH_PROPERTIES` beside the colour, not
/// on it, hence a second walk.
fn resolve_alpha(f: &StepFile, root: Value, depth: usize, seen: &mut HashSet<u32>) -> Option<f32> {
    if depth == 0 {
        return None;
    }
    match root {
        Value::Ref(id) => {
            if !seen.insert(id) {
                return None;
            }
            let p = f.params(id);
            if f.type_of(id) == "SURFACE_STYLE_TRANSPARENT" {
                let t = p.first()?.as_f64()? as f32;
                return Some(1.0 - t.clamp(0.0, 1.0));
            }
            p.iter()
                .find_map(|&sub| resolve_alpha(f, sub, depth - 1, seen))
        }
        Value::List { .. } => f
            .list(root)
            .iter()
            .find_map(|&sub| resolve_alpha(f, sub, depth - 1, seen)),
        _ => None,
    }
}

/// Maps entity id -> colour and alpha for everything a `STYLED_ITEM` mentions.
///
/// Colour and alpha are kept separate rather than defaulted, because they are
/// inherited separately: a face with a transparency and no colour takes its
/// body's colour, not the default grey.
fn color_table(f: &StepFile, clashes: &mut usize) -> HashMap<u32, (Option<[f32; 3]>, Option<f32>)> {
    let mut out: HashMap<u32, (Option<[f32; 3]>, Option<f32>)> = HashMap::new();
    for id in f.ids() {
        let ty = f.type_of(id);
        if ty != "STYLED_ITEM" && ty != "OVER_RIDING_STYLED_ITEM" {
            continue;
        }
        let p = f.params(id);
        // STYLED_ITEM(name, styles, item)
        let (Some(&styles), Some(item)) = (p.get(1), p.get(2).and_then(|v| v.as_ref_id())) else {
            continue;
        };
        let color = resolve_color(f, styles, 12, &mut HashSet::new());
        let alpha = resolve_alpha(f, styles, 12, &mut HashSet::new());
        if color.is_some() || alpha.is_some() {
            let e = out.entry(item).or_insert((None, None));
            // A second, disagreeing style on the same item. The last one wins,
            // which is arbitrary, so count it.
            if let (Some(had), Some(now)) = (e.0, color)
                && had != now
            {
                *clashes += 1;
            }
            e.0 = color.or(e.0);
            e.1 = alpha.or(e.1);
        }
    }
    out
}

pub fn build(f: &StepFile) -> Model {
    let bound = f.id_bound() as usize;
    let mut b = Builder {
        f,
        surfaces: Vec::new(),
        curves: Vec::new(),
        surf_of: vec![-1; bound],
        curve_of: vec![-1; bound],
        stats: BuildStats::default(),
        vertex_bbox: Aabb::EMPTY,
        wire_bbox: Aabb::EMPTY,
        synthetic_edges: 0,
        angle_scale: plane_angle_scale(f),
        solid_bbox: Aabb::EMPTY,
        end_slack: 0.0,
    };
    // A vertex this near an end of its curve names that end, even if another
    // point along the curve is nearer. Measured against the model rather than
    // the curve, because how loosely a vertex is written is the exporter's
    // tolerance, not the curve's. The vertices are measured up front, before
    // any edge needs the answer.
    b.end_slack = {
        let mut size = Aabb::EMPTY;
        for id in f.ids_of("VERTEX_POINT") {
            if let Some(p) = f.params(id).get(1).and_then(|&v| b.point(v)) {
                size.add_point(p);
            }
        }
        size.extent() * 1e-5
    };

    let mut clashes = 0;
    let colors = color_table(f, &mut clashes);
    b.stats.restyled = clashes;

    // Resolve the assembly first: a component may be instanced several times,
    // each at a different place, and each instance becomes its own solid.
    let occurrences = b.assembly_occurrences();
    let nodes: Vec<AsmNode> = occurrences
        .iter()
        .map(|o| AsmNode {
            name: o.name.clone(),
            parent: o.parent,
        })
        .collect();
    let mut placements: HashMap<u32, Vec<(Xform, usize)>> = HashMap::new();
    for (i, o) in occurrences.iter().enumerate() {
        for item in b.rep_items(o.rep) {
            if let Some(id) = item.as_ref_id()
                && is_solid_root(f.type_of(id))
            {
                placements.entry(id).or_default().push((o.world, i));
            }
        }
    }
    let solid_roots: Vec<u32> = f.ids().filter(|&id| is_solid_root(f.type_of(id))).collect();

    let mut solids = Vec::new();
    let mut claimed_faces: HashSet<u32> = HashSet::new();

    for id in solid_roots {
        let ty = f.type_of(id);
        let volume = is_volume(ty);
        let p = f.params(id);
        let name = f.str_of(*p.first().unwrap_or(&Value::Null)).to_string();

        let mut face_refs: Vec<(u32, bool)> = Vec::new();
        let mut shell_ids: Vec<u32> = Vec::new();
        let mut wires: Vec<EdgeUse> = Vec::new();
        let mut surface_refs: Vec<(u32, Option<Domain>)> = Vec::new();
        // A set of nothing but curves and points has no surface, and drawing
        // none is right. Reported apart from a body that should have had faces.
        let mut surfaceless = false;
        match ty {
            // Its items are faces already, with no shell in between. Loose
            // curves and points are filtered out rather than reported as
            // failed faces.
            "GEOMETRIC_SET" => {
                if let Some(&items) = p.get(1) {
                    let ids: Vec<u32> =
                        f.list(items).iter().filter_map(|v| v.as_ref_id()).collect();
                    face_refs.extend(
                        ids.iter()
                            .copied()
                            .filter(|&id| is_face_item(f.type_of(id)))
                            .map(|id| (id, false)),
                    );
                    // Whatever is left that reads as a curve goes to the overlay.
                    // A lone point has no extent to draw, and is skipped by
                    // name so it is not reported as an unsupported curve.
                    for id in ids {
                        let ty = f.type_of(id);
                        if is_face_item(ty) || ty.contains("POINT") {
                            continue;
                        }
                        // A set may hold bare surfaces as well. They become
                        // faces with no bounds, limited by their trim or
                        // natural domain; asking for a curve would misreport
                        // them as unsupported curves.
                        if ty.contains("SURFACE") || ty == "PLANE" {
                            let (basis, domain, _) = b.untrim(id);
                            if let Some(surface) = b.surface_index(basis) {
                                surface_refs.push((surface, domain));
                            }
                            continue;
                        }
                        // A composite curve is a list of segments rather than a
                        // curve the reader can evaluate, so it is drawn as its
                        // parts.
                        let parts: Vec<u32> = if ty == "COMPOSITE_CURVE" {
                            f.list(*f.params(id).get(1).unwrap_or(&Value::Null))
                                .iter()
                                .filter_map(|v| v.as_ref_id())
                                .filter_map(|s| f.params(s).get(2).and_then(|v| v.as_ref_id()))
                                .collect()
                        } else {
                            vec![id]
                        };
                        for id in parts {
                        let Some(curve) = b.curve_index(id) else {
                            continue;
                        };
                        let (t0, t1) = b.curve_range(id, &b.curves[curve as usize]);
                        let c = &b.curves[curve as usize];
                        // Kept out of `vertex_bbox`, the length scale for the
                        // chord tolerance: construction lines and drawing views
                        // can dwarf the part, and must not coarsen its faces.
                        c.extend_bounds(t0, t1, &mut b.wire_bbox);
                        wires.push(EdgeUse {
                            curve,
                            edge: id,
                            t0,
                            t1,
                        });
                        }
                    }
                }
                surfaceless = face_refs.is_empty();
            }
            "SHELL_BASED_SURFACE_MODEL" => {
                if let Some(&items) = p.get(1) {
                    shell_ids.extend(f.list(items).iter().filter_map(|v| v.as_ref_id()));
                }
            }
            _ => {
                if let Some(outer) = p.get(1).and_then(|v| v.as_ref_id()) {
                    shell_ids.push(outer);
                }
                // BREP_WITH_VOIDS adds inner void shells.
                if ty == "BREP_WITH_VOIDS"
                    && let Some(&voids) = p.get(2)
                {
                    shell_ids.extend(f.list(voids).iter().filter_map(|v| v.as_ref_id()));
                }
            }
        }

        for s in shell_ids {
            b.shell_faces(s, false, &mut face_refs);
        }

        b.solid_bbox = Aabb::EMPTY;
        // The body's own style, which each face falls back to one attribute at
        // a time.
        let (body_color, body_alpha) = colors.get(&id).copied().unwrap_or((None, None));
        let mut faces = Vec::with_capacity(face_refs.len() + surface_refs.len());
        for (surface, domain) in surface_refs {
            b.stats.faces += 1;
            faces.push(Face {
                surface,
                same_sense: true,
                bounds: Vec::new(),
                domain,
                style: None,
            });
        }
        for (fid, flip) in face_refs {
            claimed_faces.insert(fid);
            b.stats.faces += 1;
            match b.face(fid) {
                Some(mut face) => {
                    if flip {
                        face.same_sense = !face.same_sense;
                    }
                    // A `STYLED_ITEM` may name a face as readily as a body.
                    if let Some(&(c, a)) = colors.get(&fid) {
                        b.stats.styled_faces += 1;
                        face.style = Some((
                            c.or(body_color).unwrap_or(DEFAULT_COLOR),
                            a.or(body_alpha).unwrap_or(1.0),
                        ));
                    }
                    faces.push(face);
                }
                None => b.stats.skipped_faces += 1,
            }
        }
        if faces.is_empty() && wires.is_empty() {
            // A whole body vanishing is a loss nothing downstream notices: no
            // face was skipped, because none was found. It means a shell
            // spelled some way `shell_faces` does not read. A set of loose
            // construction curves is faceless by nature and not counted.
            if !surfaceless {
                b.stats.dropped_bodies += 1;
            }
            continue;
        }
        let (color, alpha) = (body_color.unwrap_or(DEFAULT_COLOR), body_alpha.unwrap_or(1.0));
        // A face may be transparent where its body is not. The body's alpha
        // routes the whole part through the transparent pass, so it takes the
        // lowest, or the face is drawn opaque.
        let alpha = faces
            .iter()
            .filter_map(|f| f.style.map(|(_, a)| a))
            .fold(alpha, f32::min);
        let local_bbox = b.solid_bbox;
        // A component the assembly never mentions is drawn at the origin. That
        // is a guess, told apart from an identity transform the file asks for.
        let linked = placements.contains_key(&id);
        let places = placements
            .get(&id)
            .cloned()
            .unwrap_or_else(|| vec![(Xform::IDENTITY, usize::MAX)]);
        b.stats.instances += places.len();
        if alpha < 1.0 {
            b.stats.translucent += places.len();
        }
        for (placement, node) in places {
            solids.push(Solid {
                source: id,
                name: name.clone(),
                color,
                alpha,
                faces: faces.clone(),
                wires: wires.clone(),
                placement,
                local_bbox,
                linked,
                node: (node != usize::MAX).then_some(node),
                volume,
            });
        }
    }

    // An ADVANCED_FACE reachable from no solid is drawn, so surface-only files
    // are not blank -- but only when there are no solids. A file may carry a
    // second, geometrically bounded description of the same part beside the
    // solid, and drawing it would lay a duplicate skin over the model.
    let orphans: Vec<u32> = if solids.is_empty() {
        f.ids()
            .filter(|id| f.type_of(*id) == "ADVANCED_FACE" && !claimed_faces.contains(id))
            .collect()
    } else {
        Vec::new()
    };
    if !orphans.is_empty() {
        b.solid_bbox = Aabb::EMPTY;
        let mut faces = Vec::new();
        for fid in orphans {
            b.stats.faces += 1;
            match b.face(fid) {
                Some(face) => faces.push(face),
                None => b.stats.skipped_faces += 1,
            }
        }
        if !faces.is_empty() {
            b.stats.instances += 1;
            solids.push(Solid {
                source: u32::MAX,
                name: "(loose faces)".into(),
                color: DEFAULT_COLOR,
                alpha: 1.0,
                faces,
                wires: Vec::new(),
                placement: Xform::IDENTITY,
                local_bbox: b.solid_bbox,
                linked: false,
                node: None,
                // Loose faces reachable from no shell: not a body at all.
                volume: false,
            });
        }
    }

    // World bounds: every solid's own bounds, moved into place.
    let mut vertex_bbox = Aabb::EMPTY;
    for s in &solids {
        if s.local_bbox.is_empty() {
            continue;
        }
        let (lo, hi) = (s.local_bbox.min, s.local_bbox.max);
        for i in 0..8 {
            let corner = v3(
                if i & 1 == 0 { lo.x } else { hi.x },
                if i & 2 == 0 { lo.y } else { hi.y },
                if i & 4 == 0 { lo.z } else { hi.z },
            );
            vertex_bbox.add_point(s.placement.point(corner));
        }
    }

    // Loose curves set the scale only when there is nothing else: a file that is
    // all construction geometry still needs a tolerance and a camera.
    if vertex_bbox.is_empty() {
        vertex_bbox = b.wire_bbox;
    }

    Model {
        solids,
        nodes,
        surfaces: b.surfaces,
        curves: b.curves,
        vertex_bbox,
        mm_per_unit: length_scale(f),
        stats: b.stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step;

    /// A unit cube built the way a real translator emits one: six planar faces,
    /// each with an outer edge loop of four line edges.
    fn cube_step() -> String {
        let mut s = String::from("DATA;\n");
        // Eight corner points, #1..#8.
        let corners = [
            (0., 0., 0.),
            (1., 0., 0.),
            (1., 1., 0.),
            (0., 1., 0.),
            (0., 0., 1.),
            (1., 0., 1.),
            (1., 1., 1.),
            (0., 1., 1.),
        ];
        for (i, (x, y, z)) in corners.iter().enumerate() {
            s += &format!("#{}=CARTESIAN_POINT('',({x:.1},{y:.1},{z:.1}));\n", i + 1);
            s += &format!("#{}=VERTEX_POINT('',#{});\n", 10 + i, i + 1);
        }
        // Faces as (corner indices ccw, plane origin, plane normal).
        let faces: [([usize; 4], usize, (f64, f64, f64)); 6] = [
            ([0, 3, 2, 1], 0, (0., 0., -1.)),
            ([4, 5, 6, 7], 4, (0., 0., 1.)),
            ([0, 1, 5, 4], 0, (0., -1., 0.)),
            ([2, 3, 7, 6], 2, (0., 1., 0.)),
            ([1, 2, 6, 5], 1, (1., 0., 0.)),
            ([3, 0, 4, 7], 3, (-1., 0., 0.)),
        ];
        let mut next = 100;
        for (quad, origin, n) in faces {
            let mut edge_ids = Vec::new();
            for k in 0..4 {
                let (a, bb) = (quad[k], quad[(k + 1) % 4]);
                let (ax, ay, az) = corners[a];
                let (bx, by, bz) = corners[bb];
                let (dx, dy, dz) = (bx - ax, by - ay, bz - az);
                s += &format!("#{}=CARTESIAN_POINT('',({ax:.1},{ay:.1},{az:.1}));\n", next);
                s += &format!("#{}=DIRECTION('',({dx:.1},{dy:.1},{dz:.1}));\n", next + 1);
                s += &format!("#{}=VECTOR('',#{},1.);\n", next + 2, next + 1);
                s += &format!("#{}=LINE('',#{},#{});\n", next + 3, next, next + 2);
                s += &format!(
                    "#{}=EDGE_CURVE('',#{},#{},#{},.T.);\n",
                    next + 4,
                    10 + a,
                    10 + bb,
                    next + 3
                );
                s += &format!("#{}=ORIENTED_EDGE('',*,*,#{},.T.);\n", next + 5, next + 4);
                edge_ids.push(next + 5);
                next += 10;
            }
            let (ox, oy, oz) = corners[origin];
            let (nx, ny, nz) = n;
            s += &format!("#{}=CARTESIAN_POINT('',({ox:.1},{oy:.1},{oz:.1}));\n", next);
            s += &format!("#{}=DIRECTION('',({nx:.1},{ny:.1},{nz:.1}));\n", next + 1);
            s += &format!("#{}=AXIS2_PLACEMENT_3D('',#{},#{},$);\n", next + 2, next, next + 1);
            s += &format!("#{}=PLANE('',#{});\n", next + 3, next + 2);
            let list = edge_ids
                .iter()
                .map(|e| format!("#{e}"))
                .collect::<Vec<_>>()
                .join(",");
            s += &format!("#{}=EDGE_LOOP('',({list}));\n", next + 4);
            s += &format!("#{}=FACE_OUTER_BOUND('',#{},.T.);\n", next + 5, next + 4);
            s += &format!(
                "#{}=ADVANCED_FACE('',(#{}),#{},.T.);\n",
                next + 6,
                next + 5,
                next + 3
            );
            next += 10;
        }
        let face_list = (0..6)
            .map(|i| format!("#{}", 100 + i * 50 + 46))
            .collect::<Vec<_>>()
            .join(",");
        s += &format!("#9000=CLOSED_SHELL('',({face_list}));\n");
        s += "#9001=MANIFOLD_SOLID_BREP('cube',#9000);\n";
        s += "ENDSEC;\n";
        s
    }

    #[test]
    fn builds_a_cube_solid() {
        let src = cube_step();
        let f = step::parse(src.as_bytes()).expect("parse");
        let m = build(&f);
        assert_eq!(m.solids.len(), 1, "expected one solid");
        let solid = &m.solids[0];
        assert_eq!(solid.name, "cube");
        assert_eq!(solid.faces.len(), 6, "expected six faces");
        assert_eq!(m.stats.skipped_faces, 0);
        assert!(m.stats.unsupported_surfaces.is_empty());
        for face in &solid.faces {
            assert_eq!(face.bounds.len(), 1);
            assert!(face.bounds[0].outer);
            assert_eq!(face.bounds[0].edges.len(), 4);
        }
        // Vertex bbox is the unit cube.
        assert!(m.vertex_bbox.min.distance(v3(0., 0., 0.)) < 1e-12);
        assert!(m.vertex_bbox.max.distance(v3(1., 1., 1.)) < 1e-12);
    }

    #[test]
    fn line_edges_get_unit_parameter_ranges() {
        let f = step::parse(cube_step().as_bytes()).unwrap();
        let m = build(&f);
        for face in &m.solids[0].faces {
            for e in &face.bounds[0].edges {
                // Every cube edge has length 1 and a unit direction vector.
                assert!((e.t1 - e.t0).abs() > 0.99 && (e.t1 - e.t0).abs() < 1.01);
            }
        }
    }

    #[test]
    fn closed_circular_edge_spans_a_full_turn() {
        let src = "DATA;\n\
            #1=CARTESIAN_POINT('',(5.,0.,0.));\n\
            #2=VERTEX_POINT('',#1);\n\
            #3=CARTESIAN_POINT('',(0.,0.,0.));\n\
            #4=AXIS2_PLACEMENT_3D('',#3,$,$);\n\
            #5=CIRCLE('',#4,5.);\n\
            #6=EDGE_CURVE('',#2,#2,#5,.T.);\n\
            #7=ORIENTED_EDGE('',*,*,#6,.T.);\n\
            #8=EDGE_LOOP('',(#7));\n\
            #9=FACE_OUTER_BOUND('',#8,.T.);\n\
            #10=PLANE('',#4);\n\
            #11=ADVANCED_FACE('',(#9),#10,.T.);\n\
            ENDSEC;";
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);
        let face = &m.solids[0].faces[0];
        let e = face.bounds[0].edges[0];
        assert!(
            (e.t1 - e.t0 - crate::geom::TWO_PI).abs() < 1e-9,
            "closed circle should span 2pi, got {}",
            e.t1 - e.t0
        );
    }

    /// The same circle, but bounded by two distinct vertex entities that happen
    /// to sit on the same point. Alone in its loop it still has to close.
    #[test]
    fn a_lone_circle_closes_even_with_distinct_vertices() {
        let src = "DATA;\n\
            #1=CARTESIAN_POINT('',(5.,0.,0.));\n\
            #2=VERTEX_POINT('',#1);\n\
            #12=CARTESIAN_POINT('',(5.,0.,0.));\n\
            #13=VERTEX_POINT('',#12);\n\
            #3=CARTESIAN_POINT('',(0.,0.,0.));\n\
            #4=AXIS2_PLACEMENT_3D('',#3,$,$);\n\
            #5=CIRCLE('',#4,5.);\n\
            #6=EDGE_CURVE('',#2,#13,#5,.T.);\n\
            #7=ORIENTED_EDGE('',*,*,#6,.T.);\n\
            #8=EDGE_LOOP('',(#7));\n\
            #9=FACE_OUTER_BOUND('',#8,.T.);\n\
            #10=PLANE('',#4);\n\
            #11=ADVANCED_FACE('',(#9),#10,.T.);\n\
            ENDSEC;";
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);
        let e = m.solids[0].faces[0].bounds[0].edges[0];
        assert!((e.t1 - e.t0 - crate::geom::TWO_PI).abs() < 1e-9);
    }

    /// The same coincident-vertex circle, but sharing its loop with real edges.
    /// Here it is a hairline of no extent; closing it would sweep a disc of
    /// radius 5 across a face a hundredth that size.
    #[test]
    fn a_hairline_beside_other_edges_is_dropped_not_closed() {
        let src = "DATA;\n\
            #1=CARTESIAN_POINT('',(5.,0.,0.));\n\
            #2=VERTEX_POINT('',#1);\n\
            #12=CARTESIAN_POINT('',(5.,0.,0.));\n\
            #13=VERTEX_POINT('',#12);\n\
            #3=CARTESIAN_POINT('',(0.,0.,0.));\n\
            #4=AXIS2_PLACEMENT_3D('',#3,$,$);\n\
            #5=CIRCLE('',#4,5.);\n\
            #6=EDGE_CURVE('',#2,#13,#5,.T.);\n\
            #7=ORIENTED_EDGE('',*,*,#6,.T.);\n\
            #20=CARTESIAN_POINT('',(5.05,0.,0.));\n\
            #21=VERTEX_POINT('',#20);\n\
            #22=DIRECTION('',(1.,0.,0.));\n\
            #23=VECTOR('',#22,1.);\n\
            #24=LINE('',#12,#23);\n\
            #25=EDGE_CURVE('',#13,#21,#24,.T.);\n\
            #26=ORIENTED_EDGE('',*,*,#25,.T.);\n\
            #27=LINE('',#20,#23);\n\
            #28=EDGE_CURVE('',#21,#2,#27,.T.);\n\
            #29=ORIENTED_EDGE('',*,*,#28,.T.);\n\
            #8=EDGE_LOOP('',(#7,#26,#29));\n\
            #9=FACE_OUTER_BOUND('',#8,.T.);\n\
            #10=PLANE('',#4);\n\
            #11=ADVANCED_FACE('',(#9),#10,.T.);\n\
            ENDSEC;";
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);
        let bound = &m.solids[0].faces[0].bounds[0];
        assert_eq!(bound.edges.len(), 2, "the hairline should have dropped out");
        assert_eq!(m.stats.dropped_edges, 1);
        for e in &bound.edges {
            assert!(
                (e.t1 - e.t0).abs() < crate::geom::TWO_PI,
                "no surviving edge should have been closed into a full turn"
            );
        }
    }

    #[test]
    fn reversed_orientation_flips_the_traversal() {
        // Same circle, but the oriented edge and the bound both run backwards.
        let mk = |oe: &str, bound: &str| {
            format!(
                "DATA;\n\
                #1=CARTESIAN_POINT('',(5.,0.,0.));\n\
                #2=VERTEX_POINT('',#1);\n\
                #3=CARTESIAN_POINT('',(0.,0.,0.));\n\
                #4=AXIS2_PLACEMENT_3D('',#3,$,$);\n\
                #5=CIRCLE('',#4,5.);\n\
                #6=EDGE_CURVE('',#2,#2,#5,.T.);\n\
                #7=ORIENTED_EDGE('',*,*,#6,.{oe}.);\n\
                #8=EDGE_LOOP('',(#7));\n\
                #9=FACE_OUTER_BOUND('',#8,.{bound}.);\n\
                #10=PLANE('',#4);\n\
                #11=ADVANCED_FACE('',(#9),#10,.T.);\n\
                ENDSEC;"
            )
        };
        let param_span = |src: String| {
            let f = step::parse(src.as_bytes()).unwrap();
            let m = build(&f);
            let e = m.solids[0].faces[0].bounds[0].edges[0];
            e.t1 - e.t0
        };
        assert!(param_span(mk("T", "T")) > 0.0);
        assert!(param_span(mk("F", "T")) < 0.0);
        assert!(param_span(mk("T", "F")) < 0.0);
        // Two flips cancel.
        assert!(param_span(mk("F", "F")) > 0.0);
    }

    #[test]
    fn implicit_knot_vectors_have_the_right_shape() {
        // Clamped: multiplicity degree+1 at both ends, simple inside.
        assert_eq!(
            implicit_knots("QUASI_UNIFORM_CURVE", 1, 2).unwrap(),
            vec![0.0, 0.0, 1.0, 1.0]
        );
        assert_eq!(
            implicit_knots("QUASI_UNIFORM_CURVE", 3, 5).unwrap(),
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0]
        );
        // Unclamped: every knot simple.
        assert_eq!(
            implicit_knots("UNIFORM_CURVE", 2, 4).unwrap(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        // A single Bezier span only exists when degree == n - 1.
        assert_eq!(
            implicit_knots("BEZIER_CURVE", 3, 4).unwrap(),
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        );
        assert!(implicit_knots("BEZIER_CURVE", 3, 5).is_none());
        // Every form must produce exactly n + degree + 1 knots.
        for form in ["QUASI_UNIFORM_CURVE", "UNIFORM_CURVE"] {
            for degree in 1..5 {
                for n in degree + 1..degree + 8 {
                    let k = implicit_knots(form, degree, n).unwrap();
                    assert_eq!(k.len(), n + degree + 1, "{form} d={degree} n={n}");
                }
            }
        }
        assert!(implicit_knots("QUASI_UNIFORM_CURVE", 3, 3).is_none());
    }

    #[test]
    fn quasi_uniform_curve_spans_its_control_points() {
        let src = "DATA;\n\
            #1=CARTESIAN_POINT('',(0.,0.,0.));\n\
            #2=CARTESIAN_POINT('',(10.,0.,0.));\n\
            #3=VERTEX_POINT('',#1);\n\
            #4=VERTEX_POINT('',#2);\n\
            #5=QUASI_UNIFORM_CURVE('',1,(#1,#2),.POLYLINE_FORM.,.F.,.U.);\n\
            #6=EDGE_CURVE('',#3,#4,#5,.T.);\n\
            #7=ORIENTED_EDGE('',*,*,#6,.T.);\n\
            #8=EDGE_LOOP('',(#7));\n\
            #9=FACE_OUTER_BOUND('',#8,.T.);\n\
            #10=CARTESIAN_POINT('',(0.,0.,0.));\n\
            #11=AXIS2_PLACEMENT_3D('',#10,$,$);\n\
            #12=PLANE('',#11);\n\
            #13=ADVANCED_FACE('',(#9),#12,.T.);\n\
            ENDSEC;";
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);
        assert!(m.stats.unsupported_curves.is_empty(), "{:?}", m.stats.unsupported_curves);
        let c = &m.curves[0];
        // Degree 1 through two points is the straight segment between them.
        assert!(c.eval(0.0).distance(v3(0.0, 0.0, 0.0)) < 1e-9);
        assert!(c.eval(1.0).distance(v3(10.0, 0.0, 0.0)) < 1e-9);
        assert!(c.eval(0.5).distance(v3(5.0, 0.0, 0.0)) < 1e-9);
    }

    #[test]
    fn surface_curve_falls_back_to_its_3d_curve() {
        // INTERSECTION_CURVE('', <curve_3d>, <associated geometry>, .CURVE_3D.)
        let src = "DATA;\n\
            #1=CARTESIAN_POINT('',(0.,0.,0.));\n\
            #2=DIRECTION('',(1.,0.,0.));\n\
            #3=VECTOR('',#2,1.);\n\
            #4=LINE('',#1,#3);\n\
            #5=INTERSECTION_CURVE('',#4,(#20,#21),.CURVE_3D.);\n\
            #6=CARTESIAN_POINT('',(4.,0.,0.));\n\
            #7=VERTEX_POINT('',#1);\n\
            #8=VERTEX_POINT('',#6);\n\
            #9=EDGE_CURVE('',#7,#8,#5,.T.);\n\
            #10=ORIENTED_EDGE('',*,*,#9,.T.);\n\
            #11=EDGE_LOOP('',(#10));\n\
            #12=FACE_OUTER_BOUND('',#11,.T.);\n\
            #13=AXIS2_PLACEMENT_3D('',#1,$,$);\n\
            #14=PLANE('',#13);\n\
            #15=ADVANCED_FACE('',(#12),#14,.T.);\n\
            ENDSEC;";
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);
        assert!(
            m.stats.unsupported_curves.is_empty(),
            "{:?}",
            m.stats.unsupported_curves
        );
        let e = m.solids[0].faces[0].bounds[0].edges[0];
        assert!((e.t1 - e.t0 - 4.0).abs() < 1e-9, "expected the 3D line, span {}", e.t1 - e.t0);
    }

    #[test]
    fn oriented_face_unwraps_and_flips() {
        // The inherited `bounds` attribute is derived, hence the `*`.
        let mk = |orient: &str| {
            format!(
                "DATA;\n\
                #1=CARTESIAN_POINT('',(0.,0.,0.));\n\
                #2=CARTESIAN_POINT('',(1.,0.,0.));\n\
                #3=VERTEX_POINT('',#1);\n\
                #4=VERTEX_POINT('',#2);\n\
                #5=DIRECTION('',(1.,0.,0.));\n\
                #6=VECTOR('',#5,1.);\n\
                #7=LINE('',#1,#6);\n\
                #8=EDGE_CURVE('',#3,#4,#7,.T.);\n\
                #9=ORIENTED_EDGE('',*,*,#8,.T.);\n\
                #10=EDGE_LOOP('',(#9));\n\
                #11=FACE_OUTER_BOUND('',#10,.T.);\n\
                #12=AXIS2_PLACEMENT_3D('',#1,$,$);\n\
                #13=PLANE('',#12);\n\
                #14=ADVANCED_FACE('',(#11),#13,.T.);\n\
                #15=ORIENTED_FACE('',*,#14,.{orient}.);\n\
                #16=CLOSED_SHELL('',(#15));\n\
                #17=MANIFOLD_SOLID_BREP('s',#16);\n\
                ENDSEC;"
            )
        };
        for (orient, expect) in [("T", true), ("F", false)] {
            let f = step::parse(mk(orient).as_bytes()).unwrap();
            let m = build(&f);
            assert!(
                m.stats.unsupported_surfaces.is_empty(),
                "the wrapped face was mistaken for a surface: {:?}",
                m.stats.unsupported_surfaces
            );
            assert_eq!(m.solids[0].faces.len(), 1);
            assert_eq!(m.solids[0].faces[0].same_sense, expect, "orientation .{orient}.");
        }
    }

    #[test]
    fn plane_angles_follow_the_declared_unit() {
        // The unit block a great many translators emit: angles in degrees,
        // stated as a conversion from the SI radian.
        let units = "#30=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.));\n\
             #31=PLANE_ANGLE_MEASURE_WITH_UNIT(PLANE_ANGLE_MEASURE(0.0174532925199433),#30);\n\
             #32=DIMENSIONAL_EXPONENTS(0.,0.,0.,0.,0.,0.,0.);\n\
             #33=(CONVERSION_BASED_UNIT('DEGREE',#31)NAMED_UNIT(#32)PLANE_ANGLE_UNIT());\n\
             #34=(GEOMETRIC_REPRESENTATION_CONTEXT(3)\
             GLOBAL_UNIT_ASSIGNED_CONTEXT((#33))REPRESENTATION_CONTEXT('',''));\n";
        let cone = "#1=CARTESIAN_POINT('',(5.,0.,0.));\n\
             #2=VERTEX_POINT('',#1);\n\
             #3=CARTESIAN_POINT('',(0.,0.,0.));\n\
             #4=AXIS2_PLACEMENT_3D('',#3,$,$);\n\
             #5=CIRCLE('',#4,5.);\n\
             #6=EDGE_CURVE('',#2,#2,#5,.T.);\n\
             #7=ORIENTED_EDGE('',*,*,#6,.T.);\n\
             #8=EDGE_LOOP('',(#7));\n\
             #9=FACE_OUTER_BOUND('',#8,.T.);\n\
             #20=CONICAL_SURFACE('',#4,5.,45.);\n\
             #21=ADVANCED_FACE('',(#9),#20,.T.);\n";

        let f = step::parse(format!("DATA;\n{units}{cone}ENDSEC;").as_bytes()).unwrap();
        assert!((plane_angle_scale(&f) - std::f64::consts::PI / 180.0).abs() < 1e-12);
        let m = build(&f);
        let Surface::Cone { half_angle, .. } = m.surfaces[0] else {
            panic!("expected a cone")
        };
        // 45 degrees, not 45 radians -- which would open the cone out by a
        // factor of tan(45 rad) = 1.6 per unit instead of exactly 1.
        assert!(
            (half_angle - std::f64::consts::FRAC_PI_4).abs() < 1e-12,
            "{half_angle}"
        );

        // With nothing declared, angles are radians and stay as written.
        let f = step::parse(format!("DATA;\n{cone}ENDSEC;").as_bytes()).unwrap();
        assert_eq!(plane_angle_scale(&f), 1.0);
        let m = build(&f);
        let Surface::Cone { half_angle, .. } = m.surfaces[0] else {
            panic!("expected a cone")
        };
        assert!((half_angle - 45.0).abs() < 1e-12, "{half_angle}");
    }

    /// An inch is written as a conversion whose factor is quoted in whatever SI
    /// unit the measure names -- 2.54 against centimetres, not 0.0254 against
    /// metres.
    #[test]
    fn length_units_follow_the_measure_they_are_quoted_in() {
        let ctx = |units: &str| {
            format!(
                "DATA;\n{units}\
                 #40=(GEOMETRIC_REPRESENTATION_CONTEXT(3)\
                 GLOBAL_UNIT_ASSIGNED_CONTEXT((#39))REPRESENTATION_CONTEXT('',''));\n\
                 ENDSEC;"
            )
        };
        let inch = "#36=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.CENTI.,.METRE.));\n\
                    #37=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(2.54),#36);\n\
                    #38=DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);\n\
                    #39=(CONVERSION_BASED_UNIT('INCH',#37)LENGTH_UNIT()NAMED_UNIT(#38));\n";
        let f = step::parse(ctx(inch).as_bytes()).unwrap();
        assert!((length_scale(&f) - 25.4).abs() < 1e-9, "{}", length_scale(&f));

        for (prefix, mm) in [(".MILLI.", 1.0), (".CENTI.", 10.0), ("$", 1000.0)] {
            let si = format!("#39=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT({prefix},.METRE.));\n");
            let f = step::parse(ctx(&si).as_bytes()).unwrap();
            assert!(
                (length_scale(&f) - mm).abs() < 1e-9,
                "{prefix} gave {}",
                length_scale(&f)
            );
        }

        // Nothing declared: assume the coordinates are already millimetres.
        let f = step::parse(b"DATA;\n#1=CARTESIAN_POINT('',(0.,0.,0.));\nENDSEC;").unwrap();
        assert_eq!(length_scale(&f), 1.0);
    }

    #[test]
    fn vertex_loop_becomes_a_point_bound() {
        // A spherical cap: one circle, plus a VERTEX_LOOP at the pole it closes
        // on. The vertex is the only thing in the file that says *which* pole,
        // so it has to survive as far as the tessellator.
        let src = "DATA;\n\
            #1=CARTESIAN_POINT('',(5.,0.,0.));\n\
            #2=VERTEX_POINT('',#1);\n\
            #3=CARTESIAN_POINT('',(0.,0.,0.));\n\
            #4=AXIS2_PLACEMENT_3D('',#3,$,$);\n\
            #5=CIRCLE('',#4,5.);\n\
            #6=EDGE_CURVE('',#2,#2,#5,.T.);\n\
            #7=ORIENTED_EDGE('',*,*,#6,.T.);\n\
            #8=EDGE_LOOP('',(#7));\n\
            #9=FACE_OUTER_BOUND('',#8,.T.);\n\
            #10=CARTESIAN_POINT('',(0.,0.,-5.));\n\
            #11=VERTEX_POINT('',#10);\n\
            #12=VERTEX_LOOP('',#11);\n\
            #13=FACE_BOUND('',#12,.T.);\n\
            #14=SPHERICAL_SURFACE('',#4,5.);\n\
            #15=ADVANCED_FACE('',(#9,#13),#14,.T.);\n\
            ENDSEC;";
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);
        assert_eq!(m.stats.dropped_bounds, 0, "a vertex loop is not a loss");
        let face = &m.solids[0].faces[0];
        assert_eq!(face.bounds.len(), 2);
        let v = face
            .bounds
            .iter()
            .find_map(|b| b.vertex)
            .expect("the vertex bound should survive");
        assert!(v.distance(v3(0.0, 0.0, -5.0)) < 1e-9, "{v:?}");
        // And it carries no edges, so nothing downstream tries to sample it.
        assert!(face.bounds.iter().any(|b| b.edges.is_empty()));
    }

    #[test]
    fn poly_loop_becomes_line_segments() {
        // A faceted brep: FACE_SURFACE bounded by a POLY_LOOP of points.
        let src = "DATA;\n\
            #1=CARTESIAN_POINT('',(0.,0.,0.));\n\
            #2=CARTESIAN_POINT('',(4.,0.,0.));\n\
            #3=CARTESIAN_POINT('',(4.,3.,0.));\n\
            #4=POLY_LOOP('',(#1,#2,#3));\n\
            #5=FACE_OUTER_BOUND('',#4,.T.);\n\
            #6=AXIS2_PLACEMENT_3D('',#1,$,$);\n\
            #7=PLANE('',#6);\n\
            #8=FACE_SURFACE('',(#5),#7,.T.);\n\
            #9=CLOSED_SHELL('',(#8));\n\
            #10=FACETED_BREP('tri',#9);\n\
            ENDSEC;";
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);
        assert_eq!(m.solids.len(), 1);
        assert_eq!(m.stats.skipped_faces, 0);
        let bound = &m.solids[0].faces[0].bounds[0];
        assert_eq!(bound.edges.len(), 3, "one segment per polygon span");
        // The synthesised segments must close the triangle.
        let mut total = 0.0;
        for e in &bound.edges {
            let c = &m.curves[e.curve as usize];
            total += c.eval(e.t0).distance(c.eval(e.t1));
        }
        assert!((total - (4.0 + 3.0 + 5.0)).abs() < 1e-9, "perimeter {total}");
        // Vertices must reach the bounding box even though there are no
        // VERTEX_POINTs in a faceted brep.
        assert!(m.vertex_bbox.max.distance(v3(4.0, 3.0, 0.0)) < 1e-9);
    }

    /// A minimal faceted triangle, as one component representation.
    fn component_step() -> String {
        "#1=CARTESIAN_POINT('',(0.,0.,0.));\n\
         #2=CARTESIAN_POINT('',(1.,0.,0.));\n\
         #3=CARTESIAN_POINT('',(0.,1.,0.));\n\
         #4=POLY_LOOP('',(#1,#2,#3));\n\
         #5=FACE_OUTER_BOUND('',#4,.T.);\n\
         #6=AXIS2_PLACEMENT_3D('',#1,$,$);\n\
         #7=PLANE('',#6);\n\
         #8=FACE_SURFACE('',(#5),#7,.T.);\n\
         #9=CLOSED_SHELL('',(#8));\n\
         #10=FACETED_BREP('part',#9);\n\
         #11=ADVANCED_BREP_SHAPE_REPRESENTATION('comp',(#10),$);\n"
            .to_string()
    }

    /// A mapped item may target a transformation operator rather than a plain
    /// placement, and that operator may carry a uniform scale.
    #[test]
    fn a_transformation_operator_places_and_scales_a_component() {
        let src = format!(
            "DATA;\n{}\
             #12=AXIS2_PLACEMENT_3D('',#1,$,$);\n\
             #13=CARTESIAN_POINT('',(5.,0.,0.));\n\
             #14=CARTESIAN_TRANSFORMATION_OPERATOR_3D('',$,$,#13,3.,$);\n\
             #15=REPRESENTATION_MAP(#12,#11);\n\
             #16=MAPPED_ITEM('',#15,#14);\n\
             #22=SHAPE_REPRESENTATION('asm',(#16),$);\n\
             ENDSEC;",
            component_step()
        );
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);
        assert_eq!(m.solids.len(), 1, "the operator must not be skipped");
        let x = &m.solids[0].placement;
        assert!(x.origin.distance(v3(5.0, 0.0, 0.0)) < 1e-9, "{:?}", x.origin);
        // The scale rides in the basis vectors, so a unit step comes out three
        // times as long and the component is drawn three times the size.
        assert!((x.dir(v3(1.0, 0.0, 0.0)).length() - 3.0).abs() < 1e-9);
        assert!(x.point(v3(2.0, 0.0, 0.0)).distance(v3(11.0, 0.0, 0.0)) < 1e-9);
    }

    #[test]
    fn mapped_item_places_and_repeats_a_component() {
        // One component instanced twice, at +5X and at +9Y rotated a quarter
        // turn about Z.
        let src = format!(
            "DATA;\n{}\
             #12=AXIS2_PLACEMENT_3D('',#1,$,$);\n\
             #13=CARTESIAN_POINT('',(5.,0.,0.));\n\
             #14=AXIS2_PLACEMENT_3D('',#13,$,$);\n\
             #15=REPRESENTATION_MAP(#12,#11);\n\
             #16=MAPPED_ITEM('',#15,#14);\n\
             #17=CARTESIAN_POINT('',(0.,9.,0.));\n\
             #18=DIRECTION('',(0.,0.,1.));\n\
             #19=DIRECTION('',(0.,1.,0.));\n\
             #20=AXIS2_PLACEMENT_3D('',#17,#18,#19);\n\
             #21=MAPPED_ITEM('',#15,#20);\n\
             #22=SHAPE_REPRESENTATION('asm',(#16,#21),$);\n\
             ENDSEC;",
            component_step()
        );
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);

        assert_eq!(m.solids.len(), 2, "one solid per occurrence");
        let mut placed: Vec<Vec3> = m.solids.iter().map(|s| s.placement.origin).collect();
        placed.sort_by(|a, b| a.x.total_cmp(&b.x));
        assert!(placed[0].distance(v3(0.0, 9.0, 0.0)) < 1e-9, "{:?}", placed[0]);
        assert!(placed[1].distance(v3(5.0, 0.0, 0.0)) < 1e-9, "{:?}", placed[1]);

        // The rotated instance turns +X into +Y.
        let rotated = m
            .solids
            .iter()
            .find(|s| s.placement.origin.y > 1.0)
            .expect("rotated instance");
        assert!(rotated.placement.dir(v3(1.0, 0.0, 0.0)).distance(v3(0.0, 1.0, 0.0)) < 1e-9);

        // Both share the same underlying geometry rather than duplicating it.
        assert_eq!(m.solids[0].faces.len(), m.solids[1].faces.len());
        // World bounds must cover both placements.
        assert!(m.vertex_bbox.max.x >= 6.0 - 1e-9 && m.vertex_bbox.max.y >= 9.0 - 1e-9);
    }

    #[test]
    fn context_dependent_relationship_places_a_component() {
        // The other spelling: a transform tied to an assembly link in the
        // product structure, with the component side identified by the NAUO.
        let src = format!(
            "DATA;\n{}\
             #22=SHAPE_REPRESENTATION('asm',(),$);\n\
             #30=AXIS2_PLACEMENT_3D('',#1,$,$);\n\
             #31=CARTESIAN_POINT('',(0.,7.,0.));\n\
             #32=AXIS2_PLACEMENT_3D('',#31,$,$);\n\
             #33=ITEM_DEFINED_TRANSFORMATION('','',#30,#32);\n\
             #34=(REPRESENTATION_RELATIONSHIP('','',#11,#22)\
             REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#33)\
             SHAPE_REPRESENTATION_RELATIONSHIP());\n\
             #35=PRODUCT_DEFINITION_SHAPE('','',#36);\n\
             #36=NEXT_ASSEMBLY_USAGE_OCCURRENCE('','','',#40,#41);\n\
             #37=CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#34,#35);\n\
             #40=PRODUCT_DEFINITION('asm','',$,$);\n\
             #41=PRODUCT_DEFINITION('comp','',$,$);\n\
             #42=PRODUCT_DEFINITION_SHAPE('','',#41);\n\
             #43=SHAPE_DEFINITION_REPRESENTATION(#42,#11);\n\
             #44=PRODUCT_DEFINITION_SHAPE('','',#40);\n\
             #45=SHAPE_DEFINITION_REPRESENTATION(#44,#22);\n\
             ENDSEC;",
            component_step()
        );
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);
        assert_eq!(m.solids.len(), 1);
        assert!(
            m.solids[0].placement.origin.distance(v3(0.0, 7.0, 0.0)) < 1e-9,
            "{:?}",
            m.solids[0].placement.origin
        );
    }

    #[test]
    fn a_component_with_no_assembly_stays_at_the_origin() {
        let src = format!("DATA;\n{}ENDSEC;", component_step());
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);
        assert_eq!(m.solids.len(), 1);
        assert!(m.solids[0].placement.is_identity());
        // Its representation has no parent, so it is a root of the assembly
        // and placed as one. Only a body belonging to no representation at all
        // falls back to a placement the reader invented.
        assert!(m.solids[0].linked);
    }

    /// An identity transform is a placement like any other, not a link the
    /// reader failed to follow.
    #[test]
    fn an_identity_transform_still_counts_as_placed() {
        let src = format!(
            "DATA;\n{}\
             #22=SHAPE_REPRESENTATION('asm',(),$);\n\
             #30=AXIS2_PLACEMENT_3D('',#1,$,$);\n\
             #33=ITEM_DEFINED_TRANSFORMATION('','',#30,#30);\n\
             #34=(REPRESENTATION_RELATIONSHIP('','',#11,#22)\
             REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#33)\
             SHAPE_REPRESENTATION_RELATIONSHIP());\n\
             #35=PRODUCT_DEFINITION_SHAPE('','',#36);\n\
             #36=NEXT_ASSEMBLY_USAGE_OCCURRENCE('','','',#40,#41);\n\
             #37=CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#34,#35);\n\
             #40=PRODUCT_DEFINITION('asm','',$,$);\n\
             #41=PRODUCT_DEFINITION('comp','',$,$);\n\
             #42=PRODUCT_DEFINITION_SHAPE('','',#41);\n\
             #43=SHAPE_DEFINITION_REPRESENTATION(#42,#11);\n\
             #44=PRODUCT_DEFINITION_SHAPE('','',#40);\n\
             #45=SHAPE_DEFINITION_REPRESENTATION(#44,#22);\n\
             ENDSEC;",
            component_step()
        );
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);
        assert_eq!(m.solids.len(), 1);
        assert!(m.solids[0].placement.is_identity());
        assert!(m.solids[0].linked, "the assembly did place it, at the origin");
    }

    #[test]
    fn surface_of_revolution_reads_its_trimmed_meridian() {
        // A line from (5,0,0) along +Z, trimmed to 0..10, revolved about Z:
        // an exact cylinder of radius 5.
        let src = "DATA;\n\
            #1=CARTESIAN_POINT('',(5.,0.,0.));\n\
            #2=DIRECTION('',(0.,0.,1.));\n\
            #3=VECTOR('',#2,1.);\n\
            #4=LINE('',#1,#3);\n\
            #5=TRIMMED_CURVE('',#4,(0.),(10.),.T.,.PARAMETER.);\n\
            #6=CARTESIAN_POINT('',(0.,0.,0.));\n\
            #7=DIRECTION('',(0.,0.,1.));\n\
            #8=AXIS1_PLACEMENT('',#6,#7);\n\
            #9=SURFACE_OF_REVOLUTION('',#5,#8);\n\
            #10=VERTEX_POINT('',#1);\n\
            #11=EDGE_CURVE('',#10,#10,#4,.T.);\n\
            #12=ORIENTED_EDGE('',*,*,#11,.T.);\n\
            #13=EDGE_LOOP('',(#12));\n\
            #14=FACE_OUTER_BOUND('',#13,.T.);\n\
            #15=ADVANCED_FACE('',(#14),#9,.T.);\n\
            ENDSEC;";
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);
        assert!(
            m.stats.unsupported_surfaces.is_empty(),
            "{:?}",
            m.stats.unsupported_surfaces
        );
        let Surface::Revolution(s) = &m.surfaces[0] else {
            panic!("expected a revolution, got {:?}", m.surfaces[0]);
        };
        // The trim, not the line's unbounded default, sets the v range.
        assert!((s.v0 - 0.0).abs() < 1e-12 && (s.v1 - 10.0).abs() < 1e-12);
        assert!(s.origin.distance(Vec3::ZERO) < 1e-12);
        assert!(s.dir.distance(v3(0.0, 0.0, 1.0)) < 1e-12);
        // And it really is the cylinder.
        assert!(m.surfaces[0].eval(0.0, 4.0).distance(v3(5.0, 0.0, 4.0)) < 1e-9);
    }

    #[test]
    fn trimmed_curve_resolves_to_its_basis_for_edges() {
        let src = "DATA;\n\
            #1=CARTESIAN_POINT('',(0.,0.,0.));\n\
            #2=CARTESIAN_POINT('',(3.,0.,0.));\n\
            #3=DIRECTION('',(1.,0.,0.));\n\
            #4=VECTOR('',#3,1.);\n\
            #5=LINE('',#1,#4);\n\
            #6=TRIMMED_CURVE('',#5,(0.),(3.),.T.,.PARAMETER.);\n\
            #7=VERTEX_POINT('',#1);\n\
            #8=VERTEX_POINT('',#2);\n\
            #9=EDGE_CURVE('',#7,#8,#6,.T.);\n\
            #10=ORIENTED_EDGE('',*,*,#9,.T.);\n\
            #11=EDGE_LOOP('',(#10));\n\
            #12=FACE_OUTER_BOUND('',#11,.T.);\n\
            #13=AXIS2_PLACEMENT_3D('',#1,$,$);\n\
            #14=PLANE('',#13);\n\
            #15=ADVANCED_FACE('',(#12),#14,.T.);\n\
            ENDSEC;";
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);
        assert!(m.stats.unsupported_curves.is_empty(), "{:?}", m.stats.unsupported_curves);
        // The edge's own vertices give the range, so it spans 0..3.
        let e = m.solids[0].faces[0].bounds[0].edges[0];
        assert!((e.t1 - e.t0 - 3.0).abs() < 1e-9, "span {}", e.t1 - e.t0);
    }

    /// A geometrically bounded face: a plane trimmed to a square by four
    /// composite-curve segments, with no vertices anywhere in the file.
    #[test]
    fn curve_bounded_surface_becomes_a_face() {
        let mut s = String::from("DATA;\n");
        s += "#1=CARTESIAN_POINT('',(0.,0.,0.));\n\
              #2=AXIS2_PLACEMENT_3D('',#1,$,$);\n\
              #3=PLANE('',#2);\n";
        // Four trimmed lines round a 4x4 square.
        let corners = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        let mut segs = Vec::new();
        let mut next = 10;
        for i in 0..4 {
            let (ax, ay) = corners[i];
            let (bx, by) = corners[(i + 1) % 4];
            s += &format!("#{}=CARTESIAN_POINT('',({ax:.1},{ay:.1},0.));\n", next);
            s += &format!(
                "#{}=DIRECTION('',({:.1},{:.1},0.));\n",
                next + 1,
                bx - ax,
                by - ay
            );
            s += &format!("#{}=VECTOR('',#{},1.);\n", next + 2, next + 1);
            s += &format!("#{}=LINE('',#{},#{});\n", next + 3, next, next + 2);
            s += &format!(
                "#{}=TRIMMED_CURVE('',#{},(0.),(1.),.T.,.PARAMETER.);\n",
                next + 4,
                next + 3
            );
            s += &format!(
                "#{}=COMPOSITE_CURVE_SEGMENT(.CONTINUOUS.,.T.,#{});\n",
                next + 5,
                next + 4
            );
            segs.push(next + 5);
            next += 10;
        }
        let list = segs
            .iter()
            .map(|i| format!("#{i}"))
            .collect::<Vec<_>>()
            .join(",");
        s += &format!("#100=OUTER_BOUNDARY_CURVE('',({list}),.F.);\n");
        s += "#101=CURVE_BOUNDED_SURFACE('',#3,(#100),.F.);\n";
        s += "#102=GEOMETRIC_SET('',(#101));\n";
        s += "ENDSEC;";

        let f = step::parse(s.as_bytes()).unwrap();
        let m = build(&f);
        assert_eq!(m.solids.len(), 1, "the geometric set should give one solid");
        assert_eq!(m.stats.skipped_faces, 0);
        let face = &m.solids[0].faces[0];
        assert_eq!(face.bounds.len(), 1);
        assert!(face.bounds[0].outer, "OUTER_BOUNDARY_CURVE is the outer bound");
        assert_eq!(face.bounds[0].edges.len(), 4);
        // There are no vertices in this model, so the range must have come
        // from each segment's own trimming.
        for e in &face.bounds[0].edges {
            assert!((e.t1 - e.t0 - 1.0).abs() < 1e-9, "span {}", e.t1 - e.t0);
        }
        // And the bounds must still be found, or the camera would frame nothing.
        assert!(m.vertex_bbox.max.distance(v3(4.0, 4.0, 0.0)) < 1e-9);
    }

    /// A bare surface in a `GEOMETRIC_SET` is a face with nothing bounding it,
    /// not an unsupported curve.
    ///
    /// Whether it can be drawn falls out of `full_domain`: a B-spline has a
    /// knot rectangle, a sphere and a torus close on themselves, and a plane or
    /// cylinder is unbounded and failed rather than guessed at.
    #[test]
    fn a_bare_surface_in_a_set_is_drawn_where_its_extent_is_known() {
        let src = "DATA;\n\
            #1=CARTESIAN_POINT('',(0.,0.,0.));\n\
            #2=AXIS2_PLACEMENT_3D('',#1,$,$);\n\
            #3=SPHERICAL_SURFACE('',#2,5.);\n\
            #4=PLANE('',#2);\n\
            #5=RECTANGULAR_TRIMMED_SURFACE('',#4,0.,10.,0.,10.,.T.,.T.);\n\
            #6=GEOMETRIC_SET('',(#3,#4,#5));\n\
            ENDSEC;";
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);
        assert_eq!(m.solids.len(), 1);
        let faces = &m.solids[0].faces;
        assert_eq!(faces.len(), 3, "all three are faces, bounded or not");
        assert!(faces.iter().all(|f| f.bounds.is_empty()));
        // Only the trimmed one knows where it stops in its own right; the sphere
        // closes on itself and the bare plane does neither.
        let trimmed: Vec<_> = faces.iter().filter(|f| f.domain.is_some()).collect();
        assert_eq!(trimmed.len(), 1);
        assert_eq!(trimmed[0].domain, Some((0.0, 10.0, 0.0, 10.0)));
        assert!(
            m.stats.unsupported_surfaces.is_empty(),
            "a surface is not a curve: {:?}",
            m.stats.unsupported_surfaces
        );
    }

    /// Units survive being written in the abbreviated form (`CNBSUN` for
    /// `CONVERSION_BASED_UNIT`, and so on): an inch file with degrees, its
    /// unit block written with short names.
    #[test]
    fn units_are_read_through_abbreviated_names() {
        let src = "DATA;\n\
            #26=(NMDUNT(*)PLANUN()SUNT($,.RADIAN.));\n\
            #27=(LNGUNT()NMDUNT(*)SUNT($,.METRE.));\n\
            #28=(LNGUNT()NMDUNT(*)SUNT(.MILLI.,.METRE.));\n\
            #29=LMWU(LENGTH_MEASURE(25.4000000000),#28);\n\
            #30=(CNBSUN('INCH',#29)LNGUNT()NMDUNT(#27));\n\
            #31=(NMDUNT(*)PLANUN()SUNT($,.RADIAN.));\n\
            #32=PAMWU(PLANE_ANGLE_MEASURE(0.0174532925199),#31);\n\
            #33=(CNBSUN('DEGREE',#32)NMDUNT(#26)PLANUN());\n\
            #48=(GMRPCN(3)GUAC((#30,#33))RPRCNT('*MASTER','WORKSPACE'));\n\
            ENDSEC;";
        let f = step::parse(src.as_bytes()).unwrap();
        assert!(
            (length_scale(&f) - 25.4).abs() < 1e-9,
            "inches: {}",
            length_scale(&f)
        );
        assert!(
            (plane_angle_scale(&f) - std::f64::consts::PI / 180.0).abs() < 1e-9,
            "degrees: {}",
            plane_angle_scale(&f)
        );
    }

    /// A `RECTANGULAR_TRIMMED_SURFACE` says where a face stops when nothing else
    /// does, and its angular trims carry the file's plane-angle unit.
    ///
    /// The trim does not reparameterise, so the basis surface is used directly
    /// and the rectangle is carried on the face. Read as radians, a quarter
    /// turn written as 90 degrees would wrap the cylinder fourteen times.
    #[test]
    fn a_trimmed_surface_carries_its_rectangle_in_the_files_own_angle() {
        let mk = |unit: &str, u1: &str| {
            format!(
                "DATA;\n\
                 #1=CARTESIAN_POINT('',(0.,0.,0.));\n\
                 #2=AXIS2_PLACEMENT_3D('',#1,$,$);\n\
                 #3=CYLINDRICAL_SURFACE('',#2,5.);\n\
                 #4=RECTANGULAR_TRIMMED_SURFACE('',#3,0.,{u1},0.,10.,.T.,.T.);\n\
                 #5=ADVANCED_FACE('',(),#4,.T.);\n\
                 #6=CLOSED_SHELL('',(#5));\n\
                 #7=MANIFOLD_SOLID_BREP('',#6);\n\
                 #10=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));\n\
                 {unit}\
                 #20=(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNIT_ASSIGNED_CONTEXT((#10,#13))REPRESENTATION_CONTEXT('',''));\n\
                 #21=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#7),#20);\n\
                 ENDSEC;"
            )
        };
        let radians = "#13=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.));\n";
        let degrees = "#11=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.));\n\
                       #12=PLANE_ANGLE_MEASURE_WITH_UNIT(PLANE_ANGLE_MEASURE(0.0174532925199433),#11);\n\
                       #13=(CONVERSION_BASED_UNIT('DEGREE',#12)NAMED_UNIT(#11)PLANE_ANGLE_UNIT());\n";
        let quarter = |src: String| {
            let f = step::parse(src.as_bytes()).unwrap();
            let m = build(&f);
            let face = &m.solids[0].faces[0];
            assert!(face.bounds.is_empty(), "nothing bounds it but the trim");
            face.domain.expect("a trimmed face carries its rectangle")
        };
        let (u0, u1, v0, v1) = quarter(mk(radians, "1.5707963267948966"));
        assert!((u1 - u0 - std::f64::consts::FRAC_PI_2).abs() < 1e-9, "u {u0}..{u1}");
        assert!((v1 - v0 - 10.0).abs() < 1e-9, "v {v0}..{v1}");
        // The same quarter turn written the way most translators write it.
        let (d0, d1, e0, e1) = quarter(mk(degrees, "90."));
        assert!((d1 - d0 - std::f64::consts::FRAC_PI_2).abs() < 1e-9, "u {d0}..{d1}");
        // And v is a length, so it is left alone.
        assert!((e1 - e0 - 10.0).abs() < 1e-9, "v {e0}..{e1}");
    }

    /// A surface the reader does not know is named and counted, not dropped.
    /// A `RECTANGULAR_TRIMMED_SURFACE` beside it is read, so it must not appear
    /// in the unsupported list and its face must survive.
    #[test]
    fn an_unknown_surface_is_named_and_its_face_counted() {
        let face = |id: u32, surface: &str| {
            format!(
                "#{id}={surface}\n\
                 #{}=ADVANCED_FACE('',(#40),#{id},.T.);\n",
                id + 1
            )
        };
        let src = format!(
            "DATA;\n\
             #1=CARTESIAN_POINT('',(0.,0.,0.));\n\
             #2=AXIS2_PLACEMENT_3D('',#1,$,$);\n\
             #3=PLANE('',#2);\n\
             #10=CARTESIAN_POINT('',(0.,0.,0.));\n\
             #11=CARTESIAN_POINT('',(1.,0.,0.));\n\
             #12=VERTEX_POINT('',#10);\n\
             #13=VERTEX_POINT('',#11);\n\
             #14=DIRECTION('',(1.,0.,0.));\n\
             #15=VECTOR('',#14,1.);\n\
             #16=LINE('',#10,#15);\n\
             #17=EDGE_CURVE('',#12,#13,#16,.T.);\n\
             #18=ORIENTED_EDGE('',*,*,#17,.T.);\n\
             #39=EDGE_LOOP('',(#18));\n\
             #40=FACE_OUTER_BOUND('',#39,.T.);\n\
             {}{}\
             #80=CLOSED_SHELL('',(#51,#61));\n\
             #81=MANIFOLD_SOLID_BREP('',#80);\n\
             ENDSEC;",
            face(50, "RECTANGULAR_TRIMMED_SURFACE('',#3,0.,1.,0.,1.,.T.,.T.);"),
            face(60, "OFFSET_SURFACE('',#3,2.,.F.);"),
        );
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);
        assert_eq!(
            m.stats.unsupported_surfaces.get("OFFSET_SURFACE"),
            Some(&1),
            "got {:?}",
            m.stats.unsupported_surfaces
        );
        assert_eq!(
            m.stats.unsupported_surfaces.len(),
            1,
            "the trimmed surface is read, so only one kind is unsupported: {:?}",
            m.stats.unsupported_surfaces
        );
        // The lost face is attributed to the surface, not to its boundary.
        assert_eq!(m.stats.faces_unsupported, 1);
        assert_eq!(m.stats.skipped_faces, 1);
        // The trimmed one survived, so the body did too.
        assert_eq!(m.solids.len(), 1);
        assert_eq!(m.solids[0].faces.len(), 1);
        assert_eq!(m.stats.dropped_bodies, 0);
    }

    /// A `GEOMETRIC_SET` of nothing but curves is a body, and its curves are
    /// kept as wires: for a wireframe-only file they are everything. A lone
    /// point in the set is skipped without being reported as unsupported.
    #[test]
    fn a_set_of_loose_curves_is_a_body_of_wires() {
        let src = "DATA;\n\
            #1=CARTESIAN_POINT('',(0.,0.,0.));\n\
            #2=DIRECTION('',(0.,0.,1.));\n\
            #3=DIRECTION('',(1.,0.,0.));\n\
            #4=AXIS2_PLACEMENT_3D('',#1,#2,#3);\n\
            #5=CIRCLE('',#4,50.);\n\
            #6=CARTESIAN_POINT('',(-50.,0.,0.));\n\
            #7=DIRECTION('',(1.,0.,0.));\n\
            #8=VECTOR('',#7,100.);\n\
            #9=LINE('',#6,#8);\n\
            #10=GEOMETRIC_SET('',(#5,#9,#1));\n\
            ENDSEC;";
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);
        assert_eq!(m.solids.len(), 1, "the set is a body");
        assert!(m.solids[0].faces.is_empty());
        assert_eq!(m.solids[0].wires.len(), 2, "the circle and the line, not the point");
        assert_eq!(m.stats.dropped_bodies, 0, "nothing was lost");
        assert!(
            m.stats.unsupported_curves.is_empty(),
            "a point is not an unsupported curve: {:?}",
            m.stats.unsupported_curves
        );
        // With no faces anywhere, the curves have to set the scale themselves or
        // there is no tolerance and no camera.
        assert!(
            (m.vertex_bbox.size().x - 100.0).abs() < 1e-9,
            "bounds {:?}",
            m.vertex_bbox.size()
        );
    }

    #[test]
    fn composite_segment_sense_reverses_the_traversal() {
        let mk = |sense: &str| {
            format!(
                "DATA;\n\
                #1=CARTESIAN_POINT('',(0.,0.,0.));\n\
                #2=AXIS2_PLACEMENT_3D('',#1,$,$);\n\
                #3=PLANE('',#2);\n\
                #4=DIRECTION('',(1.,0.,0.));\n\
                #5=VECTOR('',#4,1.);\n\
                #6=LINE('',#1,#5);\n\
                #7=TRIMMED_CURVE('',#6,(0.),(2.),.T.,.PARAMETER.);\n\
                #8=COMPOSITE_CURVE_SEGMENT(.CONTINUOUS.,.{sense}.,#7);\n\
                #9=OUTER_BOUNDARY_CURVE('',(#8),.F.);\n\
                #10=CURVE_BOUNDED_SURFACE('',#3,(#9),.F.);\n\
                #11=GEOMETRIC_SET('',(#10));\n\
                ENDSEC;"
            )
        };
        let span = |src: String| {
            let f = step::parse(src.as_bytes()).unwrap();
            let m = build(&f);
            let e = m.solids[0].faces[0].bounds[0].edges[0];
            e.t1 - e.t0
        };
        assert!((span(mk("T")) - 2.0).abs() < 1e-9);
        assert!((span(mk("F")) + 2.0).abs() < 1e-9, "a reversed segment runs backwards");
    }

    /// A rational curve whose knots are implied rather than written out. Both
    /// spellings appear in the same file, so the complex-instance reader has to
    /// accept either.
    #[test]
    fn rational_curve_with_implicit_knots() {
        // Degree 2 over 3 control points with the quarter-circle weights: the
        // quasi-uniform knot vector for that is the clamped Bezier one, so the
        // curve should trace an exact unit arc.
        let w = 1.0 / 2.0f64.sqrt();
        let src = format!(
            "DATA;\n\
            #10=CARTESIAN_POINT('',(1.,0.,0.));\n\
            #11=CARTESIAN_POINT('',(1.,1.,0.));\n\
            #12=CARTESIAN_POINT('',(0.,1.,0.));\n\
            #1=(BOUNDED_CURVE()B_SPLINE_CURVE(2,(#10,#11,#12),.UNSPECIFIED.,.F.,.F.)\
            CURVE()GEOMETRIC_REPRESENTATION_ITEM()QUASI_UNIFORM_CURVE()\
            RATIONAL_B_SPLINE_CURVE((1.,{w},1.))REPRESENTATION_ITEM(''));\n\
            #13=VERTEX_POINT('',#10);\n\
            #14=VERTEX_POINT('',#12);\n\
            #15=EDGE_CURVE('',#13,#14,#1,.T.);\n\
            #16=ORIENTED_EDGE('',*,*,#15,.T.);\n\
            #17=EDGE_LOOP('',(#16));\n\
            #18=FACE_OUTER_BOUND('',#17,.T.);\n\
            #19=CARTESIAN_POINT('',(0.,0.,0.));\n\
            #20=AXIS2_PLACEMENT_3D('',#19,$,$);\n\
            #21=PLANE('',#20);\n\
            #22=ADVANCED_FACE('',(#18),#21,.T.);\n\
            ENDSEC;"
        );
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);
        assert!(
            m.stats.unsupported_curves.is_empty(),
            "{:?}",
            m.stats.unsupported_curves
        );
        // Rational weights and implicit knots together must still give a circle.
        let c = &m.curves[0];
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let p = c.eval(t);
            assert!((p.length() - 1.0).abs() < 1e-9, "t={t} r={}", p.length());
        }
    }

    /// A void shell reaches its solid through an `ORIENTED_CLOSED_SHELL` whose
    /// inherited face set is derived, written `*`. Reading the wrong attribute
    /// would let the cavity's faces escape the solid and its placement.
    #[test]
    fn void_shell_faces_stay_with_their_solid() {
        // Two triangles: one for the outer shell, one for the void.
        let tri = |base: u32, z: f64| {
            format!(
                "#{}=CARTESIAN_POINT('',(0.,0.,{z:.1}));\n\
                 #{}=CARTESIAN_POINT('',(1.,0.,{z:.1}));\n\
                 #{}=CARTESIAN_POINT('',(0.,1.,{z:.1}));\n\
                 #{}=POLY_LOOP('',(#{},#{},#{}));\n\
                 #{}=FACE_OUTER_BOUND('',#{},.T.);\n\
                 #{}=AXIS2_PLACEMENT_3D('',#{},$,$);\n\
                 #{}=PLANE('',#{});\n\
                 #{}=FACE_SURFACE('',(#{}),#{},.T.);\n\
                 #{}=CLOSED_SHELL('',(#{}));\n",
                base, base + 1, base + 2,
                base + 3, base, base + 1, base + 2,
                base + 4, base + 3,
                base + 5, base,
                base + 6, base + 5,
                base + 7, base + 4, base + 6,
                base + 8, base + 7,
            )
        };
        let src = format!(
            "DATA;\n{}{}\
             #100=ORIENTED_CLOSED_SHELL('',*,#28,.F.);\n\
             #101=BREP_WITH_VOIDS('block',#18,(#100));\n\
             ENDSEC;",
            tri(10, 0.0),
            tri(20, 5.0)
        );
        let f = step::parse(src.as_bytes()).unwrap();
        let m = build(&f);

        assert_eq!(m.solids.len(), 1, "the void must not become its own solid");
        assert_eq!(m.solids[0].name, "block");
        assert_eq!(
            m.solids[0].faces.len(),
            2,
            "outer shell and void shell both belong to the solid"
        );
        assert!(
            !m.solids.iter().any(|s| s.name == "(loose faces)"),
            "no face should have escaped its solid"
        );
        // The void is at z = 5, so the solid's own bounds must reach it.
        assert!(m.solids[0].local_bbox.max.z >= 5.0 - 1e-9);
    }

    #[test]
    fn reads_styled_item_colours() {
        let src = "DATA;\n\
            #1=COLOUR_RGB('',0.25,0.5,0.75);\n\
            #2=FILL_AREA_STYLE_COLOUR('',#1);\n\
            #3=FILL_AREA_STYLE('',(#2));\n\
            #4=SURFACE_STYLE_FILL_AREA(#3);\n\
            #5=SURFACE_SIDE_STYLE('',(#4));\n\
            #6=SURFACE_STYLE_USAGE(.BOTH.,#5);\n\
            #7=PRESENTATION_STYLE_ASSIGNMENT((#6));\n\
            #8=STYLED_ITEM('',(#7),#9);\n\
            #9=MANIFOLD_SOLID_BREP('part',#10);\n\
            #10=CLOSED_SHELL('',());\n\
            ENDSEC;";
        let f = step::parse(src.as_bytes()).unwrap();
        let mut clashes = 0;
        let table = color_table(&f, &mut clashes);
        let (c, a) = table.get(&9).copied().expect("colour for #9");
        let c = c.expect("a colour");
        assert!((c[0] - 0.25).abs() < 1e-6 && (c[1] - 0.5).abs() < 1e-6 && (c[2] - 0.75).abs() < 1e-6);
        // None, not 1.0: a style silent on transparency must not claim
        // "opaque", since each attribute falls back to the body's separately.
        assert_eq!(a, None, "a style with no transparency states no alpha");
        assert_eq!(clashes, 0);
    }
}
