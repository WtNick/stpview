//! Turns the B-rep into triangles.
//!
//! Per face the pipeline is:
//!   1. sample every bounding edge into a 3D polyline (adaptive, chord-error);
//!   2. invert the surface to get those polylines in (u, v), unwrapping the
//!      seam of periodic surfaces so each loop stays a simple polygon;
//!   3. ear-clip the resulting polygon-with-holes in parameter space;
//!   4. refine the triangles until the surface is within tolerance, using
//!      conforming edge splits so neighbouring triangles never crack apart;
//!   5. emit positions with analytic normals.
//!
//! Boundary edges are never split in step 4: their points come straight from
//! the curve, so keeping them untouched is what makes adjacent faces meet
//! exactly instead of leaving hairline gaps.

pub(crate) use std::collections::{HashMap, HashSet};

pub(crate) use rayon::prelude::*;

pub(crate) use crate::geom::{Curve, Surface, Xform, TWO_PI};
pub(crate) use crate::math::{v2, v3, Aabb, Vec2, Vec3};
pub(crate) use crate::model::{Bound, EdgeUse, Face, Model, Solid};

/// Twenty bytes: a position, a packed normal, and a packed colour.
///
/// The position stays full f32: CAD models span a metre with features a
/// micron across. The normal is octahedrally encoded in two 16-bit numbers
/// (about 0.004 degrees; eight bits a component bands along a specular
/// highlight). Colour and opacity are one byte a channel, display-referred as
/// `COLOUR_RGB` gives them; eight bits of a linear colour would lose the dark
/// end.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    normal: [u16; 2],
    /// Colour and opacity, the latter from `SURFACE_STYLE_TRANSPARENT`. Per
    /// vertex so it reaches the shader without a uniform rewrite between draws.
    color: [u8; 4],
}

/// Octahedral normal encoding (Meyer et al.): folds the sphere onto a square.
/// The decoder in `shader.wgsl` must match.
fn oct_encode(n: Vec3) -> [u16; 2] {
    let l = n.x.abs() + n.y.abs() + n.z.abs();
    let (mut u, mut v) = if l > 1e-300 { (n.x / l, n.y / l) } else { (0.0, 0.0) };
    if n.z < 0.0 {
        // The lower hemisphere folds out into the corners of the square.
        let (fu, fv) = (1.0 - v.abs(), 1.0 - u.abs());
        u = fu.copysign(u);
        v = fv.copysign(v);
    }
    let q = |x: f64| ((x.clamp(-1.0, 1.0) * 0.5 + 0.5) * 65535.0).round() as u16;
    [q(u), q(v)]
}

fn oct_decode(e: [u16; 2]) -> Vec3 {
    let f = |x: u16| x as f64 / 65535.0 * 2.0 - 1.0;
    let (x, y) = (f(e[0]), f(e[1]));
    let z = 1.0 - x.abs() - y.abs();
    let t = (-z).max(0.0);
    v3(x - t.copysign(x), y - t.copysign(y), z).normalize_or(v3(0.0, 0.0, 1.0))
}

impl Vertex {
    pub fn new(pos: Vec3, normal: Vec3, color: [f32; 3], alpha: f32) -> Vertex {
        let b = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
        Vertex {
            pos: pos.as_f32(),
            normal: oct_encode(normal),
            color: [b(color[0]), b(color[1]), b(color[2]), b(alpha)],
        }
    }

    /// The shading normal, unpacked. The shader has its own decoder.
    pub fn normal(&self) -> Vec3 {
        oct_decode(self.normal)
    }

    pub fn position(&self) -> Vec3 {
        v3(self.pos[0] as f64, self.pos[1] as f64, self.pos[2] as f64)
    }

    /// Colour and opacity as stored.
    pub fn color_bytes(&self) -> [u8; 4] {
        self.color
    }
}

/// One occurrence of one component, and the slice of the mesh it owns.
///
/// The three ranges are into `Mesh::indices` and, twice, into `Mesh::edges`:
/// sharp edges come first in that buffer and tangent ones after, so a part's
/// share of it is two runs rather than one.
#[derive(Debug, Clone)]
pub struct Part {
    /// Where it sits in the assembly, as `parent / child / leaf`.
    pub name: String,
    /// Its node in `Model::nodes`, or `None` for a body the assembly never
    /// mentioned.
    pub node: Option<usize>,
    /// 1.0 for an opaque part; below it the part goes through the transparent
    /// pass instead.
    pub alpha: f32,
    /// Whether this body has an inside for a section cap to find.
    ///
    /// The file's own word for it where it gives one (a
    /// `SHELL_BASED_SURFACE_MODEL` has no inside). That word is not always
    /// present or right -- a closed solid can sit in a `GEOMETRIC_SET` -- so a
    /// sheet-typed body is welded and asked directly.
    pub cappable: bool,
    /// The body's colour as the file gave it, display-referred and before any
    /// face overrides. Not drawn from (the vertices carry their own); it is
    /// what a highlight has to contrast with.
    pub color: [f32; 3],
    pub tris: std::ops::Range<u32>,
    /// Its share of the vertex buffer. Not used for drawing; it maps a picked
    /// vertex index back to a component.
    pub verts: std::ops::Range<u32>,
    /// Loose curves, which bound no face and so are never nudged towards the
    /// eye the way an edge overlay is.
    pub wire: std::ops::Range<u32>,
    pub sharp: std::ops::Range<u32>,
    pub tangent: std::ops::Range<u32>,
    /// The placed bounds of the triangles and lines drawn for it: what fitting
    /// the view to a selection frames, and what decides whether a view wants
    /// it finer.
    pub bbox: Aabb,
    /// Each face's share of `tris`, in drawing order, with its bounds. Lets a
    /// view re-tessellate only the faces it can see; see `view.rs`. Empty for
    /// a mesh built by hand.
    pub faces: Vec<FaceSpan>,
}

/// One face's triangles and bounds within a part.
#[derive(Debug, Clone)]
pub struct FaceSpan {
    /// Which of the solid's faces this is.
    pub face: u32,
    /// Its triangles, as a range of `Mesh::indices` like `Part::tris`.
    pub tris: std::ops::Range<u32>,
    /// The bounds of its vertices, placed. `f32` is enough to decide whether a
    /// view can see the face.
    pub bbox: [f32; 6],
}

impl FaceSpan {
    pub fn bounds(&self) -> Aabb {
        let b = &self.bbox;
        Aabb {
            min: v3(b[0] as f64, b[1] as f64, b[2] as f64),
            max: v3(b[3] as f64, b[4] as f64, b[5] as f64),
        }
    }
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// One per solid, in the order they appear in the buffers above.
    pub parts: Vec<Part>,
    /// Line-list positions for the edge overlay, sharp edges first. Drawing
    /// only the first `sharp_edge_verts` gives the CAD-style view that hides
    /// tangent continuations; drawing all of them shows every B-rep edge.
    pub edges: Vec<[f32; 3]>,
    pub bbox: Aabb,
    pub stats: TessStats,
    /// The file's length unit, so a length read off the mesh can be quoted in
    /// millimetres. Everything else in this module works in model units.
    pub mm_per_unit: f64,
}

/// Not derived: a derived `Default` would leave `mm_per_unit` at zero and
/// report every length as nothing.
impl Default for Mesh {
    fn default() -> Mesh {
        Mesh {
            vertices: Vec::new(),
            indices: Vec::new(),
            parts: Vec::new(),
            edges: Vec::new(),
            bbox: Aabb::EMPTY,
            stats: TessStats::default(),
            mm_per_unit: 1.0,
        }
    }
}

impl Mesh {
    /// What the mesh costs in memory. Counted exactly: these buffers and the
    /// parts' face spans are the only large allocations, all `Vec`s of
    /// fixed-size elements.
    pub fn bytes(&self) -> usize {
        use std::mem::size_of;
        self.vertices.len() * size_of::<Vertex>()
            + self.indices.len() * size_of::<u32>()
            + self.edges.len() * size_of::<[f32; 3]>()
            + self.parts.len() * size_of::<Part>()
            + self.parts.iter().map(|p| p.faces.len() * size_of::<FaceSpan>()).sum::<usize>()
    }

    /// Appends a batch, shifting every range it carries to where its data
    /// landed. A batch's ranges are local to it; this is the only place that
    /// knows otherwise, so streamed and whole-model meshes stay identical.
    pub fn absorb(&mut self, other: Mesh) {
        let (vb, ib) = (self.vertices.len() as u32, self.indices.len() as u32);
        let eb = self.edges.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.indices.extend(other.indices.iter().map(|i| i + vb));
        self.edges.extend_from_slice(&other.edges);
        let shift = |r: &std::ops::Range<u32>, by: u32| r.start + by..r.end + by;
        self.parts.extend(other.parts.into_iter().map(|mut p| {
            p.tris = shift(&p.tris, ib);
            p.verts = shift(&p.verts, vb);
            p.wire = shift(&p.wire, eb);
            p.sharp = shift(&p.sharp, eb);
            p.tangent = shift(&p.tangent, eb);
            for f in &mut p.faces {
                f.tris = shift(&f.tris, ib);
            }
            p
        }));
        self.bbox = self.bbox.union(other.bbox);
        self.stats.merge(&other.stats);
        self.mm_per_unit = other.mm_per_unit;
    }

    /// Concatenates meshes verbatim: bytes untouched, only the indices and
    /// ranges shifted.
    ///
    /// This is what the instanced renderer uploads, each unique mesh once.
    /// Verbatim matters: an identity round-trip through the octahedral
    /// encoding (as `absorb_placed` does) need not return the same bits.
    /// A part-less hand-built mesh gets a covering part, as in `absorb_placed`.
    pub fn concatenated(meshes: &[&Mesh]) -> Mesh {
        let mut out = Mesh::default();
        for m in meshes {
            let (vb, ib) = (out.vertices.len() as u32, out.indices.len() as u32);
            let eb = out.edges.len() as u32;
            out.vertices.extend_from_slice(&m.vertices);
            out.indices.extend(m.indices.iter().map(|i| i + vb));
            out.edges.extend_from_slice(&m.edges);
            let shift = |r: &std::ops::Range<u32>, by: u32| r.start + by..r.end + by;
            let whole: Vec<Part>;
            let parts = if m.parts.is_empty() && !m.vertices.is_empty() {
                whole = vec![Part {
                    name: String::new(),
                    node: None,
                    alpha: 1.0,
                    cappable: false,
                    color: [0.5; 3],
                    tris: 0..m.indices.len() as u32,
                    verts: 0..m.vertices.len() as u32,
                    wire: 0..0,
                    sharp: 0..m.edges.len() as u32,
                    tangent: 0..0,
                    bbox: m.bbox,
                    faces: Vec::new(),
                }];
                &whole
            } else {
                &m.parts
            };
            out.parts.extend(parts.iter().map(|p| {
                let mut p = p.clone();
                p.tris = shift(&p.tris, ib);
                p.verts = shift(&p.verts, vb);
                p.wire = shift(&p.wire, eb);
                p.sharp = shift(&p.sharp, eb);
                p.tangent = shift(&p.tangent, eb);
                for f in &mut p.faces {
                    f.tris = shift(&f.tris, ib);
                }
                p
            }));
            out.bbox = out.bbox.union(m.bbox);
            out.stats.merge(&m.stats);
        }
        out
    }

    /// `absorb` through a placement: positions, normals and edge points are
    /// carried by `place`, and an optional tint multiplies the colour. Takes
    /// the mesh by reference since a scene places one mesh many times.
    ///
    /// - Normals go through the rotation block and are renormalised, so a
    ///   uniform scale does not shrink them.
    /// - A mirroring placement (negative determinant) swaps each triangle's
    ///   winding so it keeps agreeing with the normals; the section cap's
    ///   parity count depends on it.
    ///
    /// `mm_per_unit` is left alone: the caller knows the units of the world it
    /// is composing.
    pub fn absorb_placed(&mut self, other: &Mesh, place: &Xform, tint: Option<[f32; 4]>) {
        let (vb, ib) = (self.vertices.len() as u32, self.indices.len() as u32);
        let eb = self.edges.len() as u32;
        let mirrored = place.det() < 0.0;

        let point32 = |p: [f32; 3]| {
            let q = place.point(v3(p[0] as f64, p[1] as f64, p[2] as f64));
            [q.x as f32, q.y as f32, q.z as f32]
        };
        self.vertices.extend(other.vertices.iter().map(|v| {
            let n = place.dir(v.normal()).normalize_or(v3(0.0, 0.0, 1.0));
            let mut out = *v;
            out.pos = point32(v.pos);
            out.normal = oct_encode(n);
            if let Some(t) = tint {
                for (c, f) in out.color.iter_mut().zip(t) {
                    *c = (*c as f32 * f.clamp(0.0, 1.0)).round() as u8;
                }
            }
            out
        }));
        if mirrored {
            self.indices.extend(other.indices.chunks_exact(3).flat_map(|t| {
                [t[0] + vb, t[2] + vb, t[1] + vb]
            }));
        } else {
            self.indices.extend(other.indices.iter().map(|i| i + vb));
        }
        self.edges.extend(other.edges.iter().map(|p| point32(*p)));
        let shift = |r: &std::ops::Range<u32>, by: u32| r.start + by..r.end + by;
        // A hand-built mesh with no parts is still one part: several passes
        // (the occlusion prepass among them) slice draws by part, and geometry
        // no part claims would be drawn by some and skipped by others.
        let whole: Vec<Part>;
        let other_parts = if other.parts.is_empty() && !other.vertices.is_empty() {
            whole = vec![Part {
                name: String::new(),
                node: None,
                alpha: 1.0,
                // A hand-built slab is scenery, and scenery is not capped.
                cappable: false,
                color: [0.5; 3],
                tris: 0..other.indices.len() as u32,
                verts: 0..other.vertices.len() as u32,
                wire: 0..0,
                sharp: 0..other.edges.len() as u32,
                tangent: 0..0,
                bbox: other.bbox,
                faces: Vec::new(),
            }];
            &whole
        } else {
            &other.parts
        };
        self.parts.extend(other_parts.iter().map(|p| {
            let mut p = p.clone();
            p.tris = shift(&p.tris, ib);
            p.verts = shift(&p.verts, vb);
            p.wire = shift(&p.wire, eb);
            p.sharp = shift(&p.sharp, eb);
            p.tangent = shift(&p.tangent, eb);
            for f in &mut p.faces {
                f.tris = shift(&f.tris, ib);
                let b = placed_bbox(&f.bounds(), place);
                f.bbox = [
                    b.min.x as f32, b.min.y as f32, b.min.z as f32,
                    b.max.x as f32, b.max.y as f32, b.max.z as f32,
                ];
            }
            p.bbox = {
                let mut bb = crate::math::Aabb::EMPTY;
                // The eight corners of the old box, carried.
                for i in 0..8 {
                    let c = p.bbox;
                    let x = if i & 1 == 0 { c.min.x } else { c.max.x };
                    let y = if i & 2 == 0 { c.min.y } else { c.max.y };
                    let z = if i & 4 == 0 { c.min.z } else { c.max.z };
                    bb.add_point(place.point(v3(x, y, z)));
                }
                bb
            };
            p
        }));
        for i in 0..8 {
            let b = other.bbox;
            let x = if i & 1 == 0 { b.min.x } else { b.max.x };
            let y = if i & 2 == 0 { b.min.y } else { b.max.y };
            let z = if i & 4 == 0 { b.min.z } else { b.max.z };
            self.bbox.add_point(place.point(v3(x, y, z)));
        }
        self.stats.merge(&other.stats);
    }

    /// Hands back the vertex and index buffers once the GPU owns them.
    ///
    /// Nothing on the CPU reads them after upload: a pick is a render pass,
    /// and everything that walks the triangles (`--check`, the winding count,
    /// `closed_parts`) has run by then. Only the edges stay.
    pub fn release_uploaded(&mut self) -> usize {
        let freed = self.vertices.len() * std::mem::size_of::<Vertex>()
            + self.indices.len() * std::mem::size_of::<u32>();
        // Assigned rather than cleared: `clear` keeps the allocation.
        self.vertices = Vec::new();
        self.indices = Vec::new();
        freed
    }

    /// Which part a vertex belongs to; turns a picked vertex into a component.
    /// Linear rather than a binary search: empty parts make the starts merely
    /// non-decreasing, and it runs once per click.
    pub fn part_of_vertex(&self, v: u32) -> Option<usize> {
        self.parts.iter().position(|p| p.verts.contains(&v))
    }
}

#[derive(Debug, Default, Clone)]
pub struct TessStats {
    pub faces_ok: usize,
    /// Faces that wrap a periodic seam and were stitched into a band.
    pub seam_bands: usize,
    /// Faces closed at a degenerate parameter line (cone apex, sphere pole).
    pub seam_caps: usize,
    /// Faces hitting the per-face triangle cap.
    pub capped: usize,
    // Failure buckets, broken out so a bad file says *why* it failed.
    /// No bound survived projection into parameter space.
    pub fail_no_loops: usize,
    /// Loop circles a periodic surface in a way the seam stitcher cannot close.
    pub fail_seam: usize,
    /// Outline encloses no area in parameter space.
    pub fail_degenerate: usize,
    /// Ear clipping rejected the outline (self-intersecting in parameter space).
    pub fail_earcut: usize,
    /// Every triangulation of it covered grossly more than the outline encloses.
    pub fail_miscovered: usize,
    /// Failures of any kind, per [`Surface::kind_index`].
    pub fail_kinds: [usize; Surface::KINDS.len()],
    /// Where the first few failures were, for `--stats` to name them.
    pub failures: Vec<FaceFailure>,
    /// Faces whose triangles reach well outside the outline that bounds them:
    /// drawn, but the wrong shape.
    pub stray_faces: usize,
    pub strays: Vec<FaceFailure>,
    /// Faces where the constrained triangulation gave up and ear clipping
    /// stood in. The outline is usually worth a look.
    pub cdt_failed: usize,
    /// Faces the triangulator took only on a second attempt, with the input
    /// shaken by a distance nothing can see. See where it is used.
    pub cdt_shaken: usize,
    /// Of the faces that came out covering the wrong area, how wide the
    /// narrowest was.
    ///
    /// Narrow faces (fillets, chamfers) are ordinary; a narrow face that also
    /// failed is explained by its width, because two sides that close are
    /// chorded independently and the chords can cross. A finer `--tol` helps.
    pub narrowest: Option<f64>,
    /// Outline points that are not where their own parameters put them.
    ///
    /// Every stage that shifts, reverses or joins a loop must carry `uv` and
    /// `pts` together and only add whole periods to a parameter. Otherwise the
    /// vertex gets the normal of somewhere else, which nothing else catches:
    /// the point is on the surface, the area is right, the mesh is closed.
    pub displaced: usize,
    pub displacements: Vec<FaceFailure>,
    /// Faces whose triangles do not add up to the area their outline encloses.
    /// Too little means a piece is missing; too much means part is covered
    /// twice, which draws as two skins fighting. Neither shows as a crack.
    pub miscovered: usize,
    pub miscovers: Vec<FaceFailure>,
    /// Bounds thrown away while sampling: too few points to enclose anything,
    /// or no projection onto the surface. The face still draws, minus a hole,
    /// so no failure is counted; the only trace is a crack at the neighbour.
    pub bounds_unsampled: usize,
    pub bounds_unprojected: usize,
    /// Edges drawn in the overlay.
    pub edges_sharp: usize,
    /// Edges suppressed because the faces meeting there are tangent.
    pub edges_tangent: usize,
    /// `EDGE_CURVE`s used by exactly one face. A closed solid has none; any
    /// mean the file's own topology is open there.
    pub edges_single_use: usize,
}

impl TessStats {
    pub fn faces_failed(&self) -> usize {
        self.fail_no_loops
            + self.fail_seam
            + self.fail_degenerate
            + self.fail_earcut
            + self.fail_miscovered
    }

    fn merge(&mut self, o: &TessStats) {
        self.faces_ok += o.faces_ok;
        self.seam_bands += o.seam_bands;
        self.seam_caps += o.seam_caps;
        self.capped += o.capped;
        self.fail_no_loops += o.fail_no_loops;
        self.fail_seam += o.fail_seam;
        self.fail_degenerate += o.fail_degenerate;
        self.fail_earcut += o.fail_earcut;
        self.fail_miscovered += o.fail_miscovered;
        self.edges_sharp += o.edges_sharp;
        self.edges_tangent += o.edges_tangent;
        // Every field must be merged here, or it reads zero for ever.
        self.edges_single_use += o.edges_single_use;
        for (a, b) in self.fail_kinds.iter_mut().zip(&o.fail_kinds) {
            *a += b;
        }
        for fx in &o.failures {
            if self.failures.len() >= MAX_RECORDED_FAILURES {
                break;
            }
            self.failures.push(fx.clone());
        }
        self.stray_faces += o.stray_faces;
        self.cdt_failed += o.cdt_failed;
        self.cdt_shaken += o.cdt_shaken;
        self.miscovered += o.miscovered;
        if let Some(w) = o.narrowest
            && self.narrowest.is_none_or(|n| w < n)
        {
            self.narrowest = Some(w);
        }
        self.displaced += o.displaced;
        for fx in &o.displacements {
            if self.displacements.len() >= MAX_RECORDED_FAILURES {
                break;
            }
            self.displacements.push(fx.clone());
        }
        self.bounds_unsampled += o.bounds_unsampled;
        self.bounds_unprojected += o.bounds_unprojected;
        for fx in &o.strays {
            if self.strays.len() >= MAX_RECORDED_FAILURES {
                break;
            }
            self.strays.push(fx.clone());
        }
        for fx in &o.miscovers {
            if self.miscovers.len() >= MAX_RECORDED_FAILURES {
                break;
            }
            self.miscovers.push(fx.clone());
        }
    }

    /// Counts a failed face, and remembers where it was while few enough have
    /// failed for the list to be worth reading.
    fn fail(&mut self, why: Fail, surface: &Surface, loops: &[LoopUv], place: &Xform) {
        match why {
            Fail::NoLoops => self.fail_no_loops += 1,
            Fail::Seam => self.fail_seam += 1,
            Fail::Degenerate => self.fail_degenerate += 1,
            Fail::Earcut => self.fail_earcut += 1,
            Fail::Miscovered => self.fail_miscovered += 1,
        }
        self.fail_kinds[surface.kind_index()] += 1;
        if self.failures.len() < MAX_RECORDED_FAILURES {
            let bbox = loop_bbox(loops, place);
            // The (u, v) extent beside the 3D one tells a sub-tolerance sliver
            // (small in both) from an outline that collapsed in projection.
            let mut uv = Aabb::EMPTY;
            for l in loops {
                for p in &l.uv {
                    uv.add_point(crate::math::v3(p.x, p.y, 0.0));
                }
            }
            self.failures.push(FaceFailure {
                kind: Surface::KINDS[surface.kind_index()],
                why: why.label(),
                center: bbox.center(),
                size: bbox.size(),
                uv_size: v2(uv.size().x, uv.size().y),
            });
        }
    }

}

/// Enough to characterise what is failing; past this the per-surface counts are
/// the useful summary.
const MAX_RECORDED_FAILURES: usize = 24;

/// How many chord tolerances wide a face has to be before it can be drawn
/// faithfully. Below about this its two sides are chorded independently and
/// nothing stops the chords crossing.
pub const NARROW_FACE_TOLERANCES: f64 = 3.0;

/// Why a face could not be tessellated.
#[derive(Clone, Copy)]
pub enum Fail {
    NoLoops,
    Seam,
    Degenerate,
    Earcut,
    Miscovered,
}

impl Fail {
    fn label(self) -> &'static str {
        match self {
            Fail::NoLoops => "unclosed outline",
            Fail::Seam => "seam",
            Fail::Degenerate => "degenerate",
            Fail::Earcut => "self-intersecting",
            Fail::Miscovered => "covering the wrong area",
        }
    }
}

/// A face that did not tessellate, and where in the finished model it was.
#[derive(Debug, Clone)]
pub struct FaceFailure {
    pub kind: &'static str,
    pub why: &'static str,
    pub center: Vec3,
    pub size: Vec3,
    /// Extent of the outline in (u, v), which says how it failed.
    pub uv_size: Vec2,
}

const MAX_REFINE_PASSES: usize = 6;
const MAX_FACE_TRIS: usize = 40_000;
/// The same cap for a face tessellated for a view. Higher, because the face
/// keeps its model-tolerance triangles outside the view and the view's detail
/// comes on top. What the view adds is bounded by the screen, so this guards
/// against runaway rather than setting a budget.
const MAX_VIEW_FACE_TRIS: usize = 400_000;
const MAX_CURVE_DEPTH: u32 = 12;

// ---------------------------------------------------------------------------

mod check;
mod edges;
mod face;
mod loops;
mod refine;
mod sample;
mod view;
#[cfg(test)]
mod tests;

// The public paths (`tess::seams` and friends) used by the app and renderer.
pub use check::{closed_parts, seams, winding, Seams, Winding};
pub use view::{parts_to_refine, tessellate_for_view, ViewTolerance, REFINE_BELOW, REFINE_FLOOR};

// Submodules share everything crate-visible: re-exported here, imported back
// by each via `use super::*`.
pub(crate) use check::*;
pub(crate) use edges::*;
pub(crate) use face::*;
pub(crate) use loops::*;
pub(crate) use refine::*;
pub(crate) use sample::*;
pub(crate) use view::*;

fn placed_bbox(b: &Aabb, place: &Xform) -> Aabb {
    let mut out = Aabb::EMPTY;
    if b.is_empty() {
        return out;
    }
    for i in 0..8 {
        out.add_point(place.point(crate::math::v3(
            if i & 1 == 0 { b.min.x } else { b.max.x },
            if i & 2 == 0 { b.min.y } else { b.max.y },
            if i & 4 == 0 { b.min.z } else { b.max.z },
        )));
    }
    out
}

/// A part's name, as the path down the assembly to it.
///
/// Product names are what a user recognises; the solid's own name is usually
/// blank and names the body, not the component, so it is a fallback.
fn part_name(model: &Model, s: &Solid) -> String {
    let mut path: Vec<&str> = Vec::new();
    let mut at = s.node;
    while let Some(i) = at {
        let n = &model.nodes[i];
        if !n.name.is_empty() {
            path.push(&n.name);
        }
        at = n.parent;
    }
    path.reverse();
    if path.is_empty() {
        if s.name.is_empty() {
            return String::new();
        }
        return s.name.clone();
    }
    // The body's own name only where it adds something the path does not.
    if !s.name.is_empty() && path.last() != Some(&s.name.as_str()) {
        path.push(&s.name);
    }
    path.join(" / ")
}

/// A part's three runs of the edge buffer: loose curves, sharp edges, tangent
/// edges, in that order.
pub(crate) type EdgeSpan = (std::ops::Range<u32>, std::ops::Range<u32>, std::ops::Range<u32>);


// STEP colours are display-referred; `vs_surface` converts them to linear, so
// the vertex's eight bits a channel do not lose the dark end.

/// How many faces go into the first streamed batch, and the ceiling the batch
/// size doubles towards.
///
/// The first batch is small because it decides how long the window stands
/// empty; later ones are large because each batch has a fixed overhead (job
/// list, rayon fold, edge pass). Solids are never divided, so a solid larger
/// than the budget is a batch on its own.
const FIRST_BATCH_FACES: usize = 400;
const MAX_BATCH_FACES: usize = 16000;

/// Tessellates the model in batches of whole solids, handing each to `sink` as
/// it is finished.
///
/// Each batch is tessellated exactly as the whole model would be, so
/// concatenating them gives the same vertices and triangles in the same order
/// (`streaming_and_whole_agree`). Only the edge buffer differs: it is grouped
/// loose-then-sharp-then-tangent within a batch rather than across the model.
///
/// The first batch is made alone, since it decides how long the window stands
/// empty. The rest are made in parallel and handed over in order by whichever
/// thread finished the one owed, hence `sink: Send`
/// (`batches_arrive_in_order_as_if_made_in_turn`). This keeps cores busy
/// through each batch's serial tail.
pub fn tessellate_streaming(model: &Model, tol: f64, mut sink: impl FnMut(Mesh) + Send) {
    let all = batches(model);
    let Some((first, rest)) = all.split_first() else {
        return;
    };
    sink(tessellate_range(model, tol, None, None, first.clone()));
    // The next batch owed, the batches finished ahead of it, and the sink,
    // taken out by whichever thread is handing over. The sink is called with
    // the lock released: a sink that waited on rayon could otherwise pick up
    // another batch's work and deadlock on the lock its own thread holds.
    let owed = std::sync::Mutex::new((0usize, std::collections::BTreeMap::new(), Some(sink)));
    rest.par_iter().enumerate().for_each(|(k, sel)| {
        let batch = tessellate_range(model, tol, None, None, sel.clone());
        let mut state = owed.lock().unwrap();
        state.1.insert(k, batch);
        let Some(mut sink) = state.2.take() else {
            return;
        };
        loop {
            let next = state.0;
            let Some(batch) = state.1.remove(&next) else {
                break;
            };
            state.0 += 1;
            drop(state);
            sink(batch);
            state = owed.lock().unwrap();
        }
        state.2 = Some(sink);
    });
}

/// The runs of solids the model is tessellated in, first to last.
pub(crate) fn batches(model: &Model) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut budget = FIRST_BATCH_FACES;
    while start < model.solids.len() {
        let mut end = start;
        let mut faces = 0;
        while end < model.solids.len() && (end == start || faces < budget) {
            faces += model.solids[end].faces.len() + model.solids[end].wires.len();
            end += 1;
        }
        out.push(start..end);
        start = end;
        budget = (budget * 2).min(MAX_BATCH_FACES);
    }
    out
}

thread_local! {
    /// Set while this thread runs something whose panics are caught and
    /// expected; see [`quietly`].
    static QUIET: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f`, catching a panic and keeping its message off the terminal.
///
/// The constrained triangulator panics on some outlines; those are caught per
/// face. The process hook is wrapped once and only a thread inside this
/// function is quiet, so overlapping tessellations (a view refining while a
/// model streams in) never silence other threads or leave the hook swapped.
pub(crate) fn quietly<T>(f: impl FnOnce() -> T) -> std::thread::Result<T> {
    static WRAP: std::sync::Once = std::sync::Once::new();
    WRAP.call_once(|| {
        let loud = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !QUIET.with(|q| q.get()) {
                loud(info);
            }
        }));
    });
    QUIET.with(|q| q.set(true));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    QUIET.with(|q| q.set(false));
    r
}

/// Tessellates the whole model. `tol` is the maximum chord deviation, in model
/// units.
pub fn tessellate(model: &Model, tol: f64) -> Mesh {
    let mut mesh = Mesh {
        mm_per_unit: model.mm_per_unit,
        ..Default::default()
    };
    // Every batch is held until the last is in, so the whole mesh is sized
    // once rather than grown by doubling.
    let mut made: Vec<Mesh> = Vec::new();
    tessellate_streaming(model, tol, |batch| made.push(batch));
    mesh.vertices.reserve_exact(made.iter().map(|b| b.vertices.len()).sum());
    mesh.indices.reserve_exact(made.iter().map(|b| b.indices.len()).sum());
    mesh.edges.reserve_exact(made.iter().map(|b| b.edges.len()).sum());
    for batch in made {
        mesh.absorb(batch);
    }

    // Swapping triangle edges into the overlay shows what was actually built;
    // a shaded view of a bad triangulation still looks plausible. Six edge
    // points per triangle, in triangle order, so a part's share is its index
    // range doubled and the overlay stays sliceable by part.
    if std::env::var_os("STPVIEW_MESH_EDGES").is_some() {
        let mut wire = Vec::with_capacity(mesh.indices.len() * 2);
        for t in mesh.indices.chunks_exact(3) {
            for k in 0..3 {
                wire.push(mesh.vertices[t[k] as usize].pos);
                wire.push(mesh.vertices[t[(k + 1) % 3] as usize].pos);
            }
        }
        for p in &mut mesh.parts {
            p.wire = 0..0;
            p.tangent = 0..0;
            p.sharp = p.tris.start * 2..p.tris.end * 2;
        }
        mesh.edges = wire;
    }
    mesh
}

/// A choice of faces, by solid and by which of its faces: `tessellate_range`
/// skips every face it answers no for.
pub(crate) type FaceFilter<'a> = &'a (dyn Fn(usize, usize) -> bool + Sync);

/// Tessellates a run of solids into a mesh of their own, at `tol` -- or finer
/// wherever `view` looks, for which see `view.rs` -- and of their faces only
/// those `only` keeps, where it is given.
fn tessellate_range(
    model: &Model,
    tol: f64,
    view: Option<&ViewTolerance>,
    only: Option<FaceFilter>,
    sel: std::ops::Range<usize>,
) -> Mesh {
    // One global cap for every edge in the model, so faces sharing an edge
    // always agree on where its points go.
    let base = Tol::uniform(tol, model.vertex_bbox.extent());

    let solids = &model.solids[sel.clone()];

    // Occurrences of one component prepare once and emit per placement. The
    // key includes the colours baked into the vertices, since the body colour
    // reaches every unstyled face.
    type Key = (u32, [u32; 3], u32);
    let key = |s: &Solid| -> Key {
        (s.source, s.color.map(f32::to_bits), s.alpha.to_bits())
    };
    let mut group_of: HashMap<Key, usize> = HashMap::new();
    // Per group: the representative occurrence and how many there are.
    let mut reps: Vec<usize> = Vec::new();
    let mut occurrences: Vec<usize> = Vec::new();
    let mut group_for_si: Vec<usize> = Vec::with_capacity(solids.len());
    for (si, s) in solids.iter().enumerate() {
        // The loose-face pseudo-solid is nobody's twin. Nor is anything
        // tessellated for a view: occurrences at different distances want
        // different tessellations.
        let g = if s.source == u32::MAX || view.is_some() {
            reps.push(si);
            occurrences.push(0);
            reps.len() - 1
        } else {
            *group_of.entry(key(s)).or_insert_with(|| {
                reps.push(si);
                occurrences.push(0);
                reps.len() - 1
            })
        };
        occurrences[g] += 1;
        group_for_si.push(g);
    }

    // A face's own style beats its body's: a two-material part hangs its
    // colours off the faces.
    let face_color = |s: &Solid, f: &Face| -> [f32; 4] {
        match f.style {
            Some((c, a)) => [c[0], c[1], c[2], a],
            None => [s.color[0], s.color[1], s.color[2], s.alpha],
        }
    };

    // Phase 1: prepare each *repeated* component's faces in parallel, with
    // the representative's placement standing in for the diagnostics. A
    // component placed once gains nothing from preparing ahead and pays for
    // materialising its patches, so singles prepare fused with their emit.
    let prep_jobs: Vec<(usize, &Face, [f32; 4])> = reps
        .iter()
        .enumerate()
        .filter(|(g, _)| occurrences[*g] > 1)
        .flat_map(|(g, &si)| {
            let s = &solids[si];
            s.faces.iter().map(move |f| (g, f, face_color(s, f)))
        })
        .collect();
    let prepared: Vec<(TessStats, PreparedFace)> = prep_jobs
        .par_iter()
        .map(|(g, face, color)| {
            let mut st = TessStats::default();
            // Only ever uniform: a view makes every solid a group of one.
            let pf = prepare_face(
                face,
                &model.surfaces,
                &model.curves,
                *color,
                &base,
                &solids[reps[*g]].placement,
                &mut st,
            );
            (st, pf)
        })
        .collect();
    // Slice the flat prepared list back into per-group spans; a group of one
    // has an empty span at wherever the cursor stood.
    let mut group_faces: Vec<std::ops::Range<usize>> = Vec::with_capacity(reps.len());
    {
        let mut at = 0;
        for (g, &si) in reps.iter().enumerate() {
            let n = if occurrences[g] > 1 { solids[si].faces.len() } else { 0 };
            group_faces.push(at..at + n);
            at += n;
        }
    }

    // Phase 2: emit every occurrence, in solid order, through its own
    // placement, from the shared preparation where there is one. `si` counts
    // within the batch; `Mesh::absorb` shifts the local ranges.
    let emit_jobs: Vec<(u32, usize, Option<usize>)> = solids
        .iter()
        .enumerate()
        .flat_map(|(si, s)| {
            let g = group_for_si[si];
            let span = group_faces[g].clone();
            let shared = occurrences[g] > 1;
            (0..s.faces.len())
                .filter(move |&k| only.is_none_or(|keep| keep(sel.start + si, k)))
                .map(move |k| (si as u32, k, shared.then_some(span.start + k)))
        })
        .collect();
    let chunks: Vec<FaceMesh> = emit_jobs
        .par_iter()
        .fold(FaceMesh::default, |mut acc, (si, k, fi)| {
            let s = &solids[*si as usize];
            let before = (acc.indices.len() as u32, acc.vertices.len() as u32);
            let own = match fi {
                Some(fi) => emit_face(&prepared[*fi].1, &s.placement, &mut acc),
                None => {
                    let face = &s.faces[*k];
                    let pf = prepare_face(
                        face,
                        &model.surfaces,
                        &model.curves,
                        face_color(s, face),
                        &base.placed(view, &s.placement),
                        &s.placement,
                        &mut acc.stats,
                    );
                    emit_face(&pf, &s.placement, &mut acc)
                }
            };
            let grew = acc.indices.len() as u32 - before.0;
            acc.faces.push((*si, *k as u32, grew, own));
            let grew_v = acc.vertices.len() as u32 - before.1;
            match acc.runs.last_mut() {
                Some((s, n, v)) if s == si => {
                    *n += grew;
                    *v += grew_v;
                }
                _ => acc.runs.push((*si, grew, grew_v)),
            }
            acc
        })
        .collect();

    let mut mesh = Mesh {
        mm_per_unit: model.mm_per_unit,
        ..Default::default()
    };
    let total_v: usize = chunks.iter().map(|p| p.vertices.len()).sum();
    let total_i: usize = chunks.iter().map(|p| p.indices.len()).sum();
    mesh.vertices.reserve(total_v);
    mesh.indices.reserve(total_i);

    // Where each solid's triangles begin and end. Contiguous because rayon
    // collects an indexed iterator's runs back in order and the jobs were
    // built solid by solid; `parts_are_contiguous` checks it.
    let mut tri_span = vec![(u32::MAX, 0u32); sel.len()];
    let mut vert_span = vec![(u32::MAX, 0u32); sel.len()];
    let mut faces_of: Vec<Vec<FaceSpan>> = vec![Vec::new(); sel.len()];
    for p in chunks {
        let base = mesh.vertices.len() as u32;
        let mut at = mesh.indices.len() as u32;
        let mut at_face = at;
        for &(si, k, n, b) in &p.faces {
            faces_of[si as usize].push(FaceSpan {
                face: k,
                tris: at_face..at_face + n,
                bbox: [
                    b.min.x as f32, b.min.y as f32, b.min.z as f32,
                    b.max.x as f32, b.max.y as f32, b.max.z as f32,
                ],
            });
            at_face += n;
        }
        let mut at_v = base;
        mesh.vertices.extend_from_slice(&p.vertices);
        mesh.indices.extend(p.indices.iter().map(|i| i + base));
        for (si, n, nv) in &p.runs {
            let e = &mut tri_span[*si as usize];
            e.0 = e.0.min(at);
            at += n;
            e.1 = e.1.max(at);
            let v = &mut vert_span[*si as usize];
            v.0 = v.0.min(at_v);
            at_v += nv;
            v.1 = v.1.max(at_v);
        }
        mesh.bbox = mesh.bbox.union(p.bbox);
        mesh.stats.merge(&p.stats);
    }

    // The shared preparations' statistics, once per occurrence, so counts do
    // not depend on whether work was shared. Merged literally rather than by a
    // scaled twin of `merge`, so a new field cannot be added to one and missed
    // in the other.
    for (g, span) in group_faces.iter().enumerate() {
        for (st, _) in &prepared[span.clone()] {
            for _ in 0..occurrences[g] {
                mesh.stats.merge(st);
            }
        }
    }

    let e = extract_edges_in(model, &base, view, sel.clone());
    let edge_span = e.span;
    mesh.edges = e.lines;
    mesh.bbox = mesh.bbox.union(e.bbox);
    mesh.stats.merge(&e.stats);

    mesh.parts = model.solids[sel.clone()]
        .iter()
        .enumerate()
        .zip(faces_of)
        .map(|((si, s), faces)| {
            let (lo, hi) = tri_span[si];
            let (vlo, vhi) = vert_span[si];
            let (wire, sharp, tangent) = edge_span[si].clone();
            Part {
                name: part_name(model, s),
                node: s.node,
                alpha: s.alpha,
                cappable: s.volume,
                color: s.color,
                tris: lo.min(hi)..hi,
                verts: vlo.min(vhi)..vhi,
                wire,
                sharp,
                tangent,
                bbox: placed_bbox(&s.local_bbox, &s.placement),
                faces,
            }
        })
        .collect();

    // Each part's bounds are those of what was drawn for it, not of its B-rep
    // vertices, which can miss most of the part (a helix's vertices sit only
    // at its ends). The vertex box stays only for a part that drew nothing.
    let (vertices, edges) = (&mesh.vertices, &mesh.edges);
    mesh.parts.par_iter_mut().for_each(|p| {
        let mut b = Aabb::EMPTY;
        for v in &vertices[p.verts.start as usize..p.verts.end as usize] {
            b.add_point(v.position());
        }
        for r in [&p.wire, &p.sharp, &p.tangent] {
            for e in &edges[r.start as usize..r.end as usize] {
                b.add_point(v3(e[0] as f64, e[1] as f64, e[2] as f64));
            }
        }
        if !b.is_empty() {
            p.bbox = b;
        }
    });

    // Only the bodies the file did not call a volume need asking.
    let weld = tol * 1e-3;
    let asked: Vec<bool> = mesh
        .parts
        .par_iter()
        .map(|p| p.cappable || part_is_closed(&mesh, p, weld))
        .collect();
    for (p, c) in mesh.parts.iter_mut().zip(asked) {
        p.cappable = c;
    }
    mesh
}

