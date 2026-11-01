//! Tessellating for a view: finer than the model's own tolerance wherever a
//! camera looks closely.
//!
//! The model's tolerance is a fraction of its size, about a tenth of a pixel
//! with the whole model in view but several pixels when zoomed into a small
//! hole. So inside the frustum a chord may stray a fraction of a pixel *at its
//! own depth*; everywhere else the model's tolerance holds.
//!
//! The unit of work is the solid. Solids share no edges, so a re-tessellated
//! one drops into place without a crack, and within one every edge is sampled
//! from the same field (a function of position, not of which face asks) by
//! every face meeting along it. Only faces the view wants finer are redrawn;
//! the rest would come out identical, so their first triangles stay (see
//! `tessellate_for_view`).
use super::*;

/// Where a camera needs the tessellation finer than the model's own tolerance,
/// and how much finer. World space: the space the mesh is placed in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewTolerance {
    pub eye: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub forward: Vec3,
    /// Tangents of the view's half-angles across and up, widened a little so
    /// turning the camera slightly does not bring coarse geometry into view.
    pub tan_x: f64,
    pub tan_y: f64,
    /// Chord error allowed per unit of depth: the error in pixels times the
    /// size of a pixel at unit distance.
    pub per_depth: f64,
    /// Depths nearer than this are measured as this -- the near plane, inside
    /// which nothing is drawn -- so nothing asks for a tolerance of zero.
    pub near: f64,
    /// And nothing finer than this, however close. See `REFINE_FLOOR`.
    pub floor: f64,
}

/// How much finer than the model's own tolerance a view may ask for, at most.
/// Past this the tessellation would resolve the f32 precision of the vertex
/// positions rather than the geometry.
pub const REFINE_FLOOR: f64 = 1.0 / 1024.0;

/// A view refines a part only if it wants it at least this much finer than
/// the model's tolerance; below that the difference is under a pixel.
pub const REFINE_BELOW: f64 = 0.5;

impl ViewTolerance {
    /// The tolerance the view asks for at a world point, or `None` where it
    /// cannot see it.
    #[inline]
    pub fn at(&self, p: Vec3) -> Option<f64> {
        let d = p - self.eye;
        let z = d.dot(self.forward);
        if !(z > 0.0) {
            return None;
        }
        if d.dot(self.right).abs() > self.tan_x * z || d.dot(self.up).abs() > self.tan_y * z {
            return None;
        }
        Some((self.per_depth * z.max(self.near)).max(self.floor))
    }

    /// The finest the view asks for anywhere in a world box, or `None` if the
    /// box is out of view.
    ///
    /// Conservative both ways: a box is only out of view when all of it is
    /// behind one side of the frustum, and its nearest depth is that of its
    /// nearest corner, which the geometry inside may never reach.
    pub fn finest_in(&self, b: &Aabb) -> Option<f64> {
        if b.is_empty() {
            return None;
        }
        let mut near = f64::INFINITY;
        // One bit per side of the frustum, set while every corner so far is
        // outside it; any bit left at the end is a side all eight are behind.
        let mut out = 0b11111u8;
        for i in 0..8 {
            let c = v3(
                if i & 1 == 0 { b.min.x } else { b.max.x },
                if i & 2 == 0 { b.min.y } else { b.max.y },
                if i & 4 == 0 { b.min.z } else { b.max.z },
            );
            let d = c - self.eye;
            let z = d.dot(self.forward);
            let (x, y) = (d.dot(self.right), d.dot(self.up));
            let mut here = 0u8;
            if !(z > 0.0) {
                here |= 1;
            }
            if x > self.tan_x * z {
                here |= 2;
            }
            if -x > self.tan_x * z {
                here |= 4;
            }
            if y > self.tan_y * z {
                here |= 8;
            }
            if -y > self.tan_y * z {
                here |= 16;
            }
            out &= here;
            near = near.min(z);
        }
        if out != 0 {
            return None;
        }
        Some((self.per_depth * near.max(self.near)).max(self.floor))
    }
}

/// The chord tolerance as the tessellator reads it: one number for the whole
/// model, or finer inside a view.
///
/// Positions are in the solid's own space; the view is in world space, so each
/// solid carries its placement. With no view every answer is the model's
/// tolerance, so the tessellation is bit-identical to a plain one.
#[derive(Clone, Copy)]
pub(crate) struct Tol<'a> {
    pub(crate) base: f64,
    /// The longest chord allowed at `base`; see `max_segment_length`.
    pub(crate) max_seg: f64,
    /// What `max_segment_length` scales with, for chords finer than `base`.
    extent: f64,
    view: Option<(&'a ViewTolerance, &'a Xform)>,
}

impl<'a> Tol<'a> {
    /// The model's tolerance everywhere, with the longest chord it allows.
    #[cfg(test)]
    pub(crate) fn fixed(base: f64, max_seg: f64) -> Tol<'static> {
        Tol {
            base,
            max_seg,
            extent: 0.0,
            view: None,
        }
    }

    /// The model's tolerance, over a model of this extent.
    pub(crate) fn uniform(base: f64, extent: f64) -> Tol<'static> {
        Tol {
            base,
            max_seg: max_segment_length(base, extent),
            extent,
            view: None,
        }
    }

    /// The same, refined wherever `view` looks, for a solid placed at `place`.
    pub(crate) fn placed<'b>(&self, view: Option<&'b ViewTolerance>, place: &'b Xform) -> Tol<'b> {
        Tol {
            base: self.base,
            max_seg: self.max_seg,
            extent: self.extent,
            view: view.map(|v| (v, place)),
        }
    }

    pub(crate) fn is_uniform(&self) -> bool {
        self.view.is_none()
    }

    /// The chord tolerance at a point of the solid.
    #[inline]
    pub(crate) fn at(&self, p: Vec3) -> f64 {
        match self.view {
            None => self.base,
            Some((v, place)) => v.at(place.point(p)).map_or(self.base, |t| t.min(self.base)),
        }
    }

    /// The tolerance and the longest chord allowed for a segment from `a`
    /// through `m` to `b`: the finest any of the three asks for, so a segment
    /// reaching from far off into the view is held to the view's standard.
    #[inline]
    pub(crate) fn segment(&self, a: Vec3, m: Vec3, b: Vec3) -> (f64, f64) {
        if self.view.is_none() {
            return (self.base, self.max_seg);
        }
        let t = self.at(a).min(self.at(m)).min(self.at(b));
        if t >= self.base {
            (self.base, self.max_seg)
        } else {
            (t, max_segment_length(t, self.extent).min(self.max_seg))
        }
    }

    /// The finest tolerance anywhere in a box of the solid's own space. For the
    /// guards that need one number for a whole face.
    pub(crate) fn finest(&self, local: &Aabb) -> f64 {
        match self.view {
            None => self.base,
            Some((v, place)) => v
                .finest_in(&placed_bbox(local, place))
                .map_or(self.base, |t| t.min(self.base)),
        }
    }
}

/// The finest a view asks for anywhere on a face as first drawn, or `None`
/// where the face is out of view.
///
/// Measured on the face's bounds widened by the tolerance it was drawn at,
/// since the surface bulges up to that far past its chords.
fn finest_on(view: &ViewTolerance, f: &FaceSpan, slack: f64) -> Option<f64> {
    let mut b = f.bounds();
    if b.is_empty() {
        return None;
    }
    let s = v3(slack, slack, slack);
    b.min = b.min - s;
    b.max = b.max + s;
    view.finest_in(&b)
}

/// How far a face's surface can stand off the chords it was drawn with, in
/// the world: the tolerance, times whatever scale the placement carries.
fn slack(model: &Model, solid: usize, tol: f64) -> f64 {
    let p = &model.solids[solid].placement;
    tol * p.x.length().max(p.y.length()).max(p.z.length()).max(1.0)
}

/// The parts a view would show coarser than it asks for, finest need first.
///
/// A part qualifies only if shown and wanted more than `REFINE_BELOW` finer
/// than `tol`. Asked face by face where the part has face spans (a long part
/// is only as near as its visible faces), else of the part's bounds.
pub fn parts_to_refine(
    model: &Model,
    parts: &[Part],
    visible: &[bool],
    view: &ViewTolerance,
    tol: f64,
) -> Vec<usize> {
    let mut want: Vec<(f64, usize)> = parts
        .iter()
        .enumerate()
        .filter(|(i, p)| !p.tris.is_empty() && visible.get(*i).copied().unwrap_or(true))
        .filter_map(|(i, p)| {
            if p.faces.is_empty() || i >= model.solids.len() {
                return view.finest_in(&p.bbox).map(|t| (t, i));
            }
            let s = slack(model, i, tol);
            p.faces
                .iter()
                .filter_map(|f| finest_on(view, f, s))
                .min_by(|a, b| a.total_cmp(b))
                .map(|t| (t, i))
        })
        .filter(|&(t, _)| t < tol * REFINE_BELOW)
        .collect();
    want.sort_by(|a, b| a.0.total_cmp(&b.0));
    want.into_iter().map(|(_, i)| i).collect()
}

/// Tessellates again, finer wherever `view` looks closely at them, the faces
/// of these solids that the view can see.
///
/// `parts` is the mesh `tessellate` made, one part per solid, indexed by
/// `solids`. The result has one part per entry of `solids`, holding only the
/// faces the view asks finer than `tol` for (named in `faces`) and every edge
/// of the solid. Other faces stay as first drawn. No crack opens: a face is
/// left alone only when all of it is out of the view's reach, so its shared
/// edges are sampled at the model's tolerance by both faces.
///
/// Occurrences of a repeated component are not shared: each is at its own
/// distance from the eye.
pub fn tessellate_for_view(
    model: &Model,
    tol: f64,
    view: &ViewTolerance,
    parts: &[Part],
    solids: &[usize],
) -> Mesh {
    // Per solid asked for, which of its faces to draw again. A face without a
    // span is drawn again, which is never wrong, only slower.
    let table: HashMap<usize, Vec<bool>> = solids
        .iter()
        .map(|&si| {
            let n = model.solids.get(si).map_or(0, |s| s.faces.len());
            let mut again = vec![true; n];
            if let Some(p) = parts.get(si) {
                let s = slack(model, si, tol);
                for f in &p.faces {
                    if let Some(a) = again.get_mut(f.face as usize) {
                        *a = finest_on(view, f, s).is_some_and(|t| t < tol);
                    }
                }
            }
            (si, again)
        })
        .collect();
    let redo = |si: usize, k: usize| table.get(&si).is_none_or(|t| t.get(k).copied().unwrap_or(true));
    let mut mesh = Mesh {
        mm_per_unit: model.mm_per_unit,
        ..Default::default()
    };
    // Runs of consecutive solids go through together.
    let mut at = 0;
    while at < solids.len() {
        let mut end = at + 1;
        while end < solids.len() && solids[end] == solids[end - 1] + 1 {
            end += 1;
        }
        let range = solids[at]..solids[end - 1] + 1;
        mesh.absorb(tessellate_range(model, tol, Some(view), Some(&redo), range));
        at = end;
    }
    mesh
}
