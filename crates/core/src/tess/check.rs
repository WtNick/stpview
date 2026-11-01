//! The mesh asked about itself: welded seams, winding, closed volumes.
//! `--check` is this module, and the section cap trusts `closed_parts`.
use super::*;

/// Points within `radius` of one another taken as one point, wherever the grid
/// happens to fall and whatever order they come in.
///
/// Faces meet at copies of one vertex rounded to f32 independently, so they
/// may differ by a few ulps; rounding to a grid would split such copies
/// whenever a cell boundary runs between them. Here a point is a connected
/// piece of "within `radius` of", so a cluster wider than the radius is still
/// one point.
///
/// The grid is only an index, with cells sixteen radii across. Distinct
/// vertices stand far more than that apart, and a position needs a neighbouring
/// cell only along an axis where it is within reach of a cell face, so most
/// lookups touch one cell.
pub(crate) struct Weld {
    radius: f64,
    cell: f64,
    /// Each occupied cell's latest position; `next` chains the rest of it.
    heads: HashMap<(i64, i64, i64), u32, std::hash::BuildHasherDefault<CellHash>>,
    next: Vec<u32>,
    /// Every distinct position taken in, in arrival order.
    pos: Vec<Vec3>,
    parent: Vec<u32>,
    /// Each position's point, numbered in the order the points were first
    /// seen; filled by `settle`.
    point_of: Vec<u32>,
    /// Where each point was first seen.
    pub(crate) points: Vec<Vec3>,
    /// `add`'s scratch list of positions within reach, reused between calls.
    reach: Vec<u32>,
}

/// A multiply-rotate hash for cell keys (three small integers); the standard
/// hasher resists adversaries at several times the cost.
#[derive(Default)]
pub(crate) struct CellHash(u64);

impl std::hash::Hasher for CellHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_i64(&mut self, x: i64) {
        self.write_u64(x as u64);
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl Weld {
    const END: u32 = u32::MAX;

    pub(crate) fn new(radius: f64) -> Weld {
        Weld {
            radius,
            cell: radius * 16.0,
            heads: HashMap::default(),
            next: Vec::new(),
            pos: Vec::new(),
            parent: Vec::new(),
            point_of: Vec::new(),
            points: Vec::new(),
            reach: Vec::new(),
        }
    }

    /// Takes `p` in and returns a handle for it, which `point` turns into
    /// `p`'s point once everything is in and `settle` has run.
    pub(crate) fn add(&mut self, p: Vec3) -> u32 {
        let mut reach = std::mem::take(&mut self.reach);
        reach.clear();
        let mut same = Self::END;
        self.near_positions(p, |i| {
            if self.pos[i as usize] == p {
                same = i;
            }
            reach.push(i);
        });
        // A repeat of a position already in is that position: everything within
        // reach of it has been joined to it already, or will be when it comes.
        let i = if same != Self::END {
            same
        } else {
            let i = self.pos.len() as u32;
            let head = self.heads.insert(self.cell_of(p).0, i).unwrap_or(Self::END);
            self.next.push(head);
            self.pos.push(p);
            self.parent.push(i);
            for &j in &reach {
                self.join(i, j);
            }
            i
        };
        self.reach = reach;
        i
    }

    /// Numbers the points once everything is in, in the order each was first
    /// seen, so the same input always gets the same numbers. A piece's root is
    /// always its earliest position -- `join` keeps the earlier root on top --
    /// so a position that is its own root starts a point and any other takes
    /// the number its root was given already.
    pub(crate) fn settle(&mut self) {
        let mut point_of: Vec<u32> = Vec::with_capacity(self.pos.len());
        for i in 0..self.pos.len() {
            let root = self.root(i as u32) as usize;
            if root == i {
                point_of.push(self.points.len() as u32);
                self.points.push(self.pos[i]);
            } else {
                point_of.push(point_of[root]);
            }
        }
        self.point_of = point_of;
    }

    /// The point a handle from `add` belongs to. Only after `settle`.
    pub(crate) fn point(&self, handle: u32) -> u32 {
        self.point_of[handle as usize]
    }

    /// Calls `f` with every point within `radius` of `p`, taking nothing in:
    /// once per position there, so a point may come more than once. Only after
    /// `settle`.
    pub(crate) fn near(&self, p: Vec3, mut f: impl FnMut(u32)) {
        self.near_positions(p, |i| f(self.point_of[i as usize]));
    }

    fn near_positions(&self, p: Vec3, mut f: impl FnMut(u32)) {
        let (c, side) = self.cell_of(p);
        // Each axis offers the position's own cell, and the neighbour within
        // reach of it when there is one.
        let steps = |s: i64| -> &'static [i64] {
            match s {
                0 => &[0],
                s if s < 0 => &[0, -1],
                _ => &[0, 1],
            }
        };
        for &dx in steps(side.0) {
            for &dy in steps(side.1) {
                for &dz in steps(side.2) {
                    let key = (c.0 + dx, c.1 + dy, c.2 + dz);
                    let mut i = self.heads.get(&key).copied().unwrap_or(Self::END);
                    while i != Self::END {
                        if self.pos[i as usize].distance(p) <= self.radius {
                            f(i);
                        }
                        i = self.next[i as usize];
                    }
                }
            }
        }
    }

    /// The cell `p` is in, and along each axis the neighbour within reach of
    /// it, if one is: -1 or +1 where `p` is within `radius` of that face, 0
    /// where it is not.
    fn cell_of(&self, p: Vec3) -> ((i64, i64, i64), (i64, i64, i64)) {
        let axis = |x: f64| {
            let s = x / self.cell;
            let c = s.floor();
            let reach = self.radius / self.cell;
            let side = if s - c <= reach {
                -1
            } else if c + 1.0 - s <= reach {
                1
            } else {
                0
            };
            (c as i64, side)
        };
        let (x, y, z) = (axis(p.x), axis(p.y), axis(p.z));
        ((x.0, y.0, z.0), (x.1, y.1, z.1))
    }

    fn root(&mut self, mut i: u32) -> u32 {
        while self.parent[i as usize] != i {
            let up = self.parent[self.parent[i as usize] as usize];
            self.parent[i as usize] = up;
            i = up;
        }
        i
    }

    /// Joins two positions' pieces under the earlier of their roots.
    fn join(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.root(a), self.root(b));
        if ra != rb {
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.parent[hi as usize] = lo;
        }
    }
}

/// How well the triangles of separate faces actually meet.
pub struct Seams {
    /// Mesh edges belonging to exactly one triangle once coincident vertices are
    /// welded: a crack, or the rim of a legitimately open shell.
    pub unpaired: usize,
    /// Mesh edges shared by more than two triangles, which a manifold has none
    /// of and usually means a face was emitted twice.
    pub overshared: usize,
    pub total: usize,
    /// Where the first few unpaired edges are, so a crack can be looked at.
    pub cracks: Vec<Vec3>,
    /// Where the first few over-shared edges are. A third triangle on an edge is
    /// a second skin laid over the first, and the two draw on top of each other.
    pub folds: Vec<Vec3>,
    /// Of the unpaired edges, how many run along a B-rep edge. Those are two
    /// faces disagreeing about a boundary they share. The rest are inside a
    /// face, which would mean refinement tore it.
    pub on_brep_edge: usize,
}

/// How consistently the mesh is wound, which decides whether a section can be
/// capped.
///
/// A cap asks the GPU whether a pixel is inside the solid by counting entries
/// against exits along the view ray, which needs the triangle winding.
/// Shading is two-sided, so nothing else would notice a part wound inside out.
pub struct Winding {
    /// Triangles whose vertex order agrees with their own shading normals.
    pub outward: usize,
    /// And those wound against them: the same surface, back to front.
    pub inward: usize,
    /// Too small or too warped for the two normals to be compared.
    pub undecided: usize,
}

/// Compares each triangle's vertex order against the surface normal its
/// vertices carry. The analytic normal is already turned outward by
/// `same_sense`, so where they disagree the winding is wrong.
pub fn winding(mesh: &Mesh) -> Winding {
    let mut w = Winding { outward: 0, inward: 0, undecided: 0 };
    let get = |i: u32| {
        let v = &mesh.vertices[i as usize];
        (v.position(), v.normal())
    };
    for t in mesh.indices.chunks_exact(3) {
        let (pa, na) = get(t[0]);
        let (pb, nb) = get(t[1]);
        let (pc, nc) = get(t[2]);
        let g = (pb - pa).cross(pc - pa);
        let s = na + nb + nc;
        let (gl, sl) = (g.length(), s.length());
        if gl <= 0.0 || sl <= 0.0 {
            w.undecided += 1;
            continue;
        }
        // A well-tessellated triangle is plainly aligned with or opposed to
        // its surface normal; anything in between is too warped to vote.
        let cos = g.dot(s) / (gl * sl);
        if cos > 0.1 {
            w.outward += 1;
        } else if cos < -0.1 {
            w.inward += 1;
        } else {
            w.undecided += 1;
        }
    }
    w
}

/// Whether one part's triangles form a closed surface.
///
/// The part's own vertices are welded in index order, so the answer does not
/// depend on where the grid falls; a body the file did not call a volume is
/// capped on this answer.
pub(crate) fn part_is_closed(mesh: &Mesh, p: &Part, weld: f64) -> bool {
    if p.tris.is_empty() {
        return false;
    }
    let tris = &mesh.indices[p.tris.start as usize..p.tris.end as usize];
    let first = p.verts.start as usize;
    let mut w = Weld::new(weld);
    let own: Vec<u32> = mesh.vertices[first..p.verts.end as usize]
        .iter()
        .map(|v| w.add(v.position()))
        .collect();
    // A triangle reaching outside its part's own vertices would be a fault
    // elsewhere, but it is still answered by position.
    let mut strays: HashMap<u32, u32> = HashMap::new();
    for &i in tris {
        if own.get((i as usize).wrapping_sub(first)).is_none() {
            strays.entry(i).or_insert_with(|| w.add(mesh.vertices[i as usize].position()));
        }
    }
    w.settle();
    let id = |i: u32| {
        let handle = own.get((i as usize).wrapping_sub(first)).copied().unwrap_or_else(|| strays[&i]);
        w.point(handle)
    };
    let mut edges: Vec<u64> = Vec::with_capacity(p.tris.len());
    for t in tris.chunks_exact(3) {
        let v = [id(t[0]), id(t[1]), id(t[2])];
        for k in 0..3 {
            let (a, b) = (v[k], v[(k + 1) % 3]);
            if a != b {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                edges.push(((lo as u64) << 32) | hi as u64);
            }
        }
    }
    edges.sort_unstable();
    let mut i = 0;
    while i < edges.len() {
        let mut j = i;
        while j < edges.len() && edges[j] == edges[i] {
            j += 1;
        }
        // Exactly two triangles on every edge, or it is not a closed surface
        // and has no inside for a cap to find.
        if j - i != 2 {
            return false;
        }
        i = j;
    }
    true
}

/// Marks each part closed or not, which decides whether it can be capped.
///
/// A cap's entry/exit count only cancels for a part with no holes; an open
/// shell would be capped across its whole silhouette. `seams` welds the mesh
/// as one and cannot say which part a crack belongs to, so this asks per part.
pub fn closed_parts(mesh: &Mesh, weld: f64) -> Vec<bool> {
    mesh.parts
        .par_iter()
        .map(|p| part_is_closed(mesh, p, weld))
        .collect()
}

/// Welds the mesh by position and counts how many edges fail to pair up.
///
/// Faces never share vertices; adjacent triangles line up only because both
/// faces sampled the shared curve identically, which is why refinement may
/// not split a boundary edge. An open shell has a genuine rim, so a nonzero
/// count is not by itself a fault.
pub fn seams(mesh: &Mesh, weld: f64) -> Seams {
    let mut w = Weld::new(weld);
    let handles: Vec<u32> = mesh.vertices.iter().map(|v| w.add(v.position())).collect();
    w.settle();
    let vid: Vec<u32> = handles.iter().map(|&h| w.point(h)).collect();

    // Packed (low, high) welded ids, sorted, so counting runs is one pass and
    // the peak cost is one u64 per mesh edge rather than a hash entry.
    let mut edges: Vec<u64> = Vec::with_capacity(mesh.indices.len());
    for t in mesh.indices.chunks_exact(3) {
        for k in 0..3 {
            let (a, b) = (vid[t[k] as usize], vid[t[(k + 1) % 3] as usize]);
            if a == b {
                continue; // a degenerate sliver, not a seam
            }
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            edges.push(((lo as u64) << 32) | hi as u64);
        }
    }
    edges.sort_unstable();

    // The overlay samples every B-rep edge exactly once, so the welded points it
    // lands on are the ones that ought to lie on a shared boundary.
    let pos = &w.points;
    let mut on_edge = vec![false; pos.len()];
    for p in &mesh.edges {
        w.near(v3(p[0] as f64, p[1] as f64, p[2] as f64), |i| on_edge[i as usize] = true);
    }
    let is_boundary = |id: usize| on_edge[id];

    let (mut unpaired, mut overshared, mut total, mut on_brep_edge) = (0, 0, 0, 0);
    let mut cracks: Vec<Vec3> = Vec::new();
    let mut folds: Vec<Vec3> = Vec::new();
    let mut i = 0;
    while i < edges.len() {
        let mut j = i;
        while j < edges.len() && edges[j] == edges[i] {
            j += 1;
        }
        total += 1;
        match j - i {
            1 => {
                unpaired += 1;
                let (lo, hi) = ((edges[i] >> 32) as usize, (edges[i] & 0xffff_ffff) as usize);
                if is_boundary(lo) && is_boundary(hi) {
                    on_brep_edge += 1;
                }
                if cracks.len() < 24 {
                    cracks.push((pos[lo] + pos[hi]) * 0.5);
                }
            }
            2 => {}
            _ => {
                overshared += 1;
                if folds.len() < 24 {
                    let (lo, hi) = ((edges[i] >> 32) as usize, (edges[i] & 0xffff_ffff) as usize);
                    folds.push((pos[lo] + pos[hi]) * 0.5);
                }
            }
        }
        i = j;
    }
    Seams {
        unpaired,
        overshared,
        total,
        cracks,
        folds,
        on_brep_edge,
    }
}

