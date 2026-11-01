//! Refinement: splitting triangles until the surface is within tolerance
//! of its chords, without ever splitting a locked boundary edge.
use super::*;

// Refinement
// ---------------------------------------------------------------------------

pub(crate) struct Patch {
    pub(crate) uv: Vec<Vec2>,
    pub(crate) p3: Vec<Vec3>,
    pub(crate) tris: Vec<[u32; 3]>,
    /// Edges that came from the boundary curves; splitting them would move
    /// them off the true curve and open cracks against the neighbouring face.
    pub(crate) locked: EdgeSet,
}

#[inline]
pub(crate) fn ekey(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

/// Hashes an edge (two vertex indices) as the one word the pair makes.
///
/// SipHash resists adversarial keys at a cost comparable to refinement's own
/// arithmetic; these keys are our own vertex counters, so a multiply spreads
/// them well enough. Nothing iterates these sets, so the hash cannot affect
/// the mesh.
#[derive(Default)]
pub(crate) struct EdgeHash(u64);

impl std::hash::Hasher for EdgeHash {
    fn finish(&self) -> u64 {
        // The table indexes by the low bits and tags by the top ones, so fold
        // the product's high half, which every input bit reaches, down onto
        // its low half, which only the low input bits do.
        let h = self.0.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^ (h >> 32)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ b as u64;
        }
    }
    fn write_u32(&mut self, x: u32) {
        self.0 = (self.0 << 32) | x as u64;
    }
}

pub(crate) type EdgeSet = HashSet<(u32, u32), std::hash::BuildHasherDefault<EdgeHash>>;
pub(crate) type EdgeMap<V> = HashMap<(u32, u32), V, std::hash::BuildHasherDefault<EdgeHash>>;

impl Patch {
    fn split_triangle(&self, t: [u32; 3], m: [Option<u32>; 3], out: &mut Vec<[u32; 3]>) {
        let present = m.iter().filter(|x| x.is_some()).count();
        match present {
            0 => out.push(t),
            3 => {
                let (a, b, c) = (t[0], t[1], t[2]);
                let (ab, bc, ca) = (m[0].unwrap(), m[1].unwrap(), m[2].unwrap());
                out.push([a, ab, ca]);
                out.push([ab, b, bc]);
                out.push([ca, bc, c]);
                out.push([ab, bc, ca]);
            }
            1 => {
                // Rotate so the split edge is a->b.
                let k = m.iter().position(|x| x.is_some()).unwrap();
                let (a, b, c) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
                let ab = m[k].unwrap();
                out.push([a, ab, c]);
                out.push([ab, b, c]);
            }
            _ => {
                // Rotate so the *un*split edge is c->a.
                let k = m.iter().position(|x| x.is_none()).unwrap();
                let r = (k + 1) % 3;
                let (a, b, c) = (t[r], t[(r + 1) % 3], t[(r + 2) % 3]);
                let ab = m[r].unwrap();
                let bc = m[(r + 1) % 3].unwrap();
                out.push([ab, b, bc]);
                // Split the remaining quad a-ab-bc-c along its shorter diagonal.
                if self.p3[a as usize].distance(self.p3[bc as usize])
                    <= self.p3[ab as usize].distance(self.p3[c as usize])
                {
                    out.push([a, ab, bc]);
                    out.push([a, bc, c]);
                } else {
                    out.push([a, ab, c]);
                    out.push([ab, bc, c]);
                }
            }
        }
    }

    pub(crate) fn refine(&mut self, surface: &Surface, tol: &Tol, stats: &mut TessStats) {
        if surface.is_flat() {
            return;
        }
        // Below this parameter step the surface is provably within tolerance,
        // so a triangle that small is finished whatever the deviation test
        // says; see `Surface::uv_step`. Taken at the finest tolerance anywhere
        // on the face, since the guard only ever stops a split.
        let (mut lo, mut hi) = (v2(f64::MAX, f64::MAX), v2(f64::MIN, f64::MIN));
        for p in &self.uv {
            lo = v2(lo.x.min(p.x), lo.y.min(p.y));
            hi = v2(hi.x.max(p.x), hi.y.max(p.y));
        }
        let finest = if tol.is_uniform() {
            tol.base
        } else {
            let mut b = Aabb::EMPTY;
            for p in &self.p3 {
                b.add_point(*p);
            }
            tol.finest(&b)
        };
        let (du_min, dv_min) = surface.uv_step(finest, lo, hi);
        // A chord's error goes as the square of its length, so a tolerance k
        // times finer is log4(k) halvings further down. Three passes for each,
        // because a long thin triangle needs several passes per halving.
        // Converged regions stop splitting, so spare passes cost only a scan.
        // The cap is raised to match for a view.
        let (passes, cap) = if finest < tol.base {
            let more = (0.5 * (tol.base / finest).log2()).ceil() as usize;
            (MAX_REFINE_PASSES + 3 * more.min(6), MAX_VIEW_FACE_TRIS)
        } else {
            (MAX_REFINE_PASSES, MAX_FACE_TRIS)
        };

        // Cleared each pass rather than rebuilt, keeping their capacity.
        let mut mids: EdgeMap<u32> = EdgeMap::default();
        let mut skip = EdgeSet::default();
        for _ in 0..passes {
            if self.tris.len() >= cap {
                stats.capped += 1;
                return;
            }
            mids.clear();
            skip.clear();

            for ti in 0..self.tris.len() {
                let t = self.tris[ti];
                let (a, b, c) = (
                    self.uv[t[0] as usize],
                    self.uv[t[1] as usize],
                    self.uv[t[2] as usize],
                );
                // Already fine enough that no split could improve it.
                let span_u = a.x.max(b.x).max(c.x) - a.x.min(b.x).min(c.x);
                let span_v = a.y.max(b.y).max(c.y) - a.y.min(b.y).min(c.y);
                if span_u <= du_min && span_v <= dv_min {
                    continue;
                }
                // A triangle can bulge in its middle while all three edge
                // midpoints look fine, so the centroid is checked too. It only
                // marks the triangle's own splittable edges: forcing all three
                // would quadruple the mesh every pass, and locked boundary
                // edges cannot take the split anyway.
                let cen_uv = (self.uv[t[0] as usize] + self.uv[t[1] as usize]
                    + self.uv[t[2] as usize])
                    * (1.0 / 3.0);
                let cen_3d = (self.p3[t[0] as usize]
                    + self.p3[t[1] as usize]
                    + self.p3[t[2] as usize])
                    * (1.0 / 3.0);
                // Held to the finest the view asks for anywhere on the
                // triangle, not at its middle alone: see the edge test below.
                let cen_tol = if tol.is_uniform() {
                    tol.base
                } else {
                    t.iter()
                        .map(|&i| tol.at(self.p3[i as usize]))
                        .fold(tol.at(cen_3d), f64::min)
                };
                let bulges = surface.eval(cen_uv.x, cen_uv.y).distance(cen_3d) > cen_tol;

                for e in 0..3 {
                    let (a, b) = (t[e], t[(e + 1) % 3]);
                    let key = ekey(a, b);
                    if mids.contains_key(&key) || skip.contains(&key) {
                        continue;
                    }
                    if self.locked.contains(&key) {
                        skip.insert(key);
                        continue;
                    }
                    let (pa, pb) = (self.p3[a as usize], self.p3[b as usize]);
                    // The finest tolerance at either end or the middle, as a
                    // boundary curve is sampled. Judged at the middle alone, an
                    // edge reaching out of the view gets the coarser tolerance
                    // while its neighbours get the finer one, and splits chase
                    // each other without converging.
                    let (t, _) = tol.segment(pa, (pa + pb) * 0.5, pb);
                    let (uva, uvb) = (self.uv[a as usize], self.uv[b as usize]);
                    let uvm = (uva + uvb) * 0.5;
                    // Never chase a singularity below the tolerance: an edge
                    // along a pole row is one point all the way along. But a
                    // short chord is finished only if its midpoint is close
                    // too: an edge straight across an outline whose opposite
                    // sides are one seam (a torus, a pipe elbow, a cylinder's
                    // seam out and back) starts and ends at one point and goes
                    // all the way round.
                    if pa.distance(pb) <= t
                        && surface.eval(uvm.x, uvm.y).distance((pa + pb) * 0.5) <= t
                    {
                        skip.insert(key);
                        continue;
                    }
                    // The triangle's step guard, applied per edge: a sliver
                    // spanning half a turn in u but nothing in v fails the
                    // triangle test, and its short edges would otherwise be
                    // split every pass until the cap.
                    //
                    // The two limits combine as an ellipse, not a box: an edge
                    // at the limit in both directions is a diagonal half again
                    // as long as either bound allows.
                    let ru = (uva.x - uvb.x).abs() / du_min;
                    let rv = (uva.y - uvb.y).abs() / dv_min;
                    if ru * ru + rv * rv <= 1.0 {
                        skip.insert(key);
                        continue;
                    }
                    let pm = surface.eval(uvm.x, uvm.y);
                    if !pm.is_finite() {
                        skip.insert(key);
                        continue;
                    }
                    if bulges || pm.distance((pa + pb) * 0.5) > t {
                        let idx = self.uv.len() as u32;
                        self.uv.push(uvm);
                        self.p3.push(pm);
                        mids.insert(key, idx);
                    } else {
                        skip.insert(key);
                    }
                }
            }

            if mids.is_empty() {
                return;
            }
            let mut next = Vec::with_capacity(self.tris.len() * 2);
            for ti in 0..self.tris.len() {
                let t = self.tris[ti];
                let m = [
                    mids.get(&ekey(t[0], t[1])).copied(),
                    mids.get(&ekey(t[1], t[2])).copied(),
                    mids.get(&ekey(t[2], t[0])).copied(),
                ];
                self.split_triangle(t, m, &mut next);
            }
            self.tris = next;
        }
    }
}
