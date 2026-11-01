//! STL: a soup of triangles, read into a mesh.
//!
//! An STL file is a tessellation already. What the format leaves out --
//! which corners are one point, which way the surface faces there, and where
//! its edges are -- is recovered here. The result is an ordinary [`Mesh`],
//! one part per `solid`, taken like a tessellated B-rep.
//!
//! STL has no units. Its triangles are read as millimetres, which is what
//! nearly every writer means and what a scene's world is.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use rayon::prelude::*;

use crate::math::{v3, Aabb, Vec3};
use crate::model::DEFAULT_COLOR;
use crate::tess::{Mesh, Part, Vertex};

/// One `solid` of a file: three corners per facet, in the order written.
pub struct Solid {
    /// As the file names it, which a binary file never does.
    pub name: String,
    pub corners: Vec<[f32; 3]>,
    /// One per facet where the file colours any of them, display-referred;
    /// `None` for a facet that takes the solid's colour. Empty when the file
    /// colours none.
    pub colors: Vec<Option<[u8; 3]>>,
    /// The colour of the whole solid, where the file gives one.
    pub color: Option<[u8; 3]>,
}

impl Solid {
    fn named(name: String) -> Solid {
        Solid { name, corners: Vec::new(), colors: Vec::new(), color: None }
    }

    pub fn facets(&self) -> usize {
        self.corners.len() / 3
    }
}

/// A file as read.
pub struct Stl {
    pub solids: Vec<Solid>,
    pub binary: bool,
    /// Facets whose written normal points against the order of their corners.
    ///
    /// The order is what is drawn; the written normal is ignored, as nearly
    /// every reader ignores it (writers often leave it zero). Only the
    /// section cap, which counts by winding, can tell. A file wound inside
    /// out shows here as nearly all its facets.
    pub against: usize,
    /// Facets the file promised but did not hold: a binary file shorter than
    /// its count says, or an ASCII loop without three readable vertices.
    pub unread: usize,
}

/// A binary file's header, and each facet after the count: a normal, three
/// corners and a 16-bit attribute word.
const HEADER: usize = 80;
const FACET: usize = 50;

/// The facet count of a binary file, if its length is exactly what that count
/// says.
///
/// Told by length rather than by the first word, because a binary header is
/// free text and some writers start it with `solid`, the word that opens an
/// ASCII file. The converse does not happen by accident: four bytes of ASCII
/// text read as a count are at least 0x20202020, half a billion facets.
fn binary_count(bytes: &[u8]) -> Option<usize> {
    let n = u32::from_le_bytes(bytes.get(HEADER..HEADER + 4)?.try_into().ok()?) as usize;
    (bytes.len() == HEADER + 4 + n * FACET).then_some(n)
}

/// Whether the file opens as ASCII STL does: `solid`, then text.
///
/// "Then text" is a NUL byte anywhere in the first kilobyte, which a binary
/// file with a `solid` header has: in its count, unless that is over sixteen
/// million, and in nearly every float after it.
fn ascii(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(1024)];
    let text = head.trim_ascii_start().to_ascii_lowercase();
    text.starts_with(b"solid")
        && !head.contains(&0)
        && (contains(&text, b"facet") || contains(&text, b"endsolid"))
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

/// Whether these bytes are STL by their contents alone, which is all a shell
/// thumbnail provider has. Strict where [`parse`] is lenient: a binary file
/// whose count disagrees with its length is not claimed here.
pub fn looks_like(bytes: &[u8]) -> bool {
    binary_count(bytes).is_some() || ascii(bytes)
}

/// Reads either encoding.
pub fn parse(bytes: &[u8]) -> Result<Stl, String> {
    if let Some(n) = binary_count(bytes) {
        return Ok(parse_binary(bytes, n));
    }
    if ascii(bytes) {
        return Ok(parse_ascii(bytes));
    }
    // A binary file whose count disagrees with its length: a writer that never
    // went back to fill the count in, or a file cut short.
    if bytes.len() >= HEADER + 4 {
        let n = u32::from_le_bytes(bytes[HEADER..HEADER + 4].try_into().unwrap()) as usize;
        return Ok(parse_binary(bytes, n));
    }
    Err("not an STL file: too short to be binary, and does not open with `solid`".into())
}

fn parse_binary(bytes: &[u8], declared: usize) -> Stl {
    let held = (bytes.len() - HEADER - 4) / FACET;
    // A count of zero is a writer that never filled it in, and the length is
    // the only word left; a count past the end is a file cut short.
    let n = if declared == 0 { held } else { declared.min(held) };
    let f32_at = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let vec_at = |o: usize| [f32_at(o), f32_at(o + 4), f32_at(o + 8)];

    let mut corners = Vec::with_capacity(n * 3);
    let mut attrs = Vec::with_capacity(n);
    let mut against = 0;
    for i in 0..n {
        let o = HEADER + 4 + i * FACET;
        let (a, b, c) = (vec_at(o + 12), vec_at(o + 24), vec_at(o + 36));
        if points_against(vec_at(o), a, b, c) {
            against += 1;
        }
        corners.extend([a, b, c]);
        attrs.push(u16::from_le_bytes([bytes[o + 48], bytes[o + 49]]));
    }
    let (color, colors) = binary_colors(&bytes[..HEADER], &attrs);
    Stl {
        solids: vec![Solid { name: String::new(), corners, colors, color }],
        binary: true,
        against,
        unread: declared.saturating_sub(held),
    }
}

/// The two conventions for colour in a binary file, told apart by the header.
///
/// Materialise Magics writes `COLOR=` and four bytes into the header for the
/// whole solid, and gives a facet its own colour by *clearing* bit 15 of its
/// attribute word, red in the low five bits. VisCAM and SolidView have no
/// solid colour, and *set* bit 15, red in the high five. Magics' fourth byte
/// is of disputed meaning (opacity or unused), so it is not read: wrongly
/// opaque is a lesser fault than wrongly invisible.
///
/// Read strictly, Magics makes a zero attribute word a black facet, and zero
/// is what colour-unaware writers put there. So a file whose words are all
/// zero colours no facet.
fn binary_colors(header: &[u8], attrs: &[u16]) -> (Option<[u8; 3]>, Vec<Option<[u8; 3]>>) {
    // Five bits to eight, rounded.
    let five = |w: u16, shift: u32| ((((w >> shift) & 31) as u32 * 255 + 15) / 31) as u8;
    let magics = header
        .windows(6)
        .position(|w| w == b"COLOR=")
        .and_then(|i| header.get(i + 6..i + 9))
        .map(|c| [c[0], c[1], c[2]]);
    let colors = match magics {
        Some(_) if attrs.iter().any(|&w| w != 0) => attrs
            .iter()
            .map(|&w| (w & 0x8000 == 0).then(|| [five(w, 0), five(w, 5), five(w, 10)]))
            .collect(),
        None if attrs.iter().any(|&w| w & 0x8000 != 0) => attrs
            .iter()
            .map(|&w| (w & 0x8000 != 0).then(|| [five(w, 10), five(w, 5), five(w, 0)]))
            .collect(),
        _ => Vec::new(),
    };
    (magics, colors)
}

/// Line by line, by keyword, and forgiving: case is ignored, a missing `solid`
/// or `endsolid` costs nothing, and lines it does not know are passed over.
fn parse_ascii(bytes: &[u8]) -> Stl {
    let mut stl = Stl { solids: Vec::new(), binary: false, against: 0, unread: 0 };
    let mut solid: Option<Solid> = None;
    let mut normal = [0.0f32; 3];
    let mut ring: Vec<[f32; 3]> = Vec::new();
    let mut unreadable = false;
    for line in bytes.split(|&b| b == b'\n') {
        let line = line.trim_ascii();
        let mut words = line.split(|b| b.is_ascii_whitespace()).filter(|w| !w.is_empty());
        let Some(key) = words.next() else { continue };
        let is = |k: &[u8]| key.eq_ignore_ascii_case(k);
        if is(b"solid") {
            stl.solids.extend(solid.take());
            let name = line[key.len()..].trim_ascii();
            solid = Some(Solid::named(String::from_utf8_lossy(name).into_owned()));
        } else if is(b"endsolid") {
            stl.solids.extend(solid.take());
        } else if is(b"facet") {
            // `facet normal x y z`; the numbers may be missing.
            words.next();
            normal = numbers(words).unwrap_or([0.0; 3]);
        } else if is(b"outer") {
            ring.clear();
            unreadable = false;
        } else if is(b"vertex") {
            match numbers(words) {
                Some(p) => ring.push(p),
                None => unreadable = true,
            }
        } else if is(b"endloop") {
            let s = solid.get_or_insert_with(|| Solid::named(String::new()));
            if ring.len() < 3 || unreadable {
                stl.unread += 1;
            } else {
                // More than three is a polygon, which the format forbids and
                // some writers emit anyway. Fanned, it is the intended face
                // wherever it is convex.
                for k in 1..ring.len() - 1 {
                    s.corners.extend([ring[0], ring[k], ring[k + 1]]);
                }
                if points_against(normal, ring[0], ring[1], ring[2]) {
                    stl.against += 1;
                }
            }
            ring.clear();
            unreadable = false;
        }
    }
    // A loop the file ended in the middle of.
    if !ring.is_empty() {
        stl.unread += 1;
    }
    stl.solids.extend(solid.take());
    stl
}

/// Three numbers off the front of a line.
fn numbers<'a>(mut words: impl Iterator<Item = &'a [u8]>) -> Option<[f32; 3]> {
    let mut next = || std::str::from_utf8(words.next()?).ok()?.parse::<f32>().ok();
    Some([next()?, next()?, next()?])
}

/// Whether a written normal points against the facet its corners wind. A zero
/// normal -- the writer did not say -- is against nothing.
fn points_against(n: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> bool {
    let (a, b, c) = (point(a), point(b), point(c));
    (b - a).cross(c - a).dot(point(n)) < 0.0
}

fn point(p: [f32; 3]) -> Vec3 {
    v3(p[0] as f64, p[1] as f64, p[2] as f64)
}

// Into a mesh
// ---------------------------------------------------------------------------

/// What building the mesh found that the file did not say.
#[derive(Debug, Default, Clone)]
pub struct StlStats {
    /// Solids with facets, each a part.
    pub solids: usize,
    /// Corners as written, and the distinct points they weld to.
    pub corners: usize,
    pub points: usize,
    /// Facets two of whose corners are one point, or with a corner that is not
    /// a number: a line or nothing rather than a triangle. Left out.
    pub dropped: usize,
    /// Facets too thin for their own normal to mean anything. Drawn, but they
    /// lend nothing to their neighbours' shading and mark no crease.
    pub slivers: usize,
    /// Edges with one facet: the rim of a hole, or of a sheet.
    pub open: usize,
    /// Edges with more than two.
    pub overshared: usize,
    /// Edges whose two facets run along it the same way, so that one of them
    /// is wound against the other.
    pub flipped: usize,
    /// Edges between two facets that both have a normal; of those, the ones
    /// flat to within [`FLAT_ANGLE`], and the creases past [`CREASE_ANGLE`].
    pub between: usize,
    pub flat: usize,
    pub creases: usize,
    /// Solids too rough for their creases to be features, whose creases are
    /// left to `--edges all`.
    pub rough: usize,
}

impl StlStats {
    fn add(&mut self, o: &StlStats) {
        self.solids += o.solids;
        self.corners += o.corners;
        self.points += o.points;
        self.dropped += o.dropped;
        self.slivers += o.slivers;
        self.open += o.open;
        self.overshared += o.overshared;
        self.flipped += o.flipped;
        self.between += o.between;
        self.flat += o.flat;
        self.creases += o.creases;
        self.rough += o.rough;
    }
}

/// A facet whose cross product is under this fraction of its longest edge
/// squared -- the sine of its smallest angle, near enough -- has a normal made
/// of rounding. Relative to the facet, because a degeneracy test has to scale
/// with what it tests.
const SLIVER: f64 = 1e-6;

/// Facets meeting at more than this, in degrees, are a crease: shaded apart,
/// and an edge in the overlay.
///
/// Twice the B-rep's `TANGENT_ANGLE`, which compares true surface normals.
/// Neighbouring facets on a smooth surface differ by the tessellation step
/// instead, which CAD exporters take up to 30 degrees and small fillets
/// chorded in two or three facets push past that. A 45-degree chamfer is
/// still a crease.
pub const CREASE_ANGLE: f64 = 40.0;

/// Neighbouring facets within this, in degrees, are flat.
pub const FLAT_ANGLE: f64 = 1.0;

/// A solid whose edges between facets are less than this share flat is rough
/// at the scale of its facets, and its creases are texture, not features.
///
/// A CAD exporter splits planar faces and cylinder strips into exactly
/// coplanar triangles, so a mesh tessellated from a B-rep is typically a
/// quarter flat or more. A sculpted, scanned or displaced surface has almost
/// no coplanar neighbours. A tenth sits between the two with margin.
const SMOOTH_SHARE: f64 = 0.1;

/// Builds the mesh: one part per solid, of every facet that is a triangle.
///
/// `name` names a solid the file left unnamed, which in a binary file is all
/// of them; the caller passes the file's stem.
pub fn mesh(stl: &Stl, name: &str) -> (Mesh, StlStats) {
    let built: Vec<(Mesh, StlStats)> = stl
        .solids
        .par_iter()
        .filter(|s| !s.corners.is_empty())
        .map(|s| solid_mesh(s, if s.name.is_empty() { name } else { s.name.as_str() }))
        .collect();
    let mut mesh = Mesh::default();
    let mut stats = StlStats::default();
    for (m, s) in built {
        mesh.absorb(m);
        stats.add(&s);
    }
    (mesh, stats)
}

/// One solid, welded, shaded and edged.
///
/// Shading normals are smoothed across facets meeting at under
/// [`CREASE_ANGLE`], weighted by each facet's angle at the point, and the
/// overlay's creases are exactly the edges across which they are not: one
/// decision. Whether those lines are drawn by default is decided per solid;
/// see [`SMOOTH_SHARE`]. Open and over-shared edges are always drawn, as a
/// sheet's rim is in a B-rep.
fn solid_mesh(s: &Solid, name: &str) -> (Mesh, StlStats) {
    let mut st = StlStats { corners: s.corners.len(), solids: 1, ..Default::default() };
    let (point_of, points) = weld(&s.corners);
    st.points = points.len();
    let at = |i: u32| point(points[i as usize]);

    // The facets that are triangles, as welded points, and their colours.
    let byte = |c: f32| (c * 255.0).round() as u8;
    let plain = s.color.unwrap_or(DEFAULT_COLOR.map(byte));
    let mut tris: Vec<[u32; 3]> = Vec::with_capacity(s.facets());
    let mut paint: Vec<[u8; 3]> = Vec::with_capacity(s.facets());
    for (f, c) in point_of.chunks_exact(3).enumerate() {
        let t = [c[0], c[1], c[2]];
        let finite = t.iter().all(|&i| points[i as usize].iter().all(|x| x.is_finite()));
        if t[0] == t[1] || t[1] == t[2] || t[2] == t[0] || !finite {
            st.dropped += 1;
            continue;
        }
        tris.push(t);
        paint.push(s.colors.get(f).copied().flatten().unwrap_or(plain));
    }

    // Each facet's unit normal, by the order of its corners, and its angle at
    // each corner. A sliver's normal is zero, which nothing else can be.
    let facet: Vec<([f32; 3], [f32; 3])> = tris
        .par_iter()
        .map(|t| {
            let (a, b, c) = (at(t[0]), at(t[1]), at(t[2]));
            let g = (b - a).cross(c - a);
            let longest = (b - a).length_sq().max((c - b).length_sq()).max((a - c).length_sq());
            if !(g.length() > SLIVER * longest) {
                return ([0.0; 3], [0.0; 3]);
            }
            let angle = |p: Vec3, q: Vec3, r: Vec3| {
                let (u, v) = (q - p, r - p);
                u.cross(v).length().atan2(u.dot(v)) as f32
            };
            ((g * (1.0 / g.length())).as_f32(), [angle(a, b, c), angle(b, c, a), angle(c, a, b)])
        })
        .collect();
    st.slivers = facet.iter().filter(|f| f.0 == [0.0; 3]).count();

    // The corners at each point, as `3 * facet + k`, grouped by point in
    // point order.
    let np = points.len();
    let mut first = vec![0u32; np + 1];
    for t in &tris {
        for &v in t {
            first[v as usize + 1] += 1;
        }
    }
    for i in 0..np {
        first[i + 1] += first[i];
    }
    let mut corners = vec![0u32; tris.len() * 3];
    {
        let mut fill = first.clone();
        for (f, t) in tris.iter().enumerate() {
            for (k, &v) in t.iter().enumerate() {
                corners[fill[v as usize] as usize] = (3 * f + k) as u32;
                fill[v as usize] += 1;
            }
        }
    }

    let cos_limit = CREASE_ANGLE.to_radians().cos();
    let normal_at = |c: u32, around: &[u32]| -> Vec3 {
        let own = facet[c as usize / 3].0;
        let mut sum = Vec3::ZERO;
        for &o in around {
            let (n, angles) = &facet[o as usize / 3];
            if *n == [0.0; 3] {
                continue;
            }
            // A sliver has no normal to keep a crease by, so it takes the
            // average of everything around it.
            if own != [0.0; 3] && point(own).dot(point(*n)) < cos_limit {
                continue;
            }
            sum += point(*n) * angles[o as usize % 3] as f64;
        }
        sum.normalize_or(if own == [0.0; 3] { v3(0.0, 0.0, 1.0) } else { point(own) })
    };

    // One vertex per point and distinct normal and colour, made in parallel
    // over runs of points. Corners in one smooth region sum the same facets
    // in the same order, so they agree to the bit and share a vertex.
    const RUN: usize = 4096;
    let made: Vec<(Vec<Vertex>, Vec<u32>)> = (0..np.div_ceil(RUN))
        .into_par_iter()
        .map(|r| {
            let mut verts: Vec<Vertex> = Vec::new();
            // Each corner's vertex, as an index into `verts`, in `corners` order.
            let mut slot: Vec<u32> = Vec::new();
            for v in r * RUN..((r + 1) * RUN).min(np) {
                let around = &corners[first[v] as usize..first[v + 1] as usize];
                let base = verts.len();
                for &c in around {
                    let color = paint[c as usize / 3].map(|x| x as f32 / 255.0);
                    let vx = Vertex::new(at(v as u32), normal_at(c, around), color, 1.0);
                    let same = verts[base..]
                        .iter()
                        .position(|w| bytemuck::bytes_of(w) == bytemuck::bytes_of(&vx));
                    slot.push(match same {
                        Some(j) => (base + j) as u32,
                        None => {
                            verts.push(vx);
                            (verts.len() - 1) as u32
                        }
                    });
                }
            }
            (verts, slot)
        })
        .collect();
    let mut vertices: Vec<Vertex> = Vec::with_capacity(made.iter().map(|m| m.0.len()).sum());
    let mut indices = vec![0u32; tris.len() * 3];
    // The runs walked `corners` front to back between them, so their slots
    // line up with it end to end.
    let mut j = 0;
    for (verts, slot) in made {
        let base = vertices.len() as u32;
        for s in slot {
            indices[corners[j] as usize] = base + s;
            j += 1;
        }
        vertices.extend(verts);
    }

    // The edges, as packed (low, high) point pairs, each with its facet and
    // whether that facet runs along it low to high.
    let mut ends: Vec<(u64, u32)> = Vec::with_capacity(tris.len() * 3);
    for (f, t) in tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let (lo, hi) = (a.min(b), a.max(b));
            ends.push((((lo as u64) << 32) | hi as u64, ((f as u32) << 1) | (a < b) as u32));
        }
    }
    ends.par_sort_unstable();
    // Rims and creases, by their packed point pairs; and how many of the
    // edges between two facets are flat, which decides what a crease is.
    let cos_flat = FLAT_ANGLE.to_radians().cos();
    let (mut rims, mut creases): (Vec<u64>, Vec<u64>) = (Vec::new(), Vec::new());
    let (mut between, mut flat) = (0usize, 0usize);
    for run in ends.chunk_by(|x, y| x.0 == y.0) {
        match run {
            [(key, _)] => {
                st.open += 1;
                rims.push(*key);
            }
            [(key, x), (_, y)] => {
                if x & 1 == y & 1 {
                    st.flipped += 1;
                }
                // A sliver's normal is rounding, so an edge it shares is no
                // evidence of anything.
                let (n, m) = (facet[(x >> 1) as usize].0, facet[(y >> 1) as usize].0);
                if n == [0.0; 3] || m == [0.0; 3] {
                    continue;
                }
                let cos = point(n).dot(point(m));
                between += 1;
                if cos > cos_flat {
                    flat += 1;
                } else if cos < cos_limit {
                    creases.push(*key);
                }
            }
            [(key, _), ..] => {
                st.overshared += 1;
                rims.push(*key);
            }
            [] => unreachable!("chunk_by yields no empty runs"),
        }
    }
    st.creases = creases.len();
    st.between = between;
    st.flat = flat;
    // Rough where few neighbours are flat: there every crease is texture, and
    // drawing them all buries the surface under lines. They go in the
    // overlay's second range, which `--edges all` draws, as a B-rep's tangent
    // edges do.
    let rough = (flat as f64) < SMOOTH_SHARE * between as f64;
    st.rough = rough as usize;
    let (sharp, soft): (Vec<u64>, Vec<u64>) = if rough {
        (rims, creases)
    } else {
        (rims.into_iter().chain(creases).collect(), Vec::new())
    };
    let mut lines: Vec<[f32; 3]> = Vec::with_capacity(2 * (sharp.len() + soft.len()));
    for key in sharp.iter().chain(&soft) {
        lines.push(points[(key >> 32) as usize]);
        lines.push(points[(key & 0xffff_ffff) as usize]);
    }

    let mut bbox = Aabb::EMPTY;
    for v in &vertices {
        bbox.add_point(v.position());
    }
    let (n_sharp, n_lines) = (2 * sharp.len() as u32, lines.len() as u32);
    let part = Part {
        name: name.to_string(),
        node: None,
        alpha: 1.0,
        // Every edge two facets running opposite ways along it is a surface
        // with an inside and a known way out, which is what a cap counts.
        cappable: st.open == 0 && st.overshared == 0 && st.flipped == 0,
        color: s.color.map_or(DEFAULT_COLOR, |c| c.map(|x| x as f32 / 255.0)),
        tris: 0..indices.len() as u32,
        verts: 0..vertices.len() as u32,
        wire: 0..0,
        sharp: 0..n_sharp,
        tangent: n_sharp..n_lines,
        bbox,
        faces: Vec::new(),
    };
    let mut mesh = Mesh {
        vertices,
        indices,
        parts: vec![part],
        edges: lines,
        bbox,
        ..Default::default()
    };
    // By range, as the tessellator counts them: what `sharp` draws and what
    // `all` adds.
    mesh.stats.faces_ok = tris.len();
    mesh.stats.edges_sharp = sharp.len();
    mesh.stats.edges_tangent = soft.len();
    mesh.stats.edges_single_use = st.open;
    (mesh, st)
}

/// Gives each distinct point one number, in the order points first appear.
///
/// By exact value: a writer emits a point that several facets share as the
/// same float each time, so equal bits are one point, and nothing merely close
/// is merged -- a feature smaller than any tolerance keeps its own corners.
/// Signed zeros are one point: they are equal, and a writer that negates a
/// zero has not moved anything.
fn weld(corners: &[[f32; 3]]) -> (Vec<u32>, Vec<[f32; 3]>) {
    let mut seen: HashMap<u128, u32, BuildHasherDefault<Bits>> =
        HashMap::with_capacity_and_hasher(corners.len() / 4, Default::default());
    let mut points: Vec<[f32; 3]> = Vec::with_capacity(corners.len() / 4);
    let ids = corners
        .iter()
        .map(|p| {
            // `+ 0.0` turns -0.0 into 0.0 and leaves every other value alone.
            let [x, y, z] = p.map(|c| (c + 0.0).to_bits() as u128);
            *seen.entry((x << 64) | (y << 32) | z).or_insert_with(|| {
                points.push(*p);
                (points.len() - 1) as u32
            })
        })
        .collect();
    (ids, points)
}

/// Hashes a point's bits. The standard hasher resists collision attacks a
/// model file cannot mount, at several times the cost, and welding asks once
/// per corner.
#[derive(Default)]
struct Bits(u64);

impl Hasher for Bits {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }

    fn write_u64(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
    }

    fn write_u128(&mut self, x: u128) {
        self.write_u64(x as u64);
        self.write_u64((x >> 64) as u64);
    }

    /// A multiply only carries upwards, so the low bits a table indexes by
    /// would see only the low bits of the input, and a coordinate that is a
    /// round number has none set. MurmurHash3's finaliser folds the rest down.
    fn finish(&self) -> u64 {
        let mut h = self.0;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^ (h >> 33)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Tri = [[f32; 3]; 3];

    /// A cube of side two about the origin, wound outward: for each axis and
    /// side, a square in the other two axes, which run cyclically so that
    /// their cross product is the axis.
    fn cube() -> Vec<Tri> {
        let mut out = Vec::new();
        for a in 0..3 {
            for s in [-1.0f32, 1.0] {
                let (u, v) = ((a + 1) % 3, (a + 2) % 3);
                let p = |i: f32, j: f32| {
                    let mut q = [0.0; 3];
                    q[a] = s;
                    q[u] = i;
                    q[v] = j;
                    q
                };
                let mut q = [p(-1.0, -1.0), p(1.0, -1.0), p(1.0, 1.0), p(-1.0, 1.0)];
                if s < 0.0 {
                    q.reverse();
                }
                out.push([q[0], q[1], q[2]]);
                out.push([q[0], q[2], q[3]]);
            }
        }
        out
    }

    /// A unit sphere in latitude and longitude, wound outward, with a fan at
    /// each pole. Neighbouring facets meet at about 360 / `lon` degrees.
    fn sphere(lat: usize, lon: usize) -> Vec<Tri> {
        bumpy(lat, lon, 0.0)
    }

    /// The same, each point but the poles pushed out by up to `bump` of the
    /// radius, by a hash of where it is: a rough surface, but the same one
    /// every run.
    fn bumpy(lat: usize, lon: usize, bump: f64) -> Vec<Tri> {
        let at = |i: usize, j: usize| {
            // `j % lon`: a full turn is not bitwise where it started, and the
            // seam would open.
            let j = j % lon;
            let t = std::f64::consts::PI * i as f64 / lat as f64;
            let p = std::f64::consts::TAU * j as f64 / lon as f64;
            let noise = ((i * 7919 + j * 104_729) % 1000) as f64 / 1000.0;
            let r = if i == 0 || i == lat { 1.0 } else { 1.0 + bump * noise };
            [(r * t.sin() * p.cos()) as f32, (r * t.sin() * p.sin()) as f32, (r * t.cos()) as f32]
        };
        let mut out = Vec::new();
        for i in 0..lat {
            for j in 0..lon {
                let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
                // The poles are one point however `p` is spelled, which the
                // weld relies on: the fan's tip is written as the same bits.
                if i > 0 {
                    out.push([a, b, d]);
                }
                if i + 1 < lat {
                    out.push([b, c, d]);
                }
            }
        }
        // `sin(pi)` is not zero, so the south pole is a ring of nearly equal
        // points; pin it.
        for t in &mut out {
            for p in t.iter_mut() {
                if p[2] == -1.0 {
                    *p = [0.0, 0.0, -1.0];
                } else if p[2] == 1.0 {
                    *p = [0.0, 0.0, 1.0];
                }
            }
        }
        out
    }

    fn normal(t: &Tri) -> [f32; 3] {
        let (a, b, c) = (point(t[0]), point(t[1]), point(t[2]));
        (b - a).cross(c - a).normalize_or(Vec3::ZERO).as_f32()
    }

    fn binary(header: &[u8], tris: &[Tri], attrs: &[u16]) -> Vec<u8> {
        let mut out = vec![b' '; HEADER];
        out[..header.len()].copy_from_slice(header);
        out.extend((tris.len() as u32).to_le_bytes());
        for (k, t) in tris.iter().enumerate() {
            for p in std::iter::once(normal(t)).chain(t.iter().copied()) {
                for x in p {
                    out.extend(x.to_le_bytes());
                }
            }
            out.extend(attrs.get(k).copied().unwrap_or(0).to_le_bytes());
        }
        out
    }

    fn ascii_of(solids: &[(&str, &[Tri])]) -> Vec<u8> {
        let mut s = String::new();
        for (name, tris) in solids {
            s += &format!("solid {name}\n");
            for t in *tris {
                let n = normal(t);
                s += &format!("  facet normal {:e} {:e} {:e}\n    outer loop\n", n[0], n[1], n[2]);
                for p in t {
                    // `{:e}` writes the shortest string that reads back to
                    // the same f32, so the two encodings hold the same bits.
                    s += &format!("      vertex {:e} {:e} {:e}\n", p[0], p[1], p[2]);
                }
                s += "    endloop\n  endfacet\n";
            }
            s += &format!("endsolid {name}\n");
        }
        s.into_bytes()
    }

    fn area(m: &Mesh) -> f64 {
        m.indices
            .chunks_exact(3)
            .map(|t| {
                let p = |i: u32| m.vertices[i as usize].position();
                (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])).length() / 2.0
            })
            .sum()
    }

    /// The cube's eight corners are eight points; each is three vertices, one
    /// per face it touches, because faces meeting square are a crease; and of
    /// its eighteen edges the twelve cube edges are drawn and the six face
    /// diagonals, which are flat, are not.
    #[test]
    fn a_cube_welds_to_eight_points_and_draws_its_twelve_edges() {
        let stl = parse(&binary(b"cube", &cube(), &[])).unwrap();
        assert!(stl.binary);
        assert_eq!(stl.against, 0);
        let (m, st) = mesh(&stl, "cube");
        assert_eq!((st.corners, st.points), (36, 8));
        assert_eq!(m.vertices.len(), 24);
        assert_eq!(m.indices.len(), 36);
        assert_eq!((st.between, st.flat, st.creases, st.rough), (18, 6, 12, 0));
        assert_eq!((m.stats.edges_sharp, m.stats.edges_tangent), (12, 0));
        assert_eq!(m.edges.len(), 24);
        assert_eq!((st.open, st.overshared, st.flipped, st.dropped), (0, 0, 0, 0));
        assert!(m.parts[0].cappable);
        assert_eq!(m.parts[0].name, "cube");
        assert!((area(&m) - 24.0).abs() < 1e-9);
        // Square creases are not smoothed: every vertex carries its face's
        // normal, which on a cube is an axis.
        for v in &m.vertices {
            let n = v.normal();
            assert!(n.x.abs().max(n.y.abs()).max(n.z.abs()) > 0.9999, "{n:?}");
        }
    }

    /// On a sphere fine enough that neighbours meet under the crease angle
    /// the shading normals are the sphere's own -- the property that makes a
    /// tessellated curve look curved -- and no line is drawn across it.
    #[test]
    fn a_fine_sphere_shades_smooth_and_draws_no_edges() {
        let tris = sphere(24, 48);
        let (m, st) = mesh(&parse(&binary(b"", &tris, &[])).unwrap(), "ball");
        assert_eq!(m.stats.edges_sharp, 0);
        assert_eq!((st.open, st.flipped), (0, 0));
        assert!(m.parts[0].cappable);
        // One vertex per point: nowhere on it is a crease.
        assert_eq!(m.vertices.len(), st.points);
        for v in &m.vertices {
            let (p, n) = (v.position(), v.normal());
            let cos = p.normalize().dot(n);
            assert!(cos > 2.0f64.to_radians().cos(), "{p:?} shaded {n:?}");
        }
    }

    /// On a surface rough at the scale of its facets the creases are its
    /// texture: still there for `--edges all`, but not drawn by default, where
    /// they would bury it. The shading keeps them either way.
    #[test]
    fn a_rough_surface_leaves_its_creases_to_edges_all() {
        let (m, st) = mesh(&parse(&binary(b"", &bumpy(16, 32, 0.4), &[])).unwrap(), "rock");
        assert!(st.creases > 0, "the fixture is not rough enough to test anything");
        assert!((st.flat as f64) < SMOOTH_SHARE * st.between as f64);
        assert_eq!(st.rough, 1);
        assert_eq!((st.open, st.overshared, st.flipped), (0, 0, 0));
        assert_eq!((m.stats.edges_sharp, m.stats.edges_tangent), (0, st.creases));
        let p = &m.parts[0];
        assert_eq!(p.sharp, 0..0);
        assert_eq!(p.tangent.len(), 2 * st.creases);
        assert!(m.vertices.len() > st.points, "a crease left the shading too");
        assert!(p.cappable);
    }

    /// The same facets in either encoding are the same mesh, to the byte.
    #[test]
    fn ascii_and_binary_read_to_the_same_mesh() {
        let tris = sphere(6, 12);
        let (a, _) = mesh(&parse(&binary(b"", &tris, &[])).unwrap(), "s");
        let text = parse(&ascii_of(&[("s", &tris[..])])).unwrap();
        assert!(!text.binary);
        assert_eq!(text.against, 0);
        let (b, _) = mesh(&text, "other");
        assert_eq!(bytemuck::cast_slice::<_, u8>(&a.vertices), bytemuck::cast_slice::<_, u8>(&b.vertices));
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.edges, b.edges);
        assert_eq!(b.parts[0].name, "s");
    }

    /// Each `solid` of an ASCII file is a part of its own, under its own name.
    #[test]
    fn each_ascii_solid_is_a_part() {
        let (c, s) = (cube(), sphere(4, 8));
        let (m, _) = mesh(&parse(&ascii_of(&[("box", &c[..]), ("ball", &s[..])])).unwrap(), "file");
        assert_eq!(m.parts.len(), 2);
        assert_eq!((m.parts[0].name.as_str(), m.parts[1].name.as_str()), ("box", "ball"));
        assert_eq!(m.parts[0].tris.len(), c.len() * 3);
        assert_eq!(m.parts[1].tris, m.parts[0].tris.end..m.indices.len() as u32);
    }

    /// Some writers open a binary header with `solid`. The length says binary
    /// and wins.
    #[test]
    fn a_binary_header_that_says_solid_is_still_binary() {
        let bytes = binary(b"solid part exported as binary", &cube(), &[]);
        assert!(looks_like(&bytes));
        let stl = parse(&bytes).unwrap();
        assert!(stl.binary);
        assert_eq!(stl.solids[0].facets(), 12);
    }

    /// A file cut short is read as far as it goes, and says how much it lost.
    #[test]
    fn a_truncated_binary_file_counts_what_it_lost() {
        let mut bytes = binary(b"", &cube(), &[]);
        bytes.truncate(bytes.len() - FACET * 2 - 7);
        assert!(!looks_like(&bytes));
        let stl = parse(&bytes).unwrap();
        assert_eq!((stl.solids[0].facets(), stl.unread), (9, 3));
    }

    /// A facet that is a line is left out and counted; one that is merely thin
    /// is drawn, but bends nobody's shading.
    #[test]
    fn a_line_is_dropped_and_a_sliver_is_kept() {
        let mut tris = cube();
        let a = tris[0][0];
        tris.push([a, a, tris[0][1]]);
        let (m, st) = mesh(&parse(&binary(b"", &tris, &[])).unwrap(), "x");
        assert_eq!(st.dropped, 1);
        assert_eq!(m.indices.len(), 36);
        assert!(m.parts[0].cappable, "a dropped line opened the cube");

        let sliver = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1e-9, 0.0]];
        let (m, st) = mesh(&parse(&binary(b"", &[sliver], &[])).unwrap(), "x");
        assert_eq!((st.slivers, st.dropped), (1, 0));
        assert_eq!(m.indices.len(), 3);
    }

    /// A lone triangle is a sheet: three open edges, all drawn, and no inside
    /// to cap.
    #[test]
    fn an_open_sheet_draws_its_rim_and_is_not_capped() {
        let t = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let (m, st) = mesh(&parse(&binary(b"", &[t], &[])).unwrap(), "x");
        assert_eq!(st.open, 3);
        assert_eq!(m.stats.edges_sharp, 3);
        assert!(!m.parts[0].cappable);
    }

    /// One facet turned over: the two edges it shares with its own face run
    /// the same way as their neighbours', and the cube is no longer something
    /// a cap can count through.
    #[test]
    fn a_facet_wound_backwards_is_counted() {
        let mut tris = cube();
        tris[0].swap(1, 2);
        let bytes = binary(b"", &tris, &[]);
        let (m, st) = mesh(&parse(&bytes).unwrap(), "x");
        assert_eq!(st.flipped, 3);
        assert!(!m.parts[0].cappable);
    }

    /// The written normal is checked against the winding, not obeyed.
    #[test]
    fn normals_written_against_the_winding_are_counted() {
        let mut bytes = binary(b"", &cube(), &[]);
        for i in 0..4 {
            let o = HEADER + 4 + i * FACET;
            for k in 0..3 {
                let x = f32::from_le_bytes(bytes[o + 4 * k..o + 4 * k + 4].try_into().unwrap());
                bytes[o + 4 * k..o + 4 * k + 4].copy_from_slice(&(-x).to_le_bytes());
            }
        }
        let stl = parse(&bytes).unwrap();
        assert_eq!(stl.against, 4);
        // Nothing drawn changes.
        let (a, _) = mesh(&stl, "x");
        let (b, _) = mesh(&parse(&binary(b"", &cube(), &[])).unwrap(), "x");
        assert_eq!(bytemuck::cast_slice::<_, u8>(&a.vertices), bytemuck::cast_slice::<_, u8>(&b.vertices));
    }

    #[test]
    fn signed_zeros_weld_to_one_point() {
        let (ids, points) = weld(&[[0.0, 1.0, 0.0], [-0.0, 1.0, -0.0]]);
        assert_eq!(ids, [0, 0]);
        assert_eq!(points.len(), 1);
    }

    /// VisCAM: bit 15 set means the facet has a colour, red in the high bits.
    /// The facets left plain take the default grey, and a colour boundary
    /// splits the vertices it runs through.
    #[test]
    fn viscam_colours_set_bit_fifteen() {
        let red = 0x8000 | (31 << 10);
        let (m, _) = mesh(&parse(&binary(b"", &cube(), &[red, red])).unwrap(), "x");
        let reds = m.vertices.iter().filter(|v| v.color_bytes() == [255, 0, 0, 255]).count();
        assert_eq!(reds, 4, "the first face's four corners");
        assert_eq!(m.parts[0].color, DEFAULT_COLOR);
    }

    /// Magics: the header colours the solid, and a zero word everywhere is a
    /// writer that knows nothing of facet colour, not a black part.
    #[test]
    fn magics_colours_the_solid_from_the_header() {
        let (m, _) = mesh(&parse(&binary(b"COLOR=\x00\x80\xff\xff", &cube(), &[])).unwrap(), "x");
        assert!(m.vertices.iter().all(|v| v.color_bytes() == [0, 128, 255, 255]));
        assert_eq!(m.parts[0].color, [0.0, 128.0 / 255.0, 1.0]);

        // A cleared bit 15 is a colour of the facet's own, red low.
        let blue = 31 << 10;
        let mut attrs = vec![0x8000u16; 12];
        attrs[0] = blue;
        let (m, _) = mesh(&parse(&binary(b"COLOR=\x00\x80\xff\xff", &cube(), &attrs)).unwrap(), "x");
        assert!(m.vertices.iter().any(|v| v.color_bytes() == [0, 0, 255, 255]));
    }

    #[test]
    fn ascii_is_read_forgivingly() {
        let text = b"SOLID  Bracket \r\n\
            FACET NORMAL 0 0 1\r\n OUTER LOOP\r\n VERTEX 0 0 0\r\n VERTEX 1 0 0\r\n\
            VERTEX 1 1 0\r\n VERTEX 0 1 0\r\n ENDLOOP\r\nENDFACET\r\n\
            facet normal 0 0 1\n outer loop\n vertex 0 0 0\n vertex x 0 0\n vertex 1 1 0\n endloop\n endfacet\n\
            facet normal 0 0 1\n outer loop\n vertex 0 0 0\n vertex 1 0 0\n";
        let stl = parse(text).unwrap();
        assert_eq!(stl.solids.len(), 1);
        assert_eq!(stl.solids[0].name, "Bracket");
        // The quad fanned into two; the unreadable vertex and the loop the file
        // ended inside are counted.
        assert_eq!((stl.solids[0].facets(), stl.unread), (2, 2));
    }

    #[test]
    fn nonsense_is_not_stl() {
        assert!(!looks_like(b"ISO-10303-21;\nHEADER;"));
        assert!(parse(b"hello").is_err());
    }
}
