//! Renders a PCB-style composite scene: a board with component models placed
//! on it, each STEP file imported once however many times it is placed.
//!
//! The library API end to end -- import, compose, render, pixels -- with no
//! CLI and no window:
//!
//! ```text
//! cargo run --release -p stpview --example pcb -- <dir-with-step-files> [out.png]
//! ```
//!
//! Takes the first few .step files in the folder, lines them up on a plain
//! board, and writes a PNG.

use std::path::PathBuf;

use stpview::geom::Xform;
use stpview::math::v3;
use stpview::{Assets, Camera, RenderOptions, Renderer, Scene};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .ok_or("usage: pcb <dir-with-step-files> [out.png]")?,
    );
    let out = args.next().unwrap_or_else(|| "pcb.png".into());

    // A few distinct components, sorted so the picture is reproducible.
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("step") || s.eq_ignore_ascii_case("stp"))
        })
        .collect();
    files.sort();
    files.truncate(6);
    if files.is_empty() {
        return Err(format!("{}: no .step files here", dir.display()));
    }

    // Import in parallel; the cache tessellates each file once however often
    // it is placed.
    let assets = Assets::new();
    let meshes: Vec<_> = assets
        .import_all(&files)
        .into_iter()
        .zip(&files)
        .filter_map(|(r, p)| match r {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("skipping {}: {e}", p.display());
                None
            }
        })
        .collect();
    if meshes.is_empty() {
        return Err("nothing imported".into());
    }

    // The board is not a STEP file: a Mesh from anywhere composes with
    // imported ones. An 80x50x1.6 mm slab, FR4 green.
    let board = stpview::MeshHandle::new(slab(80.0, 50.0, 1.6, [0.05, 0.35, 0.16]));

    let mut scene = Scene::new();
    scene.add(board, Xform::IDENTITY);

    // Components in rows on the board, ignoring footprint sizes.
    let cols = 3usize;
    for (i, m) in meshes.iter().enumerate() {
        let (col, row) = (i % cols, i / cols);
        let at = v3(
            14.0 + col as f64 * 26.0,
            13.0 + row as f64 * 24.0,
            1.6,
        );
        // Carry the part's bbox centre (and base) onto its spot, whatever its
        // own origin.
        let b = m.bbox;
        let c = v3(
            (b.min.x + b.max.x) * 0.5 * m.mm_per_unit,
            (b.min.y + b.max.y) * 0.5 * m.mm_per_unit,
            b.min.z * m.mm_per_unit,
        );
        let place = Xform::translate(at - c).compose(&Xform::rotate_axis(
            2,
            (i as f64) * 30f64.to_radians(),
        ));
        scene.add(m.clone(), place);
    }

    let camera = Camera::new(&scene.bounds());
    let mut r = Renderer::offscreen_scene(1200, 900, &scene)?;
    let rgba = r.capture(&camera, &RenderOptions::default())?;

    write_png(&out, 1200, 900, &rgba)?;
    println!(
        "{}: {} components on the board, {} unique meshes",
        out,
        meshes.len(),
        assets.cached()
    );
    Ok(())
}

/// A box mesh the hard way: two triangles per face, normals out.
fn slab(w: f64, d: f64, h: f64, color: [f32; 3]) -> stpview::tess::Mesh {
    use stpview::tess::{Mesh, Vertex};
    let mut m = Mesh {
        mm_per_unit: 1.0,
        ..Default::default()
    };
    let corners = [
        v3(0.0, 0.0, 0.0),
        v3(w, 0.0, 0.0),
        v3(w, d, 0.0),
        v3(0.0, d, 0.0),
        v3(0.0, 0.0, h),
        v3(w, 0.0, h),
        v3(w, d, h),
        v3(0.0, d, h),
    ];
    // Each face gets its own four vertices so the normals stay flat.
    let faces: [([usize; 4], [f64; 3]); 6] = [
        ([3, 2, 1, 0], [0.0, 0.0, -1.0]),
        ([4, 5, 6, 7], [0.0, 0.0, 1.0]),
        ([0, 1, 5, 4], [0.0, -1.0, 0.0]),
        ([2, 3, 7, 6], [0.0, 1.0, 0.0]),
        ([1, 2, 6, 5], [1.0, 0.0, 0.0]),
        ([3, 0, 4, 7], [-1.0, 0.0, 0.0]),
    ];
    for (quad, n) in faces {
        let base = m.vertices.len() as u32;
        for k in quad {
            m.vertices.push(Vertex::new(
                corners[k],
                v3(n[0], n[1], n[2]),
                color,
                1.0,
            ));
        }
        m.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    for c in corners {
        m.bbox.add_point(c);
    }
    m
}

/// Minimal PNG out, so the example depends on nothing the library does not.
fn write_png(path: &str, w: u32, h: u32, rgba: &[u8]) -> Result<(), String> {
    // Uncompressed PNG: stored deflate blocks, no image dependency.
    fn crc(data: &[u8]) -> u32 {
        let mut table = [0u32; 256];
        for (n, t) in table.iter_mut().enumerate() {
            let mut c = n as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *t = c;
        }
        let mut c = 0xFFFF_FFFFu32;
        for &b in data {
            c = table[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
        }
        c ^ 0xFFFF_FFFF
    }
    fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
        out.extend((data.len() as u32).to_be_bytes());
        out.extend(kind);
        out.extend(data);
        let mut c = Vec::with_capacity(4 + data.len());
        c.extend(kind);
        c.extend(data);
        out.extend(crc(&c).to_be_bytes());
    }

    let mut raw = Vec::with_capacity((w as usize * 4 + 1) * h as usize);
    for row in rgba.chunks(w as usize * 4) {
        raw.push(0); // filter: none
        raw.extend(row);
    }
    // zlib stream of stored blocks.
    let mut z = vec![0x78, 0x01];
    let mut adler = (1u32, 0u32);
    for b in &raw {
        adler.0 = (adler.0 + *b as u32) % 65521;
        adler.1 = (adler.1 + adler.0) % 65521;
    }
    for (i, block) in raw.chunks(65535).enumerate() {
        let last = (i + 1) * 65535 >= raw.len();
        z.push(if last { 1 } else { 0 });
        z.extend((block.len() as u16).to_le_bytes());
        z.extend((!(block.len() as u16)).to_le_bytes());
        z.extend(block);
    }
    z.extend(((adler.1 << 16) | adler.0).to_be_bytes());

    let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend(w.to_be_bytes());
    ihdr.extend(h.to_be_bytes());
    ihdr.extend([8, 6, 0, 0, 0]); // 8-bit RGBA
    chunk(&mut png, b"IHDR", &ihdr);
    chunk(&mut png, b"IDAT", &z);
    chunk(&mut png, b"IEND", &[]);
    std::fs::write(path, png).map_err(|e| format!("{path}: {e}"))
}

