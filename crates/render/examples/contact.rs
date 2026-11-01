//! Renders a contact sheet: every model under a directory as a labelled tile,
//! so a library of thousands of parts can be looked through at a glance.
//!
//! ```text
//! cargo run --release -p stpview --example contact -- <root> [out-dir] [options]
//! ```
//!
//! One GPU context serves a batch of parts: creating a device with this frame
//! graph's pipelines and environment costs far more than importing a small
//! component. So a batch is one `Scene` holding many meshes, and each tile is
//! one capture of it with `RenderOptions::visible` cut down to a single part.
//!
//! Hiding all but one part, rather than laying parts out in world space under
//! one wide shot, gives every tile its own camera fitted to its own bounding
//! box, so small and large parts are drawn at the same size in the same pose.
//!
//! The ground plane is off: its one shadow map spans the whole scene, so a
//! small part batched with a large one would get a handful of texels, and a
//! catalogue wants a clean field anyway.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use stpview::geom::Xform;
use stpview::tess::Mesh;
use stpview::{Assets, Camera, MeshHandle, RenderOptions, Renderer, Scene};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Layout, in pixels.
struct Layout {
    /// Side of the rendered square for one part.
    tile: u32,
    /// Tiles across and down on one sheet.
    cols: u32,
    rows: u32,
    /// Strip under each tile that carries the file name.
    label: u32,
    /// Strip along the top that carries the library name.
    header: u32,
    /// Sheet background between cells, so a wide part does not read as
    /// continuing into its neighbour.
    gutter: u32,
}

impl Layout {
    fn cell_w(&self) -> u32 {
        self.tile + self.gutter
    }

    fn cell_h(&self) -> u32 {
        self.tile + self.label + self.gutter
    }

    fn per_sheet(&self) -> usize {
        (self.cols * self.rows) as usize
    }

    fn width(&self) -> u32 {
        self.cols * self.cell_w() + self.gutter
    }

    /// Height for a sheet holding `n` tiles: only the rows actually used, so
    /// the last sheet of a library is not mostly empty.
    fn height(&self, n: usize) -> u32 {
        let rows = (n as u32).div_ceil(self.cols).clamp(1, self.rows);
        self.header + rows * self.cell_h()
    }
}

struct Args {
    root: PathBuf,
    out: PathBuf,
    layout: Layout,
    /// How many meshes share one GPU context. Bigger amortises the device
    /// setup further and costs more resident mesh.
    batch: usize,
    tol: Option<f64>,
    /// Stop after this many files, for a quick look at the output.
    limit: Option<usize>,
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out).map_err(|e| format!("{}: {e}", args.out.display()))?;
    let font = Font::new();

    let groups = collect(&args.root, args.limit)?;
    if groups.is_empty() {
        return Err(format!("{}: no .step files under here", args.root.display()));
    }
    let total: usize = groups.values().map(|v| v.len()).sum();
    println!(
        "contact {} files in {} libraries -> {}",
        total,
        groups.len(),
        args.out.display()
    );

    let started = std::time::Instant::now();
    let mut sheets: Vec<Sheet> = Vec::new();
    let mut done = 0usize;
    let mut failed = 0usize;
    for (name, files) in &groups {
        let pages = files.len().div_ceil(args.layout.per_sheet());
        for (page, chunk) in files.chunks(args.layout.per_sheet()).enumerate() {
            let stem = if pages == 1 {
                name.clone()
            } else {
                format!("{name}-{:02}", page + 1)
            };
            let path = args.out.join(format!("{stem}.png"));
            let title = if pages == 1 {
                format!("{name}   {} parts", chunk.len())
            } else {
                format!(
                    "{name}   sheet {} of {pages}   {} parts",
                    page + 1,
                    chunk.len()
                )
            };
            let t0 = std::time::Instant::now();
            let bad = sheet(&args, &font, &title, chunk, &path)?;
            done += chunk.len();
            failed += bad;
            sheets.push(Sheet {
                title,
                path,
                parts: chunk.len(),
                failed: bad,
            });
            println!(
                "sheet  {} ({} parts, {bad} failed) in {:.1}s -- {done}/{total}",
                stem,
                chunk.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }

    index(&args.out, &sheets)?;
    println!(
        "done   {} sheets, {} parts drawn, {failed} could not be imported, in {:.0}s",
        sheets.len(),
        done - failed,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

struct Sheet {
    title: String,
    path: PathBuf,
    parts: usize,
    failed: usize,
}

/// Renders one sheet and writes it. Returns how many parts failed to import.
fn sheet(
    args: &Args,
    font: &Font,
    title: &str,
    files: &[PathBuf],
    out: &Path,
) -> Result<usize, String> {
    let lay = &args.layout;
    let (w, h) = (lay.width(), lay.height(files.len()));
    let mut buf = vec![0u8; (w * h * 4) as usize];
    // The renderer's own clear colour, so a tile has no edge against the sheet.
    for px in buf.chunks_mut(4) {
        px.copy_from_slice(&[13, 14, 16, 255]);
    }
    fill(&mut buf, w, 0, 0, w, lay.header, [24, 26, 30]);
    font.text(&mut buf, w, 8, (lay.header as i32 - 7) / 2, title, [210, 214, 224]);

    let mut failed = 0usize;
    // Import and render in batches: one wgpu device per batch, not per part.
    for (b, group) in files.chunks(args.batch).enumerate() {
        let base = b * args.batch;
        let assets = match args.tol {
            Some(mm) => Assets::new().with_tolerance_mm(mm),
            None => Assets::new(),
        };
        let imported = assets.import_all(group);

        // Every imported mesh at its own origin. They overlap, which does not
        // matter: only one is ever drawn at a time.
        let mut scene = Scene::new();
        let mut meshes: Vec<(usize, MeshHandle)> = Vec::new();
        for (k, r) in imported.into_iter().enumerate() {
            match r {
                Ok(m) => {
                    scene.add(m.clone(), Xform::IDENTITY);
                    meshes.push((base + k, m));
                }
                Err(e) => {
                    failed += 1;
                    eprintln!("  skip {e}");
                    cell(&mut buf, w, lay, base + k, None, &files[base + k], font);
                }
            }
        }
        if meshes.is_empty() {
            continue;
        }

        // Where each mesh's parts land in the concatenated buffer the scene
        // builds, which is what `RenderOptions::visible` indexes.
        let mut ranges = Vec::with_capacity(meshes.len());
        let mut at = 0usize;
        for (_, m) in &meshes {
            let n = part_count(m);
            ranges.push(at..at + n);
            at += n;
        }
        let total_parts = at;
        if total_parts == 0 {
            continue;
        }

        let mut r = Renderer::offscreen_scene(lay.tile, lay.tile, &scene)?;
        let mut opts = RenderOptions::default();
        // No ground plane; see the module note.
        opts.ground = false;

        for (j, (idx, m)) in meshes.iter().enumerate() {
            let mut vis = vec![false; total_parts];
            for p in ranges[j].clone() {
                vis[p] = true;
            }
            opts.visible = vis;

            // Framed on this part alone, using the scene's own notion of where
            // a placed mesh is.
            let mut one = Scene::new();
            one.add(m.clone(), Xform::IDENTITY);
            let bbox = one.bounds();
            let mut camera = Camera::new(&bbox);
            camera.close_in(&bbox);

            let rgba = r.capture(&camera, &opts)?;
            cell(&mut buf, w, lay, *idx, Some(&rgba), &files[*idx], font);
        }
    }

    write_png(out, w, h, &buf)?;
    Ok(failed)
}

/// Blits one rendered tile into the sheet and writes its label under it.
/// `rgba` of `None` means the file could not be imported: the cell is left
/// empty and its label is drawn in red.
fn cell(
    buf: &mut [u8],
    w: u32,
    lay: &Layout,
    idx: usize,
    rgba: Option<&[u8]>,
    file: &Path,
    font: &Font,
) {
    let i = idx as u32 % (lay.cols * lay.rows);
    let (col, row) = (i % lay.cols, i / lay.cols);
    let x0 = lay.gutter + col * lay.cell_w();
    let y0 = lay.header + lay.gutter + row * lay.cell_h();

    if let Some(src) = rgba {
        for y in 0..lay.tile {
            let d = ((y0 + y) * w + x0) as usize * 4;
            let s = (y * lay.tile) as usize * 4;
            let n = lay.tile as usize * 4;
            buf[d..d + n].copy_from_slice(&src[s..s + n]);
        }
    }
    // The label strip, a shade off the tile so the rows separate.
    fill(buf, w, x0, y0 + lay.tile, lay.tile, lay.label, [22, 24, 28]);

    let name = file.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
    let color = if rgba.is_some() {
        [176, 182, 194]
    } else {
        [214, 96, 96]
    };
    let per_line = ((lay.tile - 6) / Font::ADVANCE) as usize;
    for (k, line) in wrap(name, per_line, 3).iter().enumerate() {
        font.text(
            buf,
            w,
            (x0 + 3) as i32,
            (y0 + lay.tile + 2 + k as u32 * 8) as i32,
            line,
            color,
        );
    }
}

/// How many parts a mesh contributes to a scene's concatenated part list.
///
/// The instanced renderer adds a covering part for a mesh with triangles but
/// no assembly structure, and `visible` is indexed against the result; this
/// must agree or a tile shows the wrong part.
fn part_count(m: &Mesh) -> usize {
    if m.parts.is_empty() && !m.vertices.is_empty() {
        1
    } else {
        m.parts.len()
    }
}

/// Breaks a name across at most `max_lines` lines of `per_line` characters,
/// preferring underscores and hyphens. What does not fit is cut and marked.
fn wrap(s: &str, per_line: usize, max_lines: usize) -> Vec<String> {
    if per_line == 0 {
        return Vec::new();
    }
    let mut out: Vec<String> = Vec::new();
    let mut rest = s;
    while !rest.is_empty() && out.len() < max_lines {
        if rest.chars().count() <= per_line {
            out.push(rest.to_string());
            rest = "";
            break;
        }
        // Break at the last separator within the width, if there is one.
        let head: String = rest.chars().take(per_line + 1).collect();
        let cut = head
            .rfind(['_', '-'])
            .filter(|&p| p > per_line / 3)
            .map(|p| p + 1)
            .unwrap_or(per_line);
        let (a, b) = split_at_chars(rest, cut.min(rest.chars().count()));
        out.push(a.to_string());
        rest = b;
    }
    // Mark a truncation: two names can differ only in their suffix.
    if !rest.is_empty()
        && let Some(last) = out.last_mut()
    {
        let keep = per_line.saturating_sub(1).min(last.chars().count());
        let (a, _) = split_at_chars(last, keep);
        *last = format!("{a}~");
    }
    out
}

fn split_at_chars(s: &str, n: usize) -> (&str, &str) {
    let at = s.char_indices().nth(n).map(|(i, _)| i).unwrap_or(s.len());
    s.split_at(at)
}

fn fill(buf: &mut [u8], w: u32, x: u32, y: u32, dw: u32, dh: u32, c: [u8; 3]) {
    for yy in y..y + dh {
        for xx in x..x + dw {
            let i = (yy * w + xx) as usize * 4;
            if i + 3 < buf.len() {
                buf[i..i + 4].copy_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
    }
}

/// Every .step under `root`, grouped by the directory holding it, so one
/// library directory is one run of sheets.
fn collect(root: &Path, limit: Option<usize>) -> Result<BTreeMap<String, Vec<PathBuf>>, String> {
    let mut groups: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let mut here: Vec<PathBuf> = Vec::new();
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("step") || s.eq_ignore_ascii_case("stp"))
            {
                here.push(p);
            }
        }
        if here.is_empty() {
            continue;
        }
        let name = dir
            .strip_prefix(root)
            .ok()
            .filter(|p| !p.as_os_str().is_empty())
            .or_else(|| dir.file_name().map(Path::new))
            .map(|p| p.to_string_lossy().replace(['/', '\\'], "-"))
            .unwrap_or_else(|| "models".into());
        groups.entry(name).or_default().extend(here);
    }
    for v in groups.values_mut() {
        v.sort();
    }
    if let Some(n) = limit {
        let mut kept = 0usize;
        groups.retain(|_, v| {
            if kept >= n {
                return false;
            }
            v.truncate(n - kept);
            kept += v.len();
            true
        });
    }
    Ok(groups)
}

/// An HTML index beside the sheets, for browsing many large PNGs.
fn index(out: &Path, sheets: &[Sheet]) -> Result<(), String> {
    let mut h = String::from(
        "<!doctype html>\n<meta charset=\"utf-8\">\n<title>contact sheets</title>\n\
         <style>\n\
         body{background:#0d0e10;color:#cdd2dc;font:14px system-ui,sans-serif;margin:24px}\n\
         h1{font-size:18px;font-weight:600}\n\
         a{color:#8ab4f8;text-decoration:none}a:hover{text-decoration:underline}\n\
         ul{list-style:none;padding:0;columns:2;column-gap:32px}\n\
         li{margin:3px 0;break-inside:avoid}\n\
         .n{color:#71767f}.bad{color:#d66060}\n\
         </style>\n<h1>contact sheets</h1>\n<ul>\n",
    );
    for s in sheets {
        let file = s
            .path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        let bad = if s.failed > 0 {
            format!(" <span class=\"bad\">{} failed</span>", s.failed)
        } else {
            String::new()
        };
        h.push_str(&format!(
            "<li><a href=\"{file}\">{}</a> <span class=\"n\">{} parts</span>{bad}</li>\n",
            s.title, s.parts
        ));
    }
    h.push_str("</ul>\n");
    let p = out.join("index.html");
    std::fs::write(&p, h).map_err(|e| format!("{}: {e}", p.display()))
}

fn write_png(path: &Path, w: u32, h: u32, rgba: &[u8]) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .and_then(|mut w| w.write_image_data(rgba))
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn parse_args() -> Result<Args, String> {
    const USAGE: &str = "usage: contact <root> [out-dir] \
        [--tile N] [--cols N] [--rows N] [--batch N] [--tol MM] [--limit N]";
    let mut positional: Vec<String> = Vec::new();
    let mut tile = 224u32;
    let mut cols = 7u32;
    let mut rows = 8u32;
    let mut batch = 64usize;
    let mut tol = None;
    let mut limit = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = |name: &str| -> Result<String, String> {
            it.next().ok_or_else(|| format!("{name} needs a value"))
        };
        match a.as_str() {
            "--tile" => tile = val("--tile")?.parse().map_err(|_| "bad --tile")?,
            "--cols" => cols = val("--cols")?.parse().map_err(|_| "bad --cols")?,
            "--rows" => rows = val("--rows")?.parse().map_err(|_| "bad --rows")?,
            "--batch" => batch = val("--batch")?.parse().map_err(|_| "bad --batch")?,
            "--tol" => tol = Some(val("--tol")?.parse().map_err(|_| "bad --tol")?),
            "--limit" => limit = Some(val("--limit")?.parse().map_err(|_| "bad --limit")?),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            _ => positional.push(a),
        }
    }
    let root = PathBuf::from(positional.first().ok_or(USAGE)?);
    let out = positional
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("contact"));
    Ok(Args {
        root,
        out,
        layout: Layout {
            tile,
            cols: cols.max(1),
            rows: rows.max(1),
            label: 28,
            gutter: 3,
            header: 26,
        },
        batch: batch.max(1),
        tol,
        limit,
    })
}

// ---------------------------------------------------------------------------
// A 5x7 bitmap font, so a tile can carry its file name.
//
// Each line is the character followed by its seven rows, top to bottom, drawn
// as art so a wrong bit is visible in the source.
// ---------------------------------------------------------------------------

struct Font {
    glyphs: std::collections::HashMap<char, [u8; 7]>,
}

impl Font {
    /// Five pixels of glyph and one of gap.
    const ADVANCE: u32 = 6;

    fn new() -> Font {
        let mut glyphs = std::collections::HashMap::new();
        for line in GLYPHS {
            let c = line.chars().next().expect("glyph line");
            let mut rows = [0u8; 7];
            for (r, art) in line[2..].split_whitespace().enumerate().take(7) {
                let mut bits = 0u8;
                for (b, ch) in art.chars().enumerate().take(5) {
                    if ch == '#' {
                        bits |= 1 << (4 - b);
                    }
                }
                rows[r] = bits;
            }
            glyphs.insert(c, rows);
        }
        Font { glyphs }
    }

    fn text(&self, buf: &mut [u8], w: u32, x: i32, y: i32, s: &str, color: [u8; 3]) {
        let mut cx = x;
        for ch in s.chars() {
            // Missing glyphs draw as '?', not as a gap.
            let g = self
                .glyphs
                .get(&ch)
                .or_else(|| self.glyphs.get(&'?'))
                .copied()
                .unwrap_or([0; 7]);
            for (r, bits) in g.iter().enumerate() {
                for b in 0..5u32 {
                    if bits & (1 << (4 - b)) == 0 {
                        continue;
                    }
                    let (px, py) = (cx + b as i32, y + r as i32);
                    if px < 0 || py < 0 || px >= w as i32 {
                        continue;
                    }
                    let i = (py as usize * w as usize + px as usize) * 4;
                    if i + 3 < buf.len() {
                        buf[i..i + 4].copy_from_slice(&[color[0], color[1], color[2], 255]);
                    }
                }
            }
            cx += Font::ADVANCE as i32;
        }
    }
}

#[rustfmt::skip]
const GLYPHS: &[&str] = &[
"  ..... ..... ..... ..... ..... ..... .....",
"! ..#.. ..#.. ..#.. ..#.. ..#.. ..... ..#..",
"\" .#.#. .#.#. ..... ..... ..... ..... .....",
"# .#.#. .#.#. ##### .#.#. ##### .#.#. .#.#.",
"$ ..#.. .#### #.#.. .###. ..#.# ####. ..#..",
"% ##..# ##..# ...#. ..#.. .#... #..## #..##",
"& .##.. #..#. #.#.. .#... #.#.# #..#. .##.#",
"' ..#.. ..#.. ..... ..... ..... ..... .....",
"( ...#. ..#.. .#... .#... .#... ..#.. ...#.",
") .#... ..#.. ...#. ...#. ...#. ..#.. .#...",
"* ..... #.#.# .###. ##### .###. #.#.# .....",
"+ ..... ..#.. ..#.. ##### ..#.. ..#.. .....",
", ..... ..... ..... ..... ..##. ..#.. .#...",
"- ..... ..... ..... ##### ..... ..... .....",
". ..... ..... ..... ..... ..... .##.. .##..",
"/ ....# ....# ...#. ..#.. .#... #.... #....",
"0 .###. #...# #..## #.#.# ##..# #...# .###.",
"1 ..#.. .##.. ..#.. ..#.. ..#.. ..#.. .###.",
"2 .###. #...# ....# ...#. ..#.. .#... #####",
"3 ##### ...#. ..#.. ...#. ....# #...# .###.",
"4 ...#. ..##. .#.#. #..#. ##### ...#. ...#.",
"5 ##### #.... ####. ....# ....# #...# .###.",
"6 ..##. .#... #.... ####. #...# #...# .###.",
"7 ##### ....# ...#. ..#.. .#... .#... .#...",
"8 .###. #...# #...# .###. #...# #...# .###.",
"9 .###. #...# #...# .#### ....# ...#. .##..",
": ..... .##.. .##.. ..... .##.. .##.. .....",
"; ..... .##.. .##.. ..... .##.. ..#.. .#...",
"< ...#. ..#.. .#... #.... .#... ..#.. ...#.",
"= ..... ..... ##### ..... ##### ..... .....",
"> .#... ..#.. ...#. ....# ...#. ..#.. .#...",
"? .###. #...# ....# ...#. ..#.. ..... ..#..",
"@ .###. #...# ....# #.### #.#.# #.#.# .####",
"A ..#.. .#.#. #...# #...# ##### #...# #...#",
"B ####. #...# #...# ####. #...# #...# ####.",
"C .###. #...# #.... #.... #.... #...# .###.",
"D ###.. #..#. #...# #...# #...# #..#. ###..",
"E ##### #.... #.... ####. #.... #.... #####",
"F ##### #.... #.... ####. #.... #.... #....",
"G .###. #...# #.... #.### #...# #...# .####",
"H #...# #...# #...# ##### #...# #...# #...#",
"I .###. ..#.. ..#.. ..#.. ..#.. ..#.. .###.",
"J ...## ....# ....# ....# #...# #...# .###.",
"K #...# #..#. #.#.. ##... #.#.. #..#. #...#",
"L #.... #.... #.... #.... #.... #.... #####",
"M #...# ##.## #.#.# #.#.# #...# #...# #...#",
"N #...# #...# ##..# #.#.# #..## #...# #...#",
"O .###. #...# #...# #...# #...# #...# .###.",
"P ####. #...# #...# ####. #.... #.... #....",
"Q .###. #...# #...# #...# #.#.# #..#. .##.#",
"R ####. #...# #...# ####. #.#.. #..#. #...#",
"S .#### #.... #.... .###. ....# ....# ####.",
"T ##### ..#.. ..#.. ..#.. ..#.. ..#.. ..#..",
"U #...# #...# #...# #...# #...# #...# .###.",
"V #...# #...# #...# #...# #...# .#.#. ..#..",
"W #...# #...# #...# #.#.# #.#.# ##.## #...#",
"X #...# #...# .#.#. ..#.. .#.#. #...# #...#",
"Y #...# #...# .#.#. ..#.. ..#.. ..#.. ..#..",
"Z ##### ....# ...#. ..#.. .#... #.... #####",
"[ .###. .#... .#... .#... .#... .#... .###.",
"\\ #.... #.... .#... ..#.. ...#. ....# ....#",
"] .###. ...#. ...#. ...#. ...#. ...#. .###.",
"^ ..#.. .#.#. #...# ..... ..... ..... .....",
"_ ..... ..... ..... ..... ..... ..... #####",
"` .#... ..#.. ..... ..... ..... ..... .....",
"a ..... ..... .###. ....# .#### #...# .####",
"b #.... #.... ####. #...# #...# #...# ####.",
"c ..... ..... .###. #.... #.... #...# .###.",
"d ....# ....# .#### #...# #...# #...# .####",
"e ..... ..... .###. #...# ##### #.... .###.",
"f ..##. .#..# .#... ####. .#... .#... .#...",
"g ..... .#### #...# #...# .#### ....# .###.",
"h #.... #.... ####. #...# #...# #...# #...#",
"i ..#.. ..... .##.. ..#.. ..#.. ..#.. .###.",
"j ...#. ..... ..##. ...#. ...#. #..#. .##..",
"k #.... #.... #..#. #.#.. ##... #.#.. #..#.",
"l .##.. ..#.. ..#.. ..#.. ..#.. ..#.. .###.",
"m ..... ..... ##.#. #.#.# #.#.# #.#.# #.#.#",
"n ..... ..... ####. #...# #...# #...# #...#",
"o ..... ..... .###. #...# #...# #...# .###.",
"p ..... ..... ####. #...# #...# ####. #....",
"q ..... ..... .#### #...# #...# .#### ....#",
"r ..... ..... #.##. ##..# #.... #.... #....",
"s ..... ..... .#### #.... .###. ....# ####.",
"t .#... .#... ####. .#... .#... .#..# ..##.",
"u ..... ..... #...# #...# #...# #..## .##.#",
"v ..... ..... #...# #...# #...# .#.#. ..#..",
"w ..... ..... #...# #.#.# #.#.# #.#.# .#.#.",
"x ..... ..... #...# .#.#. ..#.. .#.#. #...#",
"y ..... ..... #...# #...# .#### ....# .###.",
"z ..... ..... ##### ...#. ..#.. .#... #####",
"{ ...#. ..#.. ..#.. .#... ..#.. ..#.. ...#.",
"| ..#.. ..#.. ..#.. ..#.. ..#.. ..#.. ..#..",
"} .#... ..#.. ..#.. ...#. ..#.. ..#.. .#...",
"~ ..... ..#.# .#.#. #.#.. ..... ..... .....",
];
