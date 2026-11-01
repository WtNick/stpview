//! Draws the application and document icons, and writes them as `.ico` files.
//!
//! ```text
//! cargo run --release -p stpview-app --example icon
//! ```
//!
//! The icons are committed, so this does not run in a build. Drawing them from
//! shapes keeps the two consistent: the document icon carries the application
//! icon's mark on a page.
//!
//! They are drawn rather than rendered: an icon must survive 16x16, where a
//! downsampled render is mush, and be transparent outside the shape. The mark
//! is a hand-placed isometric cube in the renderer's palette.

use std::path::{Path, PathBuf};

/// Supersampling factor; the only anti-aliasing, as shapes are point-sampled.
const SS: u32 = 8;

type Rgba = [u8; 4];

const EDGE: Rgba = [0x22, 0x2d, 0x3a, 0xff];
const FACE_TOP: Rgba = [0xdd, 0xe5, 0xef, 0xff];
const FACE_LEFT: Rgba = [0xa4, 0xb5, 0xca, 0xff];
const FACE_RIGHT: Rgba = [0x74, 0x88, 0xa3, 0xff];

const PAGE: Rgba = [0xfc, 0xfd, 0xfe, 0xff];
const PAGE_EDGE: Rgba = [0x8d, 0x9c, 0xb0, 0xff];
/// The turned corner: the only warm colour, so a STEP file stands out in a
/// list, and on the fold where it cannot be mistaken for the model.
const FOLD: Rgba = [0xe2, 0x9a, 0x45, 0xff];
const FOLD_EDGE: Rgba = [0xb4, 0x75, 0x27, 0xff];

/// A canvas of supersamples, each unpainted or opaque. On downsampling, alpha
/// is the painted fraction and colour the mean of painted samples only, which
/// avoids dark fringes.
struct Canvas {
    w: u32,
    h: u32,
    px: Vec<Option<Rgba>>,
}

impl Canvas {
    fn new(w: u32, h: u32) -> Canvas {
        Canvas {
            w,
            h,
            px: vec![None; (w * h) as usize],
        }
    }

    /// Fills a polygon, even-odd, over its own bounding box.
    fn poly(&mut self, pts: &[(f64, f64)], color: Rgba) {
        if pts.len() < 3 {
            return;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for &(x, y) in pts {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        let lo_x = (x0.floor().max(0.0)) as u32;
        let lo_y = (y0.floor().max(0.0)) as u32;
        let hi_x = (x1.ceil().min(self.w as f64)) as u32;
        let hi_y = (y1.ceil().min(self.h as f64)) as u32;

        for y in lo_y..hi_y {
            for x in lo_x..hi_x {
                // Sample at the cell centre, so an edge on an integer does
                // not depend on rounding.
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                if inside(pts, px, py) {
                    self.px[(y * self.w + x) as usize] = Some(color);
                }
            }
        }
    }

    /// A stroked segment, as a quad with a disc at each end; the discs close
    /// the joints of an outline without mitres.
    fn line(&mut self, a: (f64, f64), b: (f64, f64), width: f64, color: Rgba) {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-9 {
            return;
        }
        let (nx, ny) = (-dy / len * width * 0.5, dx / len * width * 0.5);
        self.poly(
            &[
                (a.0 + nx, a.1 + ny),
                (b.0 + nx, b.1 + ny),
                (b.0 - nx, b.1 - ny),
                (a.0 - nx, a.1 - ny),
            ],
            color,
        );
        self.disc(a, width * 0.5, color);
        self.disc(b, width * 0.5, color);
    }

    fn disc(&mut self, c: (f64, f64), r: f64, color: Rgba) {
        let lo_x = ((c.0 - r).floor().max(0.0)) as u32;
        let lo_y = ((c.1 - r).floor().max(0.0)) as u32;
        let hi_x = ((c.0 + r).ceil().min(self.w as f64)) as u32;
        let hi_y = ((c.1 + r).ceil().min(self.h as f64)) as u32;
        for y in lo_y..hi_y {
            for x in lo_x..hi_x {
                let (px, py) = (x as f64 + 0.5 - c.0, y as f64 + 0.5 - c.1);
                if px * px + py * py <= r * r {
                    self.px[(y * self.w + x) as usize] = Some(color);
                }
            }
        }
    }

    /// Box-filters the supersamples down to one pixel each, straight (not
    /// premultiplied) RGBA, which is what both the ICO and the PNG want.
    fn resolve(&self, out_w: u32, out_h: u32) -> Vec<u8> {
        let (sx, sy) = (self.w / out_w, self.h / out_h);
        let mut out = vec![0u8; (out_w * out_h * 4) as usize];
        for y in 0..out_h {
            for x in 0..out_w {
                let (mut r, mut g, mut b, mut hits) = (0u32, 0u32, 0u32, 0u32);
                for j in 0..sy {
                    for i in 0..sx {
                        let s = (y * sy + j) * self.w + (x * sx + i);
                        if let Some(c) = self.px[s as usize] {
                            r += c[0] as u32;
                            g += c[1] as u32;
                            b += c[2] as u32;
                            hits += 1;
                        }
                    }
                }
                let total = sx * sy;
                let o = ((y * out_w + x) * 4) as usize;
                // Untouched cells stay fully transparent.
                if let Some(n) = (hits > 0).then_some(hits) {
                    out[o] = (r / n) as u8;
                    out[o + 1] = (g / n) as u8;
                    out[o + 2] = (b / n) as u8;
                    out[o + 3] = (n * 255 / total) as u8;
                }
            }
        }
        out
    }
}

/// Even-odd point in polygon.
fn inside(pts: &[(f64, f64)], x: f64, y: f64) -> bool {
    let mut hit = false;
    let mut j = pts.len() - 1;
    for i in 0..pts.len() {
        let (xi, yi) = pts[i];
        let (xj, yj) = pts[j];
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            hit = !hit;
        }
        j = i;
    }
    hit
}

/// The isometric cube, centred on `c` with circumradius `r`.
///
/// Its silhouette is a regular hexagon: the top vertex is the cube's far top
/// corner, the centre is its near top corner, and the three edges radiating
/// from the centre are the "Y" that makes a hexagon read as a box.
fn cube(canvas: &mut Canvas, c: (f64, f64), r: f64, stroke: f64) {
    let k = 0.866_025_403_784_438_6; // cos 30
    let t = (c.0, c.1 - r);
    let ur = (c.0 + k * r, c.1 - 0.5 * r);
    let lr = (c.0 + k * r, c.1 + 0.5 * r);
    let b = (c.0, c.1 + r);
    let ll = (c.0 - k * r, c.1 + 0.5 * r);
    let ul = (c.0 - k * r, c.1 - 0.5 * r);

    canvas.poly(&[t, ur, c, ul], FACE_TOP);
    canvas.poly(&[ul, c, b, ll], FACE_LEFT);
    canvas.poly(&[c, ur, lr, b], FACE_RIGHT);

    for e in [(t, ur), (ur, lr), (lr, b), (b, ll), (ll, ul), (ul, t)] {
        canvas.line(e.0, e.1, stroke, EDGE);
    }
    // The three edges at the near top corner go up-left, up-right and down
    // (up would draw the cube seen from below).
    for e in [(c, ul), (c, ur), (c, b)] {
        canvas.line(e.0, e.1, stroke, EDGE);
    }
}

/// The application icon: the mark, as large as it will go.
fn app_icon(size: u32) -> Vec<u8> {
    let n = size * SS;
    let mut canvas = Canvas::new(n, n);
    let s = n as f64;
    // Relatively thinner strokes as the icon grows; 16 px needs a full pixel.
    let stroke = s * if size <= 24 { 0.055 } else { 0.035 };
    cube(&mut canvas, (s * 0.5, s * 0.5), s * 0.44, stroke);
    canvas.resolve(size, size)
}

/// The document icon: a page with the mark on it. Explorer badges thumbnails
/// with the file type's icon, so this keeps STEP files from looking like
/// programs.
fn doc_icon(size: u32) -> Vec<u8> {
    let n = size * SS;
    let mut canvas = Canvas::new(n, n);
    let s = n as f64;

    // Laid out per size rather than scaled: at 16 px true page proportions
    // leave the mark a few pixels wide, so small sizes get a wider page and a
    // bigger mark.
    let small = size <= 20;
    let (l, r) = if small { (s * 0.12, s * 0.88) } else { (s * 0.19, s * 0.81) };
    let (t, b) = if small { (s * 0.03, s * 0.97) } else { (s * 0.05, s * 0.95) };
    // The fold is kept at every size: its colour is what identifies the file
    // in a small list view.
    let fold = (r - l) * 0.34;

    canvas.poly(
        &[(l, t), (r - fold, t), (r, t + fold), (r, b), (l, b)],
        PAGE,
    );
    let edge = s * if small { 0.028 } else { 0.020 };
    for e in [
        ((l, t), (r - fold, t)),
        ((r - fold, t), (r, t + fold)),
        ((r, t + fold), (r, b)),
        ((r, b), (l, b)),
        ((l, b), (l, t)),
    ] {
        canvas.line(e.0, e.1, edge, PAGE_EDGE);
    }
    if fold > 0.0 {
        // The turned corner.
        canvas.poly(&[(r - fold, t), (r, t + fold), (r - fold, t + fold)], FOLD);
        canvas.line((r - fold, t), (r - fold, t + fold), edge, FOLD_EDGE);
        canvas.line((r - fold, t + fold), (r, t + fold), edge, FOLD_EDGE);
    }

    // The mark, below the fold and clear of the page edges.
    let (cy, rad) = if small { (0.64, 0.26) } else { (0.62, 0.24) };
    let stroke = s * if small { 0.055 } else { 0.030 };
    cube(&mut canvas, (s * 0.5, s * cy), s * rad, stroke);
    canvas.resolve(size, size)
}

/// Sizes Windows asks for, including high-DPI ones, so nothing is resampled.
const SIZES: [u32; 10] = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256];

fn main() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("icons");
    std::fs::create_dir_all(&dir).expect("create icons dir");

    for (name, draw) in [
        ("stpview.ico", app_icon as fn(u32) -> Vec<u8>),
        ("step.ico", doc_icon as fn(u32) -> Vec<u8>),
    ] {
        let images: Vec<(u32, Vec<u8>)> = SIZES.iter().map(|&s| (s, draw(s))).collect();
        let path = dir.join(name);
        write_ico(&path, &images).expect("write ico");
        println!("{}  {} sizes", path.display(), images.len());

        // A PNG of the largest, for viewing without an icon viewer.
        let png = dir.join(name.replace(".ico", "-256.png"));
        write_png(&png, 256, 256, &images.last().unwrap().1).expect("write png");

        let proof = dir.join(name.replace(".ico", "-proof.png"));
        write_proof(&proof, &images).expect("write proof");
        println!("  proof {}", proof.display());
    }
}

/// A proof sheet for the small sizes: each at 1:1 and magnified
/// (nearest-neighbour), over light and dark backgrounds.
fn write_proof(path: &Path, images: &[(u32, Vec<u8>)]) -> std::io::Result<()> {
    const ZOOM: u32 = 5;
    const GAP: u32 = 8;
    let shown: Vec<&(u32, Vec<u8>)> = images.iter().filter(|(s, _)| *s <= 64).collect();

    let band = shown.iter().map(|(s, _)| s * ZOOM + GAP + s + GAP).sum::<u32>() + GAP;
    let tall = shown.iter().map(|(s, _)| s * ZOOM).max().unwrap_or(0) + GAP * 2;
    let (w, h) = (band, tall * 2);

    let mut buf = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let o = ((y * w + x) * 4) as usize;
            let c: Rgba = if y < tall {
                [0xff, 0xff, 0xff, 0xff]
            } else {
                [0x20, 0x22, 0x26, 0xff]
            };
            buf[o..o + 4].copy_from_slice(&c);
        }
    }

    for half in 0..2u32 {
        let mut x0 = GAP;
        for (size, rgba) in &shown {
            for (scale, slot) in [(ZOOM, 0u32), (1, 1)] {
                let side = size * scale;
                let y0 = half * tall + GAP + (tall - GAP * 2 - side) / 2;
                for y in 0..side {
                    for x in 0..side {
                        let s = (((y / scale) * size + (x / scale)) * 4) as usize;
                        let (r, g, b, a) = (
                            rgba[s] as u32,
                            rgba[s + 1] as u32,
                            rgba[s + 2] as u32,
                            rgba[s + 3] as u32,
                        );
                        let o = (((y0 + y) * w + x0 + x) * 4) as usize;
                        // Straight alpha over whatever the band is.
                        for k in 0..3 {
                            let under = buf[o + k] as u32;
                            let over = [r, g, b][k];
                            buf[o + k] = ((over * a + under * (255 - a)) / 255) as u8;
                        }
                    }
                }
                x0 += side + GAP;
                let _ = slot;
            }
        }
    }
    write_png(path, w, h, &buf)
}

/// Writes a multi-resolution `.ico`.
///
/// Entries up to 64 px are 32-bit BMPs, larger ones PNGs: PNG keeps 256 px
/// small, but small PNG entries are not read by every icon consumer.
fn write_ico(path: &Path, images: &[(u32, Vec<u8>)]) -> std::io::Result<()> {
    let mut payloads: Vec<Vec<u8>> = Vec::new();
    for (size, rgba) in images {
        payloads.push(if *size <= 64 {
            bmp_entry(*size, rgba)
        } else {
            let mut png = Vec::new();
            encode_png(&mut png, *size, *size, rgba)?;
            png
        });
    }

    let mut out = Vec::new();
    out.extend_from_slice(&0u16.to_le_bytes()); // reserved
    out.extend_from_slice(&1u16.to_le_bytes()); // 1 = icon
    out.extend_from_slice(&(images.len() as u16).to_le_bytes());

    // Directory entries are fixed width, so the first payload starts after all
    // of them.
    let mut offset = 6 + 16 * images.len() as u32;
    for ((size, _), payload) in images.iter().zip(&payloads) {
        // 256 is written as 0: the field is one byte.
        let dim = if *size == 256 { 0u8 } else { *size as u8 };
        out.push(dim);
        out.push(dim);
        out.push(0); // palette size, 0 for truecolour
        out.push(0); // reserved
        out.extend_from_slice(&1u16.to_le_bytes()); // planes
        out.extend_from_slice(&32u16.to_le_bytes()); // bits per pixel
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += payload.len() as u32;
    }
    for p in &payloads {
        out.extend_from_slice(p);
    }
    std::fs::write(path, out)
}

/// One BMP icon entry: a header, the colours bottom-up, and an AND mask.
fn bmp_entry(size: u32, rgba: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&40u32.to_le_bytes()); // biSize
    out.extend_from_slice(&(size as i32).to_le_bytes()); // biWidth
    // Twice the height: colour image and mask are stacked.
    out.extend_from_slice(&((size * 2) as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // biPlanes
    out.extend_from_slice(&32u16.to_le_bytes()); // biBitCount
    out.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB
    out.extend_from_slice(&0u32.to_le_bytes()); // biSizeImage
    out.extend_from_slice(&[0u8; 16]); // resolutions, palette counts

    // Bottom-up, BGRA.
    for y in (0..size).rev() {
        for x in 0..size {
            let o = ((y * size + x) * 4) as usize;
            out.push(rgba[o + 2]);
            out.push(rgba[o + 1]);
            out.push(rgba[o]);
            out.push(rgba[o + 3]);
        }
    }
    // An all-zero AND mask: alpha does the compositing, but the mask must be
    // present or the icon reads as half its height.
    let mask_stride = size.div_ceil(32) * 4;
    out.extend(std::iter::repeat_n(0u8, (mask_stride * size) as usize));
    out
}

fn write_png(path: &Path, w: u32, h: u32, rgba: &[u8]) -> std::io::Result<()> {
    let mut out = Vec::new();
    encode_png(&mut out, w, h, rgba)?;
    std::fs::write(path, out)
}

fn encode_png(out: &mut Vec<u8>, w: u32, h: u32, rgba: &[u8]) -> std::io::Result<()> {
    let mut enc = png::Encoder::new(out, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .and_then(|mut w| w.write_image_data(rgba))
        .map_err(std::io::Error::other)
}
