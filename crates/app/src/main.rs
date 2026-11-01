//! stpview - a small 3D viewer for ISO-10303-21 (STEP) B-rep models, and for
//! STL meshes.

// `!(a > b)` is not a spelling of `a <= b` here: it is a NaN guard, since a
// degenerate case returns NaN as readily as zero and NaN must take the failing
// branch.
#![allow(clippy::neg_cmp_op_on_partial_ord)]
// Numeric code full of parallel arrays walked in step; the indexed loop is the
// clear form.
#![allow(clippy::needless_range_loop)]

mod refine;
mod scenefile;
mod select;
#[cfg(windows)]
mod shellreg;
mod spacemouse;
mod viewer;

use std::process::ExitCode;
use std::time::Instant;

// This binary parses arguments, prints reports and owns the event loop;
// geometry lives in the library.
use stpview::{camera, geom, model, render, step, stl, tess};

use stpview::math::{v3, Aabb, Vec3};
use stpview::render::AoLevel;
use stpview::tess::NARROW_FACE_TOLERANCES;

const USAGE: &str = "\
usage: stpview <file.step | file.stl | scene.toml> [options]

options:
  --scene <toml> compose several models into one picture: a list of
                 [[place]] blocks, each a file with an optional position
                 (at, mm), rotation (rotate, degrees about X, Y, Z) and
                 tint. A .toml given as the input means the same thing.
  --tol <mm>     chord tolerance for tessellating STEP
                 (default: 0.03% of the model's bounding box); an STL is
                 triangles already and is drawn as it is, in millimetres
  --ao <level>   ambient occlusion at startup: off, subtle, normal,
                 strong, heavy (default: strong)
  --edges <m>    edge overlay at startup: sharp, all, off (default: sharp)
  --wire         draw only the edges, without the shaded surfaces
  --section <s>  cut the model with a plane, as [-]x|y|z[,offset]; without an
                 offset the plane passes through the middle of the model, so
                 --section -z halves it and keeps the upper part
  --no-cap       leave the cut hollow instead of filling it
  --no-ground    do not stand the model on a shadowed floor
  --no-silhouette  do not outline the model where its surface turns away
  --supersample <n>  average a still view over this many frames, each sampling
                 its pixels somewhere else in them; 1 is none (default: 16)
  --no-refine    keep the tessellation the model was loaded at, however close
                 the view: without it, the parts a still view looks at closely
                 are tessellated again to within half a pixel on screen
  --only <text>  draw only the parts whose assembly path contains <text>
  --colors <by>  colour the parts to tell them apart: part or assembly;
                 file, the default, is the colours the file gives them
  --tree         print the assembly tree with triangle counts and exit
  --stats        print statistics and exit without opening a window
  --check        also check that adjacent faces meet without cracks
  --shot <png>   render one frame off screen to a PNG and exit
  --size <WxH>   size of that frame (default: 1200x900)
  --thumb <png>  render a square thumbnail, the picture Explorer's thumbnail
                 provider draws, through the same code it runs: 256 px unless
                 --size gives a width
  --register     register that provider for .step, .stp and .stl under HKCU,
                 so Explorer draws the models instead of a generic icon
  --unregister   take the registration back out again
  --select <x,y> select the part under that pixel of the frame, as a click
                 would, and paint it in the shot; give it once per part
  --expect <png> compare that frame against a stored one and fail if it moved
  --view <spec>  where to point the camera, for a shot or as the viewer
                 opens: yaw,pitch[,x,y,z[,radius]] in degrees and model units;
                 the position and radius frame a feature, so
                 --view 30,20,5,15,-10,6 looks at a 6 mm ball around
                 (5, 15, -10)
  --puck         print the SpaceMouse axes as they move, and nothing else
  -h, --help     show this message

controls:
  left drag            orbit
  right / middle drag  pan          (or shift + left drag)
  wheel                zoom
  spacemouse           slide to pan, push away to zoom, twist and tip to
                       orbit, roll to roll; its first button fits and levels
  click                select the part under the cursor; on empty space,
                       select nothing
  ctrl + click         add a part to the selection, or take it out again
  Up / Down            widen the selection to the assembly it belongs to,
                       and narrow it back
  Esc                  select nothing
  H                    hide the selected parts, or the one under the cursor
  I                    isolate them: hide everything else
  A                    show every part again
  C                    colours: as the file has them -> by part -> by assembly
  F                    fit the view to the selection, or to what is shown
  E                    edges: sharp -> all -> off
  S                    toggle shaded surfaces
  O                    occlusion: off -> subtle -> normal -> strong -> heavy
  X / Y / Z            section on that axis: forward -> back -> off
  [ / ]                slide the section plane (shift for a fine step)
  K                    fill the cut, or leave it hollow
  G                    toggle the ground plane
  L                    toggle the silhouette outline
  Q                    quit";

struct Args {
    path: String,
    /// The input is a scene description (`--scene`, or a `.toml` input);
    /// `path` holds it either way.
    scene: bool,
    tol: Option<f64>,
    ao: AoLevel,
    edges: render::EdgeMode,
    wire: bool,
    stats_only: bool,
    check: bool,
    shot: Option<String>,
    expect: Option<String>,
    size: (u32, u32),
    /// Leave the cut hollow, to look inside rather than read the material.
    no_cap: bool,
    no_ground: bool,
    no_silhouette: bool,
    /// Never re-tessellate for a view, and so never keep the model.
    no_refine: bool,
    /// Frames a still view is averaged over; see `RenderOptions::supersample`.
    supersample: u32,
    /// Draw only the parts whose assembly path contains this.
    only: Option<String>,
    /// Print the assembly tree and exit.
    tree: bool,
    view: Option<View>,
    /// Pixels of the rendered frame to select the parts under, as clicks
    /// would; each is reported and painted in the shot.
    selects: Vec<(f32, f32)>,
    /// How the parts are coloured.
    colors: select::ColorMode,
    /// Axis, direction, and an explicit offset. Without one the plane goes
    /// through the middle of the model, known only once it is read.
    section: Option<(render::Axis, f64, Option<f64>)>,
    /// Where to write a thumbnail, and its side. Renders through
    /// `stpview::thumb`, as the shell provider does.
    thumb: Option<(String, u32)>,
}

fn parse_section(s: &str) -> Result<(render::Axis, f64, Option<f64>), String> {
    let (spec, offset) = match s.split_once(',') {
        Some((a, o)) => (
            a,
            Some(o.trim().parse::<f64>().map_err(|_| format!("bad --section offset {o:?}"))?),
        ),
        None => (s, None),
    };
    let (axis, sign) = render::Axis::parse(spec.trim())
        .ok_or_else(|| format!("unknown --section axis {spec:?}; expected x, y, z, -x, -y or -z"))?;
    Ok((axis, sign, offset))
}

/// Where `--shot` puts the camera, and where the viewer opens. Angles in
/// degrees; an absent target frames the whole model.
struct View {
    yaw: f64,
    pitch: f64,
    at: Option<(Vec3, f64)>,
}

impl View {
    fn aim(&self, camera: &mut camera::Camera) {
        camera.yaw = self.yaw;
        camera.pitch = self.pitch;
        if let Some((at, radius)) = self.at {
            camera.target = at;
            if radius > 0.0 {
                camera.distance = radius / (camera.fov_y as f64 * 0.5).sin();
            }
        }
    }
}

fn parse_view(s: &str) -> Result<View, String> {
    let n: Vec<f64> = s
        .split(',')
        .map(|t| t.trim().parse::<f64>().map_err(|_| format!("bad --view number {t:?}")))
        .collect::<Result<_, _>>()?;
    let at = match n.len() {
        2 => None,
        5 => Some((v3(n[2], n[3], n[4]), 0.0)),
        6 => Some((v3(n[2], n[3], n[4]), n[5])),
        _ => return Err("--view wants yaw,pitch[,x,y,z[,radius]]".into()),
    };
    Ok(View {
        yaw: n[0].to_radians(),
        pitch: n[1].to_radians(),
        at,
    })
}

/// Prints the assembly tree, with what each node costs.
///
/// A part's line carries its triangles; a subassembly's carries the total
/// under it. Nodes no geometry reaches are listed too.
fn print_tree(model: &model::Model, mesh: &tess::Mesh) {
    let nodes = &model.nodes;
    // Unnamed nodes fold into their nearest named ancestor: a component's
    // geometry hangs off a nameless representation below its product.
    let named: Vec<usize> = (0..nodes.len())
        .map(|mut i| {
            while nodes[i].name.is_empty() {
                match nodes[i].parent {
                    Some(p) => i = p,
                    None => break,
                }
            }
            i
        })
        .collect();

    let mut own = vec![0usize; nodes.len()];
    let mut loose = 0usize;
    for p in &mesh.parts {
        let n = p.tris.len() / 3;
        match p.node {
            Some(i) => own[named[i]] += n,
            None => loose += n,
        }
    }
    // Totals roll up from the leaves; a node always follows its parent, so
    // one reverse pass is enough.
    let mut total = own.clone();
    for i in (0..nodes.len()).rev() {
        if let Some(p) = nodes[i].parent {
            let (from, to) = (named[i], named[p]);
            if from != to {
                total[to] += total[from];
            }
        }
    }

    let shown = |i: usize| !nodes[i].name.is_empty();
    let depth = |mut i: usize| {
        let mut d = 0;
        while let Some(p) = nodes[i].parent {
            i = p;
            d += usize::from(shown(i));
        }
        d
    };
    // Parts as well as nodes: some files have no product structure at all,
    // just one node and many bodies.
    println!(
        "tree   {} nodes, {} parts",
        nodes.iter().filter(|n| !n.name.is_empty()).count(),
        mesh.parts.len()
    );
    for i in 0..nodes.len() {
        if !shown(i) {
            continue;
        }
        // "here" only where a subassembly carries geometry of its own.
        let extra = if own[i] == total[i] {
            String::new()
        } else if own[i] > 0 {
            format!(" ({} of them here)", own[i])
        } else {
            String::new()
        };
        println!(
            "       {:indent$}{}  {} tri{extra}",
            "",
            nodes[i].name,
            total[i],
            indent = depth(i) * 2,
        );
    }
    if loose > 0 {
        println!("       (outside the assembly)  {loose} tri");
    }
}

impl Args {
    /// One flag per part, or an empty slice when everything is drawn.
    fn visible(&self, mesh: &tess::Mesh) -> Vec<bool> {
        let Some(pat) = &self.only else { return Vec::new() };
        let pat = pat.to_lowercase();
        mesh.parts
            .iter()
            .map(|p| p.name.to_lowercase().contains(&pat))
            .collect()
    }

    /// The section plane in model space. Without an explicit offset it goes
    /// through the middle of the bounding box.
    fn plane(&self, bbox: &Aabb) -> Option<render::Section> {
        let (axis, sign, offset) = self.section?;
        let normal = axis.normal(sign);
        Some(render::Section {
            normal,
            offset: offset.unwrap_or_else(|| normal.dot(bbox.center())),
        })
    }

    /// The command line as the library sees it. Visibility and paint are
    /// filled in by the caller, which knows the mesh.
    fn options(&self, bbox: &Aabb) -> stpview::RenderOptions {
        stpview::RenderOptions {
            show_faces: !self.wire,
            edges: self.edges,
            ao: self.ao,
            section: self.plane(bbox),
            cap: !self.no_cap,
            ground: !self.no_ground,
            silhouette: !self.no_silhouette,
            visible: Vec::new(),
            paint: Vec::new(),
            supersample: self.supersample,
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut path = None;
    let mut scene = false;
    let mut tol = None;
    let mut ao = AoLevel::Strong;
    let mut edges = render::EdgeMode::Sharp;
    let mut wire = false;
    let mut stats_only = false;
    let mut check = false;
    let mut shot = None;
    let mut expect = None;
    let mut size = (1200, 900);
    // Whether `--size` was given: it decides a thumbnail's side, which the
    // default frame size does not.
    let mut size_given = false;
    let mut thumb = None;
    let mut view = None;
    let mut selects = Vec::new();
    let mut colors = select::ColorMode::File;
    let mut no_cap = false;
    let mut no_ground = false;
    let mut no_silhouette = false;
    let mut no_refine = false;
    let mut supersample = stpview::render::SUPERSAMPLE;
    let mut only = None;
    let mut tree = false;
    let mut section = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--shot" => shot = Some(it.next().ok_or("--shot needs a file name")?),
            "--section" => {
                section = Some(parse_section(&it.next().ok_or("--section needs a value")?)?)
            }
            "--expect" => expect = Some(it.next().ok_or("--expect needs a file name")?),
            "--select" => {
                let v = it.next().ok_or("--select needs a value")?;
                let (x, y) = v.split_once(',').ok_or("--select wants x,y in pixels")?;
                selects.push((
                    x.trim().parse().map_err(|_| format!("bad --select x {x:?}"))?,
                    y.trim().parse().map_err(|_| format!("bad --select y {y:?}"))?,
                ));
            }
            "--colors" => {
                let v = it.next().ok_or("--colors needs a value")?;
                colors = select::ColorMode::parse(&v).ok_or_else(|| {
                    format!("unknown --colors mode {v:?}; expected file, part or assembly")
                })?;
            }
            "--view" => view = Some(parse_view(&it.next().ok_or("--view needs a value")?)?),
            "--size" => {
                let v = it.next().ok_or("--size needs a value")?;
                let (w, h) = v.split_once(['x', 'X']).ok_or("--size wants WxH")?;
                size = (
                    w.parse().map_err(|_| format!("bad width {w:?}"))?,
                    h.parse().map_err(|_| format!("bad height {h:?}"))?,
                );
                size_given = true;
            }
            "--thumb" => thumb = Some(it.next().ok_or("--thumb needs a file name")?),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            // Answered in main() before parsing begins; it can only reach this
            // loop when --help is also present, which must win.
            "--puck" | "--register" | "--unregister" => {}
            "--scene" => {
                let v = it.next().ok_or("--scene needs a file")?;
                if path.replace(v).is_some() {
                    return Err("expected only one input file".into());
                }
                scene = true;
            }
            "--wire" => wire = true,
            "--no-cap" => no_cap = true,
            "--no-ground" => no_ground = true,
            "--no-silhouette" => no_silhouette = true,
            "--no-refine" => no_refine = true,
            "--supersample" => {
                let v = it.next().ok_or("--supersample needs a count")?;
                supersample = v
                    .parse::<u32>()
                    .ok()
                    .filter(|n| (1..=256).contains(n))
                    .ok_or_else(|| format!("bad --supersample {v:?}; expected 1 to 256"))?;
            }
            "--only" => only = Some(it.next().ok_or("--only needs a pattern")?),
            "--tree" => tree = true,
            "--edges" => {
                let v = it.next().ok_or("--edges needs a value")?;
                edges = render::EdgeMode::parse(&v)
                    .ok_or_else(|| format!("unknown --edges mode {v:?}; expected sharp, all or off"))?;
            }
            "--stats" => stats_only = true,
            "--check" => check = true,
            "--tol" => {
                let v = it.next().ok_or("--tol needs a value")?;
                let v: f64 = v.parse().map_err(|_| format!("bad --tol value {v:?}"))?;
                if !(v > 0.0) {
                    return Err("--tol must be positive".into());
                }
                tol = Some(v);
            }
            "--ao" => {
                let v = it.next().ok_or("--ao needs a value")?;
                ao = AoLevel::parse(&v).ok_or_else(|| {
                    format!("unknown --ao level {v:?}; expected off, subtle, normal, strong or heavy")
                })?;
            }
            other if other.starts_with('-') => return Err(format!("unknown option {other:?}")),
            other => {
                if path.replace(other.to_string()).is_some() {
                    return Err("expected only one input file".into());
                }
            }
        }
    }
    let path = path.ok_or("no input file given")?;
    // A .toml input is a scene, which lets the visual harness hold composite
    // views without a new column.
    let scene = scene || path.to_ascii_lowercase().ends_with(".toml");
    // A thumbnail is square, so only a width is taken from `--size`.
    let thumb = thumb.map(|out| (out, if size_given { size.0.max(1) } else { 256 }));
    Ok(Args {
        path,
        scene,
        tol,
        ao,
        edges,
        wire,
        stats_only,
        check,
        shot,
        expect,
        size,
        view,
        selects,
        colors,
        section,
        no_cap,
        no_ground,
        no_silhouette,
        no_refine,
        supersample,
        only,
        tree,
        thumb,
    })
}

/// Renders one square thumbnail through exactly the path the shell provider
/// runs, and writes it as a PNG. Errors go to stderr, which the shell's
/// surrogate does not have.
fn write_thumb(path: &str, out: &str, size: u32, expect: Option<&str>) -> Result<Verdict, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let t = Instant::now();
    let rgba = stpview::thumb::render(&bytes, size, path)?;
    let elapsed = t.elapsed().as_secs_f64();

    let file = std::fs::File::create(out).map_err(|e| format!("{out}: {e}"))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), size, size);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .and_then(|mut w| w.write_image_data(&rgba))
        .map_err(|e| format!("{out}: {e}"))?;

    println!("thumb  {out} at {size}x{size} in {elapsed:.2}s");

    // `--expect` works here as for `--shot`, so a thumbnail can be a stored
    // view: its camera (`close_in`), options and byte-based import are its
    // own and could drift independently.
    match expect {
        Some(want) => compare(&rgba, (size, size), want),
        None => Ok(Verdict::Ok),
    }
}

/// Renders one frame with no window and writes it out.
///
/// `STPVIEW_STREAM=1` builds the buffers as the viewer does, one batch at a
/// time from an empty mesh, so the streaming path is held to the stored views.
///
/// `model`, with the mesh's tolerance, is what the parts this camera looks at
/// closely are re-tessellated from, as in the viewer once it settles. Without
/// one (`--no-refine`, a scene) the shot uses the mesh as it is.
fn shoot(
    mesh: &mut tess::Mesh,
    args: &Args,
    path: &str,
    model: Option<(&model::Model, f64)>,
    hierarchy: &select::Hierarchy,
) -> Result<Verdict, String> {
    let streamed = model.filter(|_| std::env::var_os("STPVIEW_STREAM").is_some());
    let visible = args.visible(mesh);
    let (w, h) = args.size;
    let mut camera = camera::Camera::new(&mesh.bbox);
    if let Some(v) = &args.view {
        v.aim(&mut camera);
    }
    let mut r = match streamed {
        Some((m, tol)) => {
            let seed = tess::Mesh {
                bbox: m.vertex_bbox,
                mm_per_unit: m.mm_per_unit,
                ..Default::default()
            };
            let mut r = stpview::Renderer::offscreen(w, h, &seed)?;
            let t = Instant::now();
            let (mut batches, mut first) = (0, 0.0);
            tess::tessellate_streaming(m, tol, |b| {
                batches += 1;
                if batches == 1 {
                    first = t.elapsed().as_secs_f64();
                }
                r.append(&b);
            });
            println!(
                "stream {batches} batches in {:.2}s, first drawable at {first:.2}s",
                t.elapsed().as_secs_f64()
            );
            r
        }
        None => stpview::Renderer::offscreen(w, h, mesh)?,
    };
    // The GPU owns the triangles from here.
    mesh.release_uploaded();
    let mesh = &*mesh;

    let mut opts = args.options(&mesh.bbox);
    opts.visible = visible.clone();

    if let Some((m, tol)) = model.filter(|_| !args.no_refine) {
        refine(&mut r, m, tol, &camera, &opts, &mesh.parts);
    }

    // Selections first, so the written frame carries them. Picking by pixel
    // runs the same code a click does, which makes selection testable.
    let mut selection = select::Selection::default();
    for &(x, y) in &args.selects {
        let part = r
            .pick(&camera, &opts, x, y)
            .and_then(|p| mesh.part_of_vertex(p.vertex));
        match part {
            Some(i) => {
                selection.toggle(i);
                println!(
                    "select {x:.0},{y:.0}  {}  ({} tri)",
                    select::part_label(&mesh.parts, i),
                    mesh.parts[i].tris.len() / 3
                );
            }
            None => println!("select {x:.0},{y:.0}  nothing there"),
        }
    }
    opts.paint = select::paint(args.colors, hierarchy, &mesh.parts, &selection, None);
    let rgba = r.capture(&camera, &opts)?;

    let file = std::fs::File::create(path).map_err(|e| format!("{path}: {e}"))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .and_then(|mut w| w.write_image_data(&rgba))
        .map_err(|e| format!("{path}: {e}"))?;
    println!("shot   {path} at {w}x{h}");

    if let Some(want) = &args.expect {
        return compare(&rgba, (w, h), want);
    }
    Ok(Verdict::Ok)
}

/// Re-tessellates, for this camera, the parts it looks at closely enough to
/// want finer than the model's tolerance, and draws those in their place.
///
/// The viewer's settled-view refinement, done synchronously, so a shot is the
/// picture the viewer settles on and refinement is held to the stored views.
fn refine(
    r: &mut stpview::Renderer,
    m: &model::Model,
    tol: f64,
    camera: &camera::Camera,
    opts: &stpview::RenderOptions,
    parts: &[tess::Part],
) {
    let t = Instant::now();
    let view = r.view_tolerance(camera, tol);
    // Only what the view can see.
    let seen = r.parts_in_frame(camera, opts, stpview::IN_FRAME_SCALE);
    let mut want = tess::parts_to_refine(m, parts, &seen, &view, tol);
    if want.is_empty() {
        return;
    }
    // In model order, so consecutive solids go through the tessellator as runs.
    want.sort_unstable();
    let fine = tess::tessellate_for_view(m, tol, &view, parts, &want);
    let before: usize = want.iter().map(|&i| parts[i].tris.len() / 3).sum();
    println!(
        "refine {} parts for this view: {} triangles where there were {}, in {:.2}s",
        want.len(),
        fine.indices.len() / 3,
        before,
        t.elapsed().as_secs_f64()
    );
    r.set_refined(&fine, &want);
}

/// A pixel differs when any channel is off by more than this. Small enough to
/// catch a shading change, large enough to ignore the last bit of a float.
const VISUAL_CHANNEL: i32 = 8;
/// And the frame differs when *any* of its pixels do.
///
/// Zero because the renderer is deterministic: same binary and machine, same
/// bytes. Any tolerance would let small but real regressions through. After a
/// driver or GPU change, re-record the references rather than loosening this.
const VISUAL_FRACTION: f64 = 0.0;

/// Measures a frame against a stored one, for `models/visual.ps1`.
///
/// Counting pixels rather than a perceptual hash: the reference comes from the
/// same code at the same size, the count says how much moved, and a coarse
/// hash would average away a few wrong pixels.
fn compare(got: &[u8], (w, h): (u32, u32), want: &str) -> Result<Verdict, String> {
    let file = std::fs::File::open(want).map_err(|e| format!("{want}: {e}"))?;
    let mut reader = png::Decoder::new(std::io::BufReader::new(file))
        .read_info()
        .map_err(|e| format!("{want}: {e}"))?;
    let mut buf = vec![0; reader.output_buffer_size().unwrap_or(0)];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("{want}: {e}"))?;
    if (info.width, info.height) != (w, h) {
        return Err(format!(
            "{want} is {}x{}, this frame is {w}x{h}",
            info.width, info.height
        ));
    }
    let stride = info.color_type.samples();
    let total = (w * h) as usize;
    let mut moved = 0usize;
    let mut worst = 0i32;
    for i in 0..total {
        let (a, b) = (&got[i * 4..i * 4 + 3], &buf[i * stride..i * stride + 3]);
        let d = (0..3)
            .map(|k| (a[k] as i32 - b[k] as i32).abs())
            .max()
            .unwrap_or(0);
        worst = worst.max(d);
        if d > VISUAL_CHANNEL {
            moved += 1;
        }
    }
    let frac = moved as f64 / total as f64;
    println!(
        "visual {} of {total} pixels moved ({:.3}%), worst channel {worst}",
        moved,
        frac * 100.0
    );
    Ok(if frac > VISUAL_FRACTION {
        Verdict::Moved
    } else {
        Verdict::Ok
    })
}

/// The composite path: a scene file in, the same outputs out. The baked mesh
/// goes through the same `shoot` and viewer, so every option means what it
/// does for one file.
fn run_scene(args: Args) -> Result<Verdict, String> {
    let t_all = Instant::now();
    let specs = scenefile::parse(std::path::Path::new(&args.path))?;
    let assets = match args.tol {
        Some(mm) => stpview::Assets::new().with_tolerance_mm(mm),
        None => stpview::Assets::new(),
    };
    let t = Instant::now();
    let scene = scenefile::build(&specs, &assets).map_err(|e| e.to_string())?;
    println!(
        "scene  {} placements of {} files in {:.2}s",
        specs.len(),
        assets.cached(),
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    let mut mesh = scene.bake();
    println!(
        "bake   {} triangles, {} parts in {:.2}s",
        mesh.indices.len() / 3,
        mesh.parts.len(),
        t.elapsed().as_secs_f64()
    );
    // Where each placement landed, in scene millimetres, so a misplaced or
    // missing one is visible in the output.
    for (spec, inst) in specs.iter().zip(&scene.instances) {
        let b = inst.mesh.bbox;
        let w = inst.place.compose(&stpview::geom::Xform::scale(inst.mesh.mm_per_unit));
        let (lo, hi) = (w.point(b.min), w.point(b.max));
        println!(
            "       {} at [{:.0}, {:.0}, {:.0}] .. [{:.0}, {:.0}, {:.0}]",
            spec.file.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
            lo.x.min(hi.x), lo.y.min(hi.y), lo.z.min(hi.z),
            lo.x.max(hi.x), lo.y.max(hi.y), lo.z.max(hi.z),
        );
    }
    if mesh.vertices.is_empty() && mesh.edges.is_empty() {
        return Err("nothing to display".into());
    }

    if let Some(path) = &args.shot {
        let v = shoot(&mut mesh, &args, path, None, &select::Hierarchy::default());
        println!("total  {:.2}s", t_all.elapsed().as_secs_f64());
        return v;
    }

    let name = std::path::Path::new(&args.path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| args.path.clone());
    let section = args.plane(&mesh.bbox);
    // The channel is born closed: the mesh is complete, and the viewer's
    // end-of-stream path prints the totals and releases uploaded buffers.
    let (_, rx) = std::sync::mpsc::channel();
    viewer::App::new(
        format!("stpview - {name}"),
        mesh,
        select::Hierarchy::default(),
        args.colors,
        args.ao,
        args.edges,
        args.supersample,
        None,
        section,
        rx,
        Box::new(|m: &tess::Mesh| {
            println!(
                "edges  {} sharp, {} tangent suppressed",
                m.stats.edges_sharp, m.stats.edges_tangent
            );
        }),
    )
    .run()?;
    Ok(Verdict::Ok)
}

/// An STL is triangles already: nothing to build, tessellate or refine. It
/// takes the scene's route to the window, and reports what welding found.
fn run_stl(args: Args, bytes: &[u8], t_all: Instant) -> Result<Verdict, String> {
    let t = Instant::now();
    let file = stl::parse(bytes).map_err(|e| format!("{}: {e}", args.path))?;
    let facets: usize = file.solids.iter().map(|s| s.facets()).sum();
    println!(
        "parse  {facets} facets in {} solids, {}, in {:.2}s",
        file.solids.len(),
        if file.binary { "binary" } else { "ASCII" },
        t.elapsed().as_secs_f64()
    );
    if file.unread > 0 {
        println!(
            "       WARNING {} facets the file promised are not in it, or not readable",
            file.unread
        );
    }
    if args.tol.is_some() {
        println!("       --tol ignored: an STL is triangles already");
    }

    let t = Instant::now();
    let (mut mesh, st) = stl::mesh(&file, &stpview::assets::stem(&args.path));
    println!(
        "mesh   {} triangles, {} points welded from {} corners, in {:.2}s",
        mesh.indices.len() / 3,
        st.points,
        st.corners,
        t.elapsed().as_secs_f64()
    );
    report_stl(&file, &st, &mesh, args.check);
    println!("total  {:.2}s", t_all.elapsed().as_secs_f64());

    if args.tree {
        println!("tree   {} solids", mesh.parts.len());
        for p in &mesh.parts {
            println!("       {}  {} tri", p.name, p.tris.len() / 3);
        }
        return Ok(Verdict::Ok);
    }
    if args.stats_only {
        println!("\nsolids:");
        for (i, (p, s)) in mesh.parts.iter().zip(file.solids.iter().filter(|s| s.facets() > 0)).enumerate() {
            let painted = s.colors.iter().flatten().count();
            println!(
                "  {i:>4}  {:>8} tri  rgb({:.2}, {:.2}, {:.2}){}{}  {}",
                p.tris.len() / 3,
                p.color[0],
                p.color[1],
                p.color[2],
                if painted > 0 { format!("  {painted} facets coloured") } else { String::new() },
                if p.cappable { "" } else { "  open" },
                p.name
            );
        }
        return Ok(Verdict::Ok);
    }
    if mesh.vertices.is_empty() && mesh.edges.is_empty() {
        return Err("nothing to display".into());
    }
    if let Some(path) = &args.shot {
        return shoot(&mut mesh, &args, path, None, &select::Hierarchy::default());
    }

    let name = std::path::Path::new(&args.path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| args.path.clone());
    let section = args.plane(&mesh.bbox);
    // Born closed, as for a scene.
    let (_, rx) = std::sync::mpsc::channel();
    viewer::App::new(
        format!("stpview - {name}"),
        mesh,
        select::Hierarchy::default(),
        args.colors,
        args.ao,
        args.edges,
        args.supersample,
        None,
        section,
        rx,
        // Reported above, before the window took over.
        Box::new(|_: &tess::Mesh| {}),
    )
    .aimed(|c| {
        if let Some(v) = &args.view {
            v.aim(c);
        }
    })
    .run()?;
    Ok(Verdict::Ok)
}

/// Everything an STL load has to say about itself. The file has no units,
/// topology or edges, so most of this is what welding found.
fn report_stl(file: &stl::Stl, st: &stl::StlStats, mesh: &tess::Mesh, check: bool) {
    let b = mesh.bbox;
    println!("       bounds {:.0?} .. {:.0?}", b.min, b.max);
    println!("       units: an STL has none; read as millimetres");
    let painted: usize = file.solids.iter().map(|s| s.colors.iter().flatten().count()).sum();
    let header = file.solids.iter().any(|s| s.color.is_some());
    if painted > 0 || header {
        println!(
            "       colour: {painted} facets coloured{}",
            if header { ", and the solid from the header" } else { "" }
        );
    }
    if st.dropped > 0 {
        println!(
            "       {} facets are lines or points rather than triangles, and are left out",
            st.dropped
        );
    }
    if st.slivers > 0 {
        println!(
            "       {} facets are too thin to have a normal: drawn, but they shade nothing",
            st.slivers
        );
    }
    let s = &mesh.stats;
    let flat = if st.between > 0 { 100.0 * st.flat as f64 / st.between as f64 } else { 0.0 };
    println!(
        "edges  {} sharp, {} more with --edges all; {} creases past {} degrees, {flat:.1}% of edges flat",
        s.edges_sharp,
        s.edges_tangent,
        st.creases,
        stl::CREASE_ANGLE
    );
    if st.rough > 0 {
        // Said, so a part drawn without edges is not mistaken for a reader
        // fault.
        println!(
            "       {} of {} solids are rough between their creases, so the creases are texture and not drawn",
            st.rough, st.solids
        );
    }
    if st.open + st.overshared + st.flipped > 0 {
        println!(
            "       {} open, {} shared by more than two facets, {} between facets wound against each other",
            st.open, st.overshared, st.flipped
        );
    }
    let facets = mesh.indices.len() / 3;
    if file.against > 0 {
        // Only a cap reads the winding, so this is the one place a file wound
        // inside out would ever be noticed.
        println!(
            "       {} of {facets} facets have a written normal against their winding{}",
            file.against,
            if file.against * 2 > facets { ": the file is wound inside out" } else { "" }
        );
    }
    let open = mesh.parts.iter().filter(|p| !p.cappable).count();
    if open > 0 {
        println!(
            "       {open} of {} solids are not closed and consistently wound, so are not capped",
            mesh.parts.len()
        );
    }
    println!(
        "memory {:.0} MB: {} vertices at {} B, {} indices, {} edge points",
        mesh.bytes() as f64 / 1e6,
        mesh.vertices.len(),
        std::mem::size_of::<tess::Vertex>(),
        mesh.indices.len(),
        mesh.edges.len(),
    );
    if check {
        // The weld above is exact. Welding within a hair (a thousandth of the
        // default tolerance, as for STEP) tells a hole from two roundings of
        // one point.
        let weld = b.extent() * 3e-7;
        let t = Instant::now();
        let seams = tess::seams(mesh, weld);
        println!(
            "check  {} of {} mesh edges unpaired, {} over-shared, welded within {weld:.1e} mm, in {:.2}s",
            seams.unpaired,
            seams.total,
            seams.overshared,
            t.elapsed().as_secs_f64()
        );
        if seams.unpaired < st.open {
            println!(
                "       {} of the {} open edges close at that: their ends are near, not equal",
                st.open - seams.unpaired,
                st.open
            );
        }
        for p in &seams.cracks {
            println!("         open near {p:.3?}");
        }
        for p in &seams.folds {
            println!("         doubled near {p:.3?}");
        }
    }
}

fn main() -> ExitCode {
    // `--puck` wants no file and no window, so it is answered before parsing.
    // `--help` still wins: watch() runs until killed.
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let helping = argv.iter().any(|a| a == "-h" || a == "--help");
    if argv.iter().any(|a| a == "--puck") && !helping {
        return match spacemouse::watch() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        };
    }

    // Likewise for registering the shell provider.
    #[cfg(windows)]
    if !helping {
        let job = if argv.iter().any(|a| a == "--register") {
            Some(shellreg::register as fn() -> Result<(), String>)
        } else if argv.iter().any(|a| a == "--unregister") {
            Some(shellreg::unregister as fn() -> Result<(), String>)
        } else {
            None
        };
        if let Some(job) = job {
            return match job() {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            };
        }
    }

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };

    match run(args) {
        // A moved frame is a result, not an error: it is already reported on
        // stdout, and `models/visual.ps1` reads the exit code. Nothing goes to
        // stderr, which would abort that script.
        Ok(Verdict::Ok) => ExitCode::SUCCESS,
        Ok(Verdict::Moved) => ExitCode::from(2),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Whether a `--expect` comparison matched. Nothing else can fail this way.
enum Verdict {
    Ok,
    Moved,
}

fn run(args: Args) -> Result<Verdict, String> {
    // A thumbnail takes its own route from bytes to pixels: --thumb runs
    // exactly what the shell provider runs.
    if let Some((out, size)) = &args.thumb {
        if args.scene {
            return Err("--thumb takes a model file; a scene has no one file to thumbnail".into());
        }
        return write_thumb(&args.path, out, *size, args.expect.as_deref());
    }
    if args.scene {
        return run_scene(args);
    }
    let t_all = Instant::now();

    let bytes = std::fs::read(&args.path).map_err(|e| format!("{}: {e}", args.path))?;
    println!("read   {:.1} MB", bytes.len() as f64 / 1e6);
    if stpview::Format::of(&bytes, &args.path) == stpview::Format::Stl {
        return run_stl(args, &bytes, t_all);
    }

    let t = Instant::now();
    let file = step::parse(&bytes).map_err(|e| e.to_string())?;
    println!(
        "parse  {} entities in {:.2}s",
        file.entity_count,
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let m = model::build(&file);
    println!(
        "build  {} solids, {} faces in {:.2}s",
        m.solids.len(),
        m.stats.faces,
        t.elapsed().as_secs_f64()
    );
    let moved = m.solids.iter().filter(|s| !s.placement.is_identity()).count();
    if moved > 0 {
        println!("       {moved} of {} solids moved by assembly", m.solids.len());
    }
    // A body no assembly relationship names is drawn at the origin, which
    // usually means a link the reader could not follow, so name them. A body
    // deliberately placed at the origin is linked and not listed.
    let stray: Vec<&model::Solid> = m.solids.iter().filter(|s| !s.linked).collect();
    for s in stray.iter().take(8) {
        let name = if s.name.is_empty() { "(unnamed)" } else { &s.name };
        println!(
            "       unplaced: {name} ({} faces, centred near {:.0?})",
            s.faces.len(),
            s.local_bbox.center()
        );
    }
    if stray.len() > 8 {
        println!("       ... and {} more unplaced", stray.len() - 8);
    }
    if m.stats.dropped_edges > 0 {
        println!("       {} loop edges dropped", m.stats.dropped_edges);
    }
    if m.stats.dropped_bodies > 0 {
        println!(
            "       WARNING {} bodies had no faces and are not drawn",
            m.stats.dropped_bodies
        );
    }
    if m.stats.dropped_bounds > 0 {
        println!(
            "       {} face bounds dropped entirely",
            m.stats.dropped_bounds
        );
    }
    if m.stats.faces_unsupported > 0 {
        println!(
            "       {} faces are on a surface type the reader does not know",
            m.stats.faces_unsupported
        );
    }
    if m.stats.skipped_faces > m.stats.faces_unsupported {
        println!(
            "       {} faces had no usable boundary",
            m.stats.skipped_faces - m.stats.faces_unsupported
        );
    }
    for (name, n) in &m.stats.unsupported_surfaces {
        println!("       unsupported surface {name} x{n}");
    }
    for (name, n) in &m.stats.unsupported_curves {
        println!("       unsupported curve {name} x{n}");
    }
    if m.solids.is_empty() {
        return Err("no geometry found in this file".into());
    }
    let wires: usize = m.solids.iter().map(|s| s.wires.len()).sum();
    if wires > 0 {
        // Curves a `GEOMETRIC_SET` carries on their own (construction
        // geometry, centrelines). Said, so a wireframe-only file does not look
        // like a reader that found nothing.
        println!("       {wires} loose curves drawn as wires");
    }

    println!(
        "       bounds {:.0?} .. {:.0?}",
        m.vertex_bbox.min, m.vertex_bbox.max
    );
    if m.mm_per_unit != 1.0 {
        println!(
            "       units: 1 unit = {} mm, so the model is {:.0} mm across",
            m.mm_per_unit,
            m.vertex_bbox.extent() * m.mm_per_unit
        );
    }

    // `--tol` is quoted in millimetres whatever the file is authored in; the
    // default is a fraction of the model and needs no conversion either way.
    let tol = args
        .tol
        .map(|mm| mm / m.mm_per_unit)
        .unwrap_or(m.vertex_bbox.extent() * 3e-4);
    // The window opens before the mesh exists; only modes that need the
    // whole mesh wait for it. Everything else streams.
    let blocking = args.tree || args.stats_only || args.shot.is_some();
    let m = std::sync::Arc::new(m);
    if !blocking {
        return show(m, tol, args);
    }
    let t = Instant::now();
    let mut mesh = tess::tessellate(&m, tol);
    println!(
        "tess   {} triangles at {:.3} mm in {:.2}s",
        mesh.indices.len() / 3,
        tol * m.mm_per_unit,
        t.elapsed().as_secs_f64()
    );
    report(&m, &mesh, tol, args.check);
    println!("total  {:.2}s", t_all.elapsed().as_secs_f64());

    if args.tree {
        print_tree(&m, &mesh);
        return Ok(Verdict::Ok);
    }
    if args.stats_only {
        println!("\nsolids:");
        for (i, s) in m.solids.iter().take(30).enumerate() {
            let name = if s.name.is_empty() { "(unnamed)" } else { &s.name };
            // How many faces override the body's colour, and with what, so a
            // grey body can be told from a coloured one with grey faces.
            let styled = s.faces.iter().filter(|f| f.style.is_some()).count();
            let over = match s.faces.iter().find_map(|f| f.style) {
                Some((c, _)) if styled > 0 => {
                    format!("  {styled} faces -> rgb({:.2}, {:.2}, {:.2})", c[0], c[1], c[2])
                }
                _ => String::new(),
            };
            println!(
                "  {i:>4}  {:>5} faces {:>5} wires  rgb({:.2}, {:.2}, {:.2}){}{over}  {name}",
                s.faces.len(),
                s.wires.len(),
                s.color[0],
                s.color[1],
                s.color[2],
                if s.alpha < 1.0 { format!(" a{:.2}", s.alpha) } else { String::new() }
            );
        }
        if m.solids.len() > 30 {
            println!("  ... and {} more", m.solids.len() - 30);
        }
        return Ok(Verdict::Ok);
    }
    if mesh.vertices.is_empty() && mesh.edges.is_empty() {
        return Err("nothing to display".into());
    }
    if let Some(path) = &args.shot {
        return shoot(
            &mut mesh,
            &args,
            path,
            Some((&*m, tol)),
            &select::Hierarchy::new(&m.nodes),
        );
    }

    // Every blocking mode (tree, stats, shot) has returned above; the
    // interactive path left through `show`.
    unreachable!("every blocking mode returned above")
}

/// Opens the window first and tessellates behind it, in batches of whole
/// solids, while the window is already interactive.
///
/// The camera is framed on the model's B-rep bounds, known before any
/// triangle, so the view does not shift as batches arrive.
fn show(m: std::sync::Arc<model::Model>, tol: f64, args: Args) -> Result<Verdict, String> {
    let name = std::path::Path::new(&args.path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| args.path.clone());
    let seed = tess::Mesh {
        bbox: m.vertex_bbox,
        mm_per_unit: m.mm_per_unit,
        ..Default::default()
    };
    let section = args.plane(&seed.bbox);

    let (tx, rx) = std::sync::mpsc::channel();
    let worker = m.clone();
    std::thread::Builder::new()
        .name("tessellate".into())
        .spawn(move || {
            // A failed send means the window has gone.
            tess::tessellate_streaming(&worker, tol, |batch| {
                let _ = tx.send(batch);
            });
        })
        .map_err(|e| format!("could not start the tessellator: {e}"))?;

    let check = args.check;
    let hierarchy = select::Hierarchy::new(&m.nodes);
    // The model is kept past the load only to refine close views (see
    // `refine.rs`). That costs its whole size, mostly B-spline control
    // points, for as long as the window is open, so it is reported;
    // `--no-refine` drops it at the end of the load.
    let refine_from = (!args.no_refine).then(|| {
        println!(
            "refine keeping the model to refine close views ({:.0} MB); --no-refine drops it",
            m.heap_bytes() as f64 / 1e6
        );
        (m.clone(), tol)
    });
    viewer::App::new(
        format!("stpview - {name}"),
        seed,
        hierarchy,
        args.colors,
        args.ao,
        args.edges,
        args.supersample,
        refine_from,
        section,
        rx,
        Box::new(move |mesh| report(&m, mesh, tol, check)),
    )
    .aimed(|c| {
        if let Some(v) = &args.view {
            v.aim(c);
        }
    })
    .run()?;
    Ok(Verdict::Ok)
}
/// Everything a load has to say about itself, printed once the mesh is
/// complete (when streaming, after the last batch).
fn report(m: &model::Model, mesh: &tess::Mesh, tol: f64, check: bool) {
    let s = &mesh.stats;
    // How much of the file is repeated components, which the tessellation
    // dedup depends on.
    let unique: std::collections::HashSet<_> = m
        .solids
        .iter()
        .map(|sd| {
            (
                sd.source,
                sd.color.map(f32::to_bits),
                sd.alpha.to_bits(),
            )
        })
        .collect();
    if unique.len() < m.solids.len() {
        println!(
            "dedup  {} unique components across {} occurrences",
            unique.len(),
            m.solids.len()
        );
    }
    println!(
        "edges  {} sharp, {} tangent suppressed",
        s.edges_sharp, s.edges_tangent
    );
    // What it all costs; on large files the mesh is the dominant allocation.
    println!(
        "memory {:.0} MB: {} vertices at {} B, {} indices, {} edge points",
        mesh.bytes() as f64 / 1e6,
        mesh.vertices.len(),
        std::mem::size_of::<tess::Vertex>(),
        mesh.indices.len(),
        mesh.edges.len(),
    );
    // A mesh reaching far outside the vertex bounds means a face was built in
    // the wrong place, which is easy to miss on a large model.
    if mesh.bbox.extent() > m.vertex_bbox.extent() * 1.5 {
        println!(
            "       WARNING mesh extends well past the vertex bounds: {:.0?} .. {:.0?}",
            mesh.bbox.min, mesh.bbox.max
        );
    }
    if check {
        // Welded much finer than the chord tolerance: points from the same
        // curve agree to the last bit, so anything this separates is
        // separate.
        let t = Instant::now();
        let seams = tess::seams(mesh, tol * 1e-3);
        println!(
            "check  {} of {} mesh edges unpaired, {} over-shared, in {:.2}s",
            seams.unpaired,
            seams.total,
            seams.overshared,
            t.elapsed().as_secs_f64()
        );
        // Distinguishes the two reasons a mesh can be open: the file's topology
        // already is, or two faces sampled a shared edge differently.
        println!(
            "       {} of the unpaired run along a B-rep edge; {} B-rep edges used by one face",
            seams.on_brep_edge, s.edges_single_use
        );
        for p in &seams.cracks {
            println!("         crack near {p:.3?}");
        }
        for p in &seams.folds {
            println!("         doubled near {p:.3?}");
        }
        // Whether a section can be capped at all: the count needs every part to
        // be a closed volume, and `seams` cannot say which part a crack is in.
        let closed = tess::closed_parts(mesh, tol * 1e-3);
        let drawn: Vec<(bool, bool)> = mesh
            .parts
            .iter()
            .zip(&closed)
            .filter(|(p, _)| !p.tris.is_empty())
            .map(|(p, c)| (p.cappable, *c))
            .collect();
        println!(
            "       {} of {} parts have no inside, so are not capped; \n             {} of the rest are not watertight",
            drawn.iter().filter(|(v, _)| !v).count(),
            drawn.len(),
            drawn.iter().filter(|(v, c)| *v && !c).count()
        );
        // Whether a section could be capped. Nothing else cares about
        // winding, so this is where an inside-out part shows.
        let w = tess::winding(mesh);
        let decided = w.outward + w.inward;
        println!(
            "       winding {} outward, {} inward ({:.1}%), {} undecided",
            w.outward,
            w.inward,
            if decided > 0 { 100.0 * w.inward as f64 / decided as f64 } else { 0.0 },
            w.undecided
        );
    }
    if s.cdt_failed > 0 {
        println!(
            "       {} faces ear clipped, their outlines refused by the constrained triangulator",
            s.cdt_failed
        );
    }
    if s.cdt_shaken > 0 {
        println!(
            "       {} faces triangulated only after shaking the input",
            s.cdt_shaken
        );
    }
    if s.displaced > 0 {
        println!(
            "       WARNING {} outline points are not where their parameters put them",
            s.displaced
        );
        for x in &s.displacements {
            println!(
                "         {} {} at {:.3?}, uv {:.4?} is {:.3?}",
                x.kind, x.why, x.center, x.uv_size, x.size
            );
        }
    }
    if s.miscovered > 0 {
        println!(
            "       WARNING {} faces triangulated to the wrong area",
            s.miscovered
        );
        for x in &s.miscovers {
            println!(
                "         {} {} at {:.2?} size {:.2?} want {:.4} got {:.4}",
                x.kind, x.why, x.center, x.size, x.uv_size.x, x.uv_size.y
            );
        }
        // The tolerance is relative to the bounding box, not to any face. A
        // narrow face whose two sides are chorded independently until the
        // chords cross cannot be fixed by the reader; report the tolerance
        // that would resolve it.
        if let Some(w) = s.narrowest {
            let mm = w * m.mm_per_unit;
            let want = mm / NARROW_FACE_TOLERANCES;
            if mm < tol * m.mm_per_unit * NARROW_FACE_TOLERANCES {
                print!(
                    "         the narrowest is {mm:.4} mm across, so its two sides cannot be \
                     chorded apart at this tolerance: "
                );
                // A face a millionth of the model wide is degenerate, not
                // under-resolved; no sensible tolerance reaches it.
                //
                // The advice is about that one face only; where the
                // wrong-area faces have other causes, a finer tolerance may
                // not help.
                if want > m.vertex_bbox.extent() * m.mm_per_unit * 1e-6 {
                    println!(
                        "--tol {want:.4} ({:.0}x finer) resolves that face",
                        tol * m.mm_per_unit / want
                    );
                } else {
                    println!("that is degenerate, not under-resolved");
                }
            }
        }
    }
    if s.bounds_unsampled + s.bounds_unprojected > 0 {
        println!(
            "       WARNING {} face bounds lost while sampling ({} too few points, {} would not project)",
            s.bounds_unsampled + s.bounds_unprojected,
            s.bounds_unsampled,
            s.bounds_unprojected
        );
    }
    if s.stray_faces > 0 {
        println!(
            "       WARNING {} faces drawn outside their own outline",
            s.stray_faces
        );
        for x in &s.strays {
            println!(
                "         {} at {:.2?} outline {:.2} -> mesh {:.2}",
                x.kind, x.center, x.uv_size.x, x.uv_size.y
            );
        }
    }
    if s.seam_bands + s.seam_caps > 0 {
        println!(
            "       {} seam bands, {} apex/pole caps",
            s.seam_bands, s.seam_caps
        );
    }
    if s.faces_failed() > 0 {
        println!(
            "       {} of {} faces not tessellated ({} unclosed outline, {} seam, {} degenerate, {} self-intersecting, {} wrong area)",
            s.faces_failed(),
            s.faces_ok + s.faces_failed(),
            s.fail_no_loops,
            s.fail_seam,
            s.fail_degenerate,
            s.fail_earcut,
            s.fail_miscovered
        );
    }
    if s.faces_failed() > 0 {
        // By surface kind: one kind failing wholesale suggests a reader bug, a
        // scatter across kinds suggests the file.
        let mut by_kind: Vec<(&str, usize)> = geom::Surface::KINDS
            .iter()
            .zip(&s.fail_kinds)
            .filter(|(_, n)| **n > 0)
            .map(|(k, n)| (*k, *n))
            .collect();
        by_kind.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
        let list: Vec<String> = by_kind.iter().map(|(k, n)| format!("{k} {n}")).collect();
        println!("       failures by surface: {}", list.join(", "));
        // Where they are and how big; usually slivers far below the tolerance.
        for x in &s.failures {
            println!(
                "         {} {} at {:.3?} size {:.4?} uv {:.3?}",
                x.why, x.kind, x.center, x.size, x.uv_size
            );
        }
        if s.faces_failed() > s.failures.len() {
            println!("         ... and {} more", s.faces_failed() - s.failures.len());
        }
    }
    if s.capped > 0 {
        println!("       {} faces hit the per-face triangle cap", s.capped);
    }
    if m.stats.restyled > 0 {
        println!(
            "       {} items are styled twice over in different colours",
            m.stats.restyled
        );
    }
    if m.stats.styled_faces > 0 {
        println!(
            "       {} of {} faces carry a colour of their own",
            m.stats.styled_faces, m.stats.faces
        );
    }
    if m.stats.translucent > 0 {
        println!(
            "       {} of {} solid instances are transparent",
            m.stats.translucent, m.stats.instances
        );
    }
}



