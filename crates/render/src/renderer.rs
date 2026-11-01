//! The public renderer and the options a frame is drawn with.
//!
//! [`Gpu`] underneath is the frame graph and knows nothing about scenes or
//! files. This wraps it: build one from a scene or a mesh, point a camera,
//! get pixels (or a picture on an attached surface).

use std::sync::mpsc::Receiver;

use crate::camera::Camera;
use crate::render::{AoLevel, EdgeMode, Frame, Gpu, Paint, Pick, Section};
use crate::scene::Scene;
use crate::tess::{Mesh, ViewTolerance, REFINE_FLOOR};

/// How far a chord may stray from the true surface on screen, in pixels, once
/// a view has had its parts tessellated for it.
///
/// Half a pixel: a silhouette that far off is invisible under the
/// supersampling, whereas the model's own tolerance can leave a close-up hole
/// visibly polygonal.
pub const REFINE_PX: f64 = 0.5;

/// How much smaller the frame is drawn to see which parts own a pixel of it,
/// before a view is refined: see [`Renderer::parts_in_frame`]. A part owning
/// no pixel at a quarter of the size is too small to show facets at full size.
pub const IN_FRAME_SCALE: u32 = 4;

/// Everything about how a frame looks, minus the camera.
///
/// The owned twin of the internal `Frame`, which borrows; a consumer keeps
/// one in its app state.
#[derive(Clone)]
pub struct RenderOptions {
    pub show_faces: bool,
    pub edges: EdgeMode,
    pub ao: AoLevel,
    pub section: Option<Section>,
    /// Fill the cut when sectioned. Needs closed solids; see the cap notes in
    /// the render module.
    pub cap: bool,
    pub ground: bool,
    pub silhouette: bool,
    /// Per-part visibility, in mesh part order; empty means everything.
    pub visible: Vec<bool>,
    /// A paint per part, in the same order, laid over the part's own colour
    /// on its surface and edges (a selection highlight). Missing entries are
    /// unpainted; empty is the fast path.
    pub paint: Vec<Paint>,
    /// How many jittered frames a still view is averaged over; 1 is none. A
    /// window gets one more frame per [`Renderer::render`] until
    /// [`Renderer::converged`]; a capture draws them all before reading back.
    pub supersample: u32,
}

impl Default for RenderOptions {
    /// The CLI's defaults; every stored reference view is recorded with them.
    fn default() -> RenderOptions {
        RenderOptions {
            show_faces: true,
            edges: EdgeMode::Sharp,
            ao: AoLevel::Strong,
            section: None,
            cap: true,
            ground: true,
            silhouette: true,
            visible: Vec::new(),
            paint: Vec::new(),
            supersample: crate::render::SUPERSAMPLE,
        }
    }
}

/// A renderer bound to one target: a window surface or an offscreen texture.
///
/// Constructed from what it draws. Changing scenes means constructing a new
/// one (cheap next to tessellation); streaming grows the current contents
/// via [`Renderer::append`].
pub struct Renderer {
    gpu: Gpu,
}

impl Renderer {
    /// Headless, rendering into a texture; [`Renderer::capture`] reads it
    /// back. No window, display or event loop.
    pub fn offscreen(width: u32, height: u32, mesh: &Mesh) -> Result<Renderer, String> {
        let gpu = pollster::block_on(Gpu::offscreen(width, height, mesh))?;
        Ok(Renderer { gpu })
    }

    /// Headless, on a `wgpu::Instance` the caller keeps rather than one built
    /// per renderer.
    ///
    /// Creating an instance is a large share of a cold start and independent
    /// of the model, so a process drawing many pictures (the shell thumbnail
    /// provider) pays it once. The picture is the one [`Renderer::offscreen`]
    /// draws.
    pub fn offscreen_on(
        instance: &wgpu::Instance,
        width: u32,
        height: u32,
        mesh: &Mesh,
    ) -> Result<Renderer, String> {
        let gpu = pollster::block_on(Gpu::offscreen_on(instance, width, height, mesh))?;
        Ok(Renderer { gpu })
    }

    /// Headless, from a scene, GPU-instanced: each unique mesh is uploaded
    /// once and drawn once per placement. [`Scene::bake`] is the reference
    /// semantics; a test holds the instanced picture to the baked one.
    pub fn offscreen_scene(width: u32, height: u32, scene: &Scene) -> Result<Renderer, String> {
        let gpu = pollster::block_on(Gpu::offscreen_scene(width, height, scene))?;
        Ok(Renderer { gpu })
    }

    /// On a surface the caller provides -- any window handle wgpu can take.
    ///
    /// The instance comes from the caller too: one that can present needs the
    /// display handle, which only the windowing layer has.
    pub fn windowed(
        instance: &wgpu::Instance,
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
        mesh: &Mesh,
    ) -> Result<Renderer, String> {
        let gpu = pollster::block_on(Gpu::for_surface(instance, target, width, height, mesh))?;
        Ok(Renderer { gpu })
    }

    /// Draws to the attached target. On a window this presents; offscreen it
    /// leaves the frame ready for [`Renderer::capture`].
    ///
    /// Drawing an unchanged view adds one more frame to its supersampled
    /// average, until [`Renderer::converged`].
    pub fn render(&mut self, camera: &Camera, opts: &RenderOptions) {
        self.gpu.render(&Renderer::frame(camera, opts));
    }

    /// Whether `opts.supersample` frames of the last view have been drawn; an
    /// event loop redraws until this is true, then sleeps.
    pub fn converged(&self) -> bool {
        self.gpu.converged()
    }

    /// Renders and reads back one frame as tightly-packed RGBA8, row by row
    /// from the top left.
    pub fn capture(&mut self, camera: &Camera, opts: &RenderOptions) -> Result<Vec<u8>, String> {
        self.gpu.capture(&Renderer::frame(camera, opts))
    }

    /// What is under a pixel: a world-space point and the vertex that won the
    /// depth test (`Mesh::part_of_vertex` gives its part). The pick is a
    /// render pass with the same options, so it agrees with the picture.
    pub fn pick(&mut self, camera: &Camera, opts: &RenderOptions, x: f32, y: f32) -> Option<Pick> {
        self.gpu.pick(&Renderer::frame(camera, opts), x, y)
    }

    /// Appends a batch of finished solids, for progressive loading.
    pub fn append(&mut self, batch: &Mesh) {
        self.gpu.append(batch);
    }

    /// The tessellation `camera` needs on this renderer's target, for a model
    /// tessellated at `tol`: chords within [`REFINE_PX`] on screen, never
    /// finer than `tol * REFINE_FLOOR`. Input to
    /// [`tess::tessellate_for_view`](crate::tess::tessellate_for_view).
    pub fn view_tolerance(&self, camera: &Camera, tol: f64) -> ViewTolerance {
        let (w, h) = self.size();
        camera.view_tolerance(w, h, REFINE_PX, tol * REFINE_FLOOR)
    }

    /// Draws these parts from their own tessellation: `mesh` holds one part
    /// per entry of `parts`, in order, each replacing that part of the
    /// original mesh. Replaces any earlier refinement. Ignored by a renderer
    /// over a scene.
    pub fn set_refined(&mut self, mesh: &Mesh, parts: &[usize]) {
        self.gpu.set_refined(mesh, parts);
    }

    /// Which parts own a pixel of this view drawn at a `scale`th of the size:
    /// the candidates for finer tessellation. Hidden parts own none.
    pub fn parts_in_frame(&mut self, camera: &Camera, opts: &RenderOptions, scale: u32) -> Vec<bool> {
        self.gpu.parts_in_frame(&Renderer::frame(camera, opts), scale)
    }

    /// Draws every part from the original mesh again.
    pub fn clear_refined(&mut self) {
        self.gpu.clear_refined();
    }

    /// What the refined parts hold on the GPU, in bytes.
    pub fn refined_bytes(&self) -> u64 {
        self.gpu.refined_bytes()
    }

    /// Drains everything a streaming import has produced so far into the
    /// renderer and the accumulated mesh. Returns whether anything arrived
    /// and whether the stream has ended.
    pub fn drain_stream(&mut self, rx: &Receiver<Mesh>, into: &mut Mesh) -> (bool, bool) {
        let mut got = false;
        loop {
            match rx.try_recv() {
                Ok(batch) => {
                    self.append(&batch);
                    into.absorb(batch);
                    got = true;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => return (got, false),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return (got, true),
            }
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.gpu.resize(width, height);
    }

    /// Target size in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.gpu.config.width, self.gpu.config.height)
    }

    /// MSAA sample count actually in use; 1 means the adapter refused it.
    pub fn samples(&self) -> u32 {
        self.gpu.samples()
    }

    /// The only place a `Frame` is made from the public types, so the pick
    /// pass always agrees with the picture.
    fn frame<'a>(camera: &'a Camera, opts: &'a RenderOptions) -> Frame<'a> {
        Frame {
            camera,
            show_faces: opts.show_faces,
            edges: opts.edges,
            ao: opts.ao,
            section: opts.section,
            cap: opts.cap,
            ground: opts.ground,
            silhouette: opts.silhouette,
            visible: &opts.visible,
            paint: &opts.paint,
            supersample: opts.supersample,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Xform;
    use crate::math::{v3, Vec3};
    use crate::tess::{Mesh, Vertex};

    /// A box with flat per-face normals. Axis-aligned normals are exact points
    /// of the octahedral codec, so the baked path's decode-rotate-encode round
    /// trip is bit-stable and byte comparisons are legitimate.
    fn box_mesh(w: f64, d: f64, h: f64, color: [f32; 3]) -> Mesh {
        let mut m = Mesh {
            mm_per_unit: 1.0,
            ..Default::default()
        };
        let c = [
            v3(0.0, 0.0, 0.0), v3(w, 0.0, 0.0), v3(w, d, 0.0), v3(0.0, d, 0.0),
            v3(0.0, 0.0, h), v3(w, 0.0, h), v3(w, d, h), v3(0.0, d, h),
        ];
        let faces: [([usize; 4], Vec3); 6] = [
            ([3, 2, 1, 0], v3(0.0, 0.0, -1.0)),
            ([4, 5, 6, 7], v3(0.0, 0.0, 1.0)),
            ([0, 1, 5, 4], v3(0.0, -1.0, 0.0)),
            ([2, 3, 7, 6], v3(0.0, 1.0, 0.0)),
            ([1, 2, 6, 5], v3(1.0, 0.0, 0.0)),
            ([3, 0, 4, 7], v3(-1.0, 0.0, 0.0)),
        ];
        for (quad, n) in faces {
            let base = m.vertices.len() as u32;
            for k in quad {
                m.vertices.push(Vertex::new(c[k], n, color, 1.0));
            }
            m.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        for p in c {
            m.bbox.add_point(p);
        }
        m
    }

    /// A 90-degree turn about Z as exact data: `rotate_axis` would leave
    /// `cos(pi/2) = 6.1e-17` in the matrix, rounded differently by the baked
    /// and instanced paths.
    fn quarter_turn_z() -> Xform {
        Xform {
            x: v3(0.0, 1.0, 0.0),
            y: v3(-1.0, 0.0, 0.0),
            z: v3(0.0, 0.0, 1.0),
            origin: Vec3::ZERO,
        }
    }

    /// The GPU-instanced scene matches its CPU bake byte for byte through the
    /// whole frame graph, including the section cap over a mirrored instance.
    /// All transforms are exact in f32 and f64, so any differing byte is a
    /// real disagreement.
    #[test]
    fn an_instanced_scene_matches_its_bake_to_the_byte() {
        let a = std::sync::Arc::new(box_mesh(4.0, 4.0, 4.0, [0.8, 0.5, 0.2]));
        let b = std::sync::Arc::new(box_mesh(2.0, 6.0, 1.0, [0.3, 0.6, 0.9]));

        let mut scene = Scene::new();
        scene.add(a.clone(), Xform::IDENTITY);
        scene.add(a.clone(), Xform::translate(v3(8.0, 0.0, 0.0)));
        scene.add(
            a.clone(),
            Xform::translate(v3(0.0, 8.0, 2.0)).compose(&Xform::scale(2.0)),
        );
        scene.add(
            b.clone(),
            Xform::translate(v3(16.0, 4.0, 0.0)).compose(&quarter_turn_z()),
        );
        // Mirrored: negative determinant, so its cap parity runs through the
        // swapped stencil pipeline.
        let mut mirror = Xform::translate(v3(-6.0, 0.0, 0.0));
        mirror.x = v3(-1.0, 0.0, 0.0);
        scene.add(b.clone(), mirror);

        let camera = Camera::new(&scene.bounds());
        let mut opts = RenderOptions::default();
        // Cut through everything, cap on: the parity count exposes a
        // mirrored instance drawn through the wrong pipeline.
        opts.section = Some(crate::render::Section {
            normal: v3(0.0, 0.0, 1.0),
            offset: 3.0,
        });

        let mut native = Renderer::offscreen_scene(480, 360, &scene).expect("gpu");
        let got = native.capture(&camera, &opts).expect("native frame");

        let mut baked = Renderer::offscreen(480, 360, &scene.bake()).expect("gpu");
        let want = baked.capture(&camera, &opts).expect("baked frame");

        assert_eq!(got.len(), want.len());
        let differing = got
            .iter()
            .zip(&want)
            .filter(|(g, w)| g != w)
            .count();
        assert_eq!(
            differing, 0,
            "{differing} of {} bytes differ between the instanced scene and its bake",
            got.len()
        );
    }

    /// Switching occlusion on moves no pixel of a convex solid. Guards two
    /// faults: flipping a normal by the sign of its view-space z (wrong under
    /// perspective for a face that faces the eye but leans away from the view
    /// axis), and a face occluding itself when seen at a slant.
    ///
    /// The camera looks along +y, turned three degrees towards -x and tipped
    /// down, so the box's -x face faces the eye while leaning away from the
    /// view axis. The top face must be in view too: the first fault shows on
    /// the neighbour along an edge, not on the flipped face itself.
    #[test]
    fn occlusion_leaves_a_convex_solid_alone() {
        let mesh = box_mesh(2.0, 16.0, 4.0, [0.7, 0.7, 0.7]);
        let mut camera = Camera::new(&mesh.bbox);
        let eye = v3(-4.0, -8.0, 7.0);
        let (yaw, pitch) = ((-87f64).to_radians(), 15f64.to_radians());
        camera.yaw = yaw;
        camera.pitch = pitch;
        camera.distance = 10.0;
        let dir = v3(pitch.cos() * yaw.cos(), pitch.cos() * yaw.sin(), pitch.sin());
        camera.target = eye - dir * camera.distance;
        // Check the premise: the face is towards the eye, and the view axis
        // runs away from it.
        let (_, _, forward) = camera.basis();
        assert!((camera.eye() - eye).length() < 1e-9);
        let face = v3(-1.0, 0.0, 0.0);
        assert!(face.dot(eye - v3(0.0, 8.0, 2.0)) > 0.0, "the -x face is not in view");
        assert!(face.dot(forward) > 0.0, "the -x face does not lean away from the axis");
        assert!(eye.z > mesh.bbox.max.z, "the top face is not in view");

        let (w, h) = (400u32, 200u32);
        let mut r = Renderer::offscreen(w, h, &mesh).expect("gpu");
        let base = RenderOptions {
            ground: false,
            ..RenderOptions::default()
        };
        let lit = r
            .capture(&camera, &RenderOptions { ao: AoLevel::Off, ..base.clone() })
            .expect("frame");
        let occluded = r.capture(&camera, &base).expect("frame");

        // It has to be in the picture for any of this to mean anything.
        let background = &lit[0..3];
        let solid = (0..lit.len() / 4)
            .filter(|&i| &lit[i * 4..i * 4 + 3] != background)
            .count();
        assert!(
            solid > (w * h / 10) as usize,
            "only {solid} pixels of the box are in view"
        );
        // To the byte: on a box every sample misses, occlusion is exactly
        // 1.0, and lighting times one moves no bit.
        let moved = (0..lit.len() / 4)
            .filter(|&i| occluded[i * 4..i * 4 + 3] != lit[i * 4..i * 4 + 3])
            .count();
        assert_eq!(
            moved, 0,
            "occlusion moved {moved} of the {solid} pixels of a convex solid"
        );
    }

    /// Two boxes side by side, one of them placed off the origin, so the
    /// second part is drawn through a twin that has to carry the placement.
    fn two_boxes() -> Scene {
        let a = std::sync::Arc::new(box_mesh(4.0, 4.0, 4.0, [0.8, 0.5, 0.2]));
        let b = std::sync::Arc::new(box_mesh(4.0, 4.0, 4.0, [0.3, 0.6, 0.9]));
        let mut scene = Scene::new();
        scene.add(a, Xform::IDENTITY);
        scene.add(b, Xform::translate(v3(8.0, 0.0, 0.0)));
        scene
    }

    /// Paint at amount zero is no paint: the frame is the unpainted one to the
    /// byte. Stored views depend on it; a zero amount gets no twin, and
    /// `c + (p - c) * a` at `a = 0` is exactly `c`.
    #[test]
    fn zero_paint_is_the_unpainted_frame_to_the_byte() {
        let scene = two_boxes();
        let camera = Camera::new(&scene.bounds());
        let mut r = Renderer::offscreen_scene(320, 240, &scene).expect("gpu");
        let plain = r.capture(&camera, &RenderOptions::default()).expect("frame");
        let zero = RenderOptions {
            paint: vec![Paint::NONE, Paint { rgb: [1.0, 0.0, 0.0], amount: 0.0 }],
            ..RenderOptions::default()
        };
        let got = r.capture(&camera, &zero).expect("frame");
        assert_eq!(got, plain, "a zero amount moved pixels");
        // And back again, after a painted frame has been drawn in between:
        // the twins come and go without leaving anything behind.
        let red = RenderOptions {
            paint: vec![Paint { rgb: [1.0, 0.0, 0.0], amount: 1.0 }],
            ..RenderOptions::default()
        };
        r.capture(&camera, &red).expect("frame");
        let again = r.capture(&camera, &RenderOptions::default()).expect("frame");
        assert_eq!(again, plain, "pixels moved after a painted frame");
    }

    /// Paint reaches the pixels of the part it names, and no others; the
    /// second box's pixels differ from the first's only if its twin carries
    /// its placement.
    #[test]
    fn paint_reaches_the_part_it_names_and_nothing_else() {
        let scene = two_boxes();
        let camera = Camera::new(&scene.bounds());
        let mut r = Renderer::offscreen_scene(320, 240, &scene).expect("gpu");
        // No floor: with it the whole lower half of the frame is a surface,
        // and the claim here is about the background.
        let base = RenderOptions {
            ground: false,
            ..RenderOptions::default()
        };
        let plain = r.capture(&camera, &base).expect("frame");
        let paint = |i: usize| {
            let mut paint = vec![Paint::NONE; 2];
            paint[i] = Paint {
                rgb: [1.0, 0.0, 0.0],
                amount: 1.0,
            };
            RenderOptions {
                paint,
                ..base.clone()
            }
        };
        let first = r.capture(&camera, &paint(0)).expect("frame");
        let second = r.capture(&camera, &paint(1)).expect("frame");

        let moved = |a: &[u8], b: &[u8]| -> Vec<usize> {
            (0..a.len() / 4)
                .filter(|&i| a[i * 4..i * 4 + 3] != b[i * 4..i * 4 + 3])
                .collect()
        };
        let (m1, m2) = (moved(&plain, &first), moved(&plain, &second));
        assert!(!m1.is_empty(), "painting the first box changed nothing");
        assert!(!m2.is_empty(), "painting the second box changed nothing");
        assert_ne!(m1, m2, "both boxes were painted, so the twin ignored its placement");
        // The background is whatever the top-left pixel is: no floor, and
        // the camera frames the boxes in the middle.
        let background = &plain[0..3];
        for i in m1.iter().chain(&m2) {
            assert_ne!(
                &plain[i * 4..i * 4 + 3],
                background,
                "pixel {i} was background and paint reached it"
            );
        }
    }

    /// The first frame of a supersampled view is the unaveraged frame to the
    /// byte: pixel centres, weight one, and a lossless trip through the
    /// half-float history. It is the frame drawn while the view moves.
    #[test]
    fn the_first_frame_of_an_average_is_the_frame_there_always_was() {
        let mesh = box_mesh(4.0, 6.0, 3.0, [0.8, 0.5, 0.2]);
        let camera = Camera::new(&mesh.bbox);
        let averaged = RenderOptions::default();
        assert!(averaged.supersample > 1, "the default does not average");
        let single = RenderOptions {
            supersample: 1,
            ..averaged.clone()
        };
        let want = Renderer::offscreen(320, 240, &mesh)
            .expect("gpu")
            .capture(&camera, &single)
            .expect("frame");
        let mut r = Renderer::offscreen(320, 240, &mesh).expect("gpu");
        r.gpu.draw(&Renderer::frame(&camera, &averaged), None);
        let got = r.gpu.read_back().expect("frame");
        let moved = (0..got.len() / 4)
            .filter(|&i| got[i * 4..i * 4 + 3] != want[i * 4..i * 4 + 3])
            .count();
        assert_eq!(moved, 0, "{moved} pixels of the first averaged frame differ");
    }

    /// A still view converges after `supersample` renders and stays converged;
    /// a view change restarts it; one frame a view converges at once.
    #[test]
    fn a_still_view_converges_and_anything_that_moves_it_starts_again() {
        let mesh = box_mesh(4.0, 6.0, 3.0, [0.8, 0.5, 0.2]);
        let mut camera = Camera::new(&mesh.bbox);
        let opts = RenderOptions::default();
        let mut r = Renderer::offscreen(160, 120, &mesh).expect("gpu");
        for i in 1..opts.supersample {
            r.render(&camera, &opts);
            assert!(!r.converged(), "converged after {i} of {} frames", opts.supersample);
        }
        r.render(&camera, &opts);
        assert!(r.converged(), "not converged after every frame of the average");
        r.render(&camera, &opts);
        assert!(r.converged(), "drawing a converged view again un-converged it");

        camera.orbit(0.01, 0.0);
        r.render(&camera, &opts);
        assert!(!r.converged(), "a moved camera did not start the average again");

        let single = RenderOptions {
            supersample: 1,
            ..opts
        };
        r.render(&camera, &single);
        assert!(r.converged(), "one frame is the whole of an unaveraged view");
    }

    /// Averaging leaves edges as sharp as one frame draws them. Edge quads
    /// already carry their pixel coverage and are not jittered, so their
    /// average is themselves.
    #[test]
    fn an_edge_is_as_sharp_averaged_as_in_one_frame() {
        // One line, alone, in the plane the camera faces, at a slope that
        // crosses the rows at every phase.
        let mut mesh = Mesh {
            mm_per_unit: 1.0,
            ..Default::default()
        };
        mesh.edges = vec![[-10.3, 0.0, -4.1], [9.7, 0.0, 3.9]];
        for p in &mesh.edges {
            mesh.bbox.add_point(v3(p[0] as f64, p[1] as f64, p[2] as f64));
        }
        mesh.parts.push(crate::tess::Part {
            name: String::new(),
            node: None,
            alpha: 1.0,
            cappable: false,
            color: [0.5; 3],
            tris: 0..0,
            verts: 0..0,
            wire: 0..0,
            sharp: 0..2,
            tangent: 0..0,
            bbox: mesh.bbox,
            faces: Vec::new(),
        });
        let mut camera = Camera::new(&mesh.bbox);
        camera.yaw = -std::f64::consts::FRAC_PI_2;
        camera.pitch = 0.0;
        let opts = RenderOptions {
            show_faces: false,
            ground: false,
            supersample: 1,
            ..RenderOptions::default()
        };
        let (w, h) = (400u32, 300u32);
        let mut r = Renderer::offscreen(w, h, &mesh).expect("gpu");
        let one = r.capture(&camera, &opts).expect("frame");
        let averaged = r
            .capture(&camera, &RenderOptions { supersample: 16, ..opts.clone() })
            .expect("frame");
        // The brightest of each row the line crosses, in the red channel.
        let peaks = |px: &[u8]| -> Vec<u8> {
            (0..h as usize)
                .map(|y| (0..w as usize).map(|x| px[(y * w as usize + x) * 4]).max().unwrap_or(0))
                .collect()
        };
        let (a, b) = (peaks(&one), peaks(&averaged));
        let bg = one[0];
        let lit: Vec<usize> = (0..h as usize).filter(|&y| a[y] > bg + 16).collect();
        assert!(lit.len() > 40, "the line crosses only {} rows", lit.len());
        let worst = lit.iter().map(|&y| a[y].abs_diff(b[y])).max().unwrap_or(0);
        assert!(
            worst <= 2,
            "averaging moved the line's peak by up to {worst} levels: it was blurred"
        );
    }

    /// Two boxes as one mesh of two parts, the second eight units along x.
    fn two_parts() -> (Mesh, Mesh) {
        let a = box_mesh(4.0, 4.0, 4.0, [0.8, 0.5, 0.2]);
        let b = box_mesh(4.0, 4.0, 4.0, [0.3, 0.6, 0.9]);
        let mut both = Mesh::default();
        both.absorb_placed(&a, &Xform::IDENTITY, None);
        both.absorb_placed(&b, &Xform::translate(v3(8.0, 0.0, 0.0)), None);
        (both, b)
    }

    /// A refined part replaces its original in every pass: refined with the
    /// same triangles it moves no pixel, with others it moves the picture,
    /// and clearing restores every byte.
    #[test]
    fn a_refined_part_is_drawn_in_place_of_its_own() {
        let (both, b) = two_parts();
        let camera = Camera::new(&both.bbox);
        let mut opts = RenderOptions::default();
        opts.supersample = 1;
        // Cut through both, so the cap's parity count sees the refined part
        // as well.
        opts.section = Some(crate::render::Section {
            normal: v3(0.0, 0.0, 1.0),
            offset: 3.0,
        });
        let mut r = Renderer::offscreen(320, 240, &both).expect("gpu");
        let plain = r.capture(&camera, &opts).expect("frame");

        let mut same = Mesh::default();
        same.absorb_placed(&b, &Xform::translate(v3(8.0, 0.0, 0.0)), None);
        r.set_refined(&same, &[1]);
        assert!(r.refined_bytes() > 0);
        let got = r.capture(&camera, &opts).expect("frame");
        assert!(got == plain, "refining a part with its own triangles moved the picture");

        let mut moved = Mesh::default();
        moved.absorb_placed(&b, &Xform::translate(v3(8.0, 0.0, 2.0)), None);
        r.set_refined(&moved, &[1]);
        let elsewhere = r.capture(&camera, &opts).expect("frame");
        assert!(elsewhere != plain, "the refined part was not drawn");

        r.clear_refined();
        assert_eq!(r.refined_bytes(), 0);
        let back = r.capture(&camera, &opts).expect("frame");
        assert!(back == plain, "clearing the refined part did not restore the picture");
    }

    /// A pick on a refined part names the original part and its vertices,
    /// not the refined buffers.
    #[test]
    fn a_pick_on_a_refined_part_names_that_part() {
        let (both, b) = two_parts();
        let camera = Camera::new(&both.bbox);
        let opts = RenderOptions {
            ground: false,
            supersample: 1,
            ..RenderOptions::default()
        };
        let (w, h) = (320u32, 240u32);
        let mut r = Renderer::offscreen(w, h, &both).expect("gpu");
        let plain = r.capture(&camera, &opts).expect("frame");
        // Lifted clear of where it was, so some pixels show it only refined.
        let mut moved = Mesh::default();
        moved.absorb_placed(&b, &Xform::translate(v3(8.0, 0.0, 6.0)), None);
        r.set_refined(&moved, &[1]);
        let refined = r.capture(&camera, &opts).expect("frame");
        let background = &plain[0..3];
        let at = (0..(w * h) as usize)
            .find(|&i| &plain[i * 4..i * 4 + 3] == background && &refined[i * 4..i * 4 + 3] != background)
            .expect("the lifted part covers no new pixel");
        let (x, y) = ((at % w as usize) as f32 + 0.5, (at / w as usize) as f32 + 0.5);
        let hit = r.pick(&camera, &opts, x, y).expect("nothing under a pixel the part covers");
        assert!(
            both.parts[1].verts.contains(&hit.vertex),
            "the pick answered vertex {}, which is not the second part's",
            hit.vertex
        );
    }
}
