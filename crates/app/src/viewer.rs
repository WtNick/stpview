//! Window creation, input handling and the event loop.

use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::refine::{Refiner, SETTLE};
use crate::select::{self, ColorMode, Hierarchy, Selection};
use crate::spacemouse::SpaceMouse;
use stpview::camera::Camera;
use stpview::math::Aabb;
use stpview::model::Model;
use stpview::render::{AoLevel, Axis, EdgeMode, Section};
use stpview::tess::{self, Mesh};
use stpview::{RenderOptions, Renderer};

/// What the viewer calls once the last batch has arrived. The caller keeps the
/// model and does the reporting.
pub type OnLoaded = Box<dyn FnOnce(&Mesh)>;

/// How far the cursor may travel between press and release and still count as
/// a click rather than a drag.
const CLICK_SLOP: f64 = 3.0;

#[derive(Default)]
struct Input {
    cursor: PhysicalPosition<f64>,
    last: Option<PhysicalPosition<f64>>,
    orbiting: bool,
    panning: bool,
    shift: bool,
    ctrl: bool,
    /// Whether the left button went down without shift, and how far the
    /// cursor has travelled since. A click that never became a drag selects.
    pressed: bool,
    travelled: f64,
}

impl Input {
    fn end_drag(&mut self) {
        self.orbiting = false;
        self.panning = false;
        self.last = None;
        self.pressed = false;
    }

    fn dragging(&self) -> bool {
        self.orbiting || self.panning
    }
}

pub struct App {
    title: String,
    mesh: Mesh,
    bbox: Aabb,
    /// The assembly the parts belong to, for walking a selection up it.
    hierarchy: Hierarchy,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    camera: Camera,
    input: Input,
    show_faces: bool,
    edges: EdgeMode,
    ao: AoLevel,
    /// Which axis the section is on and which way it faces, or none.
    section: Option<(Axis, f64)>,
    /// Where along that axis, kept across a change of axis so flipping the
    /// direction shows the other half of the same cut.
    offset: Vec<f64>,
    cap: bool,
    ground: bool,
    silhouette: bool,
    /// One flag per mesh part, or empty when everything is drawn.
    visible: Vec<bool>,
    /// What has been clicked.
    selection: Selection,
    /// The part under the cursor, painted faintly.
    hover: Option<usize>,
    /// Whether the cursor has moved since the hover was last picked. Picked
    /// between events, so a fast mouse costs one pick per frame, not per
    /// motion event.
    hover_stale: bool,
    /// How the parts are coloured: as the file says, or to tell them apart.
    colors: ColorMode,
    /// Batches still to arrive from the tessellator thread, so the window
    /// opens before it has finished. `None` once the channel has closed.
    incoming: Option<Receiver<Mesh>>,
    on_loaded: Option<OnLoaded>,
    started: Instant,
    /// The SpaceMouse, if one was found. Opened once the event loop exists,
    /// because its reader thread needs the proxy to wake it.
    puck: Option<SpaceMouse>,
    /// When the puck was last integrated, so its speed is per second, not per
    /// report or frame.
    flown: Instant,
    /// Whether the puck was off centre last time round. While it is, the loop
    /// keeps ticking and `flown` stays fresh; otherwise the loop sleeps and
    /// `flown` goes stale.
    flying: bool,
    /// Said once, not once a frame.
    warned_wheel: bool,
    /// Frames a still view is averaged over: the first in answer to whatever
    /// changed the view, the rest one per redraw after it stops.
    supersample: u32,
    /// The model and its load tolerance, kept until the event loop exists to
    /// start the refiner (see `refine.rs`). `None` for a scene or under
    /// `--no-refine`.
    refine_from: Option<(Arc<Model>, f64)>,
    refiner: Option<Refiner>,
    /// The view last drawn, and since when it has looked like that. A view
    /// still for `SETTLE` is refined, once.
    seen: Option<ViewKey>,
    still_since: Instant,
    refine_due: bool,
}

/// Everything about a view that decides what it would want refined: camera,
/// window size and what is shown. Compared by bits.
#[derive(PartialEq)]
struct ViewKey {
    camera: [u64; 8],
    size: (u32, u32),
    visible: Vec<bool>,
}

impl ViewKey {
    fn of(c: &Camera, size: (u32, u32), visible: &[bool]) -> ViewKey {
        ViewKey {
            camera: [
                c.target.x.to_bits(),
                c.target.y.to_bits(),
                c.target.z.to_bits(),
                c.distance.to_bits(),
                c.yaw.to_bits(),
                c.pitch.to_bits(),
                c.roll.to_bits(),
                (c.fov_y as f64).to_bits(),
            ],
            size,
            visible: visible.to_vec(),
        }
    }
}

/// How often the event loop looks for another batch while loading. A wait
/// rather than a spin, leaving every core to the tessellator.
const POLL: Duration = Duration::from_millis(16);

impl App {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        title: String,
        mesh: Mesh,
        hierarchy: Hierarchy,
        colors: ColorMode,
        ao: AoLevel,
        edges: EdgeMode,
        supersample: u32,
        refine_from: Option<(Arc<Model>, f64)>,
        section: Option<Section>,
        incoming: Receiver<Mesh>,
        on_loaded: OnLoaded,
    ) -> App {
        let bbox = mesh.bbox;
        let c = bbox.center();
        // One offset per axis, each starting at the middle of the model.
        let mut offset = vec![c.x, c.y, c.z];
        let start = section.map(|s| {
            let n = s.normal;
            let (axis, sign) = if n.x.abs() > 0.5 {
                (Axis::X, n.x.signum())
            } else if n.y.abs() > 0.5 {
                (Axis::Y, n.y.signum())
            } else {
                (Axis::Z, n.z.signum())
            };
            offset[axis as usize] = s.offset * sign;
            (axis, sign)
        });
        App {
            title,
            camera: Camera::new(&bbox),
            mesh,
            bbox,
            hierarchy,
            window: None,
            renderer: None,
            input: Input::default(),
            show_faces: true,
            edges,
            ao,
            section: start,
            offset,
            cap: true,
            ground: true,
            silhouette: true,
            visible: Vec::new(),
            selection: Selection::default(),
            hover: None,
            hover_stale: false,
            colors,
            incoming: Some(incoming),
            on_loaded: Some(on_loaded),
            started: Instant::now(),
            puck: None,
            flown: Instant::now(),
            flying: false,
            warned_wheel: false,
            supersample,
            refine_from,
            refiner: None,
            seen: None,
            still_since: Instant::now(),
            refine_due: false,
        }
    }

    /// Points the camera somewhere other than the whole model before the
    /// window opens (`--view`, as for a shot).
    pub fn aimed(mut self, aim: impl FnOnce(&mut Camera)) -> App {
        aim(&mut self.camera);
        self
    }

    /// Notes the view about to be drawn, and restarts the wait for it to
    /// settle if it differs from the last.
    fn note_view(&mut self) {
        let Some(r) = &self.renderer else { return };
        let key = ViewKey::of(&self.camera, r.size(), &self.visible);
        if self.seen.as_ref() != Some(&key) {
            self.seen = Some(key);
            self.still_since = Instant::now();
            self.refine_due = self.refiner.is_some();
        }
    }

    /// Asks for the parts this view looks at closely to be tessellated again,
    /// or drops the refined parts once none are wanted.
    fn plan_refinement(&mut self) {
        let opts = self.options();
        let (Some(rf), Some(r)) = (&mut self.refiner, &mut self.renderer) else { return };
        if !rf.ready() {
            return;
        }
        let view = r.view_tolerance(&self.camera, rf.tol);
        // Only what the view can see. The pass leaves its uniforms behind,
        // as a pick does; every later frame writes its own first.
        let seen = r.parts_in_frame(&self.camera, &opts, stpview::IN_FRAME_SCALE);
        let want = tess::parts_to_refine(&rf.model, &self.mesh.parts, &seen, &view, rf.tol);
        if want.is_empty() {
            if r.refined_bytes() > 0 {
                r.clear_refined();
                rf.forget();
                println!("refine none wanted at this distance; refined parts dropped");
                self.redraw();
            }
            return;
        }
        rf.ask(view, want);
    }

    /// Draws a finished refinement in place of the parts it refines.
    fn apply_refinement(&mut self) {
        let (Some(rf), Some(r)) = (&mut self.refiner, &mut self.renderer) else { return };
        let Some(done) = rf.take() else { return };
        r.set_refined(&done.mesh, &done.parts);
        println!(
            "refine {} parts for the view: {} triangles, {:.1} MB, in {:.2}s",
            done.parts.len(),
            done.mesh.indices.len() / 3,
            r.refined_bytes() as f64 / 1e6,
            done.took.as_secs_f64()
        );
        self.redraw();
    }

    /// Takes whatever the tessellator has finished since the last look.
    ///
    /// Returns whether anything arrived. Each batch is a mesh with its own
    /// ranges, shifted into place by `Mesh::absorb` here and `Gpu::append` on
    /// the GPU side; both shift by the same three totals.
    fn take_batches(&mut self) -> bool {
        let Some(rx) = &self.incoming else { return false };
        let mut got = false;
        loop {
            match rx.try_recv() {
                Ok(batch) => {
                    if let Some(r) = &mut self.renderer {
                        r.append(&batch);
                    }
                    self.mesh.absorb(batch);
                    // A part that has just arrived cannot have been hidden.
                    if !self.visible.is_empty() {
                        self.visible.resize(self.mesh.parts.len(), true);
                    }
                    got = true;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.incoming = None;
                    println!(
                        "tess   {} triangles in {:.2}s",
                        self.mesh.indices.len() / 3,
                        self.started.elapsed().as_secs_f64()
                    );
                    if let Some(f) = self.on_loaded.take() {
                        f(&self.mesh);
                    }
                    // Views are refined against the complete parts.
                    if let Some(rf) = &mut self.refiner {
                        rf.loaded(&self.mesh.parts);
                    }
                    // The triangles are the GPU's now; see `release_uploaded`.
                    let freed = self.mesh.release_uploaded();
                    if freed > 16 << 20 {
                        println!("gpu    {} MB released to the GPU", freed / (1 << 20));
                    }
                    return true;
                }
            }
        }
        got
    }

    /// The plane as the renderer wants it.
    ///
    /// The stored offset is measured along the positive axis whichever way the
    /// plane faces, so flipping the direction keeps the cut in place and swaps
    /// which half survives.
    fn plane(&self) -> Option<Section> {
        let (axis, sign) = self.section?;
        Some(Section {
            normal: axis.normal(sign),
            offset: self.offset[axis as usize] * sign,
        })
    }

    /// The viewer's state as the library wants it. Every render and every
    /// pick goes through here, so what is painted is what can be clicked.
    fn options(&self) -> RenderOptions {
        RenderOptions {
            show_faces: self.show_faces,
            edges: self.edges,
            ao: self.ao,
            section: self.plane(),
            cap: self.cap,
            ground: self.ground,
            silhouette: self.silhouette,
            visible: self.visible.clone(),
            paint: select::paint(
                self.colors,
                &self.hierarchy,
                &self.mesh.parts,
                &self.selection,
                self.hover,
            ),
            supersample: self.supersample,
        }
    }

    /// The part under a pixel, or nothing. The pick is a render pass, so
    /// hidden parts and faces cut away by the section are never picked.
    fn part_at(&mut self, at: PhysicalPosition<f64>) -> Option<usize> {
        let opts = self.options();
        let r = self.renderer.as_mut()?;
        let hit = r.pick(&self.camera, &opts, at.x as f32, at.y as f32)?;
        self.mesh.part_of_vertex(hit.vertex)
    }

    /// A click. Plain, it selects the part under the cursor, or clears the
    /// selection on empty space. With ctrl it toggles the part, and on empty
    /// space leaves the selection alone.
    fn click(&mut self, at: PhysicalPosition<f64>, extend: bool) {
        match (self.part_at(at), extend) {
            (Some(part), true) => self.selection.toggle(part),
            (Some(part), false) => self.selection.set(part),
            (None, true) => return,
            (None, false) => {
                if self.selection.is_empty() {
                    return;
                }
                self.selection.clear();
            }
        }
        self.announce();
    }

    /// Says what is selected now, on the console and in the title bar, and
    /// redraws with it painted.
    fn announce(&mut self) {
        if self.selection.is_empty() {
            println!("select nothing");
        } else {
            println!(
                "select {}",
                self.selection.describe(&self.hierarchy, &self.mesh.parts)
            );
        }
        self.retitle();
        self.redraw();
    }

    /// Up: the selection grows to the component, subassembly or model that
    /// the clicked part belongs to.
    fn widen(&mut self) {
        let nodes = select::nodes(&self.mesh.parts);
        if self.selection.widen(&self.hierarchy, &nodes) {
            self.announce();
        } else if !self.selection.is_empty() {
            println!("select nothing above this");
        }
    }

    /// Down: back again.
    fn narrow(&mut self) {
        let nodes = select::nodes(&self.mesh.parts);
        if self.selection.narrow(&self.hierarchy, &nodes) {
            self.announce();
        }
    }

    /// Picks the part under the cursor if the cursor has moved, and repaints
    /// when it changed. Not while dragging: the part under the cursor at the
    /// press stays lit until release.
    fn refresh_hover(&mut self) {
        if !self.hover_stale || self.input.dragging() {
            return;
        }
        self.hover_stale = false;
        let part = self.part_at(self.input.cursor);
        if part != self.hover {
            self.hover = part;
            self.redraw();
        }
    }

    /// The parts a key acts on: the selection, or failing that the part under
    /// the cursor.
    fn targets(&self) -> Vec<usize> {
        if self.selection.is_empty() {
            self.hover.into_iter().collect()
        } else {
            self.selection.parts().collect()
        }
    }

    fn hide(&mut self, parts: Vec<usize>) {
        if parts.is_empty() {
            println!("parts  nothing is selected or under the cursor");
            return;
        }
        let n = self.mesh.parts.len();
        self.visible.resize(n, true);
        for &p in &parts {
            if p < n {
                self.visible[p] = false;
            }
        }
        // A hidden part cannot stay selected.
        self.selection.clear();
        self.hover = None;
        self.hover_stale = true;
        let shown = self.visible.iter().filter(|v| **v).count();
        println!("parts  {} hidden; {shown} of {n} shown", parts.len());
        self.retitle();
        self.redraw();
    }

    /// Hides everything but these.
    fn isolate(&mut self, parts: Vec<usize>) {
        if parts.is_empty() {
            println!("parts  nothing is selected or under the cursor");
            return;
        }
        let n = self.mesh.parts.len();
        self.visible = (0..n).map(|i| parts.contains(&i)).collect();
        if self.hover.is_some_and(|h| !parts.contains(&h)) {
            self.hover = None;
        }
        self.hover_stale = true;
        println!("parts  {} isolated; {} hidden", parts.len(), n - parts.len());
        self.retitle();
        self.redraw();
    }

    fn show_all(&mut self) {
        self.visible.clear();
        self.hover_stale = true;
        println!("parts  all {} shown", self.mesh.parts.len());
        self.retitle();
        self.redraw();
    }

    fn cycle_colors(&mut self) {
        self.colors = self.colors.next();
        println!("colors {}", self.colors.label());
        self.retitle();
        self.redraw();
    }

    /// The title bar is the readout: what is selected, how many parts are
    /// hidden, how the parts are coloured. Nothing is drawn over the model.
    fn retitle(&self) {
        let Some(w) = &self.window else { return };
        let mut t = self.title.clone();
        if !self.selection.is_empty() {
            t.push_str(" | ");
            t.push_str(&self.selection.describe(&self.hierarchy, &self.mesh.parts));
        }
        let hidden = self.visible.iter().filter(|v| !**v).count();
        if hidden > 0 {
            t.push_str(&format!(" | {hidden} hidden"));
        }
        if self.colors != ColorMode::File {
            t.push_str(&format!(" | colours {}", self.colors.label()));
        }
        w.set_title(&t);
    }

    /// Fits the selection, or everything that is shown, and levels the
    /// horizon. `F` and the puck's first button both land here.
    fn fit_view(&mut self) {
        let boxed = |parts: &mut dyn Iterator<Item = usize>| {
            let mut b = Aabb::EMPTY;
            for i in parts {
                b = b.union(self.mesh.parts[i].bbox);
            }
            b
        };
        // What can be seen, which the depth range is measured against.
        let mut seen = if self.visible.is_empty() {
            self.bbox
        } else {
            let n = self.mesh.parts.len();
            boxed(&mut (0..n).filter(|&i| self.visible.get(i).copied().unwrap_or(true)))
        };
        if seen.is_empty() {
            seen = self.bbox;
        }
        let mut target = if self.selection.is_empty() {
            seen
        } else {
            boxed(&mut self.selection.parts())
        };
        if target.is_empty() {
            target = seen;
        }
        self.camera.fit(&target);
        // The depth range covers what can be seen, not what was framed, so
        // fitting a small part does not clip away its surroundings.
        self.camera.scene_center = seen.center();
        self.camera.scene_radius = (seen.size().length() * 0.5).max(1e-9);
        self.redraw();
    }

    /// X, Y or Z: forward, then back, then off.
    fn cycle_section(&mut self, axis: Axis) {
        self.section = match self.section {
            Some((a, s)) if a == axis && s > 0.0 => Some((axis, -1.0)),
            Some((a, _)) if a == axis => None,
            _ => Some((axis, 1.0)),
        };
        match self.section {
            Some((a, s)) => println!(
                "section: {}{} at {:.3}",
                if s > 0.0 { "+" } else { "-" },
                a.label(),
                self.offset[a as usize]
            ),
            None => println!("section: off"),
        }
        self.redraw();
    }

    /// Slides the plane along its axis, two percent of the model per press
    /// (shift: fine steps).
    fn slide(&mut self, steps: f64, fine: bool) {
        let Some((axis, _)) = self.section else { return };
        let span = self.bbox.size();
        let extent = match axis {
            Axis::X => span.x,
            Axis::Y => span.y,
            Axis::Z => span.z,
        };
        let step = extent * if fine { 0.004 } else { 0.02 };
        let lo = self.bbox.min;
        let hi = self.bbox.max;
        let (lo, hi) = match axis {
            Axis::X => (lo.x, hi.x),
            Axis::Y => (lo.y, hi.y),
            Axis::Z => (lo.z, hi.z),
        };
        // Clamped to the model: past either end the section hides everything
        // or nothing.
        let o = (self.offset[axis as usize] + steps * step).clamp(lo, hi);
        self.offset[axis as usize] = o;
        println!("section: {} at {o:.3}", axis.label());
        self.redraw();
    }

    /// Applies whatever the SpaceMouse is doing, and says whether it moved.
    ///
    /// Time is measured here, not taken from the report rate, so a slow frame
    /// moves the camera by its full duration of travel.
    fn fly(&mut self) -> bool {
        let Some(puck) = &self.puck else { return false };
        let motion = puck.motion();
        let pressed = puck.take_pressed();
        // The first frame of a deflection gets no time at all, rather than a
        // capped elapsed value: `flown` is as old as the window has been idle
        // (the loop sleeps while the puck is still), but a genuinely slow
        // frame must still move by its full duration.
        let dt = if self.flying {
            self.flown.elapsed().as_secs_f64()
        } else {
            0.0
        };
        self.flown = Instant::now();
        self.flying = !motion.is_still();

        let mut moved = false;
        if self.flying {
            self.camera.fly(&motion, dt);
            // The first frame moves nothing, but the loop must start ticking.
            moved = true;
        }
        if pressed != 0 {
            // The first button fits the view, like `F`. The rest are reported
            // as unbound rather than silently swallowed.
            if pressed & 1 != 0 {
                self.fit_view();
                moved = true;
            }
            let others = pressed & !1;
            if others != 0 {
                let names: Vec<String> = (0..32)
                    .filter(|b| others & (1 << b) != 0)
                    .map(|b| (b + 1).to_string())
                    .collect();
                println!("puck   button {} is not bound", names.join(", "));
            }
        }
        moved
    }

    pub fn run(mut self) -> Result<(), String> {
        // A user-event loop so the SpaceMouse and refine threads can wake it
        // from `Wait`.
        let event_loop = EventLoop::with_user_event()
            .build()
            .map_err(|e| format!("event loop: {e}"))?;
        event_loop.set_control_flow(ControlFlow::Wait);

        let proxy = event_loop.create_proxy();
        if let Some((model, tol)) = self.refine_from.take() {
            let wake = proxy.clone();
            self.refiner = Some(Refiner::new(model, tol, move || {
                let _ = wake.send_event(());
            }));
        }
        match SpaceMouse::open(move || {
            let _ = proxy.send_event(());
        }) {
            Ok(puck) => {
                println!("puck   {}", puck.name());
                self.puck = Some(puck);
            }
            // No puck is the ordinary case; one that cannot be opened (usually
            // held by 3DxWare) is reported.
            Err(e) if e == "no SpaceMouse found" => {}
            Err(e) => println!("puck   {e}"),
        }

        event_loop
            .run_app(&mut self)
            .map_err(|e| format!("event loop: {e}"))
    }

    fn redraw(&self) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(&self.title)
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0));

        // A window gets its class's icon, which winit leaves empty, so load
        // resource id 1 (see crates/app/build.rs) explicitly. The title bar's
        // small icon and the taskbar's big one are set separately.
        #[cfg(windows)]
        let attrs = {
            use winit::platform::windows::{IconExtWindows, WindowAttributesExtWindows};
            use winit::window::Icon;
            // A build without the resource compiler has no icon; not an error.
            let attrs = match Icon::from_resource(1, None) {
                Ok(icon) => attrs.with_window_icon(Some(icon)),
                Err(_) => attrs,
            };
            match Icon::from_resource(1, None) {
                Ok(icon) => attrs.with_taskbar_icon(Some(icon)),
                Err(_) => attrs,
            }
        };

        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("error: could not create window: {e}");
                event_loop.exit();
                return;
            }
        };
        // The instance is built here because a presentable one needs the
        // display handle, which only the windowing layer has.
        let display = event_loop.owned_display_handle();
        let instance = stpview::wgpu::Instance::new(
            stpview::wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(display)),
        );
        let size = window.inner_size();
        match Renderer::windowed(
            &instance,
            window.clone(),
            size.width.max(1),
            size.height.max(1),
            &self.mesh,
        ) {
            Ok(r) => {
                if r.samples() == 1 {
                    println!("gpu    multisampling unavailable; edges will look harder");
                }
                self.renderer = Some(r);
            }
            Err(e) => {
                eprintln!("error: {e}");
                event_loop.exit();
                return;
            }
        }
        self.window = Some(window);
        self.retitle();
        // Loading is the only thing that wants the loop to come round on its
        // own; every other frame is drawn in answer to an event.
        event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + POLL));
        self.redraw();
    }

    /// A report arrived from a worker thread; waking the loop is the point,
    /// and `about_to_wait` runs next.
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: ()) {}

    /// Between events: take whatever the tessellator has finished, look up
    /// what the cursor has come to rest on, and move the camera by however
    /// long the puck has been held.
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let loading = self.incoming.is_some();
        if loading && self.take_batches() {
            self.redraw();
        }
        self.refresh_hover();
        // A held puck keeps the loop ticking on its own, so camera speed does
        // not depend on the device's report rate.
        let flying = if self.fly() {
            self.redraw();
            true
        } else {
            false
        };
        // A view still for long enough is refined, once the load is complete.
        let mut settle_at = None;
        if !loading {
            self.apply_refinement();
            if self.refine_due && !self.input.dragging() && !flying {
                let due = self.still_since + SETTLE;
                if Instant::now() < due {
                    settle_at = Some(due);
                } else if !self.refiner.as_ref().is_some_and(|r| r.busy()) {
                    // While busy it stays due; the worker's answer wakes
                    // the loop.
                    self.refine_due = false;
                    self.plan_refinement();
                }
            }
        }
        event_loop.set_control_flow(if loading || flying {
            ControlFlow::WaitUntil(Instant::now() + POLL)
        } else if let Some(t) = settle_at {
            ControlFlow::WaitUntil(t)
        } else {
            ControlFlow::Wait
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(size.width, size.height);
                }
                self.redraw();
            }

            WindowEvent::ModifiersChanged(m) => {
                self.input.shift = m.state().shift_key();
                self.input.ctrl = m.state().control_key();
            }

            // A release outside the window never arrives, so drop any drag
            // when the pointer leaves or focus goes.
            WindowEvent::CursorLeft { .. } | WindowEvent::Focused(false) => {
                self.input.end_drag();
                // Nothing is under a cursor that is not here.
                self.hover_stale = false;
                if self.hover.take().is_some() {
                    self.redraw();
                }
            }

            WindowEvent::CursorEntered { .. } => {
                self.input.last = None;
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                if down {
                    match button {
                        // Shift+left pans, matching most CAD viewers.
                        MouseButton::Left if self.input.shift => self.input.panning = true,
                        MouseButton::Left => {
                            self.input.orbiting = true;
                            self.input.pressed = true;
                            self.input.travelled = 0.0;
                        }
                        MouseButton::Right | MouseButton::Middle => self.input.panning = true,
                        _ => {}
                    }
                    self.input.last = Some(self.input.cursor);
                } else {
                    // A left press and release without the view moving is a
                    // click; orbit and select share the button.
                    let click = button == MouseButton::Left
                        && self.input.pressed
                        && self.input.travelled < CLICK_SLOP;
                    let at = self.input.cursor;
                    self.input.end_drag();
                    if click {
                        self.click(at, self.input.ctrl);
                    }
                    // The view may have moved under the cursor.
                    self.hover_stale = true;
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.input.cursor = position;
                if !self.input.dragging() {
                    self.hover_stale = true;
                }
                let Some(last) = self.input.last else {
                    self.input.last = Some(position);
                    return;
                };
                let height = self
                    .renderer
                    .as_ref()
                    .map(|r| r.size().1.max(1) as f64)
                    .unwrap_or(800.0);
                let dx = (position.x - last.x) / height;
                let dy = (position.y - last.y) / height;
                self.input.travelled += (position.x - last.x).hypot(position.y - last.y);
                self.input.last = Some(position);
                if self.input.orbiting {
                    self.camera.orbit(dx * 3.0, dy * 3.0);
                    self.redraw();
                } else if self.input.panning {
                    self.camera.pan(dx, dy);
                    self.redraw();
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                // Ignore the wheel while the puck is deflected: 3DxWare can
                // drive the wheel from the puck, which would zoom twice.
                if self.flying {
                    if !self.warned_wheel {
                        self.warned_wheel = true;
                        println!(
                            "puck   ignoring wheel events while the puck is deflected: 3DxWare \
                             is driving the scroll wheel as well, so the zoom was being applied \
                             twice. Quit it from its tray icon to use the wheel again."
                        );
                    }
                    return;
                }
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 60.0,
                };
                self.camera.zoom(steps);
                // Zooming moves the model under a cursor that stayed put.
                self.hover_stale = true;
                self.redraw();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed {
                    return;
                }
                match event.logical_key.as_ref() {
                    // Esc deselects; Q quits.
                    Key::Named(NamedKey::Escape) => {
                        if !self.selection.is_empty() {
                            self.selection.clear();
                            self.announce();
                        }
                    }
                    Key::Character("q" | "Q") => event_loop.exit(),
                    Key::Named(NamedKey::ArrowUp) => self.widen(),
                    Key::Named(NamedKey::ArrowDown) => self.narrow(),
                    Key::Character("h" | "H") => {
                        let t = self.targets();
                        self.hide(t);
                    }
                    Key::Character("i" | "I") => {
                        let t = self.targets();
                        self.isolate(t);
                    }
                    Key::Character("a" | "A") => self.show_all(),
                    Key::Character("c" | "C") => self.cycle_colors(),
                    Key::Character("f" | "F") => self.fit_view(),
                    Key::Character("e" | "E") => {
                        self.edges = self.edges.next();
                        // Never leave the user staring at an empty window.
                        if self.edges == EdgeMode::None && !self.show_faces {
                            self.show_faces = true;
                        }
                        println!("{}", self.edges.label());
                        self.redraw();
                    }
                    Key::Character("s" | "S") => {
                        self.show_faces = !self.show_faces;
                        if !self.show_faces && self.edges == EdgeMode::None {
                            self.edges = EdgeMode::Sharp;
                        }
                        self.redraw();
                    }
                    Key::Character("o" | "O") => {
                        self.ao = self.ao.next();
                        println!("occlusion: {}", self.ao.label());
                        self.redraw();
                    }
                    Key::Character("x" | "X") => self.cycle_section(Axis::X),
                    Key::Character("y" | "Y") => self.cycle_section(Axis::Y),
                    Key::Character("z" | "Z") => self.cycle_section(Axis::Z),
                    Key::Character("[" | "{") => self.slide(-1.0, self.input.shift),
                    Key::Character("]" | "}") => self.slide(1.0, self.input.shift),
                    Key::Character("k" | "K") => {
                        self.cap = !self.cap;
                        println!("section: cut {}", if self.cap { "filled" } else { "hollow" });
                        self.redraw();
                    }
                    Key::Character("g" | "G") => {
                        self.ground = !self.ground;
                        println!("ground: {}", if self.ground { "on" } else { "off" });
                        self.redraw();
                    }
                    Key::Character("l" | "L") => {
                        self.silhouette = !self.silhouette;
                        println!("silhouette: {}", if self.silhouette { "on" } else { "off" });
                        self.redraw();
                    }
                    _ => {}
                }
            }

            WindowEvent::RedrawRequested => {
                self.note_view();
                let opts = self.options();
                if let Some(r) = &mut self.renderer {
                    r.render(&self.camera, &opts);
                    // A still view accumulates one jittered frame per redraw
                    // until converged; any view change restarts it.
                    if !r.converged() {
                        self.redraw();
                    }
                }
            }

            _ => {}
        }
    }
}
