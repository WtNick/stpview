//! Re-tessellating what a still view looks at closely, finer, on a worker
//! thread.
//!
//! The load tolerance suits the whole model in the window; zoomed in, it
//! shows as polygons. Once the view settles, the visible faces are
//! tessellated again to about half a pixel on screen and drawn in place of
//! their load tessellation. The coarse picture stays up until the finer one
//! is ready, and moving the view simply asks again once it stops.

use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use stpview::model::Model;
use stpview::tess::{self, Mesh, Part, ViewTolerance};

/// How long the view has to be still before it is refined: long enough that a
/// pause mid-orbit does not start a wasted tessellation.
pub const SETTLE: Duration = Duration::from_millis(150);

struct Job {
    view: ViewTolerance,
    parts: Vec<usize>,
    base: Arc<Vec<Part>>,
}

/// A finished refinement: `mesh` holds one part per entry of `parts`, each
/// standing in for that part of the model.
pub struct Done {
    pub parts: Vec<usize>,
    pub mesh: Mesh,
    pub took: Duration,
}

pub struct Refiner {
    /// The model and its load tolerance, in model units.
    pub model: Arc<Model>,
    pub tol: f64,
    /// The mesh's parts as loaded, which decide which faces a view can
    /// reach. Empty until the load is complete; nothing is refined before.
    base: Arc<Vec<Part>>,
    jobs: Sender<Job>,
    done: Receiver<Done>,
    /// A job is out and has not come back.
    busy: bool,
    /// The view the last job was for, so the same view is never asked twice.
    asked: Option<ViewTolerance>,
}

impl Refiner {
    /// Starts the worker. `wake` is called from the worker once a refinement
    /// is ready, so the event loop does not sleep until the next input.
    pub fn new(model: Arc<Model>, tol: f64, wake: impl Fn() + Send + 'static) -> Refiner {
        let (jobs, inbox) = channel::<Job>();
        let (outbox, done) = channel::<Done>();
        let worker = model.clone();
        let spawned = std::thread::Builder::new().name("refine".into()).spawn(move || {
            while let Ok(mut job) = inbox.recv() {
                // Only the newest queued view matters.
                loop {
                    match inbox.try_recv() {
                        Ok(newer) => job = newer,
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => return,
                    }
                }
                let t = Instant::now();
                let mesh = tess::tessellate_for_view(&worker, tol, &job.view, &job.base, &job.parts);
                let finished = Done {
                    parts: job.parts,
                    mesh,
                    took: t.elapsed(),
                };
                if outbox.send(finished).is_err() {
                    return;
                }
                wake();
            }
        });
        if let Err(e) = spawned {
            println!("refine could not start a worker ({e}); zooming in will not refine");
        }
        Refiner {
            model,
            tol,
            base: Arc::new(Vec::new()),
            jobs,
            done,
            busy: false,
            asked: None,
        }
    }

    /// Takes the mesh's parts once the load is whole.
    pub fn loaded(&mut self, parts: &[Part]) {
        self.base = Arc::new(parts.to_vec());
    }

    /// Whether there is anything to refine yet.
    pub fn ready(&self) -> bool {
        !self.base.is_empty()
    }

    /// Whether a job is out.
    pub fn busy(&self) -> bool {
        self.busy
    }

    /// Asks for these parts to be tessellated for `view`, unless that exact
    /// view was the last one asked for. Returns whether it asked.
    pub fn ask(&mut self, view: ViewTolerance, mut parts: Vec<usize>) -> bool {
        if self.asked == Some(view) {
            return false;
        }
        // In model order, so consecutive solids are tessellated as runs.
        parts.sort_unstable();
        self.asked = Some(view);
        let job = Job {
            view,
            parts,
            base: self.base.clone(),
        };
        self.busy = self.jobs.send(job).is_ok();
        self.busy
    }

    /// Forgets the last view asked for, so the same view can be asked again
    /// (after the refined parts were dropped, say).
    pub fn forget(&mut self) {
        self.asked = None;
    }

    /// A finished refinement, if one has come back.
    pub fn take(&mut self) -> Option<Done> {
        match self.done.try_recv() {
            Ok(d) => {
                self.busy = false;
                Some(d)
            }
            Err(_) => None,
        }
    }
}
