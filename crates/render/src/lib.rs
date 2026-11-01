//! stpview: a CAD rendering library.
//!
//! Import STEP and STL models (via [`stpview_core`]), compose them, and render
//! them with a wgpu frame graph: image-based lighting, SSAO, stencil-parity
//! section capping, edge overlay, picking and per-part paint, on a window
//! surface or headless into pixels.

// `!(a > b)` is a deliberate NaN guard; see crates/core/src/lib.rs.
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::needless_range_loop)]

pub mod camera;
pub mod env;
pub mod render;
pub mod renderer;
pub mod thumb;

// Lets this crate address core modules as `crate::math` etc., and lets a
// consumer depend on one crate rather than two.
pub use stpview_core::{assets, geom, math, model, scene, step, stl, tess};

// The API most consumers want, at the root: import, compose, render.
pub use camera::Camera;
pub use render::Paint;
pub use renderer::{RenderOptions, Renderer, IN_FRAME_SCALE, REFINE_PX};
pub use stpview_core::{Assets, Format, ImportError, Instance, InstanceId, MeshHandle, Scene};

// Re-exported so an embedding app uses the same wgpu version as this crate.
pub use wgpu;
