//! stpview-core: STEP (ISO 10303-21) import, B-rep geometry and tessellation.
//!
//! Everything up to and including triangles, with no GPU: a Part 21 tokenizer,
//! the entity graph resolved into placed B-rep solids, curve and surface
//! evaluation, and a tessellator with its own watertightness checks. STL is
//! read straight into the same mesh.

// `!(a > b)` is deliberately not `a <= b`: it is the NaN guard. A degenerate
// input yields NaN as readily as zero, and NaN must take the failing branch.
#![allow(clippy::neg_cmp_op_on_partial_ord)]
// Parallel arrays and basis-index arithmetic read clearest as indexed loops.
#![allow(clippy::needless_range_loop)]

pub mod assets;
pub mod geom;
pub mod math;
pub mod model;
pub mod scene;
pub mod step;
pub mod stl;
pub mod tess;

pub use assets::{Assets, Format, ImportError};
pub use scene::{Instance, InstanceId, MeshHandle, Scene};
