//! One model to one square picture: what the Windows shell thumbnail provider
//! and `stpview --thumb` both run.
//!
//! It lives in the library because a thumbnail that fails inside Explorer
//! fails silently; one copy reachable from the command line keeps a broken
//! thumbnail reproducible.

use std::sync::OnceLock;

use crate::camera::Camera;
use crate::renderer::{RenderOptions, Renderer};
use crate::Assets;

/// The wgpu instance, built once for the life of the process.
///
/// A shell surrogate draws a whole folder through one `dllhost.exe`, and the
/// instance is the only model-independent part worth sharing cheaply. The
/// adapter, device and pipelines are still built per picture; sharing them
/// would mean splitting `Gpu` into mesh-dependent and independent parts.
fn instance() -> &'static wgpu::Instance {
    static INSTANCE: OnceLock<wgpu::Instance> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env())
    })
}

/// Imports `bytes` and renders a `size` by `size` thumbnail: tightly packed
/// RGBA8, row by row from the top left, as [`Renderer::capture`] returns it.
///
/// `label` names the source in error messages only.
///
/// The ground plane is off: one shadow map cannot serve parts of wildly
/// different sizes, and a catalogue wants a clean field. Otherwise the
/// product's defaults, so a thumbnail looks like `--shot` of the same file.
pub fn render(bytes: &[u8], size: u32, label: &str) -> Result<Vec<u8>, String> {
    let size = size.max(1);
    let mesh = Assets::new()
        .import_bytes(bytes, label)
        .map_err(|e| e.to_string())?;

    let mut camera = Camera::new(&mesh.bbox);
    camera.close_in(&mesh.bbox);

    let opts = RenderOptions {
        ground: false,
        ..RenderOptions::default()
    };

    let mut r = Renderer::offscreen_on(instance(), size, size, &mesh)?;
    r.capture(&camera, &opts)
}
