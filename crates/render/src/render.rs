//! GPU setup and the frame graph.
//!
//! A frame is a forward pass of multisampled shading, preceded when needed by
//! a single-sample prepass (view-space normal and linear depth) that feeds
//! screen-space occlusion, its 4x4 blur and the silhouette, and followed by
//! the section cap, weighted-blended transparency and, for a still view, the
//! supersampling average. The draw side is in `passes.rs`.


pub(crate) use wgpu::util::DeviceExt;

pub(crate) use crate::camera::Camera;
pub(crate) use crate::math::{v3, Aabb, Mat4, Vec3};
pub(crate) use std::ops::Range;

pub(crate) use crate::tess::{Mesh, Part, Vertex};

const SAMPLE_COUNT: u32 = 4;
/// Depth without a stencil, which is all the shaded passes need.
///
/// The section cap needs a stencil, but `Depth24PlusStencil8` costs depth
/// precision and changes how z-fighting resolves. So the stencil is taken only
/// from `Depth32FloatStencil8`; where that is missing the section still cuts
/// but the cut is not filled.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const DEPTH_STENCIL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32FloatStencil8;
/// Half float rather than 8-bit: occlusion is raised to a power, which turns
/// 8-bit steps into visible mottling on large flat surfaces.
const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
const SSAO_KERNEL: usize = 16;

/// Side of the shadow map. One orthographic box covers the whole model, so
/// this is all the resolution the shadow has; filtering softens the staircase.
const SHADOW_SIZE: u32 = 2048;

/// Direction the shadow light shines, in world space.
///
/// Fixed in the world, unlike the camera-relative shading lights: the ground
/// is a world object, and a shadow that swung with the camera would read as
/// the world turning. Slightly off vertical so the shadow is not hidden
/// exactly under the part.
const SHADOW_DIR: Vec3 = v3(-0.30, -0.22, -1.0);

/// Fraction of its distance to the eye by which an edge line is pulled forward
/// so it draws over the face it borders. Small enough that an edge never
/// overtakes a genuinely nearer part of the model.
pub(crate) const EDGE_EYE_NUDGE: f32 = 8.0e-4;

/// The depth test every pass drawn from the camera uses, and the depth its
/// buffers are cleared to. Depth is reversed (near 1, far 0) so a float buffer
/// resolves about the same fraction of the distance everywhere; see
/// `Mat4::perspective_reversed`. The orthographic shadow map is not reversed.
const NEARER: wgpu::CompareFunction = wgpu::CompareFunction::GreaterEqual;
const FAR_DEPTH: f32 = 0.0;

/// How far the shaded surfaces are pushed back in depth: one pixel's worth of
/// each triangle's own depth slope. Negative, since depth is reversed.
///
/// The fixed edge nudge is not enough on a slanted face: the face is sampled
/// up to half a pixel from the line's centre, where it has risen by half a
/// pixel of its slope, and the line breaks into dashes. A slope-scaled bias is
/// zero square-on and grows as the face tilts. It sits on the surfaces because
/// depth bias applies to triangles only.
const SURFACE_BIAS: wgpu::DepthBiasState = wgpu::DepthBiasState {
    constant: 0,
    slope_scale: -1.0,
    clamp: 0.0,
};

/// Occlusion radius as a fraction of the camera's distance to its target, so
/// the effect keeps a roughly constant screen-space size.
const AO_RADIUS_FRACTION: f64 = 0.035;

/// Position, an octahedral normal, and a display-referred colour with its
/// opacity: twenty bytes rather than forty. See `tess::Vertex`.
const SURFACE_ATTRS: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Unorm16x2, 2 => Unorm8x4];
const EDGE_ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];

/// One placement of a mesh, as the shader's `Inst` block reads it: three rows
/// of the model matrix with the translation in `.w`, a colour multiplier, and
/// a paint laid over the colour. The identity is exact ones and zeros, which
/// keeps the single-model path bit-exact.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    rows: [[f32; 4]; 3],
    tint: [u8; 4],
    /// A colour the shader blends the vertex colour towards, and how far:
    /// `Paint::key`. All zero is no paint, exactly (`paint_over`).
    paint: [u8; 4],
}

impl InstanceRaw {
    const IDENTITY: InstanceRaw = InstanceRaw {
        rows: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        tint: [255; 4],
        paint: NO_PAINT,
    };

    /// `Xform` keeps columns; the shader wants rows.
    fn from_xform(x: &crate::geom::Xform, tint: Option<[f32; 4]>) -> InstanceRaw {
        let t = tint.unwrap_or([1.0; 4]);
        let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        InstanceRaw {
            rows: [
                [x.x.x as f32, x.y.x as f32, x.z.x as f32, x.origin.x as f32],
                [x.x.y as f32, x.y.y as f32, x.z.y as f32, x.origin.y as f32],
                [x.x.z as f32, x.y.z as f32, x.z.z as f32, x.origin.z as f32],
            ],
            tint: [b(t[0]), b(t[1]), b(t[2]), b(t[3])],
            paint: NO_PAINT,
        }
    }

    /// The same placement, painted.
    fn painted(mut self, paint: [u8; 4]) -> InstanceRaw {
        self.paint = paint;
        self
    }
}

/// Locations 3 to 7, continuing after the widest per-vertex layout, so one
/// shader-side `Inst` struct serves every pass whatever its vertex attributes.
const INSTANCE_ATTRS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
    3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Unorm8x4, 7 => Unorm8x4
];

/// One mesh's share of the buffers and the placements that draw it.
///
/// The single-model path is one slot with one exact-identity instance, and
/// streaming appends into it; a scene is a slot per unique mesh, each drawn
/// once per placement by the GPU. `parts` indexes into `Gpu::parts`, a single
/// flat list, so per-part visibility is model-wide.
#[derive(Clone)]
struct Slot {
    parts: Range<usize>,
    instances: Range<u32>,
    /// Index into `instances` where the mirrored placements (negative
    /// determinant) begin; they are sorted to the tail. Only the section
    /// cap's stencil count can tell a front face from a back one, so only it
    /// draws the two groups differently.
    mirror_from: u32,
}

/// Some of the model's parts tessellated again for a view, drawn in place of
/// their own triangles and edges.
///
/// Separate buffers, since they are replaced whole whenever the view settles,
/// while the model's buffers only grow. A part is the unit because parts share
/// no edges, so a finer one drops in beside its neighbours without a crack.
struct Refined {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    edge_buf: wgpu::Buffer,
    edge_bind: Option<wgpu::BindGroup>,
    /// The refined parts, with ranges into the buffers above.
    parts: Vec<Part>,
    /// Which of the model's parts each one stands in for.
    of: Vec<usize>,
    /// Per model part, whether its edges are drawn from here instead: a
    /// refined part carries every edge of its solid, sampled for the view.
    replaced: Vec<bool>,
    /// The stretches of the model's own index buffer that the faces drawn
    /// here replace, sorted and apart. The rest of a refined part is still
    /// drawn from the model's buffers, around these.
    holes: Vec<Range<u32>>,
}


/// `r` less `holes` -- sorted and apart -- as the pieces of it left.
fn around(holes: &[Range<u32>], r: Range<u32>) -> Vec<Range<u32>> {
    let mut out = Vec::new();
    let mut at = r.start;
    // The first hole that ends past the start; the rest follow in order.
    let first = holes.partition_point(|h| h.end <= r.start);
    for h in &holes[first..] {
        if h.start >= r.end {
            break;
        }
        if h.start > at {
            out.push(at..h.start);
        }
        at = at.max(h.end);
    }
    if at < r.end {
        out.push(at..r.end);
    }
    out
}

impl Refined {
    /// `r` less the holes, as the pieces of it still to draw.
    fn around_holes(&self, r: Range<u32>) -> Vec<Range<u32>> {
        around(&self.holes, r)
    }

    /// The visibility and paint the frame gives the parts these stand in for.
    fn flags(&self, visible: &[bool], paint: &[[u8; 4]]) -> (Vec<bool>, Vec<[u8; 4]>) {
        let vis = self.of.iter().map(|&i| visible.get(i).copied().unwrap_or(true)).collect();
        let keys = self.of.iter().map(|&i| paint.get(i).copied().unwrap_or(NO_PAINT)).collect();
        (vis, keys)
    }
}

/// One slot's placements re-issued with a paint on them.
///
/// The forward pass draws a painted part through these instead of the slot's
/// own records, so the paint rides on the instance and the mesh is untouched.
/// Kept until the frame asks for a different set of paints.
#[derive(Clone, PartialEq, Eq)]
struct Twin {
    slot: usize,
    paint: [u8; 4],
    instances: Range<u32>,
}

/// Side of the pick target, in pixels of the window it stands in for.
///
/// A pick draws the model with the projection shifted so the pixel under the
/// cursor lands in the middle of this square at the same scale, so the buffer
/// is tiny whatever the window size. Odd, for a true centre pixel; the slack
/// either side lets a click a little off a thin part still find it, since the
/// multisampled frame and this pass disagree about silhouette pixels.
const PICK_SIZE: u32 = 17;
/// One vertex index and a world position. Integer because a float would round
/// the vertex index on a large assembly.
const PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;

/// What a buffer that a streamed load may have to grow needs on top of its own
/// usage: read from, to copy the old contents across, and written to, for the
/// batch that goes on the end.
const GROW: wgpu::BufferUsages = wgpu::BufferUsages::COPY_SRC.union(wgpu::BufferUsages::COPY_DST);

/// What kind of growable buffer: its stride, its debug label and what the GPU
/// may bind it as. The three the mesh streams into, as named constants.
#[derive(Clone, Copy)]
struct BufKind {
    stride: u64,
    label: &'static str,
    usage: wgpu::BufferUsages,
}
const VERTICES: BufKind = BufKind {
    stride: std::mem::size_of::<Vertex>() as u64,
    label: "vertices",
    usage: wgpu::BufferUsages::VERTEX,
};
const INDICES: BufKind = BufKind {
    stride: 4,
    label: "indices",
    usage: wgpu::BufferUsages::INDEX,
};
/// Edge overlay points: three f32 each. Storage, for the vertex stage that
/// draws each segment as a quad; vertex, for the rasteriser's own lines where
/// that stage cannot run.
const EDGES: BufKind = BufKind {
    stride: 12,
    label: "edges",
    usage: wgpu::BufferUsages::VERTEX.union(wgpu::BufferUsages::STORAGE),
};

/// The edge buffer as the line quads read it, or nothing where they cannot.
///
/// Bound whole, up to what the adapter allows one binding to span, and remade
/// whenever a streamed load replaces the buffer with a larger one.
fn edge_bind_group(device: &wgpu::Device, layouts: &Layouts, buf: &wgpu::Buffer) -> Option<wgpu::BindGroup> {
    let layout = layouts.lines.as_ref()?;
    let limit = device.limits().max_storage_buffer_binding_size;
    let size = buf.size().min(limit / 4 * 4);
    Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("edges"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: buf,
                offset: 0,
                size: wgpu::BufferSize::new(size),
            }),
        }],
    }))
}

/// Bytes per row of the pick readback, which a texture copy wants aligned.
fn pick_stride() -> u32 {
    (PICK_SIZE * 16).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    eye: [f32; 4],
    lights: [[f32; 4]; 3],
    params: [f32; 4],
    ssao: [f32; 4],
    proj_info: [f32; 4],
    /// The section plane: xyz is a unit normal and w the offset along it, with
    /// `dot(n, p) <= w` the half-space that survives. A zero normal is off.
    section: [f32; 4],
    /// x: how far the cap quad reaches; yzw: the middle of the model, where it
    /// is centred, so a part far from the origin is still covered.
    cap: [f32; 4],
    /// World to the shadow light's clip space.
    light_vp: [[f32; 4]; 4],
    /// x: height of the ground plane, y: how far it reaches before it fades out,
    /// z: 1.0 when it is drawn at all, w: one shadow map texel in clip units.
    ground: [f32; 4],
    /// Irradiance of the studio, as nine spherical-harmonic coefficients: the
    /// whole of the diffuse lighting, with no texture.
    sh: [[f32; 4]; 9],
    kernel: [[f32; 4]; SSAO_KERNEL],
    /// How far this frame's projection is shifted for supersampling: xy in
    /// normalised device coordinates, zw unused. Zero on a view's first frame,
    /// the one drawn while it moves.
    jitter: [f32; 4],
}

/// Sixteen points in the +z hemisphere, packed towards the origin so nearby
/// geometry dominates. Deterministic, so a frame always looks the same.
fn ssao_kernel() -> [[f32; 4]; SSAO_KERNEL] {
    let mut state: u32 = 0x9E37_79B9;
    let mut rnd = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((state >> 8) as f64) / 16_777_216.0
    };
    let mut out = [[0.0f32; 4]; SSAO_KERNEL];
    for (i, slot) in out.iter_mut().enumerate() {
        let dir = loop {
            let v = v3(rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0, rnd());
            let len = v.length_sq();
            if (1e-4..=1.0).contains(&len) {
                break v.normalize();
            }
        };
        let t = i as f64 / SSAO_KERNEL as f64;
        let d = dir * (0.1 + 0.9 * t * t);
        *slot = [d.x as f32, d.y as f32, d.z as f32, 0.0];
    }
    out
}

/// Where the light looks from, and where the floor goes.
///
/// Separate from the rest of the setup so it can be checked without a GPU:
/// geometry outside the light's box silently casts nothing.
fn shadow_frame(bbox: &Aabb) -> (Mat4, f32, f32) {
    let center = bbox.center();
    // The bounding sphere rather than the box, so the fit does not depend on
    // which way the light happens to point.
    let radius = (bbox.size().length() * 0.5).max(1e-9);
    let dir = SHADOW_DIR.normalize();
    // Standing the light off by a full radius puts the whole model in front of
    // the near plane whichever way it faces, with the same margin behind.
    let light_vp = Mat4::orthographic(radius as f32, radius as f32, 0.0, (radius * 2.0) as f32)
        .mul(&Mat4::look_at(center - dir * radius, center, v3(0.0, 1.0, 0.0)));
    // The floor sits a hair below the model so a flat underside does not
    // z-fight it, and reaches well past the model so it can fade.
    let ground_z = (bbox.min.z - radius * 0.002) as f32;
    let ground_radius = (radius * 6.0) as f32;
    (light_vp, ground_z, ground_radius)
}

/// Puts one window pixel in the middle of the pick target, at the same scale.
///
/// Applied on the left of the ordinary view-projection, so the pick sees the
/// model through the same camera as the frame and only the framing differs.
/// See `pick_transform_frames_the_pixel_it_was_given`.
fn pick_transform(w: f32, h: f32, px: f32, py: f32, size: f32) -> Mat4 {
    // Middle of the chosen pixel, in normalised device coordinates. Framebuffer
    // y runs down the screen and clip-space y runs up it. Floored first, or a
    // fractional cursor position puts the clicked pixel off the target's
    // centre.
    let cx = 2.0 * (px.floor() + 0.5) / w - 1.0;
    let cy = 1.0 - 2.0 * (py.floor() + 0.5) / h;
    let (sx, sy) = (w / size, h / size);
    Mat4([
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-cx * sx, -cy * sy, 0.0, 1.0],
    ])
}

/// The section cap's coverage, before it is thinned.
///
/// The parity count leaks wherever the mesh has a crack (an unpaired edge in
/// `--check`): the entry never cancels, and the cap is drawn along the crack.
/// A leak is a thin stroke and a cut is an area, so the mask is written to its
/// own texture and eroded before use.
const CAP_MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// The running average of a supersampled view. Half float, since eight bits
/// would round every blended step and drift over sixteen of them.
const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// How many frames a still view is averaged over, unless the caller says
/// otherwise: see [`Frame::supersample`].
pub const SUPERSAMPLE: u32 = 16;

/// Where the `i`th frame of a supersampled view samples its pixels, as an
/// offset from their centres in pixels, each coordinate in [-0.5, 0.5).
///
/// The first is the centre itself, so the frame drawn while the view moves is
/// unshifted. The rest are the Halton sequence in bases two and three, which
/// covers the square evenly at every length.
fn jitter(i: u32) -> [f32; 2] {
    if i == 0 {
        return [0.0, 0.0];
    }
    let halton = |mut i: u32, base: u32| {
        let (mut f, mut r) = (1.0f64, 0.0f64);
        while i > 0 {
            f /= base as f64;
            r += f * (i % base) as f64;
            i /= base;
        }
        r
    };
    [(halton(i, 2) - 0.5) as f32, (halton(i, 3) - 0.5) as f32]
}

/// Everything a frame is a picture of. Frames with the same key differ only
/// in where they sample their pixels, so a still view averages them; any
/// difference starts the average over. Compared by bits: a camera that moved
/// by the last bit has moved.
#[derive(Clone, PartialEq)]
struct FrameKey {
    camera: [u64; 13],
    flags: [u8; 7],
    section: Option<[u64; 4]>,
    visible: Vec<bool>,
    paint: Vec<[u8; 4]>,
    /// Bumped whenever what is on the GPU changes under the same camera: a
    /// streamed batch, a refined part, a resize.
    generation: u64,
}

impl FrameKey {
    fn of(f: &Frame, paint: &[[u8; 4]], generation: u64) -> FrameKey {
        let c = f.camera;
        FrameKey {
            camera: [
                c.target.x.to_bits(),
                c.target.y.to_bits(),
                c.target.z.to_bits(),
                c.distance.to_bits(),
                c.yaw.to_bits(),
                c.pitch.to_bits(),
                c.roll.to_bits(),
                (c.fov_y as f64).to_bits(),
                c.scene_radius.to_bits(),
                c.scene_center.x.to_bits(),
                c.scene_center.y.to_bits(),
                c.scene_center.z.to_bits(),
                f.supersample as u64,
            ],
            flags: [
                f.show_faces as u8,
                f.edges as u8,
                f.ao as u8,
                f.cap as u8,
                f.ground as u8,
                f.silhouette as u8,
                0,
            ],
            section: f.section.map(|s| {
                [s.normal.x.to_bits(), s.normal.y.to_bits(), s.normal.z.to_bits(), s.offset.to_bits()]
            }),
            visible: f.visible.to_vec(),
            paint: paint.to_vec(),
            generation,
        }
    }
}

/// How far a still view has got with its average.
#[derive(Default)]
struct Accumulation {
    key: Option<FrameKey>,
    /// Frames in the average so far, and how many it is to have.
    done: u32,
    target: u32,
}

/// Where the weighted sum of the transparent surfaces goes. Half float: the
/// weight spans four orders of magnitude.
const ACCUM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// And the running product of (1 - alpha), which needs the same range at the
/// small end.
const REVEAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

mod passes;
mod targets;
#[cfg(test)]
mod tests;

pub(crate) use targets::*;

pub struct Gpu {
    /// Absent when rendering off screen (`--shot`); everything downstream of
    /// the target view is the same as in the viewer.
    surface: Option<wgpu::Surface<'static>>,
    /// The target when there is no surface. Carries `COPY_SRC` so it can be
    /// read back.
    offscreen: Option<wgpu::Texture>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    targets: Targets,
    samples: u32,
    gbuffer_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    layouts: Layouts,
    surface_pipeline: wgpu::RenderPipeline,
    edge_pipeline: wgpu::RenderPipeline,
    wire_pipeline: wgpu::RenderPipeline,
    gbuffer_pipeline: wgpu::RenderPipeline,
    ssao_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    /// Marks, in the stencil, where the section plane passes through material.
    /// Absent when the adapter has no float depth-stencil format.
    cap_mark_pipeline: Option<(wgpu::RenderPipeline, wgpu::RenderPipeline)>,
    /// And fills those pixels with the cut.
    cap_fill_pipeline: Option<wgpu::RenderPipeline>,
    silhouette_pipeline: wgpu::RenderPipeline,
    oit_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    /// Picking: a fixed-size target and the buffer it is read back through,
    /// untouched by a resize.
    pick_pipeline: wgpu::RenderPipeline,
    pick_color: wgpu::Texture,
    pick_view: wgpu::TextureView,
    pick_depth: wgpu::TextureView,
    pick_read: wgpu::Buffer,
    bare_bind: wgpu::BindGroup,
    /// The mesh split by component, for drawing a subset of it.
    parts: Vec<Part>,
    /// Which stretch of the buffers each unique mesh owns, and the placements
    /// that draw it. One identity slot for a single model; one per unique
    /// mesh for a scene.
    slots: Vec<Slot>,
    /// Every placement, slot by slot, mirrored ones at each slot's tail --
    /// and behind them the painted twins, whenever a frame asks for any.
    instance_buf: wgpu::Buffer,
    /// The placements as written, kept so a twin can be derived from them
    /// without reading the buffer back.
    base_instances: Vec<InstanceRaw>,
    /// Which painted twins the buffer's tail currently holds.
    twins: Vec<Twin>,
    /// The cap-mark stencil pipeline for mirrored placements: reflection
    /// turns front faces into back ones, so the parity count must swap
    /// increment and decrement or a mirrored solid subtracts itself.
    cap_mark_mirror: Option<wgpu::RenderPipeline>,
    /// Whether any part is transparent. When none is, the transparency passes
    /// are skipped.
    any_transparent: bool,
    /// What was visible when the shadow map was last filled. The map is
    /// reused, so hiding a part has to redraw it.
    shadow_for: Vec<bool>,
    /// The light's box, the floor and the cap quad are derived from these; a
    /// streamed load rebuilds them as batches arrive.
    bounds: Aabb,
    /// Radius and centre of the quad the cap is drawn on, from the model.
    scene_radius: f32,
    shadow_pipeline: wgpu::RenderPipeline,
    ground_pipeline: wgpu::RenderPipeline,
    /// World to the light's clip space, fixed once the model is loaded.
    light_vp: Mat4,
    /// Where the floor sits and how far it reaches.
    ground_z: f32,
    ground_radius: f32,
    /// One shadow texel measured in the light's clip space, for the filter.
    shadow_texel: f32,
    scene_center: [f32; 3],
    shadow_view: wgpu::TextureView,
    shadow_sampler: wgpu::Sampler,
    uniform_buf: wgpu::Buffer,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
    edge_buf: wgpu::Buffer,
    edge_count: u32,
    /// `edge_buf` bound for the line quads to read, rebuilt whenever a
    /// streamed load replaces the buffer. `None` where the adapter draws the
    /// rasteriser's lines instead.
    edge_bind: Option<wgpu::BindGroup>,
    /// How much the three buffers can hold, in elements, and how much of the
    /// vertex one is used. A streamed load grows them by doubling.
    vertex_cap: u32,
    index_cap: u32,
    edge_cap: u32,
    vertex_total: u32,
    manual_srgb: bool,
    kernel: [[f32; 4]; SSAO_KERNEL],
    sh: [[f32; 4]; 9],
    env_view: wgpu::TextureView,
    env_sampler: wgpu::Sampler,
    /// Supersampling: one blends a finished frame into the running average,
    /// the other puts the average on the target.
    accumulate_pipeline: wgpu::RenderPipeline,
    present_pipeline: wgpu::RenderPipeline,
    accumulation: Accumulation,
    /// The window's surface gave no texture last time. Until one comes, a
    /// still view reports itself converged so a viewer does not spin.
    stalled: bool,
    /// Bumped whenever what the GPU holds changes under an unchanged camera,
    /// which has to start a still view's average over.
    generation: u64,
    /// Parts tessellated again for the view, if any are.
    refined: Option<Refined>,
    /// The pick pipeline for those, which flags the vertex numbers it writes.
    pick_refined_pipeline: wgpu::RenderPipeline,
}

/// Which B-rep edges to overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeMode {
    /// Only edges where the faces meet at an angle -- what a CAD viewer shows.
    Sharp,
    /// Every edge in the model, tangent continuations included.
    All,
    None,
}

impl EdgeMode {
    pub fn next(self) -> EdgeMode {
        match self {
            EdgeMode::Sharp => EdgeMode::All,
            EdgeMode::All => EdgeMode::None,
            EdgeMode::None => EdgeMode::Sharp,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            EdgeMode::Sharp => "sharp edges",
            EdgeMode::All => "all edges",
            EdgeMode::None => "no edges",
        }
    }

    pub fn parse(s: &str) -> Option<EdgeMode> {
        match s.to_ascii_lowercase().as_str() {
            "sharp" => Some(EdgeMode::Sharp),
            "all" => Some(EdgeMode::All),
            "off" | "none" => Some(EdgeMode::None),
            _ => None,
        }
    }
}

/// How hard ambient occlusion is pushed.
///
/// Raw occlusion on mostly-convex CAD geometry lands near 1.0, so a linear
/// scale barely registers; a power curve does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AoLevel {
    Off,
    Subtle,
    Normal,
    Strong,
    Heavy,
}

impl AoLevel {
    pub const ALL: [AoLevel; 5] = [
        AoLevel::Off,
        AoLevel::Subtle,
        AoLevel::Normal,
        AoLevel::Strong,
        AoLevel::Heavy,
    ];

    /// Exponent applied to the sampled occlusion. 0 switches the effect off.
    fn power(self) -> f32 {
        match self {
            AoLevel::Off => 0.0,
            AoLevel::Subtle => 1.0,
            AoLevel::Normal => 2.0,
            AoLevel::Strong => 3.5,
            AoLevel::Heavy => 6.0,
        }
    }

    /// How much occlusion also dims *direct* light. Strictly it belongs on the
    /// indirect term alone, but under a bright key light deep bores would then
    /// stay implausibly lit.
    fn direct(self) -> f32 {
        match self {
            AoLevel::Off => 0.0,
            AoLevel::Subtle => 0.25,
            AoLevel::Normal => 0.45,
            AoLevel::Strong => 0.65,
            AoLevel::Heavy => 0.85,
        }
    }

    pub fn next(self) -> AoLevel {
        let i = AoLevel::ALL.iter().position(|&l| l == self).unwrap_or(0);
        AoLevel::ALL[(i + 1) % AoLevel::ALL.len()]
    }

    pub fn label(self) -> &'static str {
        match self {
            AoLevel::Off => "off",
            AoLevel::Subtle => "subtle",
            AoLevel::Normal => "normal",
            AoLevel::Strong => "strong",
            AoLevel::Heavy => "heavy",
        }
    }

    pub fn parse(s: &str) -> Option<AoLevel> {
        AoLevel::ALL
            .into_iter()
            .find(|l| l.label().eq_ignore_ascii_case(s))
    }
}

/// A half-space to keep: everything with `dot(normal, p) <= offset`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Section {
    pub normal: Vec3,
    pub offset: f64,
}

/// The three axes a section can lie on, each in both directions, plus off.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub fn parse(s: &str) -> Option<(Axis, f64)> {
        let (sign, rest) = match s.strip_prefix('-') {
            Some(r) => (-1.0, r),
            None => (1.0, s.strip_prefix('+').unwrap_or(s)),
        };
        let axis = match rest {
            "x" | "X" => Axis::X,
            "y" | "Y" => Axis::Y,
            "z" | "Z" => Axis::Z,
            _ => return None,
        };
        Some((axis, sign))
    }

    pub fn normal(self, sign: f64) -> Vec3 {
        match self {
            Axis::X => v3(sign, 0.0, 0.0),
            Axis::Y => v3(0.0, sign, 0.0),
            Axis::Z => v3(0.0, 0.0, sign),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Axis::X => "x",
            Axis::Y => "y",
            Axis::Z => "z",
        }
    }
}

/// A colour laid over a part: its surface and its edges are blended towards
/// `rgb` by `amount`, in linear light. Used for selection highlights and
/// colour-by-component.
///
/// A blend rather than a multiplier (that is the instance `tint`), so it shows
/// on a black part. It rides on the instance record via a twin placement, so
/// painting costs no vertex bytes or mesh upload. An amount of exactly zero
/// is no paint, and an unpainted frame issues the plain draw calls.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Paint {
    pub rgb: [f32; 3],
    pub amount: f32,
}

/// No paint, as an instance record carries it.
const NO_PAINT: [u8; 4] = [0; 4];

impl Paint {
    pub const NONE: Paint = Paint {
        rgb: [0.0; 3],
        amount: 0.0,
    };

    /// The paint as an instance record carries it, and as twins are keyed:
    /// rgb and amount in eight bits each. Anything that rounds to no amount
    /// is no paint, and draws through the base placements.
    fn key(&self) -> [u8; 4] {
        let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        // NaN takes the "none" branch: `!(x > 0)` rather than `x <= 0`.
        let a = if !(self.amount > 0.0) { 0 } else { b(self.amount) };
        if a == 0 {
            return NO_PAINT;
        }
        [b(self.rgb[0]), b(self.rgb[1]), b(self.rgb[2]), a]
    }
}

/// What the caller wants drawn this frame.
pub struct Frame<'a> {
    pub camera: &'a Camera,
    pub show_faces: bool,
    pub edges: EdgeMode,
    /// `Off` skips the three occlusion passes entirely.
    pub ao: AoLevel,
    /// The half-space to keep, or the whole model when absent.
    pub section: Option<Section>,
    /// Whether to fill the cut. Ignored without a section, and unavailable on
    /// an adapter with no float depth-stencil format.
    pub cap: bool,
    /// Whether to stand the model on a shadowed floor.
    pub ground: bool,
    /// Whether to outline the model where its surface turns away.
    pub silhouette: bool,
    /// One flag per `Mesh::parts`. An empty slice means every part is drawn,
    /// which is also the fast path: the whole buffer goes in one call.
    pub visible: &'a [bool],
    /// A paint per `Mesh::parts`, or fewer: parts past the end of the list are
    /// unpainted, as is every part with an amount of zero. Empty is the fast
    /// path -- no twins, and one draw per slot.
    pub paint: &'a [Paint],
    /// How many frames to average a still view over, each sampling its pixels
    /// at a different offset. One is none: the multisampled frame is drawn
    /// straight to the target.
    ///
    /// Multisampling covers edges but shades each pixel once, so a fillet
    /// narrower than a pixel comes out as flecks; averaging shifted frames
    /// samples the shading too. The first frame of a view is unshifted, and
    /// the rest arrive one per redraw once the view stops.
    pub supersample: u32,
}

/// What one pixel of the frame is looking at.
#[derive(Debug, Clone, Copy)]
pub struct Pick {
    pub point: Vec3,
    /// One vertex of the triangle that was hit; which one is undefined, since
    /// all three belong to the same part.
    pub vertex: u32,
}

/// Where the time goes standing a renderer up, printed when `STPVIEW_TIMING=1`.
///
/// A cold start is mostly GPU initialisation, not the model, and which stage
/// costs what is not guessable, so it is measured. The offscreen constructors
/// start the clock before `wgpu::Instance::new`; `for_surface` cannot, since
/// the windowing layer owns the instance there, so its total is `Gpu::build`
/// alone.
struct Lap {
    on: bool,
    start: std::time::Instant,
    last: std::time::Instant,
}

impl Lap {
    fn new() -> Lap {
        let now = std::time::Instant::now();
        Lap {
            on: std::env::var("STPVIEW_TIMING").is_ok_and(|v| v != "0"),
            start: now,
            last: now,
        }
    }

    fn at(&mut self, what: &str) {
        if !self.on {
            return;
        }
        let now = std::time::Instant::now();
        eprintln!(
            "timing {what:<16}{:7.1} ms",
            now.duration_since(self.last).as_secs_f64() * 1000.0
        );
        self.last = now;
    }

    fn total(&mut self) {
        if !self.on {
            return;
        }
        eprintln!(
            "timing {:<16}{:7.1} ms",
            "renderer",
            self.start.elapsed().as_secs_f64() * 1000.0
        );
    }
}

/// Coalesces the visible parts' ranges into runs that share a paint, so a
/// run of unpainted neighbours costs one draw and a painted part in the
/// middle of them costs three.
///
/// Parts are laid out in the buffers in order, so hiding one part out of a
/// hundred leaves two runs, not ninety-nine. With no paint every run carries
/// `NO_PAINT`.
fn painted_runs(
    parts: &[Part],
    visible: &[bool],
    paint: &[[u8; 4]],
    pick: impl Fn(&Part) -> Range<u32>,
) -> Vec<([u8; 4], Range<u32>)> {
    let mut out: Vec<([u8; 4], Range<u32>)> = Vec::new();
    for (i, p) in parts.iter().enumerate() {
        if !visible.get(i).copied().unwrap_or(true) {
            continue;
        }
        let r = pick(p);
        if r.is_empty() {
            continue;
        }
        let k = paint.get(i).copied().unwrap_or(NO_PAINT);
        match out.last_mut() {
            Some((last_k, last)) if *last_k == k && last.end == r.start => last.end = r.end,
            _ => out.push((k, r)),
        }
    }
    out
}

/// `painted_runs` for the passes that have no use for a colour: depth, the
/// stencil count and the pick see the same geometry whatever it is painted.
fn runs(parts: &[Part], visible: &[bool], pick: impl Fn(&Part) -> Range<u32>) -> Vec<Range<u32>> {
    painted_runs(parts, visible, &[], pick)
        .into_iter()
        .map(|(_, r)| r)
        .collect()
}

impl Gpu {
    pub fn samples(&self) -> u32 {
        self.samples
    }

    /// The visibility flags that apply to one slot's parts.
    ///
    /// Flags index the flat part list, so a slot's share is the slice at its
    /// own range; an empty or short list means everything shows. Per-part
    /// hiding is a single-model feature, whose one slot starts at zero.
    fn slot_visible<'a>(&self, s: &Slot, visible: &'a [bool]) -> &'a [bool] {
        if visible.is_empty() {
            &[]
        } else {
            visible.get(s.parts.clone()).unwrap_or(&[])
        }
    }

    /// The paint keys that apply to one slot's parts: the slice at its own
    /// range, cut short where the list is; the tail is unpainted.
    fn slot_paint<'a>(&self, s: &Slot, paint: &'a [[u8; 4]]) -> &'a [[u8; 4]] {
        let end = s.parts.end.min(paint.len());
        &paint[s.parts.start.min(end)..end]
    }

    /// The instance records a run with this paint is drawn through: the
    /// slot's own for no paint, its painted twin otherwise. A missing twin
    /// (which `sync_twins` prevents) draws the run unpainted.
    fn instances_for(&self, slot: usize, s: &Slot, paint: [u8; 4]) -> Range<u32> {
        if paint == NO_PAINT {
            return s.instances.clone();
        }
        self.twins
            .iter()
            .find(|t| t.slot == slot && t.paint == paint)
            .map(|t| t.instances.clone())
            .unwrap_or_else(|| s.instances.clone())
    }

    /// Makes sure every paint the frame puts on a visible part has its twin in
    /// the instance buffer, rewriting the buffer when the set changes.
    ///
    /// The set changes only when a selection does and the buffer is small, so
    /// it is rewritten whole rather than patched.
    fn sync_twins(&mut self, paint: &[[u8; 4]], visible: &[bool]) {
        let mut want: Vec<Twin> = Vec::new();
        let mut raws = self.base_instances.clone();
        for (si, s) in self.slots.iter().enumerate() {
            let vis = self.slot_visible(s, visible);
            for (i, &k) in self.slot_paint(s, paint).iter().enumerate() {
                if k == NO_PAINT || !vis.get(i).copied().unwrap_or(true) {
                    continue;
                }
                if want.iter().any(|t| t.slot == si && t.paint == k) {
                    continue;
                }
                let start = raws.len() as u32;
                let base = s.instances.start as usize..s.instances.end as usize;
                raws.extend(self.base_instances[base].iter().map(|r| r.painted(k)));
                want.push(Twin {
                    slot: si,
                    paint: k,
                    instances: start..raws.len() as u32,
                });
            }
        }
        if want == self.twins {
            return;
        }
        // wgpu rejects an empty buffer; a scene with no placements keeps the
        // identity record `build` put there, which nothing draws.
        if raws.is_empty() {
            raws.push(InstanceRaw::IDENTITY);
        }
        self.instance_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instances"),
            contents: bytemuck::cast_slice(&raws),
            usage: wgpu::BufferUsages::VERTEX | GROW,
        });
        self.twins = want;
    }

    /// `draw_tris` for the edge overlay's non-indexed line quads.
    fn draw_lines(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        visible: &[bool],
        paint: &[[u8; 4]],
        pick: impl Fn(&Part) -> Range<u32> + Copy,
    ) {
        // A quad is six vertices a segment and a segment is two points, so
        // three vertices a point; the rasteriser's lines take the points as
        // they are.
        let per = if self.edge_bind.is_some() { 3 } else { 1 };
        for (si, s) in self.slots.iter().enumerate() {
            let parts = &self.parts[s.parts.clone()];
            let vis = self.slot_visible(s, visible);
            for (k, r) in painted_runs(parts, vis, self.slot_paint(s, paint), pick) {
                pass.draw(r.start * per..r.end * per, self.instances_for(si, s, k));
            }
        }
    }

    /// Draws a pick of every slot's triangles, each for its own instances:
    /// one call over the whole slot when nothing is hidden or painted, its
    /// runs otherwise, with each painted run drawn through its twin. Every
    /// triangle pass goes through here, so they cannot disagree about what
    /// exists or where it stands.
    ///
    /// `around` leaves out the faces drawn refined instead, which every pass
    /// drawing from the camera wants; the shadow map, drawn once for the
    /// whole model, does not.
    fn draw_tris(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        visible: &[bool],
        paint: &[[u8; 4]],
        pick: impl Fn(&Part) -> Range<u32> + Copy,
        around: bool,
    ) {
        let refined = self.refined.as_ref().filter(|r| around && !r.holes.is_empty());
        for (si, s) in self.slots.iter().enumerate() {
            let parts = &self.parts[s.parts.clone()];
            let vis = self.slot_visible(s, visible);
            // With nothing hidden and nothing painted, `painted_runs`
            // coalesces a gap-free pick into one whole-slot call.
            for (k, r) in painted_runs(parts, vis, self.slot_paint(s, paint), pick) {
                let inst = self.instances_for(si, s, k);
                match refined {
                    None => pass.draw_indexed(r, 0, inst),
                    Some(x) => {
                        for piece in x.around_holes(r) {
                            pass.draw_indexed(piece, 0, inst.clone());
                        }
                    }
                }
            }
            if parts.is_empty() && self.slots.len() == 1 && self.index_count > 0 {
                // A hand-built mesh with no parts at all still draws whole
                // rather than not at all.
                pass.draw_indexed(0..self.index_count, 0, s.instances.clone());
            }
        }
    }

    /// `visible`, less the parts whose edges are drawn refined: what the
    /// model's own edge buffer is to draw. Borrowed when nothing is refined.
    fn unrefined<'a>(&self, visible: &'a [bool]) -> std::borrow::Cow<'a, [bool]> {
        match &self.refined {
            None => std::borrow::Cow::Borrowed(visible),
            Some(r) => std::borrow::Cow::Owned(
                (0..self.parts.len())
                    .map(|i| visible.get(i).copied().unwrap_or(true) && !r.replaced[i])
                    .collect(),
            ),
        }
    }

    /// Every triangle a pass wants: the model's own, less the faces drawn
    /// refined, then those from their own buffers. Binds the vertex and index
    /// buffers itself; the instances in slot 1 are the caller's.
    fn draw_all_tris(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        visible: &[bool],
        paint: &[[u8; 4]],
        pick: impl Fn(&Part) -> Range<u32> + Copy,
    ) {
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        self.draw_tris(pass, visible, paint, pick, true);
        self.draw_refined_tris(pass, visible, paint, pick);
    }

    /// The refined parts alone, through whatever pipeline is set. Refined
    /// parts exist only for a single model, whose one slot is the identity.
    fn draw_refined_tris(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        visible: &[bool],
        paint: &[[u8; 4]],
        pick: impl Fn(&Part) -> Range<u32> + Copy,
    ) {
        let (Some(r), Some(s)) = (&self.refined, self.slots.first()) else { return };
        let (vis, keys) = r.flags(visible, paint);
        pass.set_vertex_buffer(0, r.vertex_buf.slice(..));
        pass.set_index_buffer(r.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        for (k, run) in painted_runs(&r.parts, &vis, &keys, pick) {
            pass.draw_indexed(run, 0, self.instances_for(0, s, k));
        }
    }

    /// Every edge a pass wants, of one class: the model's own less what is
    /// refined, then the refined parts' from their own buffer. Binds the
    /// edge source itself, as `draw_all_tris` does.
    fn draw_all_lines(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        visible: &[bool],
        paint: &[[u8; 4]],
        pick: impl Fn(&Part) -> Range<u32> + Copy,
    ) {
        match &self.edge_bind {
            Some(b) => pass.set_bind_group(1, b, &[]),
            None => pass.set_vertex_buffer(0, self.edge_buf.slice(..)),
        }
        self.draw_lines(pass, &self.unrefined(visible), paint, pick);
        let (Some(r), Some(s)) = (&self.refined, self.slots.first()) else { return };
        match &r.edge_bind {
            Some(b) => pass.set_bind_group(1, b, &[]),
            None => pass.set_vertex_buffer(0, r.edge_buf.slice(..)),
        }
        let per = if r.edge_bind.is_some() { 3 } else { 1 };
        let (vis, keys) = r.flags(visible, paint);
        for (k, run) in painted_runs(&r.parts, &vis, &keys, pick) {
            pass.draw(run.start * per..run.end * per, self.instances_for(0, s, k));
        }
    }

    /// Draws these parts from a tessellation of their own from now on: `mesh`
    /// holds one part per entry of `of`, in that order, each standing in for
    /// that part of the model. Replaces whatever was refined before.
    ///
    /// Every pass but the shadow draws them: the shadow map is filled once for
    /// the whole model, and a finer tessellation would not change it visibly.
    /// A scene has no single model to refine, and ignores this.
    pub fn set_refined(&mut self, mesh: &Mesh, of: &[usize]) {
        if self.slots.len() != 1 || mesh.parts.len() != of.len() || mesh.indices.is_empty() {
            self.clear_refined();
            return;
        }
        let buf = |label: &'static str, contents: &[u8], usage: wgpu::BufferUsages| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        // Never empty: wgpu refuses a zero-sized buffer, and a refined part
        // may well have no edges.
        let edges: &[[f32; 3]] = if mesh.edges.is_empty() { &[[0.0; 3], [0.0; 3]] } else { &mesh.edges };
        let edge_buf = buf("refined edges", bytemuck::cast_slice(edges), EDGES.usage);
        let mut replaced = vec![false; self.parts.len()];
        for &i in of {
            if let Some(r) = replaced.get_mut(i) {
                *r = true;
            }
        }
        // The model's own triangles of every face drawn here instead. A part
        // that names no faces but has triangles was built whole by hand, and
        // stands in for the whole of its part.
        let mut holes: Vec<Range<u32>> = Vec::new();
        for (p, &i) in mesh.parts.iter().zip(of) {
            let Some(own) = self.parts.get(i) else { continue };
            if p.faces.is_empty() {
                if !p.tris.is_empty() {
                    holes.push(own.tris.clone());
                }
                continue;
            }
            for f in &p.faces {
                let k = f.face as usize;
                let first = own
                    .faces
                    .get(k)
                    .filter(|g| g.face == f.face)
                    .or_else(|| own.faces.iter().find(|g| g.face == f.face));
                if let Some(g) = first.filter(|g| !g.tris.is_empty()) {
                    holes.push(g.tris.clone());
                }
            }
        }
        holes.sort_by_key(|h| h.start);
        let mut merged: Vec<Range<u32>> = Vec::with_capacity(holes.len());
        for h in holes {
            match merged.last_mut() {
                Some(last) if h.start <= last.end => last.end = last.end.max(h.end),
                _ => merged.push(h),
            }
        }
        self.refined = Some(Refined {
            vertex_buf: buf("refined vertices", bytemuck::cast_slice(&mesh.vertices), wgpu::BufferUsages::VERTEX),
            index_buf: buf("refined indices", bytemuck::cast_slice(&mesh.indices), wgpu::BufferUsages::INDEX),
            edge_bind: edge_bind_group(&self.device, &self.layouts, &edge_buf),
            edge_buf,
            parts: mesh.parts.clone(),
            of: of.to_vec(),
            replaced,
            holes: merged,
        });
        self.generation += 1;
    }

    /// Back to the model's own tessellation everywhere.
    pub fn clear_refined(&mut self) {
        if self.refined.take().is_some() {
            self.generation += 1;
        }
    }

    /// What the refined parts cost on the GPU, in bytes.
    pub fn refined_bytes(&self) -> u64 {
        self.refined
            .as_ref()
            .map_or(0, |r| r.vertex_buf.size() + r.index_buf.size() + r.edge_buf.size())
    }

    /// On any surface wgpu can present to. The instance is the caller's:
    /// building one that can present needs the windowing layer's display
    /// handle.
    pub async fn for_surface(
        instance: &wgpu::Instance,
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
        mesh: &Mesh,
    ) -> Result<Gpu, String> {
        let surface = instance
            .create_surface(target)
            .map_err(|e| format!("create surface: {e}"))?;
        Gpu::build(instance, Some(surface), width, height, mesh, &mut Lap::new()).await
    }

    /// The same renderer with no window behind it, for `--shot`.
    pub async fn offscreen(width: u32, height: u32, mesh: &Mesh) -> Result<Gpu, String> {
        let mut lap = Lap::new();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        lap.at("wgpu instance");
        Gpu::build(&instance, None, width, height, mesh, &mut lap).await
    }

    /// Headless, on an instance the caller keeps.
    ///
    /// Building a `wgpu::Instance` is a large share of a cold start, so a
    /// process that stands up many renderers (the shell thumbnail provider)
    /// pays it once.
    pub async fn offscreen_on(
        instance: &wgpu::Instance,
        width: u32,
        height: u32,
        mesh: &Mesh,
    ) -> Result<Gpu, String> {
        Gpu::build(instance, None, width, height, mesh, &mut Lap::new()).await
    }

    /// Headless over a scene, GPU-instanced: each unique mesh is uploaded
    /// once, verbatim, and every placement is one instance record.
    pub async fn offscreen_scene(
        width: u32,
        height: u32,
        scene: &crate::scene::Scene,
    ) -> Result<Gpu, String> {
        use crate::scene::MeshHandle;
        use std::sync::Arc;

        // Unique meshes in first-appearance order, so the same scene always
        // builds the same buffers and draws in the same order.
        let mut unique: Vec<&MeshHandle> = Vec::new();
        let mut slot_of: Vec<usize> = Vec::with_capacity(scene.instances.len());
        for inst in &scene.instances {
            let id = Arc::as_ptr(&inst.mesh);
            let k = match unique.iter().position(|m| std::ptr::eq(Arc::as_ptr(m), id)) {
                Some(k) => k,
                None => {
                    unique.push(&inst.mesh);
                    unique.len() - 1
                }
            };
            slot_of.push(k);
        }

        let refs: Vec<&Mesh> = unique.iter().map(|m| m.as_ref()).collect();
        let combined = Mesh::concatenated(&refs);
        let mut lap = Lap::new();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        lap.at("wgpu instance");
        let mut gpu = Gpu::build(&instance, None, width, height, &combined, &mut lap).await?;

        // One slot per unique mesh; within each, plain placements first and
        // mirrored ones behind, because the cap's parity count draws the two
        // groups through different stencil pipelines.
        let mut slots = Vec::with_capacity(unique.len());
        let mut raws: Vec<InstanceRaw> = Vec::with_capacity(scene.instances.len());
        let mut part_base = 0usize;
        for (k, m) in unique.iter().enumerate() {
            let start = raws.len() as u32;
            for (i, inst) in scene.instances.iter().enumerate() {
                if slot_of[i] == k && inst.world().det() >= 0.0 {
                    raws.push(InstanceRaw::from_xform(&inst.world(), inst.tint));
                }
            }
            let mirror_from = raws.len() as u32;
            for (i, inst) in scene.instances.iter().enumerate() {
                if slot_of[i] == k && inst.world().det() < 0.0 {
                    raws.push(InstanceRaw::from_xform(&inst.world(), inst.tint));
                }
            }
            // `concatenated` synthesizes the covering part for a part-less
            // mesh, so the count here is what actually landed in `parts`.
            let n_parts = if m.parts.is_empty() && !m.vertices.is_empty() {
                1
            } else {
                m.parts.len()
            };
            slots.push(Slot {
                parts: part_base..part_base + n_parts,
                instances: start..raws.len() as u32,
                mirror_from,
            });
            part_base += n_parts;
        }
        gpu.slots = slots;
        gpu.base_instances = raws.clone();
        if !raws.is_empty() {
            gpu.instance_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("instances"),
                contents: bytemuck::cast_slice(&raws),
                usage: wgpu::BufferUsages::VERTEX | GROW,
            });
        }
        // The instanced bounds, not the union of the raw meshes' boxes.
        gpu.refit(scene.bounds());
        Ok(gpu)
    }

    async fn build(
        instance: &wgpu::Instance,
        surface: Option<wgpu::Surface<'static>>,
        width: u32,
        height: u32,
        mesh: &Mesh,
        lap: &mut Lap,
    ) -> Result<Gpu, String> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface.as_ref(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no suitable GPU adapter: {e}"))?;
        let info = adapter.get_info();
        println!("gpu    {} ({:?}, {:?})", info.name, info.device_type, info.backend);

        // A float depth buffer with a stencil, if the adapter has one. Asked
        // for, not required: without it the section still cuts, and a 24-bit
        // depth format would cost precision on every frame.
        let stencil = adapter
            .features()
            .contains(wgpu::Features::DEPTH32FLOAT_STENCIL8);
        let depth_format = if stencil { DEPTH_STENCIL_FORMAT } else { DEPTH_FORMAT };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("stpview"),
                // The mesh can be a few hundred MB, which exceeds the
                // conservative defaults; ask for what this adapter can do.
                required_limits: adapter.limits(),
                required_features: if stencil {
                    wgpu::Features::DEPTH32FLOAT_STENCIL8
                } else {
                    wgpu::Features::empty()
                },
                ..Default::default()
            })
            .await
            .map_err(|e| format!("request device: {e}"))?;

        lap.at("adapter+device");
        let (width, height) = (width.max(1), height.max(1));
        let mut config = match &surface {
            Some(s) => {
                let mut c = s
                    .get_default_config(&adapter, width, height)
                    .ok_or("surface is not supported by this adapter")?;
                // Prefer an sRGB target so the hardware does the encoding.
                let caps = s.get_capabilities(&adapter);
                if let Some(f) = caps.formats.iter().copied().find(|f| f.is_srgb()) {
                    c.format = f;
                }
                c
            }
            None => wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                width,
                height,
                present_mode: wgpu::PresentMode::Fifo,
                desired_maximum_frame_latency: 2,
                alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                color_space: wgpu::SurfaceColorSpace::Auto,
                view_formats: vec![],
            },
        };
        config.width = width;
        config.height = height;
        let manual_srgb = !config.format.is_srgb();
        if let Some(s) = &surface {
            s.configure(&device, &config);
        }
        let offscreen = surface.is_none().then(|| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("shot"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        });

        let samples = if adapter
            .get_texture_format_features(config.format)
            .flags
            .sample_count_supported(SAMPLE_COUNT)
        {
            SAMPLE_COUNT
        } else {
            1
        };

        // Full float gives the occlusion pass enough depth precision to avoid
        // self-shadowing on a large model; half float is the fallback.
        let renders = |f: wgpu::TextureFormat| {
            adapter
                .get_texture_format_features(f)
                .allowed_usages
                .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
        };
        let gbuffer_format = if renders(wgpu::TextureFormat::Rgba32Float) {
            wgpu::TextureFormat::Rgba32Float
        } else {
            wgpu::TextureFormat::Rgba16Float
        };

        lap.at("targets");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("surfaces"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let post = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ssao.wgsl").into()),
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        // Every texture here is read with textureLoad at exact pixel
        // coordinates, so none of them need to be filterable.
        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layouts = Layouts {
            scene: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene"),
                entries: &[
                    uniform_entry,
                    tex_entry(1),
                    // The reflection cube map, and the one filtering sampler:
                    // everything else is read at exact pixels, but a
                    // reflection must interpolate or it bands.
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // The shadow map, through a comparison sampler: the
                    // hardware depth-tests and averages across the footprint,
                    // so four taps give a soft edge.
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                    // The occlusion prepass's normals and linear depth, which
                    // the silhouette pass reads to find where the surface ends.
                    tex_entry(6),
                ],
            }),
            post: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("post"),
                entries: &[uniform_entry, tex_entry(1), tex_entry(2)],
            }),
            bare: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bare"),
                entries: &[uniform_entry],
            }),
            // Desktop backends read storage from a vertex shader; GL does
            // not, and draws the rasteriser's one-pixel lines instead.
            lines: adapter
                .get_downlevel_capabilities()
                .flags
                .contains(wgpu::DownlevelFlags::VERTEX_STORAGE)
                .then(|| {
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("lines"),
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }],
                    })
                }),
        };
        let bare_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bare"),
            layout: &layouts.bare,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        let bare_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bare"),
            bind_group_layouts: &[Some(&layouts.bare)],
            immediate_size: 0,
        });
        let scene_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene"),
            bind_group_layouts: &[Some(&layouts.scene)],
            immediate_size: 0,
        });
        let post_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post"),
            bind_group_layouts: &[Some(&layouts.post)],
            immediate_size: 0,
        });

        let vertex_layout = |attrs: &'static [wgpu::VertexAttribute], stride: u64| {
            wgpu::VertexBufferLayout {
                array_stride: stride,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: attrs,
            }
        };
        // Every mesh-consuming pipeline binds this as its second buffer, or
        // its pass draws the geometry unplaced
        // (`every_mesh_stage_takes_the_instance_block`).
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INSTANCE_ATTRS,
        };
        // The forward pass may carry a stencil for capping; the occlusion
        // prepass has its own buffer and never needs one.
        let depth_state = move |write: bool| wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: Some(write),
            depth_compare: Some(NEARER),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let gbuffer_depth_state = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(NEARER),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let no_cull = |topology| wgpu::PrimitiveState {
            topology,
            // No culling: two-sided shading makes it unnecessary, and it keeps
            // stray face orientations in the source file harmless.
            cull_mode: None,
            ..Default::default()
        };
        let color_target = [Some(wgpu::ColorTargetState {
            format: config.format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        // The cap mark exists only for its stencil ops, but has to declare the
        // attachment of the pass it runs in, the mask.

        let no_mask_target = [Some(wgpu::ColorTargetState {
            format: CAP_MASK_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::empty(),
        })];
        let gbuffer_target = [Some(wgpu::ColorTargetState {
            format: gbuffer_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let ao_target = [Some(wgpu::ColorTargetState {
            format: AO_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];

        let surface_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("surfaces"),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_surface"),
                compilation_options: Default::default(),
                buffers: &[Some(vertex_layout(&SURFACE_ATTRS, std::mem::size_of::<Vertex>() as u64)), Some(instance_layout.clone())],
            },
            primitive: no_cull(wgpu::PrimitiveTopology::TriangleList),
            depth_stencil: Some(wgpu::DepthStencilState {
                bias: SURFACE_BIAS,
                ..depth_state(true)
            }),
            multisample: wgpu::MultisampleState {
                count: samples,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_surface"),
                compilation_options: Default::default(),
                targets: &color_target,
            }),
            multiview_mask: None,
            cache: None,
        });

        // The overlay: each segment a quad read from the edge buffer by index,
        // covered by each pixel's distance from its centre -- or, without a
        // vertex stage that can read storage, the rasteriser's own lines.
        let lines_layout = layouts.lines.as_ref().map(|lines| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("lines"),
                bind_group_layouts: &[Some(&layouts.scene), Some(lines)],
                immediate_size: 0,
            })
        });
        let line_target = [Some(wgpu::ColorTargetState {
            format: config.format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let line_pipeline = |label: &'static str, quad: &'static str, line: &'static str| {
            let (layout, entry, buffers, topology) = match &lines_layout {
                // The instance block stays in slot 1, as in every other pass;
                // the quads read the edges as storage, so slot 0 is empty.
                Some(l) => (l, quad, vec![None, Some(instance_layout.clone())], wgpu::PrimitiveTopology::TriangleList),
                None => (
                    &scene_layout,
                    line,
                    vec![Some(vertex_layout(&EDGE_ATTRS, 12)), Some(instance_layout.clone())],
                    wgpu::PrimitiveTopology::LineList,
                ),
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    buffers: &buffers,
                },
                primitive: no_cull(topology),
                depth_stencil: Some(depth_state(false)),
                multisample: wgpu::MultisampleState {
                    count: samples,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_edge"),
                    compilation_options: Default::default(),
                    targets: &line_target,
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let edge_pipeline = line_pipeline("edges", "vs_edge", "vs_edge_line");
        // Loose curves bound no face, so they are not pulled towards the eye.
        let wire_pipeline = line_pipeline("wires", "vs_wire", "vs_wire_line");

        let gbuffer_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prepass"),
            layout: Some(&bare_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_gbuffer"),
                compilation_options: Default::default(),
                buffers: &[Some(vertex_layout(&SURFACE_ATTRS, std::mem::size_of::<Vertex>() as u64)), Some(instance_layout.clone())],
            },
            primitive: no_cull(wgpu::PrimitiveTopology::TriangleList),
            depth_stencil: Some(gbuffer_depth_state.clone()),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_gbuffer"),
                compilation_options: Default::default(),
                targets: &gbuffer_target,
            }),
            multiview_mask: None,
            cache: None,
        });

        // Picking. Same vertex buffer, same section plane, its own tiny target.
        let pick = |label: &'static str, entry: &'static str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&bare_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    buffers: &[Some(vertex_layout(&SURFACE_ATTRS, std::mem::size_of::<Vertex>() as u64)), Some(instance_layout.clone())],
                },
                primitive: no_cull(wgpu::PrimitiveTopology::TriangleList),
                depth_stencil: Some(gbuffer_depth_state.clone()),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_pick"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: PICK_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let pick_pipeline = pick("pick", "vs_pick");
        let pick_refined_pipeline = pick("pick refined", "vs_pick_refined");

        // The shadow map, drawn from the light instead of from the camera.
        //
        // A slope-scaled depth bias rather than front-face culling against
        // shadow acne: culling assumes closed casters, and sheet bodies and
        // shells would stop casting.
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow"),
            layout: Some(&bare_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shadow"),
                compilation_options: Default::default(),
                buffers: &[Some(vertex_layout(&SURFACE_ATTRS, std::mem::size_of::<Vertex>() as u64)), Some(instance_layout.clone())],
            },
            primitive: no_cull(wgpu::PrimitiveTopology::TriangleList),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.5,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });

        // The ground. Blended, because it fades out at its rim.
        let ground_target = [Some(wgpu::ColorTargetState {
            format: config.format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let ground_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ground"),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_ground"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: no_cull(wgpu::PrimitiveTopology::TriangleList),
            // Writes depth so the cap and the edge overlay sort against it,
            // and a part below the floor is hidden by it.
            depth_stencil: Some(depth_state(true)),
            multisample: wgpu::MultisampleState {
                count: samples,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_ground"),
                compilation_options: Default::default(),
                targets: &ground_target,
            }),
            multiview_mask: None,
            cache: None,
        });

        // Transparency, weighted and blended rather than sorted (McGuire and
        // Bavoil); see `fs_transparent` in shader.wgsl.
        let oit_targets = [
            Some(wgpu::ColorTargetState {
                format: ACCUM_FORMAT,
                // Straight addition: this buffer is a sum.
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: REVEAL_FORMAT,
                // And a running product: dst *= 1 - src.
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::OneMinusSrc,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::OneMinusSrc,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ];
        let oit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("transparent"),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_surface"),
                compilation_options: Default::default(),
                buffers: &[Some(vertex_layout(&SURFACE_ATTRS, std::mem::size_of::<Vertex>() as u64)), Some(instance_layout.clone())],
            },
            primitive: no_cull(wgpu::PrimitiveTopology::TriangleList),
            // Tested against the opaque depth so a pane behind a wall is
            // hidden, but writing none of its own, so it does not occlude
            // the transparent surface behind it.
            //
            // The same bias as the opaque surfaces, so a pane flush with an
            // opaque face does not win against it and film it over.
            depth_stencil: Some(wgpu::DepthStencilState {
                bias: SURFACE_BIAS,
                ..depth_state(false)
            }),
            multisample: wgpu::MultisampleState {
                count: samples,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_transparent"),
                compilation_options: Default::default(),
                targets: &oit_targets,
            }),
            multiview_mask: None,
            cache: None,
        });
        // And the pass that divides the sum and lays it over the opaque image.
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("oit composite"),
            layout: Some(&post_layout),
            vertex: wgpu::VertexState {
                module: &post,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &post,
                entry_point: Some("fs_composite"),
                compilation_options: Default::default(),
                targets: &ground_target,
            }),
            multiview_mask: None,
            cache: None,
        });

        // The silhouette, found in the picture rather than in the model:
        // one screen's work whatever the model, reading the prepass already
        // drawn for the occlusion, instead of walking every mesh edge and
        // uploading a line buffer whenever the camera moves.
        let silhouette_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("silhouette"),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: no_cull(wgpu::PrimitiveTopology::TriangleList),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: samples,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_silhouette"),
                compilation_options: Default::default(),
                targets: &ground_target,
            }),
            multiview_mask: None,
            cache: None,
        });

        // Capping, in two draws.
        //
        // The first counts, in the stencil, the surfaces on the far side of
        // the plane from the eye (`fs_cap_mark`): up for each front face, down
        // for each back face, with no depth test. Along a view ray a solid
        // wholly on that side cancels; a solid the plane passes through is left
        // at one. So a nonzero stencil means "the plane is inside material".
        //
        // No depth test: geometry on the near side must not cancel anything.
        // It is accounted for by the second draw, which is depth-tested.
        let (cap_mark_pipeline, cap_mark_mirror, cap_fill_pipeline) = if depth_format != DEPTH_STENCIL_FORMAT {
            (None, None, None)
        } else {
            let count = |op: wgpu::StencilOperation| wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::Always,
                fail_op: wgpu::StencilOperation::Keep,
                depth_fail_op: wgpu::StencilOperation::Keep,
                pass_op: op,
            };
            let mark = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("cap mark"),
                layout: Some(&scene_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_surface"),
                    compilation_options: Default::default(),
                    buffers: &[Some(vertex_layout(&SURFACE_ATTRS, std::mem::size_of::<Vertex>() as u64)), Some(instance_layout.clone())],
                },
                primitive: no_cull(wgpu::PrimitiveTopology::TriangleList),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState {
                        // Wrapping, so deeply nested walls still come back to
                        // zero.
                        front: count(wgpu::StencilOperation::IncrementWrap),
                        back: count(wgpu::StencilOperation::DecrementWrap),
                        read_mask: 0xff,
                        write_mask: 0xff,
                    },
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: samples,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_cap_mark"),
                    compilation_options: Default::default(),
                    // Same attachment as the pass it runs in, with the writes
                    // masked off: this draw exists only for its stencil ops.
                    targets: &no_mask_target,
                }),
                multiview_mask: None,
                cache: None,
            });
            // The same count with increment and decrement swapped, for
            // mirrored placements: a reflection reverses every triangle's
            // orientation, and through the plain pipeline a mirrored solid
            // would subtract itself.
            let mark_mirror = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("cap mark mirrored"),
                layout: Some(&scene_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_surface"),
                    compilation_options: Default::default(),
                    buffers: &[Some(vertex_layout(&SURFACE_ATTRS, std::mem::size_of::<Vertex>() as u64)), Some(instance_layout.clone())],
                },
                primitive: no_cull(wgpu::PrimitiveTopology::TriangleList),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState {
                        front: count(wgpu::StencilOperation::DecrementWrap),
                        back: count(wgpu::StencilOperation::IncrementWrap),
                        read_mask: 0xff,
                        write_mask: 0xff,
                    },
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: samples,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_cap_mark"),
                    compilation_options: Default::default(),
                    targets: &no_mask_target,
                }),
                multiview_mask: None,
                cache: None,
            });
            let keep_nonzero = wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::NotEqual,
                fail_op: wgpu::StencilOperation::Keep,
                depth_fail_op: wgpu::StencilOperation::Keep,
                pass_op: wgpu::StencilOperation::Keep,
            };
            // The mask: the same quad, stencil-tested, writing coverage into a
            // buffer of its own so the fill can look at its neighbours.
            let mask_target = [Some(wgpu::ColorTargetState {
                format: CAP_MASK_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })];
            let mask = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("cap mask"),
                layout: Some(&scene_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_cap"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: no_cull(wgpu::PrimitiveTopology::TriangleList),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(NEARER),
                    stencil: wgpu::StencilState {
                        front: keep_nonzero,
                        back: keep_nonzero,
                        read_mask: 0xff,
                        write_mask: 0x00,
                    },
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: samples,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_cap_mask"),
                    compilation_options: Default::default(),
                    targets: &mask_target,
                }),
                multiview_mask: None,
                cache: None,
            });
            // And the fill, which reads that mask back, erodes it, and shades
            // what survives, blended so the eroded rim is soft.
            let fill = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("cap fill"),
                layout: Some(&post_layout),
                vertex: wgpu::VertexState {
                    module: &post,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &post,
                    entry_point: Some("fs_cap_fill"),
                    compilation_options: Default::default(),
                    targets: &ground_target,
                }),
                multiview_mask: None,
                cache: None,
            });
            (Some((mark, mask)), Some(mark_mirror), Some(fill))
        };

        let fullscreen = |label: &'static str, entry: &'static str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&post_layout),
                vertex: wgpu::VertexState {
                    module: &post,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &post,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &ao_target,
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let ssao_pipeline = fullscreen("ssao", "fs_ssao");
        let blur_pipeline = fullscreen("ssao blur", "fs_blur");

        // Supersampling. The average is kept by the blend unit: frame n goes
        // in at weight 1/n over the average so far at 1 - 1/n, the weight
        // set per draw as the blend constant.
        let copy = |label: &'static str, format: wgpu::TextureFormat, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&post_layout),
                vertex: wgpu::VertexState {
                    module: &post,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &post,
                    entry_point: Some("fs_copy"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let running = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Constant,
            dst_factor: wgpu::BlendFactor::OneMinusConstant,
            operation: wgpu::BlendOperation::Add,
        };
        let accumulate_pipeline = copy(
            "accumulate",
            HISTORY_FORMAT,
            Some(wgpu::BlendState {
                color: running,
                alpha: running,
            }),
        );
        let present_pipeline = copy("present", config.format, None);

        // wgpu rejects zero-sized buffers, so keep a single dummy element.
        let fallback_vertex = [Vertex::new(Vec3::ZERO, v3(0.0, 0.0, 1.0), [0.0; 3], 1.0)];
        let vertices: &[Vertex] = if mesh.vertices.is_empty() {
            &fallback_vertex
        } else {
            &mesh.vertices
        };
        let indices: &[u32] = if mesh.indices.is_empty() {
            &[0, 0, 0]
        } else {
            &mesh.indices
        };
        let edges: &[[f32; 3]] = if mesh.edges.is_empty() {
            &[[0.0; 3], [0.0; 3]]
        } else {
            &mesh.edges
        };

        // COPY_SRC and COPY_DST because a streamed load grows these: a bigger
        // buffer is made, the old contents copied across, the batch appended.
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertices"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX | GROW,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("indices"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX | GROW,
        });
        let edge_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("edges"),
            contents: bytemuck::cast_slice(edges),
            usage: EDGES.usage | GROW,
        });
        let edge_bind = edge_bind_group(&device, &layouts, &edge_buf);
        // One exact identity: a single model is an instanced draw of one.
        let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instances"),
            contents: bytemuck::cast_slice(&[InstanceRaw::IDENTITY]),
            usage: wgpu::BufferUsages::VERTEX | GROW,
        });

        // The pick target and the buffer it comes back through. A texture copy
        // wants each row aligned, so the readback is wider than the square.
        let pick_color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pick"),
            size: wgpu::Extent3d {
                width: PICK_SIZE,
                height: PICK_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PICK_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let pick_view = pick_color.create_view(&wgpu::TextureViewDescriptor::default());
        let pick_depth = view_of(
            &device,
            "pick depth",
            PICK_SIZE,
            PICK_SIZE,
            1,
            DEPTH_FORMAT,
            wgpu::TextureUsages::empty(),
        );
        let pick_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick readback"),
            size: (pick_stride() * PICK_SIZE) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        lap.at("pipelines");
        // The studio, prefiltered on the CPU (threaded and cached; see
        // `env::prefiltered_cube`) rather than shipped as an HDR photograph.
        let env = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("studio"),
            size: wgpu::Extent3d {
                width: crate::env::CUBE_SIZE,
                height: crate::env::CUBE_SIZE,
                depth_or_array_layers: 6,
            },
            mip_level_count: crate::env::CUBE_MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (mip, data) in crate::env::prefiltered_cube().iter().enumerate() {
            let size = (crate::env::CUBE_SIZE >> mip).max(1);
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &env,
                    mip_level: mip as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size * 8),
                    rows_per_image: Some(size),
                },
                wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 6,
                },
            );
        }
        lap.at("environment");
        let env_view = env.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let env_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("studio"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            // Between levels as well as within one, or a band runs along a
            // curved face where the mip steps.
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // The shadow map, and the light that fills it. Both are fixed, so it
        // is drawn once and refilled only when visibility or bounds change.
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow map"),
            size: wgpu::Extent3d {
                width: SHADOW_SIZE,
                height: SHADOW_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // A streamed load starts with nothing, and `append` rebuilds what the
        // bounds decide as batches arrive. A unit box stands in until then, so
        // the first frame has no infinities in it.
        let bounds = if mesh.parts.is_empty() {
            let mut b = Aabb::EMPTY;
            b.add_point(v3(-1.0, -1.0, -1.0));
            b.add_point(v3(1.0, 1.0, 1.0));
            b
        } else {
            mesh.bbox
        };
        let streaming = mesh.parts.is_empty();
        let (light_vp, ground_z, ground_radius) = shadow_frame(&bounds);
        let shadow_texel = 2.0 / SHADOW_SIZE as f32;

        let targets = Targets::new(
            &device,
            &config,
            samples,
            gbuffer_format,
            depth_format,
            &layouts,
            &uniform_buf,
            &env_view,
            &env_sampler,
            &shadow_view,
            &shadow_sampler,
        );

        let mut gpu = Gpu {
            surface,
            offscreen,
            device,
            queue,
            config,
            targets,
            samples,
            gbuffer_format,
            layouts,
            surface_pipeline,
            edge_pipeline,
            wire_pipeline,
            gbuffer_pipeline,
            ssao_pipeline,
            blur_pipeline,
            cap_mark_pipeline,
            cap_fill_pipeline,
            silhouette_pipeline,
            oit_pipeline,
            composite_pipeline,
            pick_pipeline,
            pick_color,
            pick_view,
            pick_depth,
            pick_read,
            bare_bind,
            any_transparent: mesh.parts.iter().any(|p| p.alpha < 1.0),
            parts: mesh.parts.clone(),
            slots: vec![Slot {
                parts: 0..mesh.parts.len(),
                instances: 0..1,
                mirror_from: 1,
            }],
            instance_buf,
            base_instances: vec![InstanceRaw::IDENTITY],
            twins: Vec::new(),
            cap_mark_mirror,
            shadow_for: Vec::new(),
            depth_format,
            scene_radius: bounds.size().length() as f32 * 0.75,
            scene_center: bounds.center().as_f32(),
            // Empty while streaming, so the first batch sets the real bounds
            // rather than unioning with a placeholder.
            bounds: if streaming { Aabb::EMPTY } else { bounds },
            ground_pipeline,
            light_vp,
            ground_z,
            ground_radius,
            shadow_texel,
            shadow_view,
            shadow_sampler,
            uniform_buf,
            vertex_buf,
            index_buf,
            index_count: mesh.indices.len() as u32,
            edge_buf,
            edge_count: mesh.edges.len() as u32,
            edge_bind,
            vertex_cap: vertices.len() as u32,
            index_cap: indices.len() as u32,
            edge_cap: edges.len() as u32,
            // Zero when `new` was given an empty mesh and put placeholders in
            // the buffers, so the first streamed batch overwrites them.
            vertex_total: mesh.vertices.len() as u32,
            manual_srgb,
            kernel: ssao_kernel(),
            sh: crate::env::irradiance_sh9(),
            env_view,
            env_sampler,
            shadow_pipeline,
            accumulate_pipeline,
            present_pipeline,
            accumulation: Accumulation::default(),
            stalled: false,
            generation: 0,
            refined: None,
            pick_refined_pipeline,
        };
        gpu.fill_shadow(&[]);
        lap.at("shadow map");
        lap.total();
        Ok(gpu)
    }

    /// Draws the model from the light into the shadow map.
    ///
    /// Called at load and again when the visible parts or the bounds change;
    /// a hidden part must stop casting.
    fn fill_shadow(&mut self, visible: &[bool]) {
        let mut u: Uniforms = bytemuck::Zeroable::zeroed();
        u.light_vp = self.light_vp.0;
        self.queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("shadow") });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if self.index_count > 0 {
                pass.set_pipeline(&self.shadow_pipeline);
                pass.set_bind_group(0, &self.bare_bind, &[]);
                pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
                pass.set_vertex_buffer(1, self.instance_buf.slice(..));
                pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                self.draw_tris(&mut pass, visible, &[], |p| p.tris.clone(), false);
            }
        }
        self.queue.submit(Some(enc.finish()));
    }
}
