//! The render targets and bind-group layouts: everything sized to the
//! window, rebuilt whole on resize.
use super::*;

/// Everything that must be rebuilt when the window resizes.
pub(crate) struct Targets {
    pub(crate) color: wgpu::TextureView,
    /// Weighted-blended transparency: the accumulation and revealage buffers,
    /// multisampled to match the depth they test against, and their resolves
    /// for the composite to read.
    pub(crate) accum: wgpu::TextureView,
    pub(crate) accum_resolved: wgpu::TextureView,
    pub(crate) reveal: wgpu::TextureView,
    pub(crate) reveal_resolved: wgpu::TextureView,
    pub(crate) oit_bind: wgpu::BindGroup,
    /// Where the section cap covers, multisampled and resolved, so the fill
    /// can erode it before using it.
    pub(crate) cap_mask: wgpu::TextureView,
    pub(crate) cap_mask_resolved: wgpu::TextureView,
    pub(crate) cap_bind: wgpu::BindGroup,
    pub(crate) depth: wgpu::TextureView,
    pub(crate) gbuffer: wgpu::TextureView,
    pub(crate) gdepth: wgpu::TextureView,
    pub(crate) ao_raw: wgpu::TextureView,
    pub(crate) ao_blur: wgpu::TextureView,
    pub(crate) scene_bind: wgpu::BindGroup,
    pub(crate) ssao_bind: wgpu::BindGroup,
    pub(crate) blur_bind: wgpu::BindGroup,
    /// Supersampling: each frame is drawn here, in the target's own format,
    /// and blended into `history`, the running average, which is what goes on
    /// the screen. The bind groups read each through the post layout.
    pub(crate) frame: wgpu::TextureView,
    pub(crate) frame_bind: wgpu::BindGroup,
    pub(crate) history: wgpu::TextureView,
    pub(crate) history_bind: wgpu::BindGroup,
}

pub(crate) struct Layouts {
    pub(crate) scene: wgpu::BindGroupLayout,
    pub(crate) post: wgpu::BindGroupLayout,
    /// The uniform buffer alone. Used by the passes that *write* one of the
    /// textures the scene group binds -- the shadow map and the occlusion
    /// prepass -- since a texture cannot be read and written in one pass.
    pub(crate) bare: wgpu::BindGroupLayout,
    /// The edge buffer, read as storage by the vertex stage that turns each
    /// segment into a quad. Absent on an adapter that cannot read storage
    /// from a vertex shader, which then draws the rasteriser's own lines.
    pub(crate) lines: Option<wgpu::BindGroupLayout>,
}

pub(crate) fn view_of(
    device: &wgpu::Device,
    label: &str,
    w: u32,
    h: u32,
    samples: u32,
    format: wgpu::TextureFormat,
    extra: wgpu::TextureUsages,
) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | extra,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

impl Targets {
    // Every argument is a sibling field of `Gpu`; a params struct would be
    // `Gpu` again, minus the fields under construction.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        samples: u32,
        gbuffer_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        layouts: &Layouts,
        uniform_buf: &wgpu::Buffer,
        env: &wgpu::TextureView,
        env_sampler: &wgpu::Sampler,
        shadow: &wgpu::TextureView,
        shadow_sampler: &wgpu::Sampler,
    ) -> Targets {
        let (w, h) = (config.width, config.height);
        let sampled = wgpu::TextureUsages::TEXTURE_BINDING;
        let color = view_of(device, "msaa color", w, h, samples, config.format, wgpu::TextureUsages::empty());
        let depth = view_of(device, "msaa depth", w, h, samples, depth_format, wgpu::TextureUsages::empty());
        let gbuffer = view_of(device, "gbuffer", w, h, 1, gbuffer_format, sampled);
        let gdepth = view_of(device, "gbuffer depth", w, h, 1, DEPTH_FORMAT, wgpu::TextureUsages::empty());
        let ao_raw = view_of(device, "ao", w, h, 1, AO_FORMAT, sampled);
        let ao_blur = view_of(device, "ao blurred", w, h, 1, AO_FORMAT, sampled);
        let accum = view_of(device, "oit accum", w, h, samples, ACCUM_FORMAT, wgpu::TextureUsages::empty());
        let reveal = view_of(device, "oit reveal", w, h, samples, REVEAL_FORMAT, wgpu::TextureUsages::empty());
        let cap_mask = view_of(device, "cap mask", w, h, samples, CAP_MASK_FORMAT, wgpu::TextureUsages::empty());
        let cap_mask_resolved = view_of(device, "cap mask resolved", w, h, 1, CAP_MASK_FORMAT, sampled);
        let accum_resolved = view_of(device, "oit accum resolved", w, h, 1, ACCUM_FORMAT, sampled);
        let reveal_resolved = view_of(device, "oit reveal resolved", w, h, 1, REVEAL_FORMAT, sampled);
        let frame = view_of(device, "frame", w, h, 1, config.format, sampled);
        let history = view_of(device, "history", w, h, 1, HISTORY_FORMAT, sampled);

        let uniform = wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        };
        let tex = |binding, view| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(view),
        };
        let scene_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene"),
            layout: &layouts.scene,
            entries: &[
                uniform.clone(),
                tex(1, &ao_blur),
                tex(2, env),
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(env_sampler),
                },
                tex(4, shadow),
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(shadow_sampler),
                },
                tex(6, &gbuffer),
            ],
        });
        let ssao_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao"),
            layout: &layouts.post,
            // The occlusion pass does not read any occlusion; slot 2 is bound
            // only to satisfy the shared layout.
            entries: &[uniform.clone(), tex(1, &gbuffer), tex(2, &ao_blur)],
        });
        let blur_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur"),
            layout: &layouts.post,
            entries: &[uniform, tex(1, &gbuffer), tex(2, &ao_raw)],
        });
        let cap_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cap"),
            layout: &layouts.post,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                tex(1, &cap_mask_resolved),
                tex(2, &cap_mask_resolved),
            ],
        });
        let oit_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oit"),
            layout: &layouts.post,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                tex(1, &accum_resolved),
                tex(2, &reveal_resolved),
            ],
        });
        // One texture read through both slots of the shared layout, as the cap
        // does; `fs_copy` reads only the first.
        let copy_bind = |label, view| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &layouts.post,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    tex(1, view),
                    tex(2, view),
                ],
            })
        };
        let frame_bind = copy_bind("frame", &frame);
        let history_bind = copy_bind("history", &history);

        Targets {
            color,
            accum,
            accum_resolved,
            reveal,
            reveal_resolved,
            oit_bind,
            cap_mask,
            cap_mask_resolved,
            cap_bind,
            depth,
            gbuffer,
            gdepth,
            ao_raw,
            ao_blur,
            scene_bind,
            ssao_bind,
            blur_bind,
            frame,
            frame_bind,
            history,
            history_bind,
        }
    }
}

