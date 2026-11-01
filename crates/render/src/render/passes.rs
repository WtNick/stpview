//! The frame graph, pass by pass: forward, occlusion, shadow, section
//! cap, transparency, silhouette, edge overlay, pick and readback --
//! the half of `Gpu` that draws, split from the half that builds.
use super::*;

impl Gpu {
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        if let Some(s) = &self.surface {
            s.configure(&self.device, &self.config);
        }
        self.targets = Targets::new(
            &self.device,
            &self.config,
            self.samples,
            self.gbuffer_format,
            self.depth_format,
            &self.layouts,
            &self.uniform_buf,
            &self.env_view,
            &self.env_sampler,
            &self.shadow_view,
            &self.shadow_sampler,
        );
        // The average lived in the targets just replaced.
        self.generation += 1;
    }

    /// Whether the last frame drawn is as good as that view gets: every frame
    /// of its average is in. A viewer keeps redrawing a still view until it
    /// is, and can then sleep.
    pub fn converged(&self) -> bool {
        self.stalled || self.accumulation.done >= self.accumulation.target
    }

    /// `jitter` shifts the whole picture by that fraction of a pixel, for a
    /// frame of a supersampled view. Zero leaves every matrix exact to the bit.
    fn uniforms(&self, f: &Frame, jitter: [f32; 2]) -> Uniforms {
        let cam = f.camera;
        let (w, h) = (self.config.width.max(1), self.config.height.max(1));
        let aspect = w as f32 / h as f32;
        let view = cam.view();
        // In device coordinates, which span two units across the target.
        let shift = [jitter[0] * 2.0 / w as f32, jitter[1] * 2.0 / h as f32];
        let shifted = |m: Mat4| {
            if shift == [0.0, 0.0] {
                return m;
            }
            let mut t = Mat4([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]);
            t.0[3][0] = shift[0];
            t.0[3][1] = shift[1];
            t.mul(&m)
        };
        let proj = shifted(cam.proj(aspect));
        let eye = cam.eye();

        // A three-point rig fixed to the camera: key over the left shoulder,
        // a dim fill opposite it to keep shadowed faces readable, and a back
        // light that separates one part's silhouette from the next.
        let (right, up, forward) = cam.basis();
        let key = (right * -0.45 + up * 0.60 - forward).normalize();
        let fill = (right * 0.75 + up * 0.10 - forward * 0.35).normalize();
        let back = (right * 0.20 + up * 0.70 + forward * 0.55).normalize();
        let dir = |v: Vec3, i: f32| [v.x as f32, v.y as f32, v.z as f32, i];

        let radius = (cam.distance * AO_RADIUS_FRACTION)
            .clamp(cam.scene_radius * 0.002, cam.scene_radius * 0.15);
        let tan_half = (cam.fov_y * 0.5).tan();

        Uniforms {
            view_proj: shifted(cam.view_proj(aspect)).0,
            view: view.0,
            proj: proj.0,
            eye: [eye.x as f32, eye.y as f32, eye.z as f32, 1.0],
            lights: [dir(key, 0.95), dir(fill, 0.22), dir(back, 0.28)],
            params: [
                if self.manual_srgb { 1.0 } else { 0.0 },
                EDGE_EYE_NUDGE,
                if f.show_faces { 0.0 } else { 1.0 },
                f.ao.power(),
            ],
            ssao: [
                radius as f32,
                // Bias, in depth and in height above the tangent plane. The
                // hemisphere follows the analytic normal but the depth buffer
                // holds facets, which would otherwise occlude each other and
                // print the mesh onto a flat panel.
                (radius * 0.04) as f32,
                w as f32,
                h as f32,
            ],
            // Exposure puts a white part under the key light near 0.85 after
            // the tone curve, leaving headroom for the specular.
            proj_info: [tan_half * aspect, tan_half, 0.85, f.ao.direct()],
            section: match f.section {
                Some(s) => {
                    let n = s.normal.normalize();
                    [n.x as f32, n.y as f32, n.z as f32, s.offset as f32]
                }
                None => [0.0; 4],
            },
            cap: [
                self.scene_radius,
                self.scene_center[0],
                self.scene_center[1],
                self.scene_center[2],
            ],
            light_vp: self.light_vp.0,
            ground: [
                self.ground_z,
                self.ground_radius,
                if f.ground { 1.0 } else { 0.0 },
                self.shadow_texel,
            ],
            sh: self.sh,
            kernel: self.kernel,
            jitter: [shift[0], shift[1], 0.0, 0.0],
        }
    }

    pub fn render(&mut self, f: &Frame) {
        let Some(surface) = &self.surface else {
            self.draw(f, None);
            return;
        };
        let frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                let (w, h) = (self.config.width, self.config.height);
                self.resize(w, h);
                return;
            }
            // Timed out or occluded, typically minimised. `converged` reports
            // true until a frame gets through, so a viewer does not spin
            // behind a window nobody can see.
            _ => {
                self.stalled = true;
                return;
            }
        };
        self.stalled = false;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.draw(f, Some(&view));
        self.queue.present(frame);
    }

    /// One frame into `view`, or into the offscreen target when there is none.
    ///
    /// Supersampled, a still view is drawn once per call with its pixels
    /// sampled at a new offset, and the target gets the average so far. The
    /// first call samples the centres; once the average is complete, a call
    /// only puts it back on the target.
    pub(crate) fn draw(&mut self, f: &Frame, view: Option<&wgpu::TextureView>) {
        // The paints this frame puts on its parts, as the instance records
        // carry them, and the twins that carry them into the forward pass.
        let paint: Vec<[u8; 4]> = f.paint.iter().map(Paint::key).collect();
        self.sync_twins(&paint, f.visible);
        // The shadow map is filled once and kept; hiding a part is the only
        // thing that can invalidate it.
        if self.shadow_for != f.visible {
            self.shadow_for = f.visible.to_vec();
            self.fill_shadow(f.visible);
        }

        // Which frame of the average this is. Any change to the view
        // restarts it.
        let target = f.supersample.max(1);
        let key = FrameKey::of(f, &paint, self.generation);
        if self.accumulation.key.as_ref() != Some(&key) || self.accumulation.target != target {
            self.accumulation = Accumulation {
                key: Some(key),
                done: 0,
                target,
            };
        }
        let averaged = target > 1;
        // Unaveraged, every call draws: a window's surface hands out a new
        // texture for each present, and it has to be filled.
        let fresh = !averaged || self.accumulation.done < target;
        let sample = self.accumulation.done;

        let own;
        let view = match view {
            Some(v) => v,
            None => {
                let Some(tex) = &self.offscreen else { return };
                own = tex.create_view(&wgpu::TextureViewDescriptor::default());
                &own
            }
        };

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        if fresh {
            let uniforms = self.uniforms(f, jitter(if averaged { sample } else { 0 }));
            self.queue
                .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
            // Averaged, the frame is drawn aside and blended in; otherwise
            // straight onto the target.
            let dest = if averaged { &self.targets.frame } else { view };

            // The prepass feeds both the occlusion and the silhouette, so either
            // one asking for it is enough.
            let want_ao = f.ao != AoLevel::Off && f.show_faces && self.index_count > 0;
            let want_prepass = want_ao
                || (f.silhouette && f.show_faces && f.edges != EdgeMode::None && self.index_count > 0);
            if want_prepass {
                self.prepass(&mut enc, f.visible);
            }
            if want_ao {
                self.occlusion_pass(&mut enc, "ssao", &self.ssao_pipeline, &self.targets.ao_raw, &self.targets.ssao_bind);
                self.occlusion_pass(&mut enc, "ssao blur", &self.blur_pipeline, &self.targets.ao_blur, &self.targets.blur_bind);
            }
            self.forward_pass(&mut enc, dest, f, &paint);
            // The cut, in two passes of its own: mark where the plane is inside
            // material, then erode that and lay it over the frame. After the
            // forward pass, so the overlay cannot be printed across it.
            if f.show_faces && f.cap && f.section.is_some() && self.index_count > 0 {
                self.cap_mask_pass(&mut enc, f);
                self.cap_fill_pass(&mut enc, dest);
            }
            // Transparency is three passes rather than one: accumulate, then
            // divide and lay the result over what the forward pass resolved.
            if self.any_transparent && f.show_faces {
                self.transparent_pass(&mut enc, f.visible, &paint);
                self.composite_pass(&mut enc, dest);
            }
            if averaged {
                // Frame n of the average goes in at 1/n, so the first replaces
                // whatever the last view left behind.
                let weight = 1.0 / (sample + 1) as f64;
                self.copy_pass(&mut enc, "accumulate", &self.accumulate_pipeline, &self.targets.frame_bind, &self.targets.history, Some(weight));
                self.accumulation.done += 1;
            } else {
                // One frame is the whole of an unaveraged view, and `converged`
                // has to say so.
                self.accumulation.done = target;
            }
        }
        if averaged {
            self.copy_pass(&mut enc, "present", &self.present_pipeline, &self.targets.history_bind, view, None);
        }
        self.queue.submit(Some(enc.finish()));
    }

    /// A full-screen copy from one target to another: into the running average,
    /// blended at `weight`, or from it onto the frame.
    fn copy_pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &wgpu::RenderPipeline,
        bind: &wgpu::BindGroup,
        target: &wgpu::TextureView,
        weight: Option<f64>,
    ) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        if let Some(w) = weight {
            pass.set_blend_constant(wgpu::Color { r: w, g: w, b: w, a: w });
        }
        pass.draw(0..3, 0..1);
    }

    /// Draws one frame off screen and hands back the pixels, tightly packed RGBA.
    ///
    /// Supersampled, that is the whole average, restarted from the first frame
    /// so the same view always gives the same bytes.
    pub fn capture(&mut self, f: &Frame) -> Result<Vec<u8>, String> {
        self.accumulation = Accumulation::default();
        for _ in 0..f.supersample.max(1) {
            self.draw(f, None);
        }
        self.read_back()
    }

    /// The offscreen target as it stands, tightly packed RGBA.
    pub(crate) fn read_back(&self) -> Result<Vec<u8>, String> {
        let tex = self.offscreen.as_ref().ok_or("not an offscreen renderer")?;
        let (w, h) = (self.config.width, self.config.height);
        // A texture-to-buffer copy wants each row aligned, so the readback
        // buffer is wider than the image and the padding is dropped below.
        let stride = (w * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (stride * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(enc.finish()));

        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("readback: {e}"))?;
        let mapped = slice.get_mapped_range().map_err(|e| format!("readback: {e}"))?;
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for row in 0..h {
            let start = (row * stride) as usize;
            out.extend_from_slice(&mapped[start..start + (w * 4) as usize]);
        }
        drop(mapped);
        buf.unmap();
        Ok(out)
    }

    /// Adds a batch of a streamed load to what is already on the GPU.
    ///
    /// The batch arrives with its ranges local to itself, as `Mesh::absorb`
    /// receives it, and is shifted here by the same amounts. The buffers grow
    /// by doubling, so the total copying stays linear in the mesh.
    pub fn append(&mut self, batch: &Mesh) {
        if batch.parts.is_empty() {
            return;
        }
        // Where this batch lands. The first append overwrites the placeholder
        // element `new` put in each buffer to keep wgpu happy about zero sizes.
        let (vb, ib, eb) = (self.vertex_total, self.index_count, self.edge_count);
        self.grow(vb, batch.vertices.len() as u32);
        self.grow_index(ib, batch.indices.len() as u32);
        self.grow_edge(eb, batch.edges.len() as u32);

        if !batch.vertices.is_empty() {
            self.queue.write_buffer(
                &self.vertex_buf,
                vb as u64 * std::mem::size_of::<Vertex>() as u64,
                bytemuck::cast_slice(&batch.vertices),
            );
        }
        if !batch.indices.is_empty() {
            // Indices are stored relative to the batch, so they shift with it.
            let shifted: Vec<u32> = batch.indices.iter().map(|i| i + vb).collect();
            self.queue.write_buffer(&self.index_buf, ib as u64 * 4, bytemuck::cast_slice(&shifted));
        }
        if !batch.edges.is_empty() {
            self.queue.write_buffer(&self.edge_buf, eb as u64 * 12, bytemuck::cast_slice(&batch.edges));
        }

        let shift = |r: &Range<u32>, by: u32| r.start + by..r.end + by;
        for p in &batch.parts {
            let mut p = p.clone();
            p.tris = shift(&p.tris, ib);
            p.verts = shift(&p.verts, vb);
            p.wire = shift(&p.wire, eb);
            p.sharp = shift(&p.sharp, eb);
            p.tangent = shift(&p.tangent, eb);
            for f in &mut p.faces {
                f.tris = shift(&f.tris, ib);
            }
            self.any_transparent |= p.alpha < 1.0;
            self.parts.push(p);
        }
        self.vertex_total = vb + batch.vertices.len() as u32;
        self.index_count = ib + batch.indices.len() as u32;
        self.edge_count = eb + batch.edges.len() as u32;
        // Streaming grows the one identity slot; scenes never append. With
        // more than one slot this would swallow the others' parts.
        if self.slots.len() == 1
            && let Some(s) = self.slots.first_mut()
        {
            s.parts.end = self.parts.len();
        }

        // The bounds decide the light's box, the floor and the section cap's
        // quad; they come from what actually arrived, or the cap misses part
        // of the cut.
        self.refit(self.bounds.union(batch.bbox));
    }

    /// Re-derives everything the bounds decide -- the light's box, the floor,
    /// the cap quad -- and marks the shadow map stale. A scene passes its
    /// *instanced* bounds, not the union of the raw meshes' boxes.
    pub(crate) fn refit(&mut self, bounds: Aabb) {
        // Whatever arrived changes the picture under an unchanged camera.
        self.generation += 1;
        self.bounds = bounds;
        let (light_vp, ground_z, ground_radius) = shadow_frame(&self.bounds);
        self.light_vp = light_vp;
        self.ground_z = ground_z;
        self.ground_radius = ground_radius;
        self.scene_radius = self.bounds.size().length() as f32 * 0.75;
        self.scene_center = self.bounds.center().as_f32();

        // The shadow map is kept between frames, and the geometry and light
        // have just changed.
        self.shadow_for.clear();
        self.shadow_for.push(true);
    }

    /// Makes room for `add` more elements past `have`, keeping what is there:
    /// double until it fits, copy the live prefix. Returns the replacement, or
    /// `None` if it already fits. Takes `&self` so the caller can pass a borrow
    /// of the buffer being replaced.
    fn grown(
        &self,
        old: &wgpu::Buffer,
        cap: u32,
        have: u32,
        add: u32,
        kind: BufKind,
    ) -> Option<(wgpu::Buffer, u32)> {
        if have + add <= cap {
            return None;
        }
        let cap = (cap.max(1) * 2).max(have + add);
        let next = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(kind.label),
            size: cap as u64 * kind.stride,
            usage: kind.usage.union(GROW),
            mapped_at_creation: false,
        });
        self.copy_into(old, &next, have as u64 * kind.stride);
        Some((next, cap))
    }

    fn grow(&mut self, have: u32, add: u32) {
        if let Some((b, c)) = self.grown(&self.vertex_buf, self.vertex_cap, have, add, VERTICES) {
            self.vertex_buf = b;
            self.vertex_cap = c;
        }
    }

    fn grow_index(&mut self, have: u32, add: u32) {
        if let Some((b, c)) = self.grown(&self.index_buf, self.index_cap, have, add, INDICES) {
            self.index_buf = b;
            self.index_cap = c;
        }
    }

    fn grow_edge(&mut self, have: u32, add: u32) {
        if let Some((b, c)) = self.grown(&self.edge_buf, self.edge_cap, have, add, EDGES) {
            self.edge_buf = b;
            self.edge_cap = c;
            // A binding names the buffer it was made from, so remake it.
            self.edge_bind = edge_bind_group(&self.device, &self.layouts, &self.edge_buf);
        }
    }

    fn copy_into(&self, from: &wgpu::Buffer, to: &wgpu::Buffer, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("grow") });
        enc.copy_buffer_to_buffer(from, 0, to, 0, bytes);
        self.queue.submit(Some(enc.finish()));
    }

    /// What is under one pixel of the frame.
    ///
    /// A render pass rather than a ray cast: it draws the same geometry with
    /// the same section and hidden parts, so what the frame does not show
    /// cannot be picked, and it needs no acceleration structure in memory.
    ///
    /// Leaves the shifted projection in the uniform buffer. Every path that
    /// picks redraws afterwards, and `draw` rewrites the whole block.
    pub fn pick(&mut self, f: &Frame, px: f32, py: f32) -> Option<Pick> {
        if self.index_count == 0 {
            return None;
        }
        let (w, h) = (self.config.width.max(1), self.config.height.max(1));
        let mut u = self.uniforms(f, [0.0, 0.0]);
        u.view_proj = pick_transform(w as f32, h as f32, px, py, PICK_SIZE as f32)
            .mul(&f.camera.view_proj(w as f32 / h as f32))
            .0;
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pick") });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pick"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.pick_view,
                    depth_slice: None,
                    resolve_target: None,
                    // Zero is "nothing here", which is why the shader adds one
                    // to the vertex index it writes.
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.pick_depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(FAR_DEPTH),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pick_pipeline);
            pass.set_bind_group(0, &self.bare_bind, &[]);
            pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            pass.set_vertex_buffer(1, self.instance_buf.slice(..));
            pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            self.draw_tris(&mut pass, f.visible, &[], |p| p.tris.clone(), true);
            // What is drawn refined is picked refined, or a close click near a
            // silhouette would land on the coarser surface. Its own pipeline
            // flags the vertex numbers as the refined buffers'.
            if self.refined.is_some() {
                pass.set_pipeline(&self.pick_refined_pipeline);
                self.draw_refined_tris(&mut pass, f.visible, &[], |p| p.tris.clone());
            }
        }
        let stride = pick_stride();
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.pick_color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.pick_read,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(PICK_SIZE),
                },
            },
            wgpu::Extent3d {
                width: PICK_SIZE,
                height: PICK_SIZE,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(enc.finish()));

        let slice = self.pick_read.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let mapped = slice.get_mapped_range().ok()?;

        // The nearest hit to the middle, which is the clicked pixel. Searching
        // outwards lets a click a little off a thin part still find it.
        let centre = (PICK_SIZE / 2) as i32;
        let mut best: Option<(i32, Pick)> = None;
        for y in 0..PICK_SIZE as i32 {
            for x in 0..PICK_SIZE as i32 {
                let o = y as usize * stride as usize + x as usize * 16;
                let word = |k: usize| {
                    u32::from_le_bytes(mapped[o + k * 4..o + k * 4 + 4].try_into().unwrap())
                };
                let id = word(0);
                if id == 0 {
                    continue;
                }
                let d = (x - centre).pow(2) + (y - centre).pow(2);
                if best.as_ref().is_some_and(|(bd, _)| *bd <= d) {
                    continue;
                }
                // A refined vertex is answered with one of the part's own
                // vertices: the caller knows only the model's numbering, and
                // a pick is for naming the part.
                let vertex = match id - 1 {
                    v if v & 0x8000_0000 != 0 => {
                        let v = v & 0x7fff_ffff;
                        let Some(i) = self.refined.as_ref().and_then(|r| {
                            r.parts.iter().position(|p| p.verts.contains(&v)).map(|j| r.of[j])
                        }) else {
                            continue;
                        };
                        self.parts[i].verts.start
                    }
                    v => v,
                };
                best = Some((
                    d,
                    Pick {
                        vertex,
                        point: v3(
                            f32::from_bits(word(1)) as f64,
                            f32::from_bits(word(2)) as f64,
                            f32::from_bits(word(3)) as f64,
                        ),
                    },
                ));
            }
        }
        drop(mapped);
        self.pick_read.unmap();
        best.map(|(_, p)| p)
    }

    /// Which parts own at least one pixel of the frame, drawn at a `scale`th of
    /// the size: a part behind others, or out of the picture, owns none.
    ///
    /// The pick pass over the whole frame, so it agrees with the picture --
    /// section, hidden parts and refined faces as drawn. It decides what is
    /// worth tessellating again for a view; a frustum cannot, inside an
    /// assembly where most of it is hidden behind the parts in front. A part
    /// too small to own a reduced pixel is too small to show its facets.
    ///
    /// Leaves the uniform buffer as `pick` does; every caller redraws.
    pub fn parts_in_frame(&mut self, f: &Frame, scale: u32) -> Vec<bool> {
        let mut seen = vec![false; self.parts.len()];
        if self.index_count == 0 {
            return seen;
        }
        let (w, h) = (self.config.width.max(1), self.config.height.max(1));
        let scale = scale.max(1);
        let (sw, sh) = ((w / scale).max(1), (h / scale).max(1));
        let u = self.uniforms(f, [0.0, 0.0]);
        self.queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));

        let color = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("parts in frame"),
            size: wgpu::Extent3d {
                width: sw,
                height: sh,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PICK_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = view_of(&self.device, "parts in frame depth", sw, sh, 1, DEPTH_FORMAT, wgpu::TextureUsages::empty());
        let stride = (sw * 16).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let read = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("parts in frame readback"),
            size: (stride * sh) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("parts in frame") });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("parts in frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(FAR_DEPTH),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pick_pipeline);
            pass.set_bind_group(0, &self.bare_bind, &[]);
            pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            pass.set_vertex_buffer(1, self.instance_buf.slice(..));
            pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            self.draw_tris(&mut pass, f.visible, &[], |p| p.tris.clone(), true);
            if self.refined.is_some() {
                pass.set_pipeline(&self.pick_refined_pipeline);
                self.draw_refined_tris(&mut pass, f.visible, &[], |p| p.tris.clone());
            }
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &read,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(sh),
                },
            },
            wgpu::Extent3d {
                width: sw,
                height: sh,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(enc.finish()));

        let slice = read.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        if self.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
            return seen;
        }
        let Ok(mapped) = slice.get_mapped_range() else { return seen };
        // Parts own their vertices in order, so the part a vertex belongs to is
        // found by halving -- over ends rather than starts, since a part that
        // drew nothing has an empty range and starts where the next one does.
        let owner = |v: u32| -> Option<usize> {
            if v & 0x8000_0000 != 0 {
                let v = v & 0x7fff_ffff;
                let r = self.refined.as_ref()?;
                return r.parts.iter().position(|p| p.verts.contains(&v)).map(|j| r.of[j]);
            }
            let i = self.parts.partition_point(|p| p.verts.end <= v);
            self.parts.get(i).filter(|p| p.verts.contains(&v)).map(|_| i)
        };
        let mut last = 0u32;
        for y in 0..sh as usize {
            let row = &mapped[y * stride as usize..y * stride as usize + sw as usize * 16];
            for px in row.chunks_exact(16) {
                let id = u32::from_le_bytes(px[0..4].try_into().unwrap());
                // Neighbouring pixels are mostly the same triangle, and the
                // lookup is the only cost here worth saving.
                if id == 0 || id == last {
                    continue;
                }
                last = id;
                if let Some(i) = owner(id - 1)
                    && let Some(s) = seen.get_mut(i)
                {
                    *s = true;
                }
            }
        }
        drop(mapped);
        read.unmap();
        seen
    }

    fn prepass(&self, enc: &mut wgpu::CommandEncoder, visible: &[bool]) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("prepass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.targets.gbuffer,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Depth 0 in the alpha channel marks "nothing here", which
                    // the occlusion pass treats as never occluding.
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.targets.gdepth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(FAR_DEPTH),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.gbuffer_pipeline);
        pass.set_bind_group(0, &self.bare_bind, &[]);
        pass.set_vertex_buffer(1, self.instance_buf.slice(..));
        // Hidden parts, and coarse tessellations replaced by refined ones,
        // must not be here either, or the occlusion and silhouette still show
        // them.
        self.draw_all_tris(&mut pass, visible, &[], |p| p.tris.clone());
    }

    fn occlusion_pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &wgpu::RenderPipeline,
        target: &wgpu::TextureView,
        bind: &wgpu::BindGroup,
    ) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Accumulates the transparent parts, weighted by depth.
    ///
    /// Shares the forward pass's depth buffer so a pane behind a wall is
    /// hidden, and loads rather than clears it: the opaque geometry has to
    /// still be in there.
    fn transparent_pass(&self, enc: &mut wgpu::CommandEncoder, visible: &[bool], paint: &[[u8; 4]]) {
        fn attach<'a>(
            view: &'a wgpu::TextureView,
            resolve: &'a wgpu::TextureView,
            clear: wgpu::Color,
        ) -> Option<wgpu::RenderPassColorAttachment<'a>> {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: Some(resolve),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })
        }
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("transparent"),
            color_attachments: &[
                // The sum starts at nothing, the product at one.
                attach(&self.targets.accum, &self.targets.accum_resolved, wgpu::Color::TRANSPARENT),
                attach(&self.targets.reveal, &self.targets.reveal_resolved, wgpu::Color::WHITE),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.targets.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.oit_pipeline);
        pass.set_bind_group(0, &self.targets.scene_bind, &[]);
        pass.set_vertex_buffer(1, self.instance_buf.slice(..));
        self.draw_all_tris(&mut pass, visible, paint, |p| {
            if p.alpha < 1.0 { p.tris.clone() } else { 0..0 }
        });
    }

    /// Divides the sum by its own weight and blends it over the opaque image.
    fn composite_pass(&self, enc: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("oit composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                // Straight onto the frame the forward pass already resolved.
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, &self.targets.oit_bind, &[]);
        pass.draw(0..3, 0..1);
    }

    fn forward_pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        f: &Frame,
        paint: &[[u8; 4]],
    ) {
        let msaa = self.samples > 1;
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("forward"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: if msaa { &self.targets.color } else { view },
                depth_slice: None,
                resolve_target: if msaa { Some(view) } else { None },
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.052,
                        g: 0.056,
                        b: 0.064,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.targets.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(FAR_DEPTH),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: if self.depth_format == DEPTH_STENCIL_FORMAT {
                    Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: wgpu::StoreOp::Discard,
                    })
                } else {
                    None
                },
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &self.targets.scene_bind, &[]);
        // The floor first, so its blended rim composites over the background.
        // Not under a wireframe, where it would hide the mesh being read.
        if f.ground && f.show_faces && self.index_count > 0 {
            pass.set_pipeline(&self.ground_pipeline);
            pass.draw(0..6, 0..1);
        }
        // Everything, or the visible parts coalesced into as few runs as they
        // will go, a painted part drawn through its twin. The common case
        // costs one call per slot.
        if f.show_faces && self.index_count > 0 {
            pass.set_pipeline(&self.surface_pipeline);
            pass.set_vertex_buffer(1, self.instance_buf.slice(..));
            self.draw_all_tris(&mut pass, f.visible, paint, |p| {
                if p.alpha < 1.0 { 0..0 } else { p.tris.clone() }
            });
        }
        // The silhouette over the shaded surfaces, under the B-rep overlay: the
        // exact lines win wherever both would draw.
        if f.silhouette && f.show_faces && f.edges != EdgeMode::None && self.index_count > 0 {
            pass.set_pipeline(&self.silhouette_pipeline);
            pass.draw(0..3, 0..1);
        }
        // Each part's edges run loose curves, then sharp, then tangent, so the
        // modes differ only in how much is drawn. Curves have their own
        // pipeline, which does not pull them towards the eye: they border no
        // face.
        if f.edges != EdgeMode::None && self.edge_count > 0 {
            pass.set_vertex_buffer(1, self.instance_buf.slice(..));
            // Always by part: a streamed load's batches are each grouped on
            // their own, so there is no model-wide split point. With
            // everything visible the runs coalesce to one draw per class.
            pass.set_pipeline(&self.wire_pipeline);
            self.draw_all_lines(&mut pass, f.visible, paint, |p| p.wire.clone());
            pass.set_pipeline(&self.edge_pipeline);
            self.draw_all_lines(&mut pass, f.visible, paint, |p| p.sharp.clone());
            if f.edges == EdgeMode::All {
                self.draw_all_lines(&mut pass, f.visible, paint, |p| p.tangent.clone());
            }
        }
    }

    /// Marks where the section plane passes through material, and writes that
    /// coverage to a buffer of its own.
    ///
    /// Its own pass because the mask needs a colour attachment the forward pass
    /// does not have, and it loads the depth and stencil the forward pass left.
    fn cap_mask_pass(&self, enc: &mut wgpu::CommandEncoder, f: &Frame) {
        let (Some((mark, mask)), _) = (&self.cap_mark_pipeline, ()) else { return };
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cap mask"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.targets.cap_mask,
                depth_slice: None,
                resolve_target: Some(&self.targets.cap_mask_resolved),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.targets.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0),
                    store: wgpu::StoreOp::Discard,
                }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &self.targets.scene_bind, &[]);
        pass.set_stencil_reference(0);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.set_vertex_buffer(1, self.instance_buf.slice(..));
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        // Volumes only. A sheet has no inside, so counting its entries would
        // fill its whole silhouette with cap.
        let cappable = |p: &Part| {
            if p.cappable && p.alpha >= 1.0 { p.tris.clone() } else { 0..0 }
        };
        // Mirrored placements go through the swapped-count pipeline: a
        // reflection turns every entry into an exit.
        //
        // A part drawn refined is counted refined, so the cap matches the
        // surfaces the forward pass drew.
        for s in &self.slots {
            let parts = &self.parts[s.parts.clone()];
            let vis = self.slot_visible(s, f.visible);
            let ranges: Vec<Range<u32>> = match &self.refined {
                Some(r) => runs(parts, vis, cappable).into_iter().flat_map(|x| r.around_holes(x)).collect(),
                None => runs(parts, vis, cappable),
            };
            let plain = s.instances.start..s.mirror_from;
            let mirrored = s.mirror_from..s.instances.end;
            if !plain.is_empty() {
                pass.set_pipeline(mark);
                for r in &ranges {
                    pass.draw_indexed(r.clone(), 0, plain.clone());
                }
            }
            if !mirrored.is_empty()
                && let Some(mm) = &self.cap_mark_mirror
            {
                pass.set_pipeline(mm);
                for r in &ranges {
                    pass.draw_indexed(r.clone(), 0, mirrored.clone());
                }
            }
        }
        // Refined parts belong to a single model, whose one placement is the
        // identity and never mirrored.
        if self.refined.is_some() {
            pass.set_pipeline(mark);
            self.draw_refined_tris(&mut pass, f.visible, &[], cappable);
        }
        pass.set_pipeline(mask);
        pass.draw(0..6, 0..1);
    }

    /// Erodes that mask and lays the cut over the frame.
    fn cap_fill_pass(&self, enc: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let Some(fill) = &self.cap_fill_pipeline else { return };
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cap fill"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(fill);
        pass.set_bind_group(0, &self.targets.cap_bind, &[]);
        pass.draw(0..3, 0..1);
    }

}
