//! Recording work: the frame's encoder, and the passes opened on it.

use std::marker::PhantomData;
use std::ops::Range;

use log::trace;

use crate::binding::BindGroup;
use crate::buffer::{IndexBuffer, IndirectBuffer, VertexSource};
use crate::device::Device;
use crate::pipeline::{ComputePipeline, RenderPipeline};
use crate::texture::{DepthTexture, TextureView};

/// A colour in linear space, 0..1 per channel.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl Color {
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };

    /// Fully transparent, for a target that will be composited over something.
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };
}

/// What a render pass does with the target's existing contents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Load {
    /// Overwrite everything with this colour first.
    Clear(Color),
    /// Keep what is there and draw over it.
    Keep,
}

/// What a render pass does with the depth buffer's existing contents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DepthLoad {
    /// Reset every depth to this value first; 1.0 is the far plane.
    Clear(f32),
    /// Keep the depths an earlier pass left, to draw more into the same scene.
    Keep,
}

/// A frame's recorded work, built up and then submitted in one go.
///
/// Open it with [`Device::encoder`] and hand it to [`Device::submit`] when the
/// frame is complete.
pub struct Encoder {
    inner: wgpu::CommandEncoder,
}

impl Encoder {
    pub(crate) fn new(device: &Device, label: &str) -> Self {
        Self {
            inner: device
                .raw()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) }),
        }
    }

    /// Opens a compute pass. Drop it to end the pass.
    pub fn compute_pass(&mut self, label: &str) -> ComputePass<'_> {
        let pass = self
            .inner
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
        ComputePass {
            inner: pass.forget_lifetime(),
            _encoder: PhantomData,
        }
    }

    /// Opens a render pass drawing into `target`. Drop it to end the pass.
    pub fn render_pass<'a>(
        &'a mut self,
        label: &str,
        target: &TextureView,
        load: Load,
    ) -> RenderPass<'a> {
        self.begin_render_pass(label, target, load, None)
    }

    /// Opens a render pass drawing into `target` and depth-testing against
    /// `depth`, which must be the same size. Pipelines drawn in it must have
    /// been created with a [`Depth`](crate::Depth).
    pub fn render_pass_with_depth<'a>(
        &'a mut self,
        label: &str,
        target: &TextureView,
        load: Load,
        depth: &DepthTexture,
        depth_load: DepthLoad,
    ) -> RenderPass<'a> {
        self.begin_render_pass(label, target, load, Some((depth, depth_load)))
    }

    fn begin_render_pass<'a>(
        &'a mut self,
        label: &str,
        target: &TextureView,
        load: Load,
        depth: Option<(&DepthTexture, DepthLoad)>,
    ) -> RenderPass<'a> {
        let pass = self.inner.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target.raw(),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: match load {
                        Load::Clear(c) => wgpu::LoadOp::Clear(wgpu::Color {
                            r: c.r,
                            g: c.g,
                            b: c.b,
                            a: c.a,
                        }),
                        Load::Keep => wgpu::LoadOp::Load,
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: depth.map(|(depth, load)| {
                wgpu::RenderPassDepthStencilAttachment {
                    view: depth.view().raw(),
                    depth_ops: Some(wgpu::Operations {
                        load: match load {
                            DepthLoad::Clear(value) => wgpu::LoadOp::Clear(value),
                            DepthLoad::Keep => wgpu::LoadOp::Load,
                        },
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }
            }),
            ..Default::default()
        });
        RenderPass {
            inner: pass.forget_lifetime(),
            _encoder: PhantomData,
        }
    }

    pub(crate) fn finish(self) -> wgpu::CommandBuffer {
        self.inner.finish()
    }

    /// Interop: the underlying wgpu encoder.
    pub fn raw_mut(&mut self) -> &mut wgpu::CommandEncoder {
        &mut self.inner
    }
}

/// A compute pass, open until it is dropped.
pub struct ComputePass<'a> {
    inner: wgpu::ComputePass<'static>,
    // The pass borrows the encoder for as long as it is open, even though the
    // wgpu pass inside has had its lifetime erased.
    _encoder: PhantomData<&'a mut Encoder>,
}

impl ComputePass<'_> {
    /// Selects the pipeline for the dispatches that follow. Together with
    /// [`set_bind_group`](Self::set_bind_group) and
    /// [`dispatch_workgroups`](Self::dispatch_workgroups), this is the long
    /// form of [`dispatch`](Self::dispatch), for a pipeline with more than one
    /// bind group.
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) {
        self.inner.set_pipeline(pipeline.raw());
    }

    /// Binds `bindings` as `@group(index)`.
    pub fn set_bind_group(&mut self, index: u32, bindings: &BindGroup) {
        self.inner.set_bind_group(index, bindings.raw(), &[]);
    }

    /// Runs `x`×`y`×`z` workgroups of the current pipeline.
    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        trace!("dispatching {x}x{y}x{z} workgroups");
        self.inner.dispatch_workgroups(x, y, z);
    }

    /// Runs `workgroups` workgroups of `pipeline`.
    pub fn dispatch(
        &mut self,
        pipeline: &ComputePipeline,
        bindings: &BindGroup,
        workgroups: (u32, u32, u32),
    ) {
        let (x, y, z) = workgroups;
        trace!("dispatching {x}x{y}x{z} workgroups");
        self.inner.set_pipeline(pipeline.raw());
        self.inner.set_bind_group(0, bindings.raw(), &[]);
        self.inner.dispatch_workgroups(x, y, z);
    }

    /// Runs enough workgroups of `workgroup_size` to cover a `width` x `height`
    /// grid, rounding up. The shader is responsible for discarding the overhang.
    pub fn dispatch_grid(
        &mut self,
        pipeline: &ComputePipeline,
        bindings: &BindGroup,
        (width, height): (u32, u32),
        (group_width, group_height): (u32, u32),
    ) {
        self.dispatch(
            pipeline,
            bindings,
            (width.div_ceil(group_width), height.div_ceil(group_height), 1),
        );
    }

    /// Interop: the underlying wgpu pass.
    pub fn raw_mut(&mut self) -> &mut wgpu::ComputePass<'static> {
        &mut self.inner
    }
}

/// A render pass, open until it is dropped.
pub struct RenderPass<'a> {
    inner: wgpu::RenderPass<'static>,
    _encoder: PhantomData<&'a mut Encoder>,
}

impl RenderPass<'_> {
    /// Draws `vertices` vertices with `pipeline`. With a vertexless pipeline
    /// such as [`RenderPipeline::screen`], `0..3` is a fullscreen triangle.
    pub fn draw(
        &mut self,
        pipeline: &RenderPipeline,
        bindings: &BindGroup,
        vertices: Range<u32>,
    ) {
        self.inner.set_pipeline(pipeline.raw());
        self.inner.set_bind_group(0, bindings.raw(), &[]);
        self.inner.draw(vertices, 0..1);
    }

    /// Selects the pipeline for the draws that follow. With the other setters
    /// below, this is the long form of [`draw`](Self::draw), for meshes.
    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
        self.inner.set_pipeline(pipeline.raw());
    }

    /// Binds `bindings` as `@group(index)`.
    pub fn set_bind_group(&mut self, index: u32, bindings: &BindGroup) {
        self.inner.set_bind_group(index, bindings.raw(), &[]);
    }

    /// Reads vertex buffer `slot` of the pipeline's layout from `buffer`.
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &impl VertexSource) {
        self.inner
            .set_vertex_buffer(slot, buffer.vertex_buffer().slice(..));
    }

    pub fn set_index_buffer(&mut self, buffer: &IndexBuffer) {
        self.inner
            .set_index_buffer(buffer.raw().slice(..), buffer.format().raw());
    }

    /// Draws `indices` of the current index buffer, `instances` times, with
    /// `base_vertex` added to each index.
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.inner.draw_indexed(indices, base_vertex, instances);
    }

    /// Draws with the arguments in draw `index` of `buffer`, as whatever pass
    /// wrote them left them.
    pub fn draw_indexed_indirect(&mut self, buffer: &IndirectBuffer, index: u32) {
        self.inner
            .draw_indexed_indirect(buffer.raw(), buffer.offset_of(index));
    }

    /// Interop: the underlying wgpu pass.
    pub fn raw_mut(&mut self) -> &mut wgpu::RenderPass<'static> {
        &mut self.inner
    }
}
