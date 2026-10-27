//! The compute pass that draws the chart.

use bytemuck::{Pod, Zeroable};
use etib::{
    BindGroup, BindLayout, ComputePipeline, Device, Encoder, Shader, Stage, StorageFormat,
    StorageTexture, TextureView, UniformBuffer,
};

/// Must match the workgroup size declared in chart_compute.wgsl.
const WORKGROUP_SIZE: (u32, u32) = (8, 8);

/// Mirror of the Params struct in chart_compute.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Params {
    inset: f32,
    _padding: [f32; 3],
}

pub struct ChartCompute {
    pipeline: ComputePipeline,
    layout: BindLayout,
    uniforms: UniformBuffer<Params>,
    target: StorageTexture,
    bindings: BindGroup,
    /// Fraction of the width the panel occupies, which the chart lays itself
    /// out beside.
    inset: f32,
    /// The chart only changes when the target or the layout does, so it is
    /// drawn on demand rather than every frame. The texture holds the result in
    /// between, which is the whole point of drawing into one.
    dirty: bool,
}

impl ChartCompute {
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        let shader = Shader::wgsl(device, "chart", include_str!("chart_compute.wgsl"));

        // Float, not 8-bit: most of the chart is brighter than the display can
        // show, which is the whole subject of the exercise.
        let target = StorageTexture::new(
            device,
            "chart target",
            width,
            height,
            StorageFormat::Rgba16Float,
        );

        let uniforms = UniformBuffer::new(device, "chart params");

        let layout = BindLayout::build(device, "chart")
            .storage_texture(Stage::Compute, target.format())
            .uniform::<Params>(Stage::Compute)
            .finish();
        let bindings = layout.group(device, "chart", &[target.view(), &uniforms]);

        let pipeline = ComputePipeline::new(device, "chart", &shader, "main", &layout);

        Self {
            pipeline,
            layout,
            uniforms,
            target,
            bindings,
            inset: 0.0,
            dirty: true,
        }
    }

    /// Tells the chart how much of the width the panel is using, so it can lay
    /// itself out in what is left. Redraws only when the figure actually moves.
    pub fn set_inset(&mut self, inset: f32) {
        let inset = inset.clamp(0.0, 0.8);
        if (inset - self.inset).abs() > 0.001 {
            self.inset = inset;
            self.dirty = true;
        }
    }

    /// The texture the post-processing pass reads.
    pub fn target_view(&self) -> &TextureView {
        self.target.view()
    }

    /// Returns `true` when the target view changed and the post-processing pass
    /// has to rebind.
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) -> bool {
        if !self.target.resize(device, width, height) {
            return false;
        }
        self.bindings = self
            .layout
            .group(device, "chart", &[self.target.view(), &self.uniforms]);
        self.dirty = true;
        true
    }

    /// Redraws the chart if anything invalidated it.
    pub fn dispatch(&mut self, device: &Device, encoder: &mut Encoder) {
        if !self.dirty {
            return;
        }
        self.dirty = false;

        self.uniforms.write(
            device,
            &Params {
                inset: self.inset,
                _padding: [0.0; 3],
            },
        );

        encoder.compute_pass("chart").dispatch_grid(
            &self.pipeline,
            &self.bindings,
            self.target.size(),
            WORKGROUP_SIZE,
        );
    }
}
