//! The compute-shader triangle rasterizer.

use bytemuck::{Pod, Zeroable};
use etib::{
    BindGroup, BindLayout, ComputePipeline, Device, Encoder, Shader, Stage, StorageFormat,
    StorageTexture, TextureView, UniformBuffer,
};

/// Must match the workgroup size declared in triangle_compute.wgsl.
const WORKGROUP_SIZE: (u32, u32) = (8, 8);

/// Mirror of the Params struct in the compute shader. repr(C) field order and
/// the 16-byte-aligned colour keep it laid out identically to the WGSL side.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Params {
    resolution: [f32; 2],
    angle: f32,
    radius: f32,
    /// Linear RGBA.
    color: [f32; 4],
    intensity: f32,
    /// WGSL rounds the struct up to a multiple of 16 bytes; bytemuck will not
    /// derive Pod for a struct with padding it cannot see, so it is spelled out.
    _padding: [f32; 3],
}

/// This frame's triangle, as the UI describes it.
pub struct Triangle {
    pub angle: f32,
    pub radius: f32,
    /// Linear RGB.
    pub color: [f32; 3],
    /// Multiplier on the fill colour. Above 1.0 the triangle is scene-referred
    /// and only the post-processing pass can bring it back into range.
    pub intensity: f32,
}

pub struct TriangleCompute {
    pipeline: ComputePipeline,
    layout: BindLayout,
    uniforms: UniformBuffer<Params>,
    target: StorageTexture,
    bindings: BindGroup,
}

impl TriangleCompute {
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        let shader = Shader::wgsl(device, "triangle", include_str!("triangle_compute.wgsl"));

        let target = StorageTexture::new(
            device,
            "triangle target",
            width,
            height,
            // Float, not 8-bit: the triangle is written brighter than the
            // display can show and tone mapped afterwards.
            StorageFormat::Rgba16Float,
        );
        let uniforms = UniformBuffer::new(device, "triangle params");

        let layout = BindLayout::build(device, "triangle")
            .storage_texture(Stage::Compute, target.format())
            .uniform::<Params>(Stage::Compute)
            .finish();
        let bindings = layout.group(device, "triangle", &[target.view(), &uniforms]);

        let pipeline = ComputePipeline::new(device, "triangle", &shader, "main", &layout);

        Self {
            pipeline,
            layout,
            uniforms,
            target,
            bindings,
        }
    }

    /// The texture the blit pass samples from.
    pub fn target_view(&self) -> &TextureView {
        self.target.view()
    }

    /// Recreates the storage texture when the window size changes. The pipeline
    /// and layout are size-independent and are reused. Returns `true` when the
    /// target view changed and the blit pass has to rebind.
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) -> bool {
        if !self.target.resize(device, width, height) {
            return false;
        }
        self.bindings = self
            .layout
            .group(device, "triangle", &[self.target.view(), &self.uniforms]);
        true
    }

    /// Uploads this frame's parameters and records the dispatch.
    pub fn dispatch(&self, device: &Device, encoder: &mut Encoder, triangle: &Triangle) {
        let (width, height) = self.target.size();
        self.uniforms.write(
            device,
            &Params {
                resolution: [width as f32, height as f32],
                angle: triangle.angle,
                radius: triangle.radius,
                color: [
                    triangle.color[0],
                    triangle.color[1],
                    triangle.color[2],
                    1.0,
                ],
                intensity: triangle.intensity,
                _padding: [0.0; 3],
            },
        );

        // One workgroup per 8x8 block of texels, rounded up; the shader
        // discards the overhang.
        encoder.compute_pass("triangle").dispatch_grid(
            &self.pipeline,
            &self.bindings,
            self.target.size(),
            WORKGROUP_SIZE,
        );
    }
}
