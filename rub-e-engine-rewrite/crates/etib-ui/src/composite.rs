//! Drawing the interface in standard range, then compositing it onto an
//! extended-range surface.
//!
//! egui picks how to encode its output from the target's format: for an sRGB
//! format it writes linear values and lets the hardware encode, and for
//! anything else it writes sRGB-encoded values directly. Neither is right for a
//! float extended-range surface, which carries a linear signal that is not
//! limited to 0..1 — handed one, egui takes the second path and the interface
//! comes out washed out.
//!
//! So on such a surface the interface gets its own sRGB texture, which is the
//! case egui already handles correctly, and this composites the result.

use bytemuck::{Pod, Zeroable};
use etib_core::{
    BindGroup, BindLayout, Blend, Device, Encoder, Filter, Load, Output, RenderPipeline,
    RenderTexture, Sampler, Shader, Stage, TextureFormat, TextureView, UniformBuffer,
};
use log::debug;

/// Mirror of the Params struct in composite.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Params {
    sdr_white: f32,
    _padding: [f32; 3],
}

pub(crate) struct Composite {
    target: RenderTexture,
    pipeline: RenderPipeline,
    layout: BindLayout,
    sampler: Sampler,
    uniforms: UniformBuffer<Params>,
    bindings: BindGroup,
}

impl Composite {
    /// The format egui paints into. Also what its renderer has to be built for.
    pub(crate) const FORMAT: TextureFormat = TextureFormat::RGBA8_SRGB;

    pub(crate) fn new(device: &Device, surface: TextureFormat, width: u32, height: u32) -> Self {
        debug!("interface will be drawn in standard range and composited");
        let shader = Shader::wgsl(device, "ui composite", include_str!("composite.wgsl"));

        let target = RenderTexture::new(device, "ui", width.max(1), height.max(1), Self::FORMAT);

        let layout = BindLayout::build(device, "ui composite")
            .texture(Stage::Fragment)
            .sampler(Stage::Fragment)
            .uniform::<Params>(Stage::Fragment)
            .finish();

        let pipeline = RenderPipeline::screen(
            device,
            "ui composite",
            &shader,
            ("vs_main", "fs_main"),
            surface,
            // egui's own output is premultiplied, and so is what it left in the
            // texture, so this blends it over the scene the same way.
            Blend::Premultiplied,
            &layout,
        );

        // Source and target are the same size, so this never interpolates.
        let sampler = Sampler::new(device, "ui composite", Filter::Nearest);
        let uniforms = UniformBuffer::new(device, "ui composite params");
        let bindings = layout.group(device, "ui composite", &[target.view(), &sampler, &uniforms]);

        Self {
            target,
            pipeline,
            layout,
            sampler,
            uniforms,
            bindings,
        }
    }

    /// The view egui paints into. Resizes to match the surface first.
    pub(crate) fn prepare(&mut self, device: &Device, width: u32, height: u32) -> &TextureView {
        if self.target.resize(device, width, height) {
            self.bindings = self.layout.group(
                device,
                "ui composite",
                &[self.target.view(), &self.sampler, &self.uniforms],
            );
        }
        self.target.view()
    }

    /// How a pass painting the interface should start: nothing of the scene is
    /// in this texture, and what egui does not cover has to stay transparent.
    pub(crate) const LOAD: Load = Load::Clear(etib_core::Color::TRANSPARENT);

    /// Draws the interface texture over `target`.
    pub(crate) fn draw(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        target: &TextureView,
        output: Output,
    ) {
        self.uniforms.write(
            device,
            &Params {
                sdr_white: output.sdr_white.max(0.0),
                _padding: [0.0; 3],
            },
        );

        encoder
            .render_pass("ui composite", target, Load::Keep)
            .draw(&self.pipeline, &self.bindings, 0..3);
    }
}
