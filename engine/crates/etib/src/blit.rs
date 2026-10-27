//! Presents an offscreen texture on the swapchain image.
//!
//! This copies what it is given, so it belongs on a standard-range surface,
//! where the values in a display-referred texture already mean what the surface
//! wants. On an extended-range one they would come out at a fraction of the
//! desktop's brightness; [`PostProcess`](crate::PostProcess) is the pass that
//! knows where to put them.

use etib_core::{
    BindGroup, BindLayout, Blend, Color, Device, Encoder, Filter, Load, RenderPipeline, Sampler,
    Shader, Stage, TextureFormat, TextureView,
};
use log::debug;

/// A 1:1 copy of a display-referred texture onto the swapchain image, with no
/// tone curve: see the module documentation for where it belongs.
pub struct Blit {
    pipeline: RenderPipeline,
    layout: BindLayout,
    sampler: Sampler,
    bindings: BindGroup,
}

impl Blit {
    /// A pass drawing `source` into targets of `target`'s format. Call
    /// [`rebind`](Self::rebind) when the source is recreated.
    pub fn new(device: &Device, target: TextureFormat, source: &TextureView) -> Self {
        let shader = Shader::wgsl(device, "blit", include_str!("shaders/blit.wgsl"));

        let layout = BindLayout::build(device, "blit")
            .texture(Stage::Fragment)
            .sampler(Stage::Fragment)
            .finish();

        let pipeline = RenderPipeline::screen(
            device,
            "blit",
            &shader,
            ("vs_main", "fs_main"),
            target,
            Blend::Replace,
            &layout,
        );

        // The source is always the same size as the target, so this is a 1:1
        // copy and the filter mode never actually kicks in.
        let sampler = Sampler::new(device, "blit", Filter::Linear);
        let bindings = layout.group(device, "blit", &[source, &sampler]);

        Self {
            pipeline,
            layout,
            sampler,
            bindings,
        }
    }

    /// Rebinds after the source texture has been recreated on resize.
    pub fn rebind(&mut self, device: &Device, source: &TextureView) {
        debug!("rebinding blit to the new source texture");
        self.bindings = self.layout.group(device, "blit", &[source, &self.sampler]);
    }

    /// Records the fullscreen draw. This clears the target, so it must run
    /// before anything that paints on top of it.
    pub fn draw(&self, encoder: &mut Encoder, target: &TextureView) {
        encoder
            .render_pass("blit", target, Load::Clear(Color::BLACK))
            .draw(&self.pipeline, &self.bindings, 0..3);
    }
}
