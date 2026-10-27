//! The egui layer itself: input plumbing and the paint pass.

use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use etib_core::{ColorSpace, Device, Encoder, Load, Output, TextureFormat, TextureView};
use log::{debug, info, trace};
use winit::window::Window;

use crate::composite::Composite;

/// Owns the egui context, the winit input bridge and the paint backend.
///
/// The widgets themselves are none of this type's business: [`EguiLayer::render`]
/// takes a closure that builds the frame's UI.
pub struct EguiLayer {
    context: egui::Context,
    state: egui_winit::State,
    renderer: Renderer,
    /// Present only for a surface egui cannot paint into directly, which is
    /// every extended-range one. See [`crate::composite`].
    composite: Option<Composite>,
}

impl EguiLayer {
    /// `target` and `color_space` describe the surface the interface ends up
    /// on. On an extended-range surface egui paints into a standard-range
    /// texture that is composited afterwards, which costs one texture and one
    /// full-target pass; on an sRGB surface it paints straight into it.
    pub fn new(
        device: &Device,
        target: TextureFormat,
        color_space: ColorSpace,
        window: &Window,
    ) -> Self {
        info!("initialising egui for {target:?} {color_space:?}");
        let context = egui::Context::default();

        let state = egui_winit::State::new(
            context.clone(),
            context.viewport_id(),
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            // Let egui pick a texture-atlas limit from the device limits.
            Some(device.max_texture_size() as usize),
        );

        let size = window.inner_size();
        let composite = color_space
            .is_hdr()
            .then(|| Composite::new(device, target, size.width, size.height));

        // egui encodes for the format it is given, so it is built for whichever
        // one it actually paints into.
        let egui_target = match &composite {
            Some(_) => Composite::FORMAT,
            None => target,
        };

        let renderer = Renderer::new(
            device.raw(),
            egui_target.raw(),
            RendererOptions {
                // egui already antialiases via feathering and nothing 3D is
                // drawn inside an egui panel, so MSAA would only cost memory.
                msaa_samples: 1,
                ..Default::default()
            },
        );
        info!("egui renderer ready");

        Self {
            context,
            state,
            renderer,
            composite,
        }
    }

    /// The underlying context, for callers that need it outside of a frame.
    pub fn context(&self) -> &egui::Context {
        &self.context
    }

    /// Feeds a window event to egui. Returns true when egui consumed it, in
    /// which case the caller should not act on it itself.
    pub fn on_window_event(&mut self, window: &Window, event: &winit::event::WindowEvent) -> bool {
        let response = self.state.on_window_event(window, event);
        if response.consumed {
            trace!("egui consumed a window event");
        }
        response.consumed
    }

    /// Builds the frame's UI with `build_ui` and records its draw commands into
    /// the encoder. Must run after whatever it paints over, since the interface
    /// is composited onto the target rather than replacing it.
    ///
    /// `build_ui` is handed the frame's root [`egui::Ui`], which is what panels
    /// attach to; `ui.ctx()` reaches the context for windows and settings.
    ///
    /// `output` says where SDR white sits on the target surface, so that the
    /// interface lands at the same brightness as the rest of the desktop;
    /// [`Output::SDR`] is right for an ordinary sRGB surface.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &Device,
        encoder: &mut Encoder,
        window: &Window,
        target: &TextureView,
        size_in_pixels: (u32, u32),
        output: Output,
        mut build_ui: impl FnMut(&mut egui::Ui),
    ) {
        let raw_input = self.state.take_egui_input(window);
        let painted = self.context.run_ui(raw_input, |ui| build_ui(ui));

        self.state
            .handle_platform_output(window, painted.platform_output);

        let paint_jobs = self
            .context
            .tessellate(painted.shapes, painted.pixels_per_point);

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [size_in_pixels.0, size_in_pixels.1],
            pixels_per_point: painted.pixels_per_point,
        };

        // TexturesDelta asserts on drop that every delta was handled, so take
        // ownership here and clear it once the deltas have been applied.
        let mut textures_delta = painted.textures_delta;

        // Each texture can accumulate several deltas in one pass; they have to
        // be applied in order.
        for (id, image_deltas) in &textures_delta.set {
            for image_delta in image_deltas {
                self.renderer
                    .update_texture(device.raw(), device.raw_queue(), *id, image_delta);
            }
        }

        // May return copy commands that have to be submitted before ours.
        let user_buffers = self.renderer.update_buffers(
            device.raw(),
            device.raw_queue(),
            encoder.raw_mut(),
            &paint_jobs,
            &screen_descriptor,
        );
        if !user_buffers.is_empty() {
            debug!(
                "submitting {} egui user command buffer(s)",
                user_buffers.len()
            );
            device.raw_queue().submit(user_buffers);
        }

        // Either paint into the target directly, or into the interface texture
        // and composite that over the target afterwards.
        let (paint_target, load) = match self.composite.as_mut() {
            Some(composite) => (
                composite.prepare(device, size_in_pixels.0, size_in_pixels.1),
                Composite::LOAD,
            ),
            // Keep, not clear: the scene is already in this attachment.
            None => (target, Load::Keep),
        };

        {
            let mut pass = encoder.render_pass("egui", paint_target, load);
            self.renderer
                .render(pass.raw_mut(), &paint_jobs, &screen_descriptor);
        }

        if let Some(composite) = self.composite.as_ref() {
            composite.draw(device, encoder, target, output);
        }

        for id in &textures_delta.free {
            self.renderer.free_texture(id);
        }
        textures_delta.clear();
    }
}
