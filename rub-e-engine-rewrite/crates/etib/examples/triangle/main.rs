//! A rotating triangle rasterized entirely in a compute shader.
//!
//! Run it with:
//!
//! ```text
//! cargo run --example triangle
//! ```
//!
//! The frame pipeline is three passes:
//!   1. a compute pass writes the triangle into an Rgba16Float storage texture,
//!      brighter than the display can show,
//!   2. a post-processing pass tone maps and grades that texture onto the
//!      swapchain image, optionally through a look-up table,
//!   3. egui paints its control panel on top, after the grade rather than
//!      through it.
//!
//! If the platform offers an extended-range surface, that is what gets
//! configured, and the same three passes present to it: the tone curve then
//! maps onto the display's real headroom instead of onto white, and the panel
//! is composited at SDR white. The panel reports which of the two is in use.
//!
//! Everything that is not about the triangle comes from the workspace's three
//! library crates: `etib-core` owns the GPU objects (textures, bindings,
//! pipelines, passes), `etib` owns the window and the frame loop, and the
//! optional `etib-ui` owns the egui panel. This example wires those pieces
//! together and implements [`etib::App`]; it names no graphics API of its own.
//!
//! `--lut teal` grades through a look-up table built into the example, and
//! `--lut <file.cube>` through one exported from a grading tool. The panel
//! offers the same choice, with a slider for how much of it to mix in.
//!
//! Pass `--sdr` to force a standard-range surface, which is worth comparing
//! against on an HDR display: the same scene, tone mapped to white instead of
//! to the display's peak. Pass `--hdr` to take an extended-range surface even
//! where the platform reports nothing about the display, which is the only way
//! to reach that path on the backends that report no luminance at all.
//!
//! Logging is on throughout. Set RUST_LOG to change the level, e.g.
//!   RUST_LOG=triangle=trace,etib_core=trace  per-frame dispatch detail
//!   RUST_LOG=etib_core=debug                 resource allocation and surface changes
//!   RUST_LOG=debug                           the above plus wgpu and winit internals

mod compute;
mod ui;

// Shared with the chart example, which offers the same tables.
#[path = "../common/lut.rs"]
mod lut;

use anyhow::Result;
use etib::winit::event::WindowEvent;
use etib::winit::window::Window;
use etib::ui::EguiLayer;
use etib::{App, Config, Frame, Gpu, Hdr, PostProcess};
use log::info;

use crate::compute::{Triangle, TriangleCompute};
use crate::lut::Tables;
use crate::ui::{UiState, build_panel};

fn main() -> Result<()> {
    init_logging();

    info!(
        "starting the triangle example (etib {}, target {} {})",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    );

    // An extended-range surface is taken when the platform offers one and the
    // display has headroom for it. `--sdr` opts out, which is how to see the
    // two side by side; `--hdr` takes one regardless of what the display says.
    let hdr = if std::env::args().any(|arg| arg == "--sdr") {
        Hdr::Off
    } else if std::env::args().any(|arg| arg == "--hdr") {
        Hdr::Force
    } else {
        Hdr::Allow
    };

    let tables = Tables::from_args()?;

    etib::run(
        Config {
            title: "etib - compute shader triangle".to_owned(),
            hdr,
            ..Default::default()
        },
        move |gpu| Ok(TriangleApp::new(gpu, tables.clone())),
    )
}

/// Default to info for this example and the library, and warn for everything
/// else, so the wgpu and naga backends stay quiet unless RUST_LOG asks for them.
fn init_logging() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,etib=info,etib_core=info,etib_ui=info,triangle=info"),
    )
    .format_timestamp_millis()
    .init();
}

struct TriangleApp {
    compute: TriangleCompute,
    post: PostProcess,
    egui: EguiLayer,
    state: UiState,
    /// The tables the panel can choose between, and which of them is on the
    /// GPU.
    luts: Tables,
    /// Drives the rotation. Advances only while unpaused.
    angle: f32,
    /// Seconds since the display was last asked what it can show. Its answer
    /// moves with the brightness slider, so it is worth re-reading, but not
    /// every frame.
    since_display_poll: f32,
}

impl TriangleApp {
    fn new(gpu: &Gpu, mut luts: Tables) -> Self {
        let (width, height) = gpu.size();
        let compute = TriangleCompute::new(&gpu.device, width, height);
        let mut post = PostProcess::new(&gpu.device, gpu.format(), compute.target_view());
        let egui = EguiLayer::new(&gpu.device, gpu.format(), gpu.color_space(), &gpu.window);

        // A table named on the command line was asked for, so it starts
        // applied; otherwise the example opens looking as it always has.
        let choice = luts.initial();
        luts.apply(&mut post, &gpu.device, compute.target_view(), choice);

        Self {
            compute,
            post,
            egui,
            state: UiState::new(gpu, choice, luts.file_name().map(str::to_owned)),
            luts,
            angle: 0.0,
            since_display_poll: 0.0,
        }
    }
}

impl App for TriangleApp {
    fn window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        // egui gets first refusal so its widgets can capture clicks and
        // keystrokes before the runner reacts to them.
        self.egui.on_window_event(window, event)
    }

    fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        if self.compute.resize(&gpu.device, width, height) {
            self.post.rebind(&gpu.device, self.compute.target_view());
        }
    }

    fn update(&mut self, dt: f32) {
        if !self.state.paused {
            self.angle += dt * self.state.rotation_speed;
            self.angle = self.angle.rem_euclid(std::f32::consts::TAU);
        }
        self.since_display_poll += dt;
    }

    fn render(&mut self, frame: Frame<'_>) {
        let Frame {
            gpu,
            encoder,
            view,
            stats,
        } = frame;

        // 1. Compute pass writes the triangle into the storage texture.
        self.compute.dispatch(
            &gpu.device,
            encoder,
            &Triangle {
                angle: self.angle,
                radius: self.state.radius,
                color: self.state.color_linear(),
                intensity: self.state.intensity,
            },
        );

        // Follow the display: the SDR white level and the headroom both move
        // with the brightness slider, and the window can be dragged to another
        // monitor entirely.
        if self.since_display_poll >= 1.0 {
            self.since_display_poll = 0.0;
            self.state.poll_display(gpu);
        }

        // The panel may have chosen a different table since the last frame.
        self.luts.apply(
            &mut self.post,
            &gpu.device,
            self.compute.target_view(),
            self.state.lut,
        );

        // 2. Post-processing brings that scene-referred image into the display's
        //    range and grades it onto the swapchain image.
        self.post
            .draw(&gpu.device, encoder, view, &self.state.grade, self.state.output);

        // 3. egui paints over the top.
        // Copied out before the panel borrows the state it lives in.
        let output = self.state.output;
        let state = &mut self.state;
        self.egui.render(
            &gpu.device,
            encoder,
            &gpu.window,
            view,
            gpu.size(),
            output,
            |ui| build_panel(ui, state, &stats),
        );
    }
}
