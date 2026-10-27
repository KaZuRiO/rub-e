//! A tone-mapping test chart: the same scene under every curve, so the
//! differences between them are something you can look at rather than reason
//! about.
//!
//! Run it with:
//!
//! ```text
//! cargo run --bin chart
//! ```
//!
//! A compute pass draws four zones of known, scene-referred values — most of
//! them far brighter than any display can show — and the post-processing pass
//! is the only thing standing between those values and the window. Switching
//! the tone curve therefore changes the picture and nothing else does.
//!
//! What to look for is written into the panel, but in short: zone 1 shows where
//! each curve puts the midtones and where it runs out of range, zone 2 shows
//! whether it uses an HDR display's headroom, zone 3 shows what it does to
//! saturated colour, and zone 4 shows the shape of the roll-off.
//!
//! `--sdr` forces a standard-range surface, which is the comparison worth
//! making on an HDR display: run it twice and watch zone 2 collapse. `--hdr`
//! takes an extended-range surface even where the platform reports nothing
//! about the display. `--curve none|reinhard|aces` picks which curve to start
//! on, for looking at one without reaching for the panel. `--lut teal` grades
//! through a look-up table built into the examples, and `--lut <file.cube>`
//! through one exported from a grading tool — the chart is a good place to see
//! what a table does, since every colour it touches is on screen at once.

mod compute;
mod ui;

// Shared with the triangle example, which offers the same tables.
#[path = "../common/lut.rs"]
mod lut;

use anyhow::Result;
use etib::ui::EguiLayer;
use etib::winit::event::WindowEvent;
use etib::winit::window::Window;
use etib::{App, Config, Frame, Gpu, Hdr, PostProcess, Tonemap};
use log::info;

use crate::compute::ChartCompute;
use crate::lut::Tables;
use crate::ui::{UiState, build_panel};

fn main() -> Result<()> {
    init_logging();

    info!(
        "starting the tone-mapping chart (etib {}, target {} {})",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    );

    let hdr = if std::env::args().any(|arg| arg == "--sdr") {
        Hdr::Off
    } else if std::env::args().any(|arg| arg == "--hdr") {
        Hdr::Force
    } else {
        Hdr::Allow
    };

    let curve = curve_from_args();
    let tables = Tables::from_args()?;

    etib::run(
        Config {
            title: "etib - tone mapping chart".to_owned(),
            size: (1300, 900),
            hdr,
            ..Default::default()
        },
        move |gpu| Ok(ChartApp::new(gpu, curve, tables.clone())),
    )
}

/// `--curve <name>`, defaulting to whatever the grade defaults to.
fn curve_from_args() -> Tonemap {
    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        if arg != "--curve" {
            continue;
        }
        let Some(name) = args.next() else { break };
        match Tonemap::ALL.iter().find(|c| c.name().eq_ignore_ascii_case(&name)) {
            Some(curve) => return *curve,
            None => info!("unknown curve {name:?}; keeping the default"),
        }
    }
    Tonemap::default()
}

fn init_logging() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,etib=info,etib_core=info,chart=info"),
    )
    .format_timestamp_millis()
    .init();
}

struct ChartApp {
    chart: ChartCompute,
    post: PostProcess,
    egui: EguiLayer,
    state: UiState,
    /// The tables the panel can choose between, and which of them is on the
    /// GPU.
    luts: Tables,
    /// Seconds since the display was last asked what it can show.
    since_display_poll: f32,
}

impl ChartApp {
    fn new(gpu: &Gpu, curve: Tonemap, mut luts: Tables) -> Self {
        let (width, height) = gpu.size();
        let chart = ChartCompute::new(&gpu.device, width, height);
        let mut post = PostProcess::new(&gpu.device, gpu.format(), chart.target_view());
        let egui = EguiLayer::new(&gpu.device, gpu.format(), gpu.color_space(), &gpu.window);

        // Whatever `--lut` named starts applied; with no argument the chart
        // opens showing the curve and nothing else, as it always has.
        let choice = luts.initial();
        luts.apply(&mut post, &gpu.device, chart.target_view(), choice);

        let mut state = UiState::new(gpu, choice, luts.file_name().map(str::to_owned));
        state.grade.tonemap = curve;

        Self {
            chart,
            post,
            egui,
            state,
            luts,
            since_display_poll: 0.0,
        }
    }
}

impl App for ChartApp {
    fn window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        self.egui.on_window_event(window, event)
    }

    fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        if self.chart.resize(&gpu.device, width, height) {
            self.post.rebind(&gpu.device, self.chart.target_view());
        }
    }

    fn update(&mut self, dt: f32) {
        self.since_display_poll += dt;
    }

    fn render(&mut self, frame: Frame<'_>) {
        let Frame {
            gpu,
            encoder,
            view,
            stats,
        } = frame;

        if self.since_display_poll >= 1.0 {
            self.since_display_poll = 0.0;
            self.state.poll_display(gpu);
        }

        // 1. Draw the chart, which only happens when something invalidated it:
        //    a resize, or the panel changing width.
        self.chart.set_inset(self.state.inset);
        self.chart.dispatch(&gpu.device, encoder);

        // The panel may have chosen a different table since the last frame.
        self.luts.apply(
            &mut self.post,
            &gpu.device,
            self.chart.target_view(),
            self.state.lut,
        );

        // 2. The pass under test.
        self.post
            .draw(&gpu.device, encoder, view, &self.state.grade, self.state.output);

        // 3. The panel, composited after the grade rather than through it.
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
