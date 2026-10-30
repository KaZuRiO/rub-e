//! Replayable benchmark: a deterministic cube world seen along a fixed camera
//! path, so two runs of the engine can be compared frame for frame.
//!
//! Run it from `engine/` with:
//!
//! ```text
//! cargo run --release --bin bench -- --save before
//! cargo run --release --bin bench -- --save after
//! cargo run --release --bin bench -- --compare before after
//! ```
//!
//! `--save` opens a window, plays the scenario, writes
//! `bench-results/<NAME>.json` and quits; Escape stops it without writing
//! anything, and so does a change of the window's size. `--compare` prints the
//! two runs side by side, without a window.

mod collect;
mod compare;
mod results;
mod scenario;

use std::cell::Cell;
use std::process::ExitCode;
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result};
use clap::Parser;
use etib::{App, Frame, Gpu, Grade, Hdr, Output, PostProcess, Vsync};
use etib_scene::glam::{Mat4, Vec3};
use etib_scene::{Camera, ModelCube, Projection, Scene};
use log::{error, info, warn};

use collect::Collector;
use results::{RunResults, ScenarioInfo};
use scenario::{FAR_DISTANCE, MEASURED_FRAMES, SCENARIO_VERSION, WARMUP_FRAMES, WINDOW};

const SKY: &[u8] = include_bytes!("../../assets/sky.hdr");
/// Texels along each side of a skybox face, as in `hello`.
const SKY_FACE: u32 = 1080;

#[derive(Parser)]
#[command(about = "Replays a fixed scenario and compares runs of it")]
#[command(group = clap::ArgGroup::new("mode").required(true).args(["save", "compare"]))]
struct Args {
    /// Play the scenario and save its results under this name.
    #[arg(long, value_name = "NAME")]
    save: Option<String>,
    /// Print two saved runs side by side, the first as the base.
    #[arg(long, num_args = 2, value_names = ["A", "B"])]
    compare: Option<Vec<String>>,
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(
        env_logger::Env::default()
            .default_filter_or("warn,etib=info,etib_core=info,etib_scene=info,bench=info"),
    )
    .format_timestamp_millis()
    .init();

    let args = Args::parse();
    let result = match (args.save, args.compare.as_deref()) {
        (Some(name), _) => save_run(name),
        (None, Some([a, b])) => compare_runs(a, b).map(|()| true),
        _ => unreachable!("clap requires exactly one of --save and --compare <A> <B>"),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn compare_runs(a: &str, b: &str) -> Result<()> {
    let dir = results::results_dir();
    let (a, b) = (results::load(a, &dir)?, results::load(b, &dir)?);
    print!("{}", compare::compare(&a, &b));
    Ok(())
}

/// How a `--save` run ended, shared between the app and `main`, which only
/// learns it once `etib::run` has returned.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Running,
    Saved,
    Failed,
}

/// Plays the scenario; `Ok(false)` when nothing was saved.
fn save_run(name: String) -> Result<bool> {
    let cubes = scenario::world();
    info!("{} cubes generated", cubes.len());
    let outcome = Rc::new(Cell::new(Outcome::Running));

    let app_outcome = Rc::clone(&outcome);
    etib::run(
        etib::Config {
            title: "etib - bench".to_owned(),
            size: WINDOW,
            hdr: Hdr::Off,
            vsync: Vsync::Off,
            fullscreen: false,
            exit_on_escape: true,
        },
        move |gpu| BenchApp::new(gpu, &cubes, name.clone(), Rc::clone(&app_outcome)),
    )?;

    Ok(match outcome.get() {
        Outcome::Saved => true,
        Outcome::Failed => false,
        // The window was closed, or Escape pressed, before the end.
        Outcome::Running => {
            eprintln!("interrupted; nothing was saved");
            false
        }
    })
}

struct BenchApp {
    name: String,
    scene: Scene,
    camera: Camera,
    post: PostProcess,
    grade: Grade,
    output: Output,
    collector: Collector,
    surface_size: (u32, u32),
    generated_cubes: u32,
    /// Frames rendered so far, which picks the camera pose.
    bench_frame: u32,
    /// The step of the frame being rendered, in seconds.
    dt: f32,
    outcome: Rc<Cell<Outcome>>,
}

impl BenchApp {
    fn new(
        gpu: &Gpu,
        cubes: &[ModelCube],
        name: String,
        outcome: Rc<Cell<Outcome>>,
    ) -> Result<Self> {
        let generated_cubes = u32::try_from(cubes.len()).context("too many cubes")?;
        let (width, height) = gpu.size();

        // Nothing dynamic: the scene only needs to be valid.
        let mut scene = Scene::new(&gpu.device, cubes, 1, width, height);
        scene
            .set_skybox_from_hdr(&gpu.device, SKY, SKY_FACE, Mat4::IDENTITY)
            .context("loading the skybox")?;
        let (eye, target) = scenario::camera_at(0);
        let camera = Camera::new(
            eye,
            target,
            Vec3::Y,
            width as f32 / height as f32,
            Projection::Perspective { fovy: 45.0 },
            0.1,
            FAR_DISTANCE,
        );
        let post = PostProcess::new(&gpu.device, gpu.format(), scene.target_view());

        if !gpu.device.gpu_timing_available() {
            warn!("this device has no timestamp queries: only CPU frame times are measured");
        }

        Ok(Self {
            name,
            scene,
            camera,
            post,
            grade: Grade::default(),
            output: Output::detect(gpu.color_space(), &gpu.surface.display()),
            collector: Collector::new((width, height), gpu.device.gpu_timing_available()),
            surface_size: (width, height),
            generated_cubes,
            bench_frame: 0,
            dt: 0.0,
            outcome,
        })
    }

    /// Writes the results file and reports where it went.
    fn save(&self, gpu: &Gpu) -> Outcome {
        let measured = self.collector.finish();
        if let Some(gpu_total) = &measured.gpu_total
            && gpu_total.samples < MEASURED_FRAMES
        {
            info!(
                "GPU timings arrived for {} of {MEASURED_FRAMES} measured frames",
                gpu_total.samples
            );
        }
        let run = RunResults {
            name: self.name.clone(),
            unix_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            adapter: gpu.device.adapter_name().to_owned(),
            backend: gpu.device.backend().to_owned(),
            surface_size: self.surface_size,
            git: results::git_info(),
            scenario: ScenarioInfo {
                version: SCENARIO_VERSION,
                warmup_frames: WARMUP_FRAMES,
                measured_frames: MEASURED_FRAMES,
                generated_cubes: self.generated_cubes,
                kept_cubes: self.scene.static_count(),
            },
            gpu_timing: gpu.device.gpu_timing_available(),
            cpu_frame: measured.cpu_frame,
            gpu_total: measured.gpu_total,
            passes: measured.passes,
        };
        match results::save(&run, &results::results_dir()) {
            Ok((path, overwrote)) => {
                let note = if overwrote { " (overwrote)" } else { "" };
                println!("saved {}{note}", path.display());
                Outcome::Saved
            }
            Err(err) => {
                error!("{err:#}");
                Outcome::Failed
            }
        }
    }
}

impl App for BenchApp {
    fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        // The collector fails the run on the next frame; until then, draw at
        // the new size rather than stretch.
        if self.scene.resize(&gpu.device, width, height) {
            self.post.rebind(&gpu.device, self.scene.target_view());
        }
    }

    fn update(&mut self, dt: f32) {
        self.dt = dt;
    }

    fn exit_requested(&self) -> bool {
        self.outcome.get() != Outcome::Running
    }

    fn render(&mut self, frame: Frame<'_>) {
        let Frame {
            gpu,
            encoder,
            view,
            stats,
        } = frame;

        (self.camera.eye, self.camera.target) = scenario::camera_at(self.bench_frame);
        self.scene.render(&gpu.device, encoder, &mut self.camera);
        self.post
            .draw(&gpu.device, encoder, view, &self.grade, self.output);

        // The collector ignores results it has already counted.
        if let Some(timings) = &stats.gpu {
            self.collector.timings(timings);
        }
        if let Err(err) = self.collector.frame_rendered(
            stats.frame_index,
            self.bench_frame,
            self.dt,
            stats.surface_size,
        ) {
            error!("{err}");
            self.outcome.set(Outcome::Failed);
            return;
        }
        self.bench_frame += 1;

        if self.collector.is_done(self.bench_frame) {
            self.outcome.set(self.save(gpu));
        }
    }
}
