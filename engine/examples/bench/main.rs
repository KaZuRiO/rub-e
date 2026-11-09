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
//! `--scenario` picks what `--save` plays: `terrain` (the default), `scattered`,
//! `moving` or `few-changes`. Each run goes under its own name, so one per
//! scenario is `--save before-moving --scenario moving`; comparing two
//! scenarios warns that the numbers are not comparable.
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
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result};
use clap::Parser;
use etib::{App, Frame, Gpu, Grade, Hdr, Output, PostProcess, Vsync};
use etib_scene::glam::{Mat4, Vec3};
use etib_scene::{Camera, DynamicId, DynamicModel, ModelCube, Projection, Scene};
use log::{error, info, warn};

use collect::Collector;
use results::{RunResults, ScenarioInfo};
use scenario::{
    FAR_DISTANCE, MEASURED_FRAMES, SCENARIO_VERSION, Scenario, WARMUP_FRAMES, WINDOW,
    model_position,
};

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
    /// What to play, with --save.
    #[arg(long, value_enum, default_value_t, conflicts_with = "compare")]
    scenario: Scenario,
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
        (Some(name), _) => save_run(name, args.scenario),
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
fn save_run(name: String, scenario: Scenario) -> Result<bool> {
    results::check_name(&name)?;
    let existing = results::results_dir().join(format!("{name}.json"));
    if existing.exists() {
        warn!("{} will be overwritten", existing.display());
    }
    let cubes = scenario.static_cubes();
    let models = scenario.models();
    info!(
        "{}: {} cubes generated, {} dynamic models",
        scenario.name(),
        cubes.len(),
        models.len()
    );
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
        move |gpu| {
            BenchApp::new(
                gpu,
                scenario,
                &cubes,
                &models,
                name.clone(),
                Rc::clone(&app_outcome),
            )
        },
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
    scenario: Scenario,
    scene: Scene,
    /// The dynamic models, in the order the scenario counts them.
    models: Vec<DynamicId>,
    camera: Camera,
    post: PostProcess,
    grade: Grade,
    output: Output,
    collector: Collector,
    surface_size: (u32, u32),
    generated_cubes: u32,
    dynamic_cubes: u32,
    /// Frames rendered so far, which picks the camera pose.
    bench_frame: u32,
    /// When the previous frame was rendered. The bench times frames itself:
    /// `App::update`'s step is capped at `MAX_DT`, which would hide stalls.
    last_render: Option<Instant>,
    outcome: Rc<Cell<Outcome>>,
}

impl BenchApp {
    fn new(
        gpu: &Gpu,
        scenario: Scenario,
        cubes: &[ModelCube],
        models: &[DynamicModel],
        name: String,
        outcome: Rc<Cell<Outcome>>,
    ) -> Result<Self> {
        let generated_cubes = u32::try_from(cubes.len()).context("too many cubes")?;
        let (width, height) = gpu.size();

        let dynamic_cubes: usize = models.iter().map(DynamicModel::cube_count).sum();
        let mut scene = Scene::new(
            &gpu.device,
            cubes,
            cubes.len(),
            dynamic_cubes,
            width,
            height,
        );
        let models = models
            .iter()
            .map(|model| scene.add_dynamic(model.clone()))
            .collect::<Result<Vec<_>, _>>()
            .context("adding the dynamic models")?;
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
            scenario,
            scene,
            models,
            camera,
            post,
            grade: Grade::default(),
            output: Output::detect(gpu.color_space(), &gpu.surface.display()),
            collector: Collector::new((width, height), gpu.device.gpu_timing_available()),
            surface_size: (width, height),
            generated_cubes,
            dynamic_cubes: u32::try_from(dynamic_cubes).context("too many dynamic cubes")?,
            bench_frame: 0,
            last_render: None,
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
                name: self.scenario.name().to_owned(),
                version: SCENARIO_VERSION,
                warmup_frames: WARMUP_FRAMES,
                measured_frames: MEASURED_FRAMES,
                generated_cubes: self.generated_cubes,
                kept_cubes: self.scene.static_count(),
                dynamic_cubes: self.dynamic_cubes,
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

        let now = Instant::now();
        // The time since the previous frame; the first one is never measured.
        let dt = self
            .last_render
            .replace(now)
            .map_or(0.0, |last| (now - last).as_secs_f32());

        (self.camera.eye, self.camera.target) = scenario::camera_at(self.bench_frame);
        for index in self.scenario.moved_models(self.bench_frame) {
            if let Some(model) = self.scene.dynamic_mut(self.models[index]) {
                model.position = model_position(index, self.bench_frame);
            }
        }
        self.scene.render(&gpu.device, encoder, &mut self.camera);
        self.post
            .draw(&gpu.device, encoder, view, &self.grade, self.output);

        // Every result, not just the newest: several can arrive at once. The
        // collector ignores those it has already counted.
        for timings in stats.gpu_arrived {
            self.collector.timings(timings);
        }
        if let Err(err) = self.collector.frame_rendered(
            stats.frame_index,
            self.bench_frame,
            dt,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scenario_defaults_to_the_terrain() {
        let args = Args::try_parse_from(["bench", "--save", "base"]).unwrap();
        assert_eq!(args.scenario, Scenario::Terrain);
    }

    #[test]
    fn a_scenario_is_chosen_by_its_name() {
        let args =
            Args::try_parse_from(["bench", "--save", "base", "--scenario", "few-changes"]).unwrap();
        assert_eq!(args.scenario, Scenario::FewChanges);
    }

    #[test]
    fn a_scenario_means_nothing_to_a_comparison() {
        let args = ["bench", "--compare", "a", "b", "--scenario", "moving"];
        assert!(Args::try_parse_from(args).is_err());
    }

    #[test]
    fn a_scenario_needs_a_run_to_apply_to() {
        assert!(Args::try_parse_from(["bench", "--scenario", "moving"]).is_err());
    }
}
