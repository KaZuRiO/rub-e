//! Sounds demo — a red cube at the origin that beeps, heard from wherever the
//! camera is. The port of the engine's `sounds`.
//!
//! ```text
//! cargo run -p etib-games --bin sounds
//! ```
//!
//! Fly around it with headphones on: the beep moves between the ears as you
//! turn, and fades as you leave. `C` captures the mouse, `Escape` releases it
//! (and quits when nothing is captured), W/A/S/D fly.

#[path = "common/fly.rs"]
mod fly;

use anyhow::{Context as _, Result};
use etib::winit::event::{DeviceEvent, WindowEvent};
use etib::winit::window::Window;
use etib::{App, Frame, Gpu, Grade, Output, PostProcess};
use etib_audio::{Audio, Listener, SingleCubeStrategy, Sound, SoundManager};
use etib_input::Input;
use etib_scene::{Camera, DynamicModel, ModelCube, Projection, Scene};
use glam::{Mat4, Vec3};

use fly::FlyCamera;

const SKY: &[u8] = include_bytes!("../assets/sky.hdr");
const SKY_FACE: u32 = 1080;
const BEEP: &[u8] = include_bytes!("../assets/beep.wav");

struct SoundsDemo {
    scene: Scene,
    camera: Camera,
    fly: FlyCamera,
    post: PostProcess,
    grade: Grade,
    output: Output,
    input: Input,
    sounds: SoundManager,
}

impl SoundsDemo {
    fn new(gpu: &Gpu) -> Result<Self> {
        let (width, height) = gpu.size();
        let camera = Camera::new(
            Vec3::new(0.0, 6.0, 18.0),
            Vec3::ZERO,
            Vec3::Y,
            width as f32 / height as f32,
            Projection::Perspective { fovy: 45.0 },
            0.1,
            500.0,
        );
        let fly = FlyCamera::new(10.0, &camera);

        let mut scene = Scene::new(&gpu.device, &[], 1, width, height);
        scene
            .set_skybox_from_hdr(&gpu.device, SKY, SKY_FACE, Mat4::IDENTITY)
            .context("loading the skybox")?;
        scene.add_dynamic(DynamicModel::from_cubes(vec![ModelCube {
            position: Vec3::ZERO,
            color: Vec3::new(1.0, 0.2, 0.2),
        }]));

        // The emitter sits on the cube, which does not move.
        let mut sounds = SoundManager::new(Audio::new_or_silent());
        let beep = Sound::from_wav(BEEP).context("decoding beep.wav")?;
        sounds.add_group(vec![Vec3::ZERO], &beep, Box::new(SingleCubeStrategy));

        let post = PostProcess::new(&gpu.device, gpu.format(), scene.target_view());
        Ok(Self {
            scene,
            camera,
            fly,
            post,
            grade: Grade::default(),
            output: Output::detect(gpu.color_space(), &gpu.surface.display()),
            input: Input::new(),
            sounds,
        })
    }
}

impl App for SoundsDemo {
    fn window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        self.input.window_event(event);
        self.fly.window_event(window, event)
    }

    fn device_event(&mut self, event: &DeviceEvent) {
        self.input.device_event(event);
    }

    fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        if self.scene.resize(&gpu.device, width, height) {
            self.post.rebind(&gpu.device, self.scene.target_view());
        }
    }

    fn update(&mut self, dt: f32) {
        let dt = dt.min(0.05);
        self.fly.update(&self.input, &mut self.camera, dt);
        self.sounds
            .update(Listener::looking_at(self.camera.eye, self.camera.target));
        self.input.clear_frame_state();
    }

    fn render(&mut self, frame: Frame<'_>) {
        let Frame {
            gpu, encoder, view, ..
        } = frame;
        self.scene.render(&gpu.device, encoder, &mut self.camera);
        self.post
            .draw(&gpu.device, encoder, view, &self.grade, self.output);
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
    etib::run(
        etib::Config {
            title: "Sounds Demo".to_owned(),
            ..Default::default()
        },
        SoundsDemo::new,
    )
}
