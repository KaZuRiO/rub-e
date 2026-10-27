//! A cube world with a cat orbiting in it, under an HDR sky.
//!
//! Run it with:
//!
//! ```text
//! cargo run --bin hello
//! cargo run --bin hello -- --static scene.model --dynamic cat.model
//! ```
//!
//! The port of `unported/engine/etib/examples/hello.rs`. The frame is two passes:
//!   1. `Scene::render` culls the static cubes on the GPU and draws them, the
//!      dynamic cat and the skybox, depth-tested, into an Rgba16Float texture,
//!   2. `PostProcess` tone maps that texture onto the swapchain.
//!
//! Without `--static` / `--dynamic` the world is generated in place — the
//! terrain and the cat the engine's Python scripts make — so nothing has to be
//! prepared first. For a performance comparison with the engine, pass the
//! output of `unported/engine/etib/examples/generate_scene.py` to `--static`.
//!
//! Controls: `C` captures the mouse to look around, `Escape` releases it (and
//! quits when nothing is captured). W/A/S/D or the arrows move — by physical
//! position, so Z/Q/S/D on an AZERTY keyboard — Space and Shift rise and fall,
//! and the wheel changes speed, or zooms with `--isometric`. The keys and the
//! mouse are read through `etib_input::Input`.
//!
//! `--sdr` and `--hdr` force the surface either way, as in the other examples.
//! `--fullscreen` opens borderless fullscreen, and `--no-vsync` lifts the cap
//! at the display's refresh rate, which a frame rate comparison wants. The
//! window title shows the frame rate and the cube counts.

mod world;

use anyhow::{Context as _, Result};
use etib::winit::event::{DeviceEvent, ElementState, WindowEvent};
use etib::winit::keyboard::PhysicalKey;
use etib::winit::window::{CursorGrabMode, Window};
use etib::{App, Frame, Gpu, Grade, Hdr, Output, PostProcess, Vsync};
use etib_input::{Input, KeyCode};
use etib_scene::glam::{Mat4, Quat, Vec3};
use etib_scene::{
    Camera, CameraController, CameraMode, DynamicId, DynamicModel, ModelCube, Move, Projection,
    Scene, load_model,
};
use log::info;

const SKY: &[u8] = include_bytes!("../../assets/sky.hdr");
/// Texels along each side of a skybox face.
const SKY_FACE: u32 = 1080;

/// The keys that move the camera, by physical position.
const BINDINGS: &[(Move, &[KeyCode])] = &[
    (Move::Forward, &[KeyCode::KeyW, KeyCode::ArrowUp]),
    (Move::Back, &[KeyCode::KeyS, KeyCode::ArrowDown]),
    (Move::Left, &[KeyCode::KeyA, KeyCode::ArrowLeft]),
    (Move::Right, &[KeyCode::KeyD, KeyCode::ArrowRight]),
    (Move::Up, &[KeyCode::Space]),
    (Move::Down, &[KeyCode::ShiftLeft, KeyCode::ShiftRight]),
];

struct Args {
    static_model: Option<String>,
    dynamic_model: Option<String>,
    isometric: bool,
    hdr: Hdr,
    fullscreen: bool,
    vsync: Vsync,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self {
            static_model: None,
            dynamic_model: None,
            isometric: false,
            hdr: Hdr::Allow,
            fullscreen: false,
            vsync: Vsync::On,
        };
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--static" => {
                    args.static_model = Some(iter.next().context("--static needs a file")?)
                }
                "--dynamic" => {
                    args.dynamic_model = Some(iter.next().context("--dynamic needs a file")?)
                }
                "--isometric" => args.isometric = true,
                "--sdr" => args.hdr = Hdr::Off,
                "--hdr" => args.hdr = Hdr::Force,
                "--fullscreen" => args.fullscreen = true,
                "--no-vsync" => args.vsync = Vsync::Off,
                other => anyhow::bail!("unknown argument '{other}'"),
            }
        }
        Ok(args)
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default()
            .default_filter_or("warn,etib=info,etib_core=info,etib_scene=info,hello=info"),
    )
    .format_timestamp_millis()
    .init();

    let args = Args::parse()?;
    let static_cubes = match &args.static_model {
        Some(path) => load_model(path).with_context(|| format!("loading {path}"))?,
        None => world::floor(100),
    };
    let cat = match &args.dynamic_model {
        Some(path) => load_model(path).with_context(|| format!("loading {path}"))?,
        None => world::cat(),
    };
    info!(
        "{} static cubes, {} in the orbiting model",
        static_cubes.len(),
        cat.len()
    );

    etib::run(
        etib::Config {
            title: "etib - hello".to_owned(),
            hdr: args.hdr,
            fullscreen: args.fullscreen,
            vsync: args.vsync,
            ..Default::default()
        },
        move |gpu| HelloApp::new(gpu, &static_cubes, &cat, args.isometric),
    )
}

struct HelloApp {
    scene: Scene,
    camera: Camera,
    controller: CameraController,
    input: Input,
    post: PostProcess,
    grade: Grade,
    output: Output,
    orbiting: DynamicId,
    cursor_captured: bool,
    elapsed: f32,
    /// Seconds since the title and the display were last refreshed.
    since_poll: f32,
}

impl HelloApp {
    fn new(
        gpu: &Gpu,
        static_cubes: &[ModelCube],
        cat: &[ModelCube],
        isometric: bool,
    ) -> Result<Self> {
        let (width, height) = gpu.size();
        let aspect = width as f32 / height as f32;

        let mut scene = Scene::new(&gpu.device, static_cubes, cat.len(), width, height);
        let orbiting = scene
            .add_dynamic(DynamicModel::from_cubes(cat.to_vec()))
            .expect("the scene has room for the cat");
        scene
            .set_skybox_from_hdr(&gpu.device, SKY, SKY_FACE, Mat4::IDENTITY)
            .context("loading the skybox")?;

        let mut controller = CameraController::new(20.0, 0.003);
        let camera = if isometric {
            controller.mode = CameraMode::Isometric;
            // A true isometric view: down the (1, 1, 1) diagonal.
            Camera::new(
                Vec3::splat(100.0),
                Vec3::ZERO,
                Vec3::Y,
                aspect,
                Projection::Orthographic { scale: 50.0 },
                -200.0,
                400.0,
            )
        } else {
            let eye = Vec3::new(0.0, 30.0, 80.0);
            let target = Vec3::new(0.0, 10.0, 0.0);
            let forward = (target - eye).normalize();
            controller.yaw = forward.z.atan2(forward.x);
            controller.pitch = forward.y.asin();
            Camera::new(
                eye,
                target,
                Vec3::Y,
                aspect,
                Projection::Perspective { fovy: 45.0 },
                0.1,
                500.0,
            )
        };

        let post = PostProcess::new(&gpu.device, gpu.format(), scene.target_view());

        Ok(Self {
            scene,
            camera,
            controller,
            input: Input::new(),
            post,
            grade: Grade::default(),
            output: Output::detect(gpu.color_space(), &gpu.surface.display()),
            orbiting,
            cursor_captured: false,
            elapsed: 0.0,
            // Due straight away, so the title shows the counts from the start.
            since_poll: 1.0,
        })
    }

    fn capture_cursor(&mut self, window: &Window, capture: bool) {
        self.cursor_captured = capture;
        let mode = if capture {
            CursorGrabMode::Confined
        } else {
            CursorGrabMode::None
        };
        // Some platforms only offer one of the two grab modes.
        if window.set_cursor_grab(mode).is_err() && capture {
            let _ = window.set_cursor_grab(CursorGrabMode::Locked);
        }
        window.set_cursor_visible(!capture);
    }
}

impl App for HelloApp {
    fn window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        self.input.window_event(event);
        // Capturing the mouse needs the window, which `update` does not have.
        let WindowEvent::KeyboardInput { event, .. } = event else {
            return false;
        };
        if event.state != ElementState::Pressed {
            return false;
        }
        match event.physical_key {
            PhysicalKey::Code(KeyCode::KeyC) => {
                self.capture_cursor(window, true);
                true
            }
            // Escape releases the mouse first, and only quits when there is
            // nothing left to release.
            PhysicalKey::Code(KeyCode::Escape) if self.cursor_captured => {
                self.capture_cursor(window, false);
                true
            }
            _ => false,
        }
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
        self.elapsed += dt;
        self.since_poll += dt;
        let t = self.elapsed * 0.5;
        if let Some(cat) = self.scene.dynamic_mut(self.orbiting) {
            cat.position = Vec3::new(t.sin() * 30.0, 6.0, t.cos() * 30.0);
            cat.rotation = Quat::from_rotation_y(-t);
        }

        for &(direction, keys) in BINDINGS {
            let held = keys.iter().any(|&key| self.input.is_key_pressed(key));
            self.controller.set(direction, held);
        }
        if self.cursor_captured {
            let delta = self.input.mouse_delta();
            self.controller.look(delta.x, delta.y);
        }
        let scroll = self.input.scroll();
        if scroll != 0.0 {
            self.controller.scroll(scroll);
        }
        self.controller.update(&mut self.camera, dt);
        self.input.clear_frame_state();
    }

    fn render(&mut self, frame: Frame<'_>) {
        let Frame {
            gpu,
            encoder,
            view,
            stats,
        } = frame;

        if self.since_poll >= 0.5 {
            self.since_poll = 0.0;
            self.output = Output::detect(gpu.color_space(), &gpu.surface.display());
            gpu.window.set_title(&format!(
                "etib - hello - {:.0} fps | static: {} cubes | dynamic: {} cubes",
                stats.fps,
                self.scene.static_count(),
                self.scene.dynamic_count(),
            ));
        }

        self.scene.render(&gpu.device, encoder, &mut self.camera);
        self.post
            .draw(&gpu.device, encoder, view, &self.grade, self.output);
    }
}
