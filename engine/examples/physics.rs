//! Physics demo — rigid-body cubes with gravity, angular momentum, wall bounce,
//! and cube-to-cube impulse collision. The port of the engine's `physics`.
//!
//! ```text
//! cargo run --bin physics
//! ```
//!
//! Controls: `C` captures the mouse to look around, `Escape` releases it (and
//! quits when nothing is captured), W/A/S/D fly, `R` drops the cubes again from
//! new places. The window title shows the frame rate.

#[path = "common/fly.rs"]
mod fly;

use anyhow::{Context as _, Result};
use etib::winit::event::{DeviceEvent, ElementState, WindowEvent};
use etib::winit::keyboard::{KeyCode, PhysicalKey};
use etib::winit::window::Window;
use etib::{App, Frame, Gpu, Grade, Output, PostProcess};
use etib_input::Input;
use etib_physics::{RigidBody, apply_cube_collision};
use etib_scene::{Camera, DynamicId, DynamicModel, ModelCube, Projection, Scene};
use glam::{Mat4, Quat, Vec3};

use fly::FlyCamera;

const SKY: &[u8] = include_bytes!("../assets/sky.hdr");
const SKY_FACE: u32 = 1080;

const GROUND_HALF: i32 = 15;

/// Y of the floor surface (cubes centred at y=0 → top face at +0.5).
const FLOOR_TOP: f32 = 0.5;

/// Half-side of each cube model (2×2×2 block → 1 unit centre-to-face).
const CUBE_HALF: f32 = 1.0;

/// Physics wall bound.  Cube corners are kept strictly inside ±WALL on X and Z.
/// Set equal to GROUND_HALF−1.5 so the outermost floor tiles act as the wall face.
const WALL: f32 = (GROUND_HALF as f32) - 1.5;

const CUBE_COUNT: usize = 8;

const COLORS: [(f32, f32, f32); CUBE_COUNT] = [
    (1.0, 0.20, 0.20),
    (1.0, 0.55, 0.05),
    (1.0, 0.90, 0.10),
    (0.20, 1.0, 0.20),
    (0.10, 0.80, 1.0),
    (0.20, 0.40, 1.0),
    (0.75, 0.20, 1.0),
    (1.0, 1.0, 1.0),
];

fn make_floor() -> Vec<ModelCube> {
    let mut cubes = Vec::new();
    for x in -GROUND_HALF..GROUND_HALF {
        for z in -GROUND_HALF..GROUND_HALF {
            let dx = (x + GROUND_HALF).min(GROUND_HALF - 1 - x);
            let dz = (z + GROUND_HALF).min(GROUND_HALF - 1 - z);
            let gray = match dx.min(dz) {
                0 => 0.06, // Outermost ring — very dark wall indicator
                1 => 0.14, // Second ring — darker shadow
                _ if (x + z).rem_euclid(2) == 0 => 0.65,
                _ => 0.30,
            };
            cubes.push(ModelCube {
                position: Vec3::new(x as f32, 0.0, z as f32),
                color: Vec3::splat(gray),
            });
        }
    }
    cubes
}

/// Thin visible wall panels — one cube tall, placed just outside the play area.
/// Coloured a muted blue-grey so they read as transparent-ish barriers.
fn make_wall_panels() -> Vec<ModelCube> {
    let half = GROUND_HALF;
    let color = Vec3::new(0.18, 0.22, 0.40);
    let mut cubes = Vec::new();

    // North / south walls (full width)
    for x in -half..half {
        for z in [-(half as f32), (half - 1) as f32] {
            cubes.push(ModelCube {
                position: Vec3::new(x as f32, 1.0, z),
                color,
            });
        }
    }
    // East / west walls (skip corners already covered above)
    for z in -(half - 1)..half - 1 {
        for x in [-(half as f32), (half - 1) as f32] {
            cubes.push(ModelCube {
                position: Vec3::new(x, 1.0, z as f32),
                color,
            });
        }
    }
    cubes
}

fn make_cube_model(r: f32, g: f32, b: f32) -> DynamicModel {
    let mut voxels = Vec::new();
    for xi in 0..2 {
        for yi in 0..2 {
            for zi in 0..2 {
                voxels.push(ModelCube {
                    position: Vec3::new(xi as f32 - 0.5, yi as f32 - 0.5, zi as f32 - 0.5),
                    color: Vec3::new(r, g, b),
                });
            }
        }
    }
    DynamicModel::from_cubes(voxels)
}

struct Cube {
    id: DynamicId,
    body: RigidBody,
}

struct PhysicsDemo {
    scene: Scene,
    camera: Camera,
    fly: FlyCamera,
    post: PostProcess,
    grade: Grade,
    output: Output,
    input: Input,
    cubes: Vec<Cube>,
    rng_state: u64,
    since_title: f32,
}

impl PhysicsDemo {
    fn new(gpu: &Gpu) -> Result<Self> {
        let (width, height) = gpu.size();
        let camera = Camera::new(
            Vec3::new(0.0, 30.0, 45.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::Y,
            width as f32 / height as f32,
            Projection::Perspective { fovy: 55.0 },
            0.1,
            500.0,
        );
        let fly = FlyCamera::new(18.0, &camera);

        // Build static geometry: floor + visible wall panels
        let mut static_cubes = make_floor();
        static_cubes.extend(make_wall_panels());

        let mut scene = Scene::new(&gpu.device, &static_cubes, CUBE_COUNT * 8, width, height);
        scene
            .set_skybox_from_hdr(&gpu.device, SKY, SKY_FACE, Mat4::IDENTITY)
            .context("loading the skybox")?;

        let cubes = COLORS
            .iter()
            .map(|&(r, g, b)| Cube {
                id: scene
                    .add_dynamic(make_cube_model(r, g, b))
                    .expect("the scene has room for every cube"),
                body: RigidBody::new()
                    .with_restitution(0.85) // bouncy — preserves most impact energy
                    .with_friction(1.0)
                    .with_half_side(CUBE_HALF)
                    .with_mass(3.0) // solid feel; inertia = 2.0 kg·m²
                    .with_bounds(-WALL, WALL, -WALL, WALL),
            })
            .collect();

        let post = PostProcess::new(&gpu.device, gpu.format(), scene.target_view());
        let mut demo = Self {
            scene,
            camera,
            fly,
            post,
            grade: Grade::default(),
            output: Output::detect(gpu.color_space(), &gpu.surface.display()),
            input: Input::new(),
            cubes,
            rng_state: seed_from_time(),
            since_title: 1.0,
        };
        demo.reset_scene();
        Ok(demo)
    }

    fn reset_scene(&mut self) {
        // Spawn inside the play area, leaving margin so the cube never starts intersecting a wall.
        let margin = CUBE_HALF + 0.6;
        let (min_xz, max_xz) = (-WALL + margin, WALL - margin);

        // Keep initial cube centres separated so we don't start with deep penetrations.
        let min_sep = (CUBE_HALF * 2.0) * 1.35;
        let min_sep_sq = min_sep * min_sep;

        let mut spawns: Vec<Vec3> = Vec::with_capacity(self.cubes.len());
        for _ in 0..self.cubes.len() {
            let mut chosen = None;
            for _attempt in 0..200 {
                let candidate = Vec3::new(
                    self.rand_f32(min_xz, max_xz),
                    0.0,
                    self.rand_f32(min_xz, max_xz),
                );
                if spawns
                    .iter()
                    .all(|p| (candidate - *p).length_squared() >= min_sep_sq)
                {
                    chosen = Some(candidate);
                    break;
                }
            }
            // Fallback: accept a crowded spot.
            let chosen = chosen.unwrap_or_else(|| {
                Vec3::new(
                    self.rand_f32(min_xz, max_xz),
                    0.0,
                    self.rand_f32(min_xz, max_xz),
                )
            });
            spawns.push(chosen);
        }

        let mut drops = Vec::with_capacity(self.cubes.len());
        let mut velocities = Vec::with_capacity(self.cubes.len());
        for _ in 0..self.cubes.len() {
            drops.push(self.rand_f32(6.0, 16.0));
            let mut vx = self.rand_f32(-10.0, 10.0);
            let mut vz = self.rand_f32(-10.0, 10.0);
            if vx.abs() + vz.abs() < 2.0 {
                vx = 6.0;
                vz = -4.0;
            }
            velocities.push((vx, vz));
        }

        for (i, cube) in self.cubes.iter_mut().enumerate() {
            let (vx, vz) = velocities[i];
            cube.body.linear_velocity = Vec3::new(vx, 0.0, vz);
            cube.body.angular_velocity = Vec3::ZERO;
            cube.body.on_ground = false;
            if let Some(model) = self.scene.dynamic_mut(cube.id) {
                model.position =
                    Vec3::new(spawns[i].x, FLOOR_TOP + CUBE_HALF + drops[i], spawns[i].z);
                model.rotation = Quat::IDENTITY;
            }
        }
    }

    fn rand_u32(&mut self) -> u32 {
        // xorshift64* (small, fast, good enough for demo randomness)
        let mut x = self.rng_state;
        if x == 0 {
            x = 0x9e3779b97f4a7c15;
        }
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        ((x.wrapping_mul(0x2545F4914F6CDD1D)) >> 32) as u32
    }

    fn rand_f32(&mut self, min: f32, max: f32) -> f32 {
        let t = (self.rand_u32() as f32) / (u32::MAX as f32);
        min + (max - min) * t
    }

    fn resolve_collisions(&mut self) {
        // Snapshot current positions into a working buffer
        let mut positions: Vec<Vec3> = self
            .cubes
            .iter()
            .map(|cube| {
                self.scene
                    .dynamic(cube.id)
                    .map_or(Vec3::ZERO, |m| m.position)
            })
            .collect();

        // Broad-phase + narrow-phase: O(n²), fine for n=8
        let n = self.cubes.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let min_dist =
                    (self.cubes[i].body.half_side() + self.cubes[j].body.half_side()) * 1.2;
                if (positions[i] - positions[j]).length_squared() >= min_dist * min_dist {
                    continue;
                }
                // Split both slices at j to get four non-overlapping mut refs
                let (left, right) = self.cubes.split_at_mut(j);
                let (left_pos, right_pos) = positions.split_at_mut(j);
                apply_cube_collision(
                    &mut left[i].body,
                    &mut left_pos[i],
                    &mut right[0].body,
                    &mut right_pos[0],
                );
            }
        }

        // Write corrected positions back to the scene
        for (cube, &pos) in self.cubes.iter().zip(&positions) {
            if let Some(m) = self.scene.dynamic_mut(cube.id) {
                m.position = pos;
            }
        }
    }
}

impl App for PhysicsDemo {
    fn window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        self.input.window_event(event);
        if let WindowEvent::KeyboardInput { event, .. } = event
            && event.state == ElementState::Pressed
            && event.physical_key == PhysicalKey::Code(KeyCode::KeyR)
        {
            self.reset_scene();
            return true;
        }
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
        let dt = dt.min(0.05); // cap dt to avoid explosion on lag spikes
        self.since_title += dt;

        for cube in &mut self.cubes {
            if let Some(model) = self.scene.dynamic_mut(cube.id) {
                cube.body
                    .update(&mut model.position, &mut model.rotation, dt, FLOOR_TOP);
            }
        }
        self.resolve_collisions();

        self.fly.update(&self.input, &mut self.camera, dt);
        self.input.clear_frame_state();
    }

    fn render(&mut self, frame: Frame<'_>) {
        let Frame {
            gpu,
            encoder,
            view,
            stats,
        } = frame;
        if self.since_title >= 0.5 {
            self.since_title = 0.0;
            self.output = Output::detect(gpu.color_space(), &gpu.surface.display());
            gpu.window
                .set_title(&format!("Physics Demo — {:.0} FPS", stats.fps));
        }
        self.scene.render(&gpu.device, encoder, &mut self.camera);
        self.post
            .draw(&gpu.device, encoder, view, &self.grade, self.output);
    }
}

fn seed_from_time() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x1234_5678_9abc_def0);
    nanos ^ (nanos.rotate_left(17)) ^ 0x9e37_79b9_7f4a_7c15
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
    etib::run(
        etib::Config {
            title: "Physics Demo".to_owned(),
            ..Default::default()
        },
        PhysicsDemo::new,
    )
}
