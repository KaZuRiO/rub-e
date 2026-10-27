//! The game as an `etib::App`: the scene, the rules, the effects, the crowd
//! and the menus, put together frame by frame.

use std::f32::consts::PI;

use anyhow::{Context as _, Result};
use etib::ui::EguiLayer;
use etib::winit::event::{DeviceEvent, WindowEvent};
use etib::winit::window::Window;
use etib::{App, Frame, Gpu, Grade, Output, PostProcess};
use etib_audio::Listener;
use etib_input::{Button, Input, KeyCode};
use etib_scene::{Camera, DynamicModel, Projection, Scene};
use glam::{Mat4, Quat, Vec3};

use crate::demo::{DEMO_GOAL_CAM_DURATION, DemoMode, demo_paddle_input, update_demo_camera};
use crate::input::{client_input, left_input, right_input};
use crate::net::Client;
use crate::sound::PongSounds;
use crate::stadium::{
    HIDDEN, LEFT_COLOR, RIGHT_COLOR, SCORE_POSITION, make_ball, make_paddle, make_score_model,
    make_shadow, make_static_geometry, make_text_model, unit_cube,
};
use crate::state::{
    BALL_GRAVITY, Ball, FIELD_HALF_H, FIELD_HALF_W, Player, reset_ball, step_physics,
};
use crate::ui::{UiAction, hud, main_menu, pause_menu};

const SKY: &[u8] = include_bytes!("../../assets/sky.hdr");
/// Texels along each side of a skybox face.
const SKY_FACE: u32 = 1080;

// Camera — perspective from a low angle to reveal cube depth
const CAM_EYE: Vec3 = Vec3::new(0.0, -60.0, 40.0);
const CAM_TARGET: Vec3 = Vec3::new(0.0, 2.0, 0.0);
const CAM_FOVY: f32 = 55.0;

const TRAIL_LENGTH: usize = 10;
const TRAIL_COLOR: Vec3 = Vec3::new(0.0, 0.3, 0.0);
/// Just above the grass, which is at -2.
const GROUND_MARK_Z: f32 = -1.95;
const PARTICLES_PER_GOAL: usize = 60;
/// Spectators are two cubes each; more than this would be more cubes than the
/// scene has room for.
const MAX_SPECTATORS: usize = 4000;
/// Dynamic cubes beyond the paddles and the ball: the scores, the particles
/// and the spectators, as the engine sized it.
const DYNAMIC_HEADROOM: usize = 100 + 2000 + 10_000;

struct Particle {
    id: usize,
    velocity: Vec3,
    life: f32,
}

struct Spectator {
    id: usize,
    base_pos: Vec3,
    phase: f32,
    speed: f32,
    jump_timer: f32,
}

/// A player and the models that show them.
struct Side {
    player: Player,
    paddle_id: usize,
    score_id: usize,
    rendered_score: u32,
    color: Vec3,
    /// Where the score is centred along X.
    score_x: f32,
}

pub struct PongApp {
    scene: Scene,
    camera: Camera,
    post: PostProcess,
    grade: Grade,
    output: Output,
    egui: EguiLayer,
    input: Input,

    ball: Ball,
    ball_id: usize,
    left: Side,
    right: Side,

    game_started: bool,
    paused: bool,
    /// Attract/demo mode: UI hidden, paddles and camera driven programmatically.
    demo: DemoMode,
    /// Countdown (seconds) for the demo follow-cam's post-goal rise-and-return.
    demo_goal_timer: f32,

    client: Option<Client>,

    trail_ids: Vec<usize>,
    trail_idx: usize,
    last_trail_pos: (i32, i32),
    shadow_id: usize,
    particles: Vec<Particle>,
    title_id: usize,
    spectators: Vec<Spectator>,

    /// Seconds since the game started running.
    time: f32,
    /// Seconds since the display was last asked what it can show.
    since_display_poll: f32,
    sounds: PongSounds,
}

impl PongApp {
    pub fn new(gpu: &Gpu, demo: DemoMode, client: Option<Client>) -> Result<Self> {
        let (width, height) = gpu.size();
        let camera = Camera::new(
            CAM_EYE,
            CAM_TARGET,
            Vec3::Y,
            width as f32 / height as f32,
            Projection::Perspective { fovy: CAM_FOVY },
            0.1,
            200.0,
        );

        let ball_model = make_ball();
        let left_paddle = make_paddle(LEFT_COLOR);
        let right_paddle = make_paddle(RIGHT_COLOR);
        let capacity = left_paddle.cube_count()
            + right_paddle.cube_count()
            + ball_model.cube_count()
            + DYNAMIC_HEADROOM;

        let (walls, seats) = make_static_geometry();
        let mut scene = Scene::new(&gpu.device, &walls, capacity, width, height);
        scene
            .set_skybox_from_hdr(
                &gpu.device,
                SKY,
                SKY_FACE,
                Mat4::from_rotation_x((-75f32).to_radians()),
            )
            .context("loading the skybox")?;

        let ball_id = scene.add_dynamic(ball_model);
        let mut side = |player: Player, paddle: DynamicModel, color: Vec3, score_x: f32| Side {
            player,
            paddle_id: scene.add_dynamic(paddle),
            score_id: scene.add_dynamic(make_score_model(0, score_x, color)),
            rendered_score: 0,
            color,
            score_x,
        };
        let left = side(Player::left(), left_paddle, LEFT_COLOR, -10.0);
        let right = side(Player::right(), right_paddle, RIGHT_COLOR, 10.0);

        // Hide models initially for the startup scene. In demo mode there is no
        // menu, so leave everything visible and let `update` position it.
        if !demo.is_active() {
            for id in [
                left.paddle_id,
                right.paddle_id,
                ball_id,
                left.score_id,
                right.score_id,
            ] {
                if let Some(model) = scene.dynamic_mut(id) {
                    model.position = HIDDEN;
                }
            }
        }

        let trail_ids = (0..TRAIL_LENGTH)
            .map(|_| {
                let mut cube = DynamicModel::from_cubes(vec![unit_cube(
                    0.0,
                    0.0,
                    0.0,
                    TRAIL_COLOR.x,
                    TRAIL_COLOR.y,
                    TRAIL_COLOR.z,
                )]);
                cube.position = HIDDEN;
                scene.add_dynamic(cube)
            })
            .collect();
        let shadow_id = scene.add_dynamic(make_shadow());

        let mut title = make_text_model("PONG 3D", -14.0, Vec3::new(0.0, 0.8, 1.0));
        title.position = Vec3::new(0.0, 0.0, 15.0); // Center of field, floating
        let title_id = scene.add_dynamic(title);

        let spectators = add_spectators(&mut scene, &seats);
        let post = PostProcess::new(&gpu.device, gpu.format(), scene.target_view());
        let egui = EguiLayer::new(&gpu.device, gpu.format(), gpu.color_space(), &gpu.window);

        Ok(Self {
            scene,
            camera,
            post,
            grade: Grade::default(),
            output: Output::detect(gpu.color_space(), &gpu.surface.display()),
            egui,
            input: Input::new(),
            ball: Ball::default(),
            ball_id,
            left,
            right,
            game_started: demo.is_active(),
            paused: false,
            demo,
            demo_goal_timer: 0.0,
            client,
            trail_ids,
            trail_idx: 0,
            last_trail_pos: (0, 0),
            shadow_id,
            particles: Vec::new(),
            title_id,
            spectators,
            time: 0.0,
            since_display_poll: 0.0,
            sounds: PongSounds::new(),
        })
    }

    /// One frame of the game; the input is cleared afterwards whatever this
    /// returns early for.
    fn step(&mut self, dt: f32) {
        self.time += dt;

        // Demo mode drives the camera programmatically.
        if self.demo.is_active() {
            update_demo_camera(
                &mut self.camera,
                self.demo,
                &self.ball,
                self.time,
                &mut self.demo_goal_timer,
                dt,
            );
        }

        // The ambience starts with the game: there is no crowd in the menu.
        if self.game_started {
            self.sounds.start_ambience();
        }
        self.sounds
            .update(Listener::looking_at(self.camera.eye, self.camera.target));

        if let Some(title) = self.scene.dynamic_mut(self.title_id) {
            if self.game_started {
                title.position.z = HIDDEN.z;
            } else {
                // Floating animation for the title
                title.position.z = 15.0 + (self.time * 2.0).sin() * 2.0;
                title.rotation =
                    Quat::from_rotation_z(((self.time * 20.0).sin() * 5.0).to_radians());
            }
        }

        if !self.game_started {
            return;
        }

        // Demo mode never pauses and shows no menus.
        if !self.demo.is_active() {
            if self.input.is_key_just_pressed(KeyCode::Escape)
                || self.input.gamepads().any_just_pressed(Button::Start)
            {
                self.paused = !self.paused;
            }
            if self.paused {
                return;
            }
        }

        let prev_ball_pos = self.ball.p;

        // The rules run here unless a server runs them.
        if self.client.is_none() {
            let (left_input, right_input) = if self.demo.is_active() {
                (
                    demo_paddle_input(&self.left.player, &self.ball, self.time, true),
                    demo_paddle_input(&self.right.player, &self.ball, self.time, false),
                )
            } else {
                (left_input(&self.input), right_input(&self.input))
            };
            step_physics(
                &mut self.ball,
                &mut self.left.player,
                &mut self.right.player,
                &left_input,
                &right_input,
                dt,
            );
        }

        // A goal: the new score, an explosion where the ball went in, the
        // crowd on that side jumping, and a cheer.
        if self.left.player.score != self.left.rendered_score {
            self.goal(true, prev_ball_pos);
        }
        if self.right.player.score != self.right.rendered_score {
            self.goal(false, prev_ball_pos);
        }

        self.place_models();
        self.update_trail();
        self.update_particles(dt);
        self.update_spectators(dt);

        // The server's state replaces ours.
        if let Some(client) = &mut self.client
            && let Some(snapshot) = client.exchange(client_input(&self.input))
        {
            snapshot.apply(
                &mut self.ball,
                &mut self.left.player,
                &mut self.right.player,
            );
        }
    }

    /// The left player scored when `left_scored`, so the ball went in at the
    /// right-hand goal.
    fn goal(&mut self, left_scored: bool, prev_ball_pos: Vec3) {
        let side = if left_scored {
            &mut self.left
        } else {
            &mut self.right
        };
        self.scene.remove_dynamic(side.score_id);
        side.score_id = self.scene.add_dynamic(make_score_model(
            side.player.score,
            side.score_x,
            side.color,
        ));
        side.rendered_score = side.player.score;

        let goal_x = if left_scored {
            FIELD_HALF_W
        } else {
            -FIELD_HALF_W
        };
        let color = side.color;
        self.spawn_explosion(Vec3::new(goal_x, prev_ball_pos.y, prev_ball_pos.z), color);

        for s in &mut self.spectators {
            let on_scoring_side = if left_scored {
                s.base_pos.x < -FIELD_HALF_W
            } else {
                s.base_pos.x > FIELD_HALF_W
            };
            let is_central = s.base_pos.y > FIELD_HALF_H;
            if on_scoring_side || (is_central && s.id % 3 == 0) {
                s.jump_timer = 1.0;
            }
        }

        let side_offset = if left_scored { -8.0 } else { 8.0 };
        self.sounds
            .cheer(self.camera.eye + Vec3::new(side_offset, 0.0, 0.0));
        self.demo_goal_timer = DEMO_GOAL_CAM_DURATION;
    }

    fn spawn_explosion(&mut self, origin: Vec3, color: Vec3) {
        let n = PARTICLES_PER_GOAL;
        for i in 0..n {
            let fi = i as f32;
            let mut model = DynamicModel::from_cubes(vec![unit_cube(
                -0.5, -0.5, -0.5, color.x, color.y, color.z,
            )]);
            model.position =
                origin + Vec3::new((fi * 1.3).sin(), (fi * 1.7).cos(), (fi * 2.3).sin()) * 2.0;
            let id = self.scene.add_dynamic(model);

            let angle1 = fi * PI * 2.0 / n as f32;
            let angle2 = (fi * 13.0).sin() * std::f32::consts::FRAC_PI_4;
            let speed = 40.0 + (fi * 7.0).sin().abs() * 60.0;
            self.particles.push(Particle {
                id,
                velocity: Vec3::new(
                    angle1.cos() * angle2.cos() * speed,
                    angle1.sin() * angle2.cos() * speed,
                    angle2.sin().abs() * speed + 20.0,
                ),
                life: 1.0 + (fi * 5.0).sin().abs(),
            });
        }
    }

    /// The ball and the paddles where the rules put them; a paddle flashes
    /// white while dashing and cycles through colours while boosted.
    fn place_models(&mut self) {
        if let Some(ball) = self.scene.dynamic_mut(self.ball_id) {
            ball.position = self.ball.p;
        }
        for side in [&self.left, &self.right] {
            let p = &side.player;
            let color = if p.dash_timer > 0.0 {
                Vec3::ONE
            } else if p.boost_active {
                let t = self.time * 15.0;
                Vec3::new(t.sin(), (t + 2.094).sin(), (t + 4.188).sin()) * 0.5 + 0.5
            } else {
                side.color
            };
            if let Some(paddle) = self.scene.dynamic_mut(side.paddle_id) {
                paddle.position = Vec3::new(p.x, p.y, 0.0);
                for cube in paddle.cubes_mut() {
                    cube.color = color;
                }
            }
        }
    }

    /// Marks on the grass behind a rolling ball, a shadow under a flying one.
    fn update_trail(&mut self) {
        let grid = (self.ball.p.x.round() as i32, self.ball.p.y.round() as i32);
        let shadow = if self.ball.p.z > 0.1 {
            for &id in &self.trail_ids {
                if let Some(mark) = self.scene.dynamic_mut(id) {
                    mark.position = HIDDEN;
                }
            }
            self.last_trail_pos = grid;
            Vec3::new(self.ball.p.x, self.ball.p.y, GROUND_MARK_Z)
        } else {
            if grid != self.last_trail_pos {
                self.last_trail_pos = grid;
                let id = self.trail_ids[self.trail_idx];
                if let Some(mark) = self.scene.dynamic_mut(id) {
                    mark.position = Vec3::new(grid.0 as f32, grid.1 as f32, GROUND_MARK_Z);
                }
                self.trail_idx = (self.trail_idx + 1) % TRAIL_LENGTH;
            }
            HIDDEN
        };
        if let Some(model) = self.scene.dynamic_mut(self.shadow_id) {
            model.position = shadow;
        }
    }

    fn update_particles(&mut self, dt: f32) {
        let scene = &mut self.scene;
        self.particles.retain_mut(|p| {
            p.life -= dt;
            if p.life <= 0.0 {
                scene.remove_dynamic(p.id);
                return false;
            }
            p.velocity.z -= BALL_GRAVITY * 1.5 * dt;
            if let Some(model) = scene.dynamic_mut(p.id) {
                model.position += p.velocity * dt;
                // floor bounce
                if model.position.z < 0.0 {
                    model.position.z = 0.0;
                    p.velocity.z *= -0.5;
                    p.velocity.x *= 0.8;
                    p.velocity.y *= 0.8;
                }
                // fade
                if let Some(cube) = model.cubes_mut().first_mut() {
                    cube.color *= 0.95;
                }
            }
            true
        });
    }

    fn update_spectators(&mut self, dt: f32) {
        for s in &mut self.spectators {
            if s.jump_timer > 0.0 {
                s.jump_timer -= dt * 2.0;
            }
            if let Some(model) = self.scene.dynamic_mut(s.id) {
                let jump = if s.jump_timer > 0.0 {
                    (s.jump_timer * PI).sin() * 2.0
                } else {
                    0.0
                };
                let bob = (self.time * s.speed + s.phase).sin() * 0.2;
                model.position.z = s.base_pos.z + bob + jump;
            }
        }
    }

    fn act(&mut self, action: UiAction) {
        match action {
            UiAction::Start => {
                self.game_started = true;
                for id in [self.left.score_id, self.right.score_id] {
                    if let Some(model) = self.scene.dynamic_mut(id) {
                        model.position = SCORE_POSITION;
                    }
                }
            }
            UiAction::Resume => self.paused = false,
            UiAction::Restart => {
                self.paused = false;
                for side in [&mut self.left, &mut self.right] {
                    side.player.score = 0;
                    side.player.y = 0.0;
                }
                reset_ball(&mut self.ball, true);
            }
            // As the engine did: there is nothing to save, and the runner has
            // no way for an application to ask it to stop.
            UiAction::Quit => std::process::exit(0),
        }
    }
}

/// Two-cube spectators, body and head, on up to [`MAX_SPECTATORS`] seats
/// spread over the stands. They wait out of sight until the game starts.
fn add_spectators(scene: &mut Scene, seats: &[Vec3]) -> Vec<Spectator> {
    let count = seats.len().min(MAX_SPECTATORS);
    if count == 0 {
        return Vec::new();
    }
    let skin_tones = [
        Vec3::new(1.0, 0.8, 0.6),
        Vec3::new(0.8, 0.6, 0.4),
        Vec3::new(0.6, 0.4, 0.2),
    ];
    let jersey_colors = [
        Vec3::new(1.0, 1.0, 1.0), // White
        Vec3::new(0.4, 0.8, 1.0), // Marseille Blue
        Vec3::new(0.1, 0.1, 0.1), // Dark
    ];

    (0..count)
        .map(|i| {
            let pos = seats[i * (seats.len() / count)];
            let jersey = jersey_colors[i % jersey_colors.len()];
            let skin = skin_tones[(i / 3) % skin_tones.len()];
            let mut model = DynamicModel::from_cubes(vec![
                unit_cube(0.0, 0.0, 0.0, jersey.x, jersey.y, jersey.z),
                unit_cube(0.0, 0.0, 1.0, skin.x, skin.y, skin.z),
            ]);
            model.position = Vec3::new(pos.x, pos.y, HIDDEN.z);
            Spectator {
                id: scene.add_dynamic(model),
                base_pos: pos,
                // Pseudo-random, but the same every run.
                phase: ((i * 555) % 100) as f32 / 100.0 * PI * 2.0,
                speed: 1.5 + ((i * 777) % 100) as f32 / 100.0 * 2.0,
                jump_timer: 0.0,
            }
        })
        .collect()
}

impl App for PongApp {
    fn window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        self.input.window_event(event);
        self.egui.on_window_event(window, event)
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
        self.since_display_poll += dt;
        self.step(dt);
        self.input.clear_frame_state();
    }

    fn render(&mut self, frame: Frame<'_>) {
        let Frame {
            gpu, encoder, view, ..
        } = frame;

        if self.since_display_poll >= 1.0 {
            self.since_display_poll = 0.0;
            self.output = Output::detect(gpu.color_space(), &gpu.surface.display());
        }

        self.scene.render(&gpu.device, encoder, &mut self.camera);
        self.post
            .draw(&gpu.device, encoder, view, &self.grade, self.output);

        // Demo mode hides all menus and HUD.
        let mut action = None;
        if !self.demo.is_active() {
            let (started, paused) = (self.game_started, self.paused);
            let (left, right) = (&self.left.player, &self.right.player);
            self.egui.render(
                &gpu.device,
                encoder,
                &gpu.window,
                view,
                gpu.size(),
                self.output,
                |ui| {
                    action = if !started {
                        main_menu(ui.ctx())
                    } else if paused {
                        pause_menu(ui.ctx())
                    } else {
                        hud(ui, left, right);
                        None
                    };
                },
            );
        }
        if let Some(action) = action {
            self.act(action);
        }
    }
}
