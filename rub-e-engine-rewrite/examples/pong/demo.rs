//! Attract mode: paddles that play themselves, and a camera that moves on its
//! own.

use etib_scene::Camera;
use glam::Vec3;

use crate::state::{Ball, PADDLE_HALF_H, PADDLE_X, Player, PlayerInput, WALL_LIMIT};

/// How long (seconds) the demo follow-cam rises up after a goal before settling
/// back onto the ball.
pub const DEMO_GOAL_CAM_DURATION: f32 = 2.5;

/// How the demo/attract mode is driven.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoMode {
    /// Not in demo mode — normal interactive play.
    Off,
    /// `--demo`: the camera slowly orbits the pitch.
    Orbit,
    /// `--demo2`: the camera chases the ball from behind.
    Follow,
}

impl DemoMode {
    /// Whether any demo mode is active (UI hidden, paddles self-playing).
    pub fn is_active(self) -> bool {
        self != DemoMode::Off
    }
}

/// Predict the ball's Y when it reaches `plane_x`, accounting for top/bottom wall
/// bounces, so a paddle can position itself precisely instead of chasing.
pub fn predict_ball_y(ball: &Ball, plane_x: f32) -> f32 {
    if ball.v.x.abs() < 0.01 {
        return ball.p.y;
    }
    let t = (plane_x - ball.p.x) / ball.v.x;
    if t <= 0.0 {
        return ball.p.y; // ball moving away — just hold the current line
    }
    let unbounded = ball.p.y + ball.v.y * t;

    // Fold the unbounded prediction back into [-WALL_LIMIT, WALL_LIMIT] as a
    // triangle wave (mirror reflections off each wall).
    let span = 2.0 * WALL_LIMIT;
    let mut m = (unbounded + WALL_LIMIT).rem_euclid(2.0 * span);
    if m > span {
        m = 2.0 * span - m;
    }
    m - WALL_LIMIT
}

/// Programmatically control a paddle for demo mode.
///
/// The paddle *predicts* where the ball will arrive (so rallies last) and adds a
/// small wandering offset so contact lands across the paddle, including the edges
/// and side faces, for angled returns instead of flat centre bounces. The result
/// is fed into `step_physics` exactly like a human or networked input would be,
/// so all collisions and scoring still apply.
pub fn demo_paddle_input(player: &Player, ball: &Ball, time: f32, is_left: bool) -> PlayerInput {
    // Desync the two paddles so they never act in lockstep.
    let phase = if is_left { 0.0 } else { 2.2 };

    let approaching = if is_left {
        ball.v.x < 0.0
    } else {
        ball.v.x > 0.0
    };
    let dist_x = (ball.p.x - player.x).abs();

    // Aim at the predicted arrival point so the paddle is almost always there in
    // time → far fewer misses, much longer exchanges.
    let predicted_y = predict_ball_y(ball, player.x);

    // Don't react instantly: while the ball is still far away the paddle lazily
    // tracks its *current* line, and only commits to the predicted landing spot as
    // the ball gets close.
    const REACT_DIST: f32 = 45.0; // start reacting once within this X distance
    const REACT_RANGE: f32 = 28.0; // ramp from lazy → committed over this span
    let react = ((REACT_DIST - dist_x) / REACT_RANGE).clamp(0.0, 1.0);
    let aim_y = ball.p.y + (predicted_y - ball.p.y) * react;

    // Small wandering offset for angle variety. Kept well inside the paddle height
    // so the ball still connects (it lands off-centre / on an edge, not past it).
    let wobble = ((time * 1.3 + phase).sin() * 0.6 + (time * 0.7 + phase).sin() * 0.4)
        * (PADDLE_HALF_H * 0.45);
    let target_y = aim_y + wobble;
    let dy = target_y - player.y;

    // Snappy vertical tracking. A tiny swipe keeps the paddle sweeping through the
    // ball so the top/bottom faces still catch it, without flinging it off target.
    let swipe = (time * 5.0 + phase).sin() * 0.2;
    let move_y = (dy * 1.5 + swipe).clamp(-1.0, 1.0);

    // Forward/backward motion along the paddle's own axis. "Forward" is toward the
    // centre line. The paddle lunges forward to meet an incoming ball and retreats
    // toward its home column afterward, with a constant slow sway.
    let home_x = if is_left { -PADDLE_X } else { PADDLE_X };
    let forward_dir = if is_left { 1.0 } else { -1.0 };
    let lunge = if approaching {
        forward_dir * react * 9.0
    } else {
        0.0
    };
    let sway = forward_dir * (time * 0.8 + phase).sin() * 3.5;
    let target_x = home_x + lunge + sway;
    let move_x = ((target_x - player.x) * 1.5).clamp(-1.0, 1.0);

    // Lob on *some* strikes, not all: a slow sine opens a lob "window" roughly a
    // third of the time. The 6s lob cooldown rate-limits it further.
    let near_strike = approaching && dist_x < 11.0;
    let lob_pressed = near_strike && ((time + phase) * 0.6).sin() > 0.5;

    // Boost now and then for flashy colour pulses (its own cooldown rate-limits it).
    let boost_pressed = ((time + phase) * 0.5).sin() > 0.9;

    PlayerInput {
        move_y,
        move_x,
        lob_pressed,
        boost_pressed,
        // No smashes: the paddle never dashes, so every return is a normal-speed shot.
        dash_pressed: false,
        sequence_number: 0,
    }
}

/// Moves the camera for the demo mode. Z is up in this scene, so the camera's
/// up vector is kept on it for a level horizon.
///
/// `goal_timer` counts down the follow-cam's rise after a goal.
pub fn update_demo_camera(
    camera: &mut Camera,
    mode: DemoMode,
    ball: &Ball,
    time: f32,
    goal_timer: &mut f32,
    dt: f32,
) {
    camera.up = Vec3::Z;

    match mode {
        DemoMode::Orbit => {
            // Centre roughly on the playfield, slightly above the ground.
            let center = Vec3::new(0.0, 2.0, 4.0);
            let angle = time * 0.15; // slow azimuth sweep
            let radius = 75.0 + (time * 0.3).sin() * 15.0; // gentle dolly in/out
            let height = 35.0 + (time * 0.5).sin() * 10.0; // gentle vertical bob
            camera.orbit(center, radius, angle, height);
        }
        DemoMode::Follow => {
            // Horizontal heading of the ball; fall back to +X when nearly still.
            let flat = Vec3::new(ball.v.x, ball.v.y, 0.0);
            let heading = if flat.length() < 0.001 {
                Vec3::X
            } else {
                flat.normalize()
            };

            // Goal celebration: a sine over the countdown gives lift 0 → 1 → 0,
            // so the camera rises high above the pitch right after a goal and
            // then settles back down onto the ball.
            *goal_timer = (*goal_timer - dt).max(0.0);
            let progress = 1.0 - *goal_timer / DEMO_GOAL_CAM_DURATION;
            let lift = (progress * std::f32::consts::PI).sin();

            // Desired eye sits close behind and just above the ball, relative
            // to its heading — a tight chase cam — plus the celebration lift.
            const FOLLOW_DIST: f32 = 14.0;
            const FOLLOW_HEIGHT: f32 = 6.0;
            const GOAL_CAM_RISE: f32 = 70.0;
            let desired =
                ball.p - heading * FOLLOW_DIST + Vec3::Z * (FOLLOW_HEIGHT + lift * GOAL_CAM_RISE);

            // Smooth the eye toward the target so direction flips on bounces
            // don't snap the camera (frame-rate independent exponential easing).
            let t = 1.0 - (-dt * 6.0).exp();
            let eye = camera.eye + (desired - camera.eye) * t;
            camera.set_position(eye);
            // Aim slightly ahead of the ball along its heading for a chase feel.
            camera.look_at(ball.p + heading * 6.0);
        }
        DemoMode::Off => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moving(p: Vec3, v: Vec3) -> Ball {
        Ball {
            p,
            v,
            ..Ball::default()
        }
    }

    #[test]
    fn prediction_folds_off_the_walls() {
        // Straight on to y = 100, which is two reflections later y = 12.
        let ball = moving(Vec3::ZERO, Vec3::new(10.0, 10.0, 0.0));
        assert!((predict_ball_y(&ball, 100.0) - 12.0).abs() < 1e-3);
        // No bounce needed.
        assert!((predict_ball_y(&ball, 5.0) - 5.0).abs() < 1e-4);
    }

    #[test]
    fn a_ball_going_away_is_not_predicted() {
        let ball = moving(Vec3::new(0.0, 3.0, 0.0), Vec3::new(10.0, 10.0, 0.0));
        assert_eq!(predict_ball_y(&ball, -30.0), 3.0);
        let still = moving(Vec3::new(0.0, 3.0, 0.0), Vec3::new(0.0, 10.0, 0.0));
        assert_eq!(predict_ball_y(&still, -30.0), 3.0);
    }

    #[test]
    fn demo_paddles_never_dash_and_stay_in_range() {
        let ball = moving(Vec3::new(-20.0, 5.0, 0.0), Vec3::new(-30.0, 8.0, 0.0));
        for i in 0..200 {
            let input = demo_paddle_input(&Player::left(), &ball, i as f32 * 0.1, true);
            assert!(!input.dash_pressed);
            assert!(input.move_x.abs() <= 1.0 && input.move_y.abs() <= 1.0);
        }
    }

    #[test]
    fn the_follow_cam_rises_after_a_goal() {
        let ball = moving(Vec3::ZERO, Vec3::new(20.0, 0.0, 0.0));
        let mut camera = Camera::new(
            Vec3::new(-14.0, 0.0, 6.0),
            Vec3::ZERO,
            Vec3::Z,
            1.0,
            etib_scene::Projection::Perspective { fovy: 55.0 },
            0.1,
            200.0,
        );
        let mut timer = 0.0;
        update_demo_camera(
            &mut camera,
            DemoMode::Follow,
            &ball,
            0.0,
            &mut timer,
            1.0 / 60.0,
        );
        let resting = camera.eye.z;

        let mut timer = DEMO_GOAL_CAM_DURATION / 2.0;
        for _ in 0..30 {
            update_demo_camera(
                &mut camera,
                DemoMode::Follow,
                &ball,
                0.0,
                &mut timer,
                1.0 / 60.0,
            );
        }
        assert!(
            camera.eye.z > resting + 10.0,
            "{} vs {resting}",
            camera.eye.z
        );
        assert_eq!(camera.up, Vec3::Z);
    }
}
