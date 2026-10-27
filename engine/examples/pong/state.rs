//! The game itself: the ball, the two players, and one step of the rules.
//!
//! Nothing here draws or plays anything, so the same code runs the local game,
//! the demo, and the headless server.

use std::f32::consts::FRAC_PI_4;

use glam::Vec3;
use serde::{Deserialize, Serialize};

// Field
pub const FIELD_HALF_W: f32 = 40.0;
pub const FIELD_HALF_H: f32 = 24.0;

// Paddles
pub const PADDLE_X: f32 = 36.0;
/// Half-height of a paddle in cube units (9 cubes → ±4.5).
pub const PADDLE_HALF_H: f32 = 4.5;
const PADDLE_SPEED: f32 = 24.0;
const SPEED_BOOST_MULTIPLIER: f32 = 2.0;
pub const SPEED_BOOST_DURATION: f32 = 2.0;
pub const SPEED_BOOST_COOLDOWN: f32 = 15.0;
const DASH_SPEED: f32 = 160.0;
pub const DASH_DURATION: f32 = 0.15;
pub const DASH_COOLDOWN: f32 = 3.0;

/// Maximum paddle center Y so the paddle never clips through a wall.
/// Wall inner face: FIELD_HALF_H − 1.0. Outermost paddle cube face: center + PADDLE_HALF_H + 1.0.
const PADDLE_MAX_Y: f32 = FIELD_HALF_H - PADDLE_HALF_H - 2.0;

// Ball
pub const BALL_SPEED_INIT: f32 = 20.0;
const BALL_SPEED_MAX: f32 = 80.0;
const BALL_SPEED_INC: f32 = 3.0;
pub const BALL_GRAVITY: f32 = 100.0;
const BALL_LOB_SPEED: f32 = 60.0;
const BALL_LOB_X_FACTOR: f32 = 0.9;
/// Ball center Y limit so it never clips into a wall (wall face − ball radius).
pub const WALL_LIMIT: f32 = FIELD_HALF_H - 2.0;

const DASH_SMASH_BONUS: f32 = 30.0;
pub const LOB_COOLDOWN: f32 = 6.0;
const PADDLE_HIT_COOLDOWN: f32 = 0.08;
/// Above this the ball has been lobbed over the paddles.
const BALL_HITTABLE_Z: f32 = 3.0;

/// What one player asks for this step. Also the client's message to the
/// server, so the field order is the wire format.
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct PlayerInput {
    pub move_y: f32, // -1.0 to 1.0 (Down, None, Up)
    pub move_x: f32, // -1.0 to 1.0 (Left, None, Right)
    pub lob_pressed: bool,
    pub boost_pressed: bool,
    pub dash_pressed: bool,
    pub sequence_number: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ball {
    pub p: Vec3,
    pub v: Vec3,
    pub left_hit_cd: f32,
    pub right_hit_cd: f32,
}

impl Default for Ball {
    fn default() -> Self {
        Self {
            p: Vec3::ZERO,
            v: Vec3::new(BALL_SPEED_INIT, BALL_SPEED_INIT * 0.4, 0.0),
            left_hit_cd: 0.0,
            right_hit_cd: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Player {
    pub x: f32,
    pub y: f32,
    pub boost_active: bool,
    pub boost_timer: f32,
    pub cooldown_timer: f32,
    pub dash_timer: f32,
    pub dash_cooldown: f32,
    pub lob_cooldown: f32,
    pub score: u32,
}

impl Player {
    pub fn new(x: f32) -> Self {
        Self {
            x,
            y: 0.0,
            boost_active: false,
            boost_timer: 0.0,
            cooldown_timer: 0.0,
            dash_timer: 0.0,
            dash_cooldown: 0.0,
            lob_cooldown: 0.0,
            score: 0,
        }
    }

    pub fn left() -> Self {
        Self::new(-PADDLE_X)
    }

    pub fn right() -> Self {
        Self::new(PADDLE_X)
    }

    /// Runs the timers down, then starts a boost or a dash if one is asked for
    /// and ready.
    fn tick(&mut self, input: &PlayerInput, dt: f32) {
        if self.cooldown_timer > 0.0 {
            self.cooldown_timer -= dt;
        }
        if self.boost_timer > 0.0 {
            self.boost_timer -= dt;
            if self.boost_timer <= 0.0 {
                self.boost_active = false;
            }
        }
        if self.dash_cooldown > 0.0 {
            self.dash_cooldown -= dt;
        }
        if self.dash_timer > 0.0 {
            self.dash_timer -= dt;
        }
        if self.lob_cooldown > 0.0 {
            self.lob_cooldown -= dt;
        }

        if input.boost_pressed && self.cooldown_timer <= 0.0 {
            self.boost_active = true;
            self.boost_timer = SPEED_BOOST_DURATION;
            self.cooldown_timer = SPEED_BOOST_COOLDOWN;
        }
        if input.dash_pressed && self.dash_cooldown <= 0.0 {
            self.dash_timer = DASH_DURATION;
            self.dash_cooldown = DASH_COOLDOWN;
        }
    }

    fn speed(&self) -> f32 {
        if self.dash_timer > 0.0 {
            DASH_SPEED
        } else if self.boost_active {
            PADDLE_SPEED * SPEED_BOOST_MULTIPLIER
        } else {
            PADDLE_SPEED
        }
    }

    /// Moves within `x_range`, never across the centre line or into a wall. A
    /// diagonal is no faster than a straight line.
    fn steer(&mut self, input: &PlayerInput, x_range: (f32, f32), dt: f32) {
        let (mut mx, mut my) = (input.move_x, input.move_y);
        let mag = (mx * mx + my * my).sqrt();
        if mag > 1.0 {
            mx /= mag;
            my /= mag;
        }
        let speed = self.speed();
        self.x = (self.x + mx * speed * dt).clamp(x_range.0, x_range.1);
        self.y = (self.y + my * speed * dt).clamp(-PADDLE_MAX_Y, PADDLE_MAX_Y);
    }
}

/// The ball's speed across the pitch, ignoring height.
pub fn ball_speed(ball: &Ball) -> f32 {
    (ball.v.x.powi(2) + ball.v.y.powi(2)).sqrt()
}

/// Puts the ball back on the centre spot, served toward one side.
pub fn reset_ball(ball: &mut Ball, toward_right: bool) {
    ball.p = Vec3::ZERO;
    ball.left_hit_cd = 0.0;
    ball.right_hit_cd = 0.0;
    let vx = if toward_right {
        BALL_SPEED_INIT
    } else {
        -BALL_SPEED_INIT
    };
    ball.v = Vec3::new(vx, BALL_SPEED_INIT * 0.4, 0.0);
}

/// One step of the rules, `dt` seconds long.
pub fn step_physics(
    ball: &mut Ball,
    left: &mut Player,
    right: &mut Player,
    left_input: &PlayerInput,
    right_input: &PlayerInput,
    dt: f32,
) {
    left.tick(left_input, dt);
    right.tick(right_input, dt);
    if ball.left_hit_cd > 0.0 {
        ball.left_hit_cd -= dt;
    }
    if ball.right_hit_cd > 0.0 {
        ball.right_hit_cd -= dt;
    }

    left.steer(left_input, (-FIELD_HALF_W + 0.5, -2.0), dt);
    right.steer(right_input, (2.0, FIELD_HALF_W - 1.5), dt);

    // --- Ball movement ---
    ball.p += ball.v * dt;
    ball.v.z -= BALL_GRAVITY * dt;

    // Floor bounce / clamp
    if ball.p.z <= 0.0 {
        ball.p.z = 0.0;
        if ball.v.z < -5.0 {
            ball.v.z *= -0.5;
        } else {
            ball.v.z = 0.0;
        }
    }

    // --- Top / bottom wall bounce ---
    if ball.p.y > WALL_LIMIT {
        ball.p.y = WALL_LIMIT;
        ball.v.y = -ball.v.y.abs();
    } else if ball.p.y < -WALL_LIMIT {
        ball.p.y = -WALL_LIMIT;
        ball.v.y = ball.v.y.abs();
    }

    // --- Paddle collisions ---
    // The paddle cubes span x..x+2 on the left and x-1..x+1 on the right, so the
    // boxes are not mirror images of each other.
    if ball.p.z < BALL_HITTABLE_Z {
        let mut cd = ball.left_hit_cd;
        hit_paddle(
            ball,
            &mut cd,
            left,
            left_input,
            (left.x - 0.5, left.x + 2.5),
            1.0,
        );
        ball.left_hit_cd = cd;

        let mut cd = ball.right_hit_cd;
        hit_paddle(
            ball,
            &mut cd,
            right,
            right_input,
            (right.x - 1.5, right.x + 1.0),
            -1.0,
        );
        ball.right_hit_cd = cd;
    }

    // --- Scoring ---
    if ball.p.x > FIELD_HALF_W {
        left.score += 1;
        reset_ball(ball, true); // serve toward right (they just missed)
    } else if ball.p.x < -FIELD_HALF_W {
        right.score += 1;
        reset_ball(ball, false); // serve toward left (they just missed)
    }
}

/// Bounces the ball off one paddle's box, `x_span` wide, if it is inside it.
///
/// `facing` is the direction the paddle's front face points along X: +1 for
/// the left paddle, -1 for the right. Which face the ball came through is the
/// one it crossed most recently — penetration divided by speed on each axis.
fn hit_paddle(
    ball: &mut Ball,
    hit_cd: &mut f32,
    player: &mut Player,
    input: &PlayerInput,
    (px_min, px_max): (f32, f32),
    facing: f32,
) {
    let py_min = player.y - PADDLE_HALF_H - 0.5;
    let py_max = player.y + PADDLE_HALF_H + 0.5;

    if *hit_cd > 0.0
        || ball.p.x <= px_min
        || ball.p.x >= px_max
        || ball.p.y <= py_min
        || ball.p.y >= py_max
    {
        return;
    }
    *hit_cd = player.dash_timer.max(PADDLE_HIT_COOLDOWN) + 0.3;

    let (front_x, back_x) = if facing > 0.0 {
        (px_max, px_min)
    } else {
        (px_min, px_max)
    };
    let pen_front = (front_x - ball.p.x).abs();
    let pen_back = (ball.p.x - back_x).abs();
    let pen_top = py_max - ball.p.y;
    let pen_bot = ball.p.y - py_min;

    let vx = ball.v.x.abs().max(0.01);
    let vy = ball.v.y.abs().max(0.01);
    let t_x = pen_front.min(pen_back) / vx;
    let t_y = pen_top.min(pen_bot) / vy;

    if t_x <= t_y {
        if pen_front < pen_back {
            ball.p.x = front_x;
            let base_speed = (ball_speed(ball) + BALL_SPEED_INC).min(BALL_SPEED_MAX);
            let new_speed = if player.dash_timer > 0.0 {
                (base_speed + DASH_SMASH_BONUS).min(BALL_SPEED_MAX + DASH_SMASH_BONUS)
            } else {
                base_speed
            };
            let offset = ((ball.p.y - player.y) / PADDLE_HALF_H).clamp(-1.0, 1.0);
            let angle = offset * FRAC_PI_4;
            if input.lob_pressed && player.lob_cooldown <= 0.0 {
                ball.v.z = BALL_LOB_SPEED;
                ball.v.x = facing * (new_speed * BALL_LOB_X_FACTOR) * angle.cos();
                player.lob_cooldown = LOB_COOLDOWN;
            } else {
                ball.v.x = facing * new_speed * angle.cos();
            }
            ball.v.y = new_speed * angle.sin();
        } else {
            ball.p.x = back_x;
            ball.v.x = -facing * ball.v.x.abs();
        }
    } else {
        // Side face — redirect strongly in Y, small X from hit position along face
        let speed = (ball_speed(ball) + BALL_SPEED_INC).min(BALL_SPEED_MAX);
        let mid_x = (px_min + px_max) * 0.5;
        let half_x = (px_max - px_min) * 0.5;
        let offset = ((ball.p.x - mid_x) / half_x).clamp(-1.0, 1.0);
        let angle = offset * FRAC_PI_4;
        if pen_top < pen_bot {
            ball.p.y = py_max;
            ball.v.y = speed * angle.cos(); // upward
        } else {
            ball.p.y = py_min;
            ball.v.y = -speed * angle.cos(); // downward
        }
        ball.v.x = speed * angle.sin();
    }
}

/// The engine's `step_physics`, copied as it was (cgmath → glam only), to hold
/// the restructured one above to the same results.
#[cfg(test)]
#[allow(clippy::all)]
mod reference {
    use std::f32::consts::FRAC_PI_4;

    use glam::Vec3;

    use super::*;

    fn ref_ball_speed(ball: &Ball) -> f32 {
        (ball.v.x.powi(2) + ball.v.y.powi(2)).sqrt()
    }

    fn ref_reset_ball(ball: &mut Ball, toward_right: bool) {
        ball.p = Vec3::new(0.0, 0.0, 0.0);
        ball.left_hit_cd = 0.0;
        ball.right_hit_cd = 0.0;
        let vx = if toward_right {
            BALL_SPEED_INIT
        } else {
            -BALL_SPEED_INIT
        };
        ball.v = Vec3::new(vx, BALL_SPEED_INIT * 0.4, 0.0);
    }

    pub fn ref_step_physics(
        ball: &mut Ball,
        left: &mut Player,
        right: &mut Player,
        left_input: &PlayerInput,
        right_input: &PlayerInput,
        dt: f32,
    ) {
        // --- Boost timers ---
        if left.cooldown_timer > 0.0 {
            left.cooldown_timer -= dt;
        }
        if left.boost_timer > 0.0 {
            left.boost_timer -= dt;
            if left.boost_timer <= 0.0 {
                left.boost_active = false;
            }
        }
        if right.cooldown_timer > 0.0 {
            right.cooldown_timer -= dt;
        }
        if right.boost_timer > 0.0 {
            right.boost_timer -= dt;
            if right.boost_timer <= 0.0 {
                right.boost_active = false;
            }
        }

        // --- Dash timers ---
        if left.dash_cooldown > 0.0 {
            left.dash_cooldown -= dt;
        }
        if left.dash_timer > 0.0 {
            left.dash_timer -= dt;
        }
        if right.dash_cooldown > 0.0 {
            right.dash_cooldown -= dt;
        }
        if right.dash_timer > 0.0 {
            right.dash_timer -= dt;
        }

        // --- Lob cooldowns ---
        if left.lob_cooldown > 0.0 {
            left.lob_cooldown -= dt;
        }
        if right.lob_cooldown > 0.0 {
            right.lob_cooldown -= dt;
        }
        if ball.left_hit_cd > 0.0 {
            ball.left_hit_cd -= dt;
        }
        if ball.right_hit_cd > 0.0 {
            ball.right_hit_cd -= dt;
        }

        // --- Boost activation ---
        if left_input.boost_pressed && left.cooldown_timer <= 0.0 {
            left.boost_active = true;
            left.boost_timer = SPEED_BOOST_DURATION;
            left.cooldown_timer = SPEED_BOOST_COOLDOWN;
        }
        if right_input.boost_pressed && right.cooldown_timer <= 0.0 {
            right.boost_active = true;
            right.boost_timer = SPEED_BOOST_DURATION;
            right.cooldown_timer = SPEED_BOOST_COOLDOWN;
        }

        // --- Dash activation ---
        if left_input.dash_pressed && left.dash_cooldown <= 0.0 {
            left.dash_timer = DASH_DURATION;
            left.dash_cooldown = DASH_COOLDOWN;
        }
        if right_input.dash_pressed && right.dash_cooldown <= 0.0 {
            right.dash_timer = DASH_DURATION;
            right.dash_cooldown = DASH_COOLDOWN;
        }

        let left_speed = if left.dash_timer > 0.0 {
            DASH_SPEED
        } else if left.boost_active {
            PADDLE_SPEED * SPEED_BOOST_MULTIPLIER
        } else {
            PADDLE_SPEED
        };
        let right_speed = if right.dash_timer > 0.0 {
            DASH_SPEED
        } else if right.boost_active {
            PADDLE_SPEED * SPEED_BOOST_MULTIPLIER
        } else {
            PADDLE_SPEED
        };

        // --- Paddle movement ---
        let mut left_mx = left_input.move_x;
        let mut left_my = left_input.move_y;
        let left_mag = (left_mx * left_mx + left_my * left_my).sqrt();
        if left_mag > 1.0 {
            left_mx /= left_mag;
            left_my /= left_mag;
        }

        left.x = (left.x + left_mx * left_speed * dt).clamp(-FIELD_HALF_W + 0.5, -2.0); // Don't cross centre
        left.y = (left.y + left_my * left_speed * dt).clamp(-PADDLE_MAX_Y, PADDLE_MAX_Y);

        let mut right_mx = right_input.move_x;
        let mut right_my = right_input.move_y;
        let right_mag = (right_mx * right_mx + right_my * right_my).sqrt();
        if right_mag > 1.0 {
            right_mx /= right_mag;
            right_my /= right_mag;
        }

        right.x = (right.x + right_mx * right_speed * dt).clamp(2.0, FIELD_HALF_W - 1.5); // Don't cross centre
        right.y = (right.y + right_my * right_speed * dt).clamp(-PADDLE_MAX_Y, PADDLE_MAX_Y);

        // --- Ball movement ---
        ball.p += ball.v * dt;
        ball.v.z -= BALL_GRAVITY * dt;

        // Floor bounce / clamp
        if ball.p.z <= 0.0 {
            ball.p.z = 0.0;
            if ball.v.z < -5.0 {
                ball.v.z *= -0.5;
            } else {
                ball.v.z = 0.0;
            }
        }

        // --- Top / bottom wall bounce ---
        if ball.p.y > WALL_LIMIT {
            ball.p.y = WALL_LIMIT;
            ball.v.y = -ball.v.y.abs();
        } else if ball.p.y < -WALL_LIMIT {
            ball.p.y = -WALL_LIMIT;
            ball.v.y = ball.v.y.abs();
        }

        // --- Paddle collisions ---
        // Ball can only hit paddles if it is low enough (e.g. not lobbed over them)
        let ball_is_hittable = ball.p.z < 3.0;

        // Left paddle — AABB face detection (velocity-weighted penetration)
        {
            let px_min = left.x - 0.5;
            let px_max = left.x + 2.5;
            let py_min = left.y - PADDLE_HALF_H - 0.5;
            let py_max = left.y + PADDLE_HALF_H + 0.5;

            if ball_is_hittable
                && ball.left_hit_cd <= 0.0
                && ball.p.x > px_min
                && ball.p.x < px_max
                && ball.p.y > py_min
                && ball.p.y < py_max
            {
                ball.left_hit_cd = left.dash_timer.max(PADDLE_HIT_COOLDOWN) + 0.3;

                let pen_front = px_max - ball.p.x;
                let pen_back = ball.p.x - px_min;
                let pen_top = py_max - ball.p.y;
                let pen_bot = ball.p.y - py_min;

                // Divide by speed on each axis: gives "time since the ball crossed that face".
                // Smallest value = face most recently crossed = entry face.
                let vx = ball.v.x.abs().max(0.01);
                let vy = ball.v.y.abs().max(0.01);
                let t_x = pen_front.min(pen_back) / vx;
                let t_y = pen_top.min(pen_bot) / vy;

                if t_x <= t_y {
                    if pen_front < pen_back {
                        ball.p.x = px_max;
                        let base_speed =
                            (ref_ball_speed(ball) + BALL_SPEED_INC).min(BALL_SPEED_MAX);
                        let new_speed = if left.dash_timer > 0.0 {
                            (base_speed + DASH_SMASH_BONUS).min(BALL_SPEED_MAX + DASH_SMASH_BONUS)
                        } else {
                            base_speed
                        };
                        let offset = ((ball.p.y - left.y) / PADDLE_HALF_H).clamp(-1.0, 1.0);
                        let angle = offset * FRAC_PI_4;
                        if left_input.lob_pressed && left.lob_cooldown <= 0.0 {
                            ball.v.z = BALL_LOB_SPEED;
                            ball.v.x = (new_speed * BALL_LOB_X_FACTOR) * angle.cos();
                            left.lob_cooldown = LOB_COOLDOWN;
                        } else {
                            ball.v.x = new_speed * angle.cos();
                        }
                        ball.v.y = new_speed * angle.sin();
                    } else {
                        ball.p.x = px_min;
                        ball.v.x = -ball.v.x.abs();
                    }
                } else {
                    // Side face — redirect strongly in Y, small X from hit position along face
                    let speed = (ref_ball_speed(ball) + BALL_SPEED_INC).min(BALL_SPEED_MAX);
                    let mid_x = (px_min + px_max) * 0.5;
                    let half_x = (px_max - px_min) * 0.5;
                    let offset = ((ball.p.x - mid_x) / half_x).clamp(-1.0, 1.0);
                    let angle = offset * FRAC_PI_4;
                    if pen_top < pen_bot {
                        ball.p.y = py_max;
                        ball.v.y = speed * angle.cos(); // upward
                    } else {
                        ball.p.y = py_min;
                        ball.v.y = -speed * angle.cos(); // downward
                    }
                    ball.v.x = speed * angle.sin();
                }
            }
        }

        // Right paddle — AABB face detection (velocity-weighted penetration)
        {
            let px_min = right.x - 1.5;
            let px_max = right.x + 1.0;
            let py_min = right.y - PADDLE_HALF_H - 0.5;
            let py_max = right.y + PADDLE_HALF_H + 0.5;

            if ball_is_hittable
                && ball.right_hit_cd <= 0.0
                && ball.p.x > px_min
                && ball.p.x < px_max
                && ball.p.y > py_min
                && ball.p.y < py_max
            {
                ball.right_hit_cd = right.dash_timer.max(PADDLE_HIT_COOLDOWN) + 0.3;

                let pen_front = ball.p.x - px_min;
                let pen_back = px_max - ball.p.x;
                let pen_top = py_max - ball.p.y;
                let pen_bot = ball.p.y - py_min;

                let vx = ball.v.x.abs().max(0.01);
                let vy = ball.v.y.abs().max(0.01);
                let t_x = pen_front.min(pen_back) / vx;
                let t_y = pen_top.min(pen_bot) / vy;

                if t_x <= t_y {
                    if pen_front < pen_back {
                        ball.p.x = px_min;
                        let base_speed =
                            (ref_ball_speed(ball) + BALL_SPEED_INC).min(BALL_SPEED_MAX);
                        let new_speed = if right.dash_timer > 0.0 {
                            (base_speed + DASH_SMASH_BONUS).min(BALL_SPEED_MAX + DASH_SMASH_BONUS)
                        } else {
                            base_speed
                        };
                        let offset = ((ball.p.y - right.y) / PADDLE_HALF_H).clamp(-1.0, 1.0);
                        let angle = offset * FRAC_PI_4;
                        if right_input.lob_pressed && right.lob_cooldown <= 0.0 {
                            ball.v.z = BALL_LOB_SPEED;
                            ball.v.x = -(new_speed * BALL_LOB_X_FACTOR) * angle.cos();
                            right.lob_cooldown = LOB_COOLDOWN;
                        } else {
                            ball.v.x = -new_speed * angle.cos();
                        }
                        ball.v.y = new_speed * angle.sin();
                    } else {
                        ball.p.x = px_max;
                        ball.v.x = ball.v.x.abs();
                    }
                } else {
                    // Side face — redirect strongly in Y, small X from hit position along face
                    let speed = (ref_ball_speed(ball) + BALL_SPEED_INC).min(BALL_SPEED_MAX);
                    let mid_x = (px_min + px_max) * 0.5;
                    let half_x = (px_max - px_min) * 0.5;
                    let offset = ((ball.p.x - mid_x) / half_x).clamp(-1.0, 1.0);
                    let angle = offset * FRAC_PI_4;
                    if pen_top < pen_bot {
                        ball.p.y = py_max;
                        ball.v.y = speed * angle.cos(); // upward
                    } else {
                        ball.p.y = py_min;
                        ball.v.y = -speed * angle.cos(); // downward
                    }
                    ball.v.x = speed * angle.sin();
                }
            }
        }

        // --- Scoring ---
        if ball.p.x > FIELD_HALF_W {
            left.score += 1;
            ref_reset_ball(ball, true); // serve toward right (they just missed)
        } else if ball.p.x < -FIELD_HALF_W {
            right.score += 1;
            ref_reset_ball(ball, false); // serve toward left (they just missed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 60.0;

    fn idle() -> PlayerInput {
        PlayerInput::default()
    }

    fn step(ball: &mut Ball, left: &mut Player, right: &mut Player) {
        step_physics(ball, left, right, &idle(), &idle(), DT);
    }

    /// A ball heading straight at `x`, at height `z`, from a little way off.
    fn ball_at(x: f32, vx: f32, z: f32) -> Ball {
        Ball {
            p: Vec3::new(x, 0.0, z),
            v: Vec3::new(vx, 0.0, 0.0),
            ..Ball::default()
        }
    }

    #[test]
    fn a_reset_serves_from_the_centre() {
        let mut ball = ball_at(10.0, -50.0, 3.0);
        ball.left_hit_cd = 1.0;
        reset_ball(&mut ball, false);
        assert_eq!(ball.p, Vec3::ZERO);
        assert_eq!(
            ball.v,
            Vec3::new(-BALL_SPEED_INIT, BALL_SPEED_INIT * 0.4, 0.0)
        );
        assert_eq!(ball.left_hit_cd, 0.0);
    }

    #[test]
    fn missing_the_ball_scores_for_the_other_side() {
        let (mut left, mut right) = (Player::left(), Player::right());
        // Past the right paddle, below it.
        let mut ball = Ball {
            p: Vec3::new(FIELD_HALF_W - 0.1, -20.0, 0.0),
            v: Vec3::new(30.0, 0.0, 0.0),
            ..Ball::default()
        };
        step(&mut ball, &mut left, &mut right);
        assert_eq!((left.score, right.score), (1, 0));
        assert_eq!(ball.p, Vec3::ZERO);
        assert!(ball.v.x > 0.0, "served toward the side that missed");
    }

    #[test]
    fn the_walls_turn_the_ball_back() {
        let (mut left, mut right) = (Player::left(), Player::right());
        let mut ball = Ball {
            p: Vec3::new(0.0, WALL_LIMIT - 0.1, 0.0),
            v: Vec3::new(0.0, 30.0, 0.0),
            ..Ball::default()
        };
        step(&mut ball, &mut left, &mut right);
        assert_eq!(ball.p.y, WALL_LIMIT);
        assert!(ball.v.y < 0.0);
    }

    #[test]
    fn the_front_of_a_paddle_returns_the_ball_faster() {
        let (mut left, mut right) = (Player::left(), Player::right());
        // Just in front of the left paddle's face, coming at it.
        let mut ball = ball_at(left.x + 2.6, -30.0, 0.0);
        step(&mut ball, &mut left, &mut right);
        assert!(ball.v.x > 0.0, "sent back toward the right");
        assert!(
            (ball_speed(&ball) - 33.0).abs() < 1e-3,
            "{}",
            ball_speed(&ball)
        );
        assert!(ball.left_hit_cd > 0.0);

        let mut ball = ball_at(right.x - 1.6, 30.0, 0.0);
        step(&mut ball, &mut left, &mut right);
        assert!(ball.v.x < 0.0, "and the right paddle sends it left");
    }

    #[test]
    fn the_back_of_a_paddle_sends_the_ball_on() {
        let (mut left, mut right) = (Player::left(), Player::right());
        // Coming from behind the left paddle, out of its own goal.
        let mut ball = ball_at(left.x - 0.4, 6.0, 0.0);
        step(&mut ball, &mut left, &mut right);
        assert_eq!(ball.p.x, left.x - 0.5);
        assert!(ball.v.x < 0.0);
    }

    #[test]
    fn a_lob_goes_up_and_over() {
        let (mut left, mut right) = (Player::left(), Player::right());
        let lob = PlayerInput {
            lob_pressed: true,
            ..idle()
        };
        let mut ball = ball_at(left.x + 2.6, -30.0, 0.0);
        step_physics(&mut ball, &mut left, &mut right, &lob, &idle(), DT);
        assert!(ball.v.z > 0.0);
        assert_eq!(left.lob_cooldown, LOB_COOLDOWN);

        // A ball in the air passes through a paddle.
        let mut high = ball_at(right.x - 1.6, 30.0, BALL_HITTABLE_Z + 5.0);
        step(&mut high, &mut left, &mut right);
        assert!(high.v.x > 0.0);
    }

    #[test]
    fn a_boost_lasts_and_then_has_to_recharge() {
        let mut player = Player::left();
        let boost = PlayerInput {
            boost_pressed: true,
            ..idle()
        };
        player.tick(&boost, DT);
        assert!(player.boost_active);
        assert_eq!(player.speed(), PADDLE_SPEED * SPEED_BOOST_MULTIPLIER);

        let mut t = 0.0;
        while t < SPEED_BOOST_DURATION + DT {
            player.tick(&boost, DT);
            t += DT;
        }
        assert!(!player.boost_active, "still cooling down, so not restarted");
        assert!(player.cooldown_timer > 0.0);
    }

    #[test]
    fn a_dash_is_the_fastest_thing() {
        let mut player = Player::left();
        let dash = PlayerInput {
            dash_pressed: true,
            ..idle()
        };
        player.tick(&dash, DT);
        assert_eq!(player.speed(), DASH_SPEED);
        assert_eq!(player.dash_cooldown, DASH_COOLDOWN);
    }

    #[test]
    fn paddles_stay_on_their_side() {
        let (mut left, mut right) = (Player::left(), Player::right());
        let charge = PlayerInput {
            move_x: 1.0,
            move_y: 1.0,
            ..idle()
        };
        let retreat = PlayerInput {
            move_x: -1.0,
            move_y: -1.0,
            ..idle()
        };
        let mut ball = Ball::default();
        for _ in 0..600 {
            step_physics(&mut ball, &mut left, &mut right, &charge, &retreat, DT);
        }
        assert_eq!(left.x, -2.0, "not across the centre line");
        assert_eq!(left.y, PADDLE_MAX_Y);
        assert_eq!(right.x, 2.0);
        assert_eq!(right.y, -PADDLE_MAX_Y);
    }

    #[test]
    fn a_diagonal_is_no_faster() {
        let mut player = Player::new(-20.0);
        let diagonal = PlayerInput {
            move_x: 1.0,
            move_y: 1.0,
            ..idle()
        };
        player.steer(&diagonal, (-FIELD_HALF_W, FIELD_HALF_W), 0.1);
        let moved = ((player.x + 20.0).powi(2) + player.y.powi(2)).sqrt();
        assert!((moved - PADDLE_SPEED * 0.1).abs() < 1e-4);
    }

    /// A repeatable stream of inputs, held for a few frames at a time like a
    /// person's.
    struct Inputs(u64);

    impl Inputs {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }

        fn input(&mut self) -> PlayerInput {
            let axis = |n: u32| [-1.0, -0.5, 0.0, 0.7, 1.0][(n % 5) as usize];
            PlayerInput {
                move_x: axis(self.next()),
                move_y: axis(self.next()),
                lob_pressed: self.next().is_multiple_of(4),
                boost_pressed: self.next().is_multiple_of(50),
                dash_pressed: self.next().is_multiple_of(30),
                sequence_number: 0,
            }
        }
    }

    #[test]
    fn plays_exactly_as_the_engine_did() {
        let mut inputs = Inputs(0x9e37_79b9_7f4a_7c15);
        let (mut ball, mut left, mut right) = (Ball::default(), Player::left(), Player::right());
        let (mut ref_ball, mut ref_left, mut ref_right) =
            (ball.clone(), left.clone(), right.clone());
        let (mut left_input, mut right_input) = (idle(), idle());
        let mut hits = 0;

        for frame in 0..20_000 {
            if frame % 6 == 0 {
                left_input = inputs.input();
                right_input = inputs.input();
            }
            // Short frames and long ones.
            let dt = [DT, DT * 0.5, DT * 2.0][frame % 3];
            let before = ball.v.x.signum();
            step_physics(
                &mut ball,
                &mut left,
                &mut right,
                &left_input,
                &right_input,
                dt,
            );
            reference::ref_step_physics(
                &mut ref_ball,
                &mut ref_left,
                &mut ref_right,
                &left_input,
                &right_input,
                dt,
            );
            assert_eq!(ball, ref_ball, "the ball, frame {frame}");
            assert_eq!(left, ref_left, "the left player, frame {frame}");
            assert_eq!(right, ref_right, "the right player, frame {frame}");
            if ball.v.x.signum() != before {
                hits += 1;
            }
        }
        // The run covered the game: rallies, goals, both sides.
        assert!(hits > 30, "{hits} returns");
        assert!(
            left.score > 0 && right.score > 0,
            "{} - {}",
            left.score,
            right.score
        );
    }

    #[test]
    fn the_input_wire_format_is_bincode_1() {
        // Five fields of 4 bytes or 1: what the engine's clients and server send.
        let input = PlayerInput {
            move_y: 1.0,
            sequence_number: 7,
            ..idle()
        };
        let bytes = bincode::serialize(&input).unwrap();
        assert_eq!(bytes.len(), 4 + 4 + 1 + 1 + 1 + 4);
        assert_eq!(bincode::deserialize::<PlayerInput>(&bytes).unwrap(), input);
    }
}
