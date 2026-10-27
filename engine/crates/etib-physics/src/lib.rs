//! Continuous rigid-body physics: gravity, angular momentum, impulse-based
//! collision, and corner-accurate floor/wall resolution.
//!
//! The port of the engine's `physics` module, from cgmath to glam, with the
//! same constants and the same behaviour. It knows nothing of the GPU or the
//! window: a body is advanced with a position and a rotation the caller owns,
//! typically those of a `DynamicModel`.
//!
//! ## Quaternion rotation system
//!
//! All orientation state is stored as a **unit quaternion** (`Quat`).
//! Angular velocity `ω` is a world-space vector whose direction is the spin axis and
//! whose magnitude is the spin rate in rad/s.  Each frame, `integrate_rotation`
//! builds a delta quaternion `dq = from_axis_angle(ω̂, |ω|·dt)` and pre-multiplies
//! it onto the current orientation: `q ← (dq · q).normalize()`.  This avoids
//! gimbal lock, is numerically stable for any spin axis, and composes cleanly with
//! the impulse torques added by [`apply_cube_collision`].

use glam::{Quat, Vec3};

/// Gravitational acceleration in world units / s².
/// Set higher than real gravity (9.81) to give cubes authority on landing.
pub const GRAVITY: f32 = 32.0;

/// Air resistance fraction removed per second from horizontal and angular motion.
/// Keeps cubes from sliding infinitely on a perfectly frictionless surface.
const AIR_DAMPING: f32 = 0.2;

/// Below this vertical speed a floor bounce is suppressed and the cube sticks.
/// Higher value = cube stops bouncing sooner, feels denser.
const BOUNCE_STOP: f32 = 0.25;

/// Horizontal speed below which the settling phase kicks in (m/s).
const SETTLE_LIN_THRESH: f32 = 0.35;
/// Angular speed below which the settling phase kicks in (rad/s).
const SETTLE_ANG_THRESH: f32 = 0.75;
/// Fraction of the remaining angle to cover per second during settling.
const SETTLE_RATE: f32 = 5.0;
/// Extra velocity damping coefficient applied while settling.
const SETTLE_DAMP: f32 = 3.0;
/// Angle (radians) below which the cube is snapped flat and frozen.
const SETTLE_SNAP_ANGLE: f32 = 0.015;
/// Speed (m/s, rad/s) below which velocities are zeroed on snap.
const SETTLE_SNAP_VEL: f32 = 0.04;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The 8 corner offsets of a cube with given half-side.
fn corners(h: f32) -> [Vec3; 8] {
    [
        Vec3::new(-h, -h, -h),
        Vec3::new(h, -h, -h),
        Vec3::new(-h, h, -h),
        Vec3::new(h, h, -h),
        Vec3::new(-h, -h, h),
        Vec3::new(h, -h, h),
        Vec3::new(-h, h, h),
        Vec3::new(h, h, h),
    ]
}

/// Return the nearest of the 24 valid face-aligned cube orientations to `q`.
///
/// Used during settling to find the target orientation the cube should tip onto.
fn nearest_face_rotation(q: Quat) -> Quat {
    use std::f32::consts::{FRAC_PI_2, PI};

    let faces: [Quat; 6] = [
        Quat::IDENTITY,
        Quat::from_axis_angle(Vec3::X, FRAC_PI_2),
        Quat::from_axis_angle(Vec3::X, -FRAC_PI_2),
        Quat::from_axis_angle(Vec3::X, PI),
        Quat::from_axis_angle(Vec3::Z, FRAC_PI_2),
        Quat::from_axis_angle(Vec3::Z, -FRAC_PI_2),
    ];
    let spins: [Quat; 4] = [
        Quat::IDENTITY,
        Quat::from_axis_angle(Vec3::Y, FRAC_PI_2),
        Quat::from_axis_angle(Vec3::Y, PI),
        Quat::from_axis_angle(Vec3::Y, -FRAC_PI_2),
    ];

    let mut best = Quat::IDENTITY;
    let mut best_dot = f32::NEG_INFINITY;
    for &face in &faces {
        for &spin in &spins {
            let c = (face * spin).normalize();
            let dot = c.dot(q).abs();
            if dot > best_dot {
                best_dot = dot;
                best = c;
            }
        }
    }
    best
}

/// Angle in radians between two unit quaternions (treating q and −q as equal).
fn quat_angle(a: Quat, b: Quat) -> f32 {
    let dot = a.dot(b).abs().min(1.0);
    2.0 * dot.acos()
}

/// Spherical interpolation from `a` to `b`, as cgmath computes it: along the
/// shorter arc, and normalised linear interpolation once the two are within
/// `acos(0.9995)` of each other.
///
/// glam's own `slerp` only falls back to linear interpolation at `1 - ε`, which
/// is exactly the range the settling phase finishes in, so it is not used.
fn slerp(a: Quat, mut b: Quat, t: f32) -> Quat {
    let mut dot = a.dot(b);
    if dot < 0.0 {
        b = -b;
        dot = -dot;
    }
    if dot > 0.9995 {
        (a * (1.0 - t) + b * t).normalize()
    } else {
        let theta = dot.clamp(-1.0, 1.0).acos();
        (a * (theta * (1.0 - t)).sin() + b * (theta * t).sin()).normalize()
    }
}

/// Integrate angular velocity `omega` (rad/s, world-space) into a quaternion for `dt` seconds.
///
/// Builds a delta quaternion `dq = from_axis_angle(ω̂, |ω|·dt)` then
/// updates `rotation ← (dq · rotation).normalize()`.
/// Pre-multiplying keeps the spin in world space, matching how angular
/// impulses from collisions are expressed.
fn integrate_rotation(rotation: &mut Quat, omega: Vec3, dt: f32) {
    let speed = omega.length();
    if speed < 1e-5 {
        return;
    }
    let axis = omega / speed;
    let dq = Quat::from_axis_angle(axis, speed * dt);
    // Pre-multiply: dq is in world space, rotation maps local→world.
    *rotation = (dq * *rotation).normalize();
}

// ---------------------------------------------------------------------------
// RigidBody
// ---------------------------------------------------------------------------

/// Continuous rigid-body state for a cube.
#[derive(Clone, Debug)]
pub struct RigidBody {
    /// Linear velocity in world units / s.
    pub linear_velocity: Vec3,
    /// Angular velocity (axis × rad/s).
    pub angular_velocity: Vec3,
    /// Bounce coefficient [0, 1].
    pub restitution: f32,
    /// Coulomb friction coefficient used for surface contacts (floor + other cubes).
    pub floor_friction: f32,
    /// Half the cube side (centre to face).
    half_side: f32,
    /// Mass in kg.
    mass: f32,
    /// True while the cube is in contact with the floor.
    pub on_ground: bool,
    /// Wall bounds [x_min, x_max, z_min, z_max].
    pub bounds: [f32; 4],
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            restitution: 0.20,   // dense hardwood — absorbs most impact energy
            floor_friction: 1.0, // Coulomb μ; 0=ice, ~1=grippy rubber
            half_side: 1.0,
            mass: 3.0, // heavier cube; inertia = m·a²/6 = 3·4/6 = 2.0 kg·m²
            on_ground: false,
            bounds: [
                f32::NEG_INFINITY,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::INFINITY,
            ],
        }
    }
}

impl RigidBody {
    /// Create a body with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set bounce coefficient (builder style).
    pub fn with_restitution(mut self, r: f32) -> Self {
        self.restitution = r.clamp(0.0, 1.0);
        self
    }

    /// Set Coulomb friction coefficient μ for contacts (builder style).
    pub fn with_friction(mut self, f: f32) -> Self {
        self.floor_friction = f.max(0.0);
        self
    }

    /// Set half-side length (builder style).
    pub fn with_half_side(mut self, h: f32) -> Self {
        self.half_side = h.max(0.001);
        self
    }

    /// Set mass in kg (builder style).
    pub fn with_mass(mut self, m: f32) -> Self {
        self.mass = m.max(0.001);
        self
    }

    /// Set invisible wall bounds (builder style).
    pub fn with_bounds(mut self, x_min: f32, x_max: f32, z_min: f32, z_max: f32) -> Self {
        self.bounds = [x_min, x_max, z_min, z_max];
        self
    }

    /// Half the cube side (centre to face). Set with
    /// [`with_half_side`](Self::with_half_side), which keeps it positive.
    pub fn half_side(&self) -> f32 {
        self.half_side
    }

    /// Mass in kg. Set with [`with_mass`](Self::with_mass), which keeps it
    /// positive: every impulse divides by it.
    pub fn mass(&self) -> f32 {
        self.mass
    }

    /// Moment of inertia for a uniform solid cube rotating about an axis
    /// through its centre: I = m·a² / 6  where a = side length.
    pub fn inertia(&self) -> f32 {
        let a = self.half_side * 2.0;
        self.mass * a * a / 6.0
    }

    /// Advance one frame.
    pub fn update(&mut self, position: &mut Vec3, rotation: &mut Quat, dt: f32, floor_y: f32) {
        debug_assert!(self.mass > 0.0, "a rigid body's mass must be positive");
        // Gravity (only on Y)
        self.linear_velocity.y -= GRAVITY * dt;

        // Light air damping on horizontal and angular motion
        let ad = (1.0 - AIR_DAMPING * dt).max(0.0);
        self.linear_velocity.x *= ad;
        self.linear_velocity.z *= ad;
        self.angular_velocity *= ad;

        // Integrate position and orientation
        *position += self.linear_velocity * dt;
        integrate_rotation(rotation, self.angular_velocity, dt);

        // Floor collision
        self.resolve_floor(position, rotation, floor_y, dt);

        // Wall collision
        self.resolve_walls(position, rotation);

        // Face settling — smoothly tips the cube onto its nearest face once slow
        if self.on_ground {
            self.settle_to_face(position, rotation, floor_y, dt);
        }
    }

    fn settle_to_face(&mut self, position: &mut Vec3, rotation: &mut Quat, floor_y: f32, dt: f32) {
        let horiz_sq = self.linear_velocity.x * self.linear_velocity.x
            + self.linear_velocity.z * self.linear_velocity.z;
        let ang_sq = self.angular_velocity.length_squared();

        // Only settle when both linear and angular motion are slow enough
        if horiz_sq > SETTLE_LIN_THRESH * SETTLE_LIN_THRESH
            || ang_sq > SETTLE_ANG_THRESH * SETTLE_ANG_THRESH
        {
            return;
        }

        // Find the nearest valid face-aligned orientation
        let target = nearest_face_rotation(*rotation);

        // Slerp toward it — the blend fraction is capped so motion stays smooth
        let t = (SETTLE_RATE * dt).min(0.5);
        *rotation = slerp(*rotation, target, t).normalize();

        // Extra damping while settling so the cube bleeds energy naturally
        let damp = (1.0 - SETTLE_DAMP * dt).max(0.0);
        self.linear_velocity *= damp;
        self.angular_velocity *= damp;

        // As the cube rotates, keep its lowest corner exactly on the floor
        // so it appears to roll onto the face rather than sink or float.
        let min_rot_y = corners(self.half_side)
            .iter()
            .map(|&c| (*rotation * c).y)
            .fold(f32::INFINITY, f32::min);
        position.y = floor_y - min_rot_y;

        // Once close enough, snap perfectly flat and zero all motion
        if quat_angle(*rotation, target) < SETTLE_SNAP_ANGLE
            && horiz_sq < SETTLE_SNAP_VEL * SETTLE_SNAP_VEL
            && ang_sq < SETTLE_SNAP_VEL * SETTLE_SNAP_VEL
        {
            *rotation = target;
            self.linear_velocity = Vec3::ZERO;
            self.angular_velocity = Vec3::ZERO;

            // Recompute floor contact with the exact final orientation
            let min_exact_y = corners(self.half_side)
                .iter()
                .map(|&c| (*rotation * c).y)
                .fold(f32::INFINITY, f32::min);
            position.y = floor_y - min_exact_y;
        }
    }

    fn resolve_floor(&mut self, position: &mut Vec3, rotation: &Quat, floor_y: f32, dt: f32) {
        let h = self.half_side;
        let world_corners: [Vec3; 8] = corners(h).map(|c| *position + *rotation * c);
        let min_y = world_corners
            .iter()
            .map(|w| w.y)
            .fold(f32::INFINITY, f32::min);

        if min_y < floor_y {
            // Push up out of floor
            position.y += floor_y - min_y;
            self.on_ground = true;

            // Approximate contact patch: average all corners that are (nearly) at the lowest Y.
            // This gives a surface point (not the centre) so impulses generate torque.
            let eps = 1e-3;
            let mut contact = Vec3::ZERO;
            let mut count = 0.0;
            for w in &world_corners {
                if (w.y - min_y).abs() <= eps {
                    contact += *w;
                    count += 1.0;
                }
            }
            if count > 0.0 {
                contact /= count;
            } else {
                // Fallback (shouldn't happen): use the first corner.
                contact = world_corners[0];
            }
            contact.y = floor_y;

            let n = Vec3::Y;
            let r = contact - *position;
            let inv_m = 1.0 / self.mass;
            let i = self.inertia();

            // --- Normal impulse (bounce) applied at the contact point ---
            // Includes rotational contribution so angled hits tumble correctly.
            let v_contact = self.linear_velocity + self.angular_velocity.cross(r);
            let v_n = v_contact.dot(n);

            // Normal impulse magnitude used for Coulomb friction clamping.
            let jn_mag: f32;

            if v_n < 0.0 {
                let rn = r.cross(n);
                let denom_n = inv_m + rn.dot(rn) / i;
                let jn = -(1.0 + self.restitution) * v_n / denom_n;
                let impulse_n = n * jn;

                self.linear_velocity += impulse_n * inv_m;
                self.angular_velocity += r.cross(impulse_n) / i;
                jn_mag = jn.abs();

                // Small-bounce suppression: if the post-collision normal speed is tiny, stick.
                let v_post = self.linear_velocity + self.angular_velocity.cross(r);
                let v_post_n = v_post.dot(n);
                if v_post_n.abs() < BOUNCE_STOP {
                    // Cancel remaining normal motion at the contact.
                    let j_cancel = -v_post_n / denom_n;
                    let impulse_cancel = n * j_cancel;
                    self.linear_velocity += impulse_cancel * inv_m;
                    self.angular_velocity += r.cross(impulse_cancel) / i;
                }
            } else {
                // Resting contact: approximate the per-frame normal impulse from gravity.
                jn_mag = (self.mass * GRAVITY * dt).max(0.0);
            }

            // --- Tangential friction impulse (drives spin) ---
            // Opposes the relative surface velocity at the contact point.
            let v_contact = self.linear_velocity + self.angular_velocity.cross(r);
            let v_t = v_contact - n * v_contact.dot(n);
            let vt_mag = v_t.length();
            if vt_mag > 1e-4 {
                let t = v_t / vt_mag;
                let rt = r.cross(t);
                let denom_t = inv_m + rt.dot(rt) / i;

                // Impulse to bring tangential contact velocity toward zero (static friction target)
                let mut jt = -v_t.dot(t) / denom_t;
                let jt_max = self.floor_friction * jn_mag;
                jt = jt.clamp(-jt_max, jt_max);

                let impulse_t = t * jt;
                self.linear_velocity += impulse_t * inv_m;
                self.angular_velocity += r.cross(impulse_t) / i;
            }
        } else {
            self.on_ground = false;
        }
    }

    fn resolve_walls(&mut self, position: &mut Vec3, rotation: &Quat) {
        let [x_min, x_max, z_min, z_max] = self.bounds;
        let h = self.half_side;

        // Compute rotated extents once
        let (mut rmax_x, mut rmin_x, mut rmax_z, mut rmin_z) = (
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
        );
        for &c in &corners(h) {
            let w = *rotation * c;
            rmax_x = rmax_x.max(w.x);
            rmin_x = rmin_x.min(w.x);
            rmax_z = rmax_z.max(w.z);
            rmin_z = rmin_z.min(w.z);
        }

        if position.x + rmax_x > x_max {
            position.x = x_max - rmax_x;
            if self.linear_velocity.x > 0.0 {
                self.linear_velocity.x = -self.linear_velocity.x * self.restitution;
                // Wall imparts a small spin in proportion to impact speed
                self.angular_velocity.z -= self.linear_velocity.x.abs() * 0.2;
            }
        }
        if position.x + rmin_x < x_min {
            position.x = x_min - rmin_x;
            if self.linear_velocity.x < 0.0 {
                self.linear_velocity.x = -self.linear_velocity.x * self.restitution;
                self.angular_velocity.z += self.linear_velocity.x.abs() * 0.2;
            }
        }
        if position.z + rmax_z > z_max {
            position.z = z_max - rmax_z;
            if self.linear_velocity.z > 0.0 {
                self.linear_velocity.z = -self.linear_velocity.z * self.restitution;
                self.angular_velocity.x += self.linear_velocity.z.abs() * 0.2;
            }
        }
        if position.z + rmin_z < z_min {
            position.z = z_min - rmin_z;
            if self.linear_velocity.z < 0.0 {
                self.linear_velocity.z = -self.linear_velocity.z * self.restitution;
                self.angular_velocity.x -= self.linear_velocity.z.abs() * 0.2;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cube-to-cube impulse collision
// ---------------------------------------------------------------------------

/// Resolve a collision between two cubes using impulse-based rigid-body physics.
///
/// Detects contact using a sphere approximation (circumradius = `half_side × √3`),
/// then computes a physically correct impulse that affects both linear and angular
/// velocity proportionally to the impact angle and speed.
///
/// Call this every frame for every pair of cubes that may be close.
pub fn apply_cube_collision(
    a: &mut RigidBody,
    pos_a: &mut Vec3,
    b: &mut RigidBody,
    pos_b: &mut Vec3,
) {
    let diff = *pos_a - *pos_b;
    let dist_sq = diff.length_squared();
    if dist_sq < 1e-8 {
        return;
    }

    // Contact threshold: use the sum of face-diagonal radii for a good
    // balance between catching corner hits and avoiding phantom collisions.
    let r_a = a.half_side * 1.2;
    let r_b = b.half_side * 1.2;
    let min_dist = r_a + r_b;

    let dist = dist_sq.sqrt();
    if dist >= min_dist {
        return;
    }

    // Collision normal: from B toward A
    let n = diff / dist;
    let penetration = min_dist - dist;

    // Push the two cubes apart proportional to inverse mass
    debug_assert!(
        a.mass > 0.0 && b.mass > 0.0,
        "a rigid body's mass must be positive"
    );
    let inv_a = 1.0 / a.mass;
    let inv_b = 1.0 / b.mass;
    let share = 1.0 / (inv_a + inv_b);
    *pos_a += n * (penetration * inv_a * share);
    *pos_b -= n * (penetration * inv_b * share);

    // Contact point (approximate midpoint on the collision surface)
    let contact = (*pos_a + *pos_b) * 0.5;
    let r_vec_a = contact - *pos_a;
    let r_vec_b = contact - *pos_b;

    // Relative velocity at the contact point (includes rotational contribution)
    let v_a = a.linear_velocity + a.angular_velocity.cross(r_vec_a);
    let v_b = b.linear_velocity + b.angular_velocity.cross(r_vec_b);
    let v_rel = v_a - v_b;
    let v_rel_n = v_rel.dot(n);

    // Already separating — nothing to do
    if v_rel_n >= 0.0 {
        return;
    }

    let e = (a.restitution + b.restitution) * 0.5;
    let i_a = a.inertia();
    let i_b = b.inertia();

    // Angular contribution to the effective mass at the contact point
    let ra_x_n = r_vec_a.cross(n);
    let rb_x_n = r_vec_b.cross(n);
    let ang = ra_x_n.dot(ra_x_n) / i_a + rb_x_n.dot(rb_x_n) / i_b;
    let denom = inv_a + inv_b + ang;

    // Impulse magnitude
    let j = -(1.0 + e) * v_rel_n / denom;
    let impulse = n * j;

    // Apply linear impulse
    a.linear_velocity += impulse * inv_a;
    b.linear_velocity -= impulse * inv_b;

    // Apply angular impulse — this is what makes cubes spin on oblique hits
    a.angular_velocity += r_vec_a.cross(impulse) / i_a;
    b.angular_velocity -= r_vec_b.cross(impulse) / i_b;

    // --- Tangential friction impulse (drives spin on sliding/angled impacts) ---
    // Recompute contact velocity after the normal impulse for stability.
    let v_a2 = a.linear_velocity + a.angular_velocity.cross(r_vec_a);
    let v_b2 = b.linear_velocity + b.angular_velocity.cross(r_vec_b);
    let v_rel2 = v_a2 - v_b2;
    let v_rel2_n = v_rel2.dot(n);
    let v_t = v_rel2 - n * v_rel2_n;
    let vt_mag = v_t.length();
    if vt_mag > 1e-4 {
        let t = v_t / vt_mag;
        let ra_x_t = r_vec_a.cross(t);
        let rb_x_t = r_vec_b.cross(t);
        let ang_t = ra_x_t.dot(ra_x_t) / i_a + rb_x_t.dot(rb_x_t) / i_b;
        let denom_t = inv_a + inv_b + ang_t;

        let mut jt = -v_rel2.dot(t) / denom_t;
        let mu = (a.floor_friction + b.floor_friction) * 0.5;
        let jt_max = mu * j.abs();
        jt = jt.clamp(-jt_max, jt_max);

        let impulse_t = t * jt;
        a.linear_velocity += impulse_t * inv_a;
        b.linear_velocity -= impulse_t * inv_b;
        a.angular_velocity += r_vec_a.cross(impulse_t) / i_a;
        b.angular_velocity -= r_vec_b.cross(impulse_t) / i_b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Quat {
        Quat::IDENTITY
    }

    #[test]
    fn a_zero_mass_is_refused() {
        assert_eq!(RigidBody::new().with_mass(0.0).mass(), 0.001);
        assert_eq!(RigidBody::new().with_half_side(0.0).half_side(), 0.001);
    }

    #[test]
    fn a_body_built_from_extreme_values_stays_finite() {
        let mut body = RigidBody::new().with_mass(0.0).with_half_side(0.0);
        let mut position = Vec3::new(0.0, 10.0, 0.0);
        let mut rotation = Quat::IDENTITY;
        for _ in 0..600 {
            body.update(&mut position, &mut rotation, 1.0 / 60.0, 0.0);
            assert!(position.is_finite(), "{position}");
            assert!(rotation.is_finite(), "{rotation}");
        }
    }

    // ── Builder / defaults ────────────────────────────────────────────────────

    #[test]
    fn default_values() {
        let b = RigidBody::new();
        assert_eq!(b.restitution, 0.20);
        assert_eq!(b.floor_friction, 1.0);
        assert_eq!(b.half_side, 1.0);
        assert_eq!(b.mass, 3.0);
        assert!(!b.on_ground);
    }

    #[test]
    fn builder_restitution_clamped() {
        assert_eq!(RigidBody::new().with_restitution(2.0).restitution, 1.0);
        assert_eq!(RigidBody::new().with_restitution(-0.5).restitution, 0.0);
    }

    #[test]
    fn builder_half_side_floored() {
        assert_eq!(RigidBody::new().with_half_side(0.0).half_side, 0.001);
    }

    #[test]
    fn builder_mass_floored() {
        assert_eq!(RigidBody::new().with_mass(0.0).mass, 0.001);
    }

    #[test]
    fn builder_friction_non_negative() {
        assert_eq!(RigidBody::new().with_friction(-1.0).floor_friction, 0.0);
    }

    #[test]
    fn builder_bounds_stored() {
        let b = RigidBody::new().with_bounds(-3.0, 3.0, -5.0, 5.0);
        assert_eq!(b.bounds, [-3.0, 3.0, -5.0, 5.0]);
    }

    // ── Inertia ───────────────────────────────────────────────────────────────

    #[test]
    fn inertia_formula() {
        let b = RigidBody::new().with_mass(3.0).with_half_side(1.0);
        // I = m * (2h)^2 / 6 = 3 * 4 / 6 = 2.0
        let expected = 3.0 * 4.0_f32 / 6.0;
        assert!((b.inertia() - expected).abs() < 1e-5);
    }

    #[test]
    fn inertia_scales_with_mass() {
        let b1 = RigidBody::new().with_mass(1.0).with_half_side(1.0);
        let b2 = RigidBody::new().with_mass(2.0).with_half_side(1.0);
        assert!((b2.inertia() - b1.inertia() * 2.0).abs() < 1e-5);
    }

    // ── Gravity / update ──────────────────────────────────────────────────────

    #[test]
    fn gravity_accelerates_downward() {
        let mut body = RigidBody::new();
        let mut pos = Vec3::new(0.0, 100.0, 0.0);
        let mut rot = identity();
        body.update(&mut pos, &mut rot, 0.016, 0.0);
        assert!(body.linear_velocity.y < 0.0);
    }

    #[test]
    fn floor_collision_pushes_cube_up() {
        let mut body = RigidBody::new().with_half_side(1.0);
        // pos.y = 0.5 → lowest corner at -0.5 → penetrates floor at y=0
        let mut pos = Vec3::new(0.0, 0.5, 0.0);
        let mut rot = identity();
        body.update(&mut pos, &mut rot, 0.001, 0.0);
        // Lowest corner (pos.y - half_side with identity rotation) must be >= 0
        assert!(pos.y - body.half_side >= -1e-3);
        assert!(body.on_ground);
    }

    #[test]
    fn on_ground_cleared_when_airborne() {
        let mut body = RigidBody::new().with_half_side(1.0);
        let mut pos = Vec3::new(0.0, 100.0, 0.0);
        let mut rot = identity();
        body.update(&mut pos, &mut rot, 0.001, 0.0);
        assert!(!body.on_ground);
    }

    #[test]
    fn wall_x_max_clamps_position() {
        let mut body = RigidBody::new()
            .with_half_side(1.0)
            .with_bounds(-50.0, 5.0, -50.0, 50.0);
        body.linear_velocity = Vec3::new(10.0, 0.0, 0.0);
        let mut pos = Vec3::new(10.0, 5.0, 0.0);
        let mut rot = identity();
        // floor_y far below so floor doesn't interfere
        body.update(&mut pos, &mut rot, 0.001, -100.0);
        // rightmost extent must not exceed x_max=5.0
        assert!(pos.x + body.half_side <= 5.0 + 1e-3);
    }

    #[test]
    fn wall_x_min_clamps_position() {
        let mut body = RigidBody::new()
            .with_half_side(1.0)
            .with_bounds(-5.0, 50.0, -50.0, 50.0);
        body.linear_velocity = Vec3::new(-10.0, 0.0, 0.0);
        let mut pos = Vec3::new(-10.0, 5.0, 0.0);
        let mut rot = identity();
        body.update(&mut pos, &mut rot, 0.001, -100.0);
        assert!(pos.x - body.half_side >= -5.0 - 1e-3);
    }

    // ── Cube-to-cube collision ────────────────────────────────────────────────

    #[test]
    fn collision_separates_overlapping_cubes() {
        let mut a = RigidBody::new().with_half_side(1.0);
        let mut b = RigidBody::new().with_half_side(1.0);
        let mut pos_a = Vec3::new(0.5, 0.0, 0.0);
        let mut pos_b = Vec3::new(-0.5, 0.0, 0.0);

        apply_cube_collision(&mut a, &mut pos_a, &mut b, &mut pos_b);

        let dist = (pos_a - pos_b).length();
        let min_dist = a.half_side * 1.2 + b.half_side * 1.2;
        assert!(dist >= min_dist - 1e-4);
    }

    #[test]
    fn collision_no_change_when_far_apart() {
        let mut a = RigidBody::new().with_half_side(1.0);
        let mut b = RigidBody::new().with_half_side(1.0);
        let mut pos_a = Vec3::new(100.0, 0.0, 0.0);
        let mut pos_b = Vec3::ZERO;
        let vel_a_before = a.linear_velocity;
        let vel_b_before = b.linear_velocity;

        apply_cube_collision(&mut a, &mut pos_a, &mut b, &mut pos_b);

        assert_eq!(a.linear_velocity, vel_a_before);
        assert_eq!(b.linear_velocity, vel_b_before);
    }

    #[test]
    fn collision_bounces_approaching_cubes() {
        let mut a = RigidBody::new().with_half_side(1.0).with_restitution(1.0);
        let mut b = RigidBody::new().with_half_side(1.0).with_restitution(1.0);
        // A moves left, B moves right — head-on, cubes are overlapping
        a.linear_velocity = Vec3::new(-5.0, 0.0, 0.0);
        b.linear_velocity = Vec3::new(5.0, 0.0, 0.0);
        let mut pos_a = Vec3::new(1.0, 0.0, 0.0);
        let mut pos_b = Vec3::new(-1.0, 0.0, 0.0);

        apply_cube_collision(&mut a, &mut pos_a, &mut b, &mut pos_b);

        // Elastic equal-mass head-on: velocities exchange along collision axis
        assert!(a.linear_velocity.x > 0.0, "A should rebound rightward");
        assert!(b.linear_velocity.x < 0.0, "B should rebound leftward");
    }

    #[test]
    fn collision_skipped_when_already_separating() {
        let mut a = RigidBody::new().with_half_side(1.0);
        let mut b = RigidBody::new().with_half_side(1.0);
        // Cubes overlap but move apart — no impulse should be applied
        a.linear_velocity = Vec3::new(10.0, 0.0, 0.0);
        b.linear_velocity = Vec3::new(-10.0, 0.0, 0.0);
        let mut pos_a = Vec3::new(1.0, 0.0, 0.0);
        let mut pos_b = Vec3::new(-1.0, 0.0, 0.0);
        let vel_a_before = a.linear_velocity;
        let vel_b_before = b.linear_velocity;

        apply_cube_collision(&mut a, &mut pos_a, &mut b, &mut pos_b);

        // Separation pushes positions apart but linear velocity should be unchanged
        // (the velocity-relative check: v_rel_n >= 0 → return early after position fix)
        assert_eq!(a.linear_velocity, vel_a_before);
        assert_eq!(b.linear_velocity, vel_b_before);
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    #[test]
    fn corners_returns_8_points_at_correct_distance() {
        let h = 1.5_f32;
        let pts = corners(h);
        assert_eq!(pts.len(), 8);
        let expected_dist = (3.0_f32 * h * h).sqrt();
        for p in &pts {
            let dist = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
            assert!((dist - expected_dist).abs() < 1e-5);
        }
    }

    #[test]
    fn quat_angle_identity_to_itself_is_zero() {
        let q = identity();
        assert!(quat_angle(q, q) < 1e-5);
    }

    #[test]
    fn slerp_takes_the_shorter_arc() {
        let half = slerp(identity(), -Quat::from_rotation_y(1.0), 0.5);
        assert!(quat_angle(half, Quat::from_rotation_y(0.5)) < 1e-4);
        // Within the linear range too.
        let near = slerp(identity(), Quat::from_rotation_y(0.01), 0.5);
        assert!(quat_angle(near, Quat::from_rotation_y(0.005)) < 1e-4);
    }

    #[test]
    fn a_tilted_cube_settles_flat_on_the_floor() {
        let mut body = RigidBody::new();
        let mut pos = Vec3::new(0.0, 1.5, 0.0);
        let mut rot = Quat::from_rotation_x(0.3);
        for _ in 0..600 {
            body.update(&mut pos, &mut rot, 1.0 / 60.0, 0.0);
        }
        assert!(body.on_ground);
        assert!(quat_angle(rot, nearest_face_rotation(rot)) < SETTLE_SNAP_ANGLE);
        assert!((pos.y - body.half_side).abs() < 1e-3, "resting on a face");
        assert!(body.angular_velocity.length() < SETTLE_SNAP_VEL);
    }

    #[test]
    fn quat_angle_negated_quaternion_is_same_rotation() {
        let q = identity();
        let neg_q = Quat::from_xyzw(-q.x, -q.y, -q.z, -q.w);
        // q and -q represent the same rotation; angle should be 0
        assert!(quat_angle(q, neg_q) < 1e-5);
    }
}
