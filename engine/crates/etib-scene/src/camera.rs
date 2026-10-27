//! The camera, and a controller that moves it from abstract input.
//!
//! Neither touches the GPU: [`Camera`] is the maths, and [`Scene`](crate::Scene)
//! uploads what [`Camera::raw`] produces. Nor does either know about a
//! keyboard — [`CameraController`] is told which way to [`Move`], and the
//! application decides which keys mean that.

use glam::{Mat4, Vec3};

/// How the camera projects the world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Projection {
    Perspective {
        /// Vertical field of view, in degrees.
        fovy: f32,
    },
    Orthographic {
        /// Half the visible height, in world units.
        scale: f32,
    },
}

/// Where the camera is, where it looks, and how it projects.
///
/// Right-handed, with the depth range wgpu uses: the near plane lands at 0 and
/// the far plane at 1.
#[derive(Debug, Clone)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    /// Width over height. [`Scene::render`](crate::Scene::render) keeps it in
    /// step with the scene's size.
    pub aspect: f32,
    pub projection: Projection,
    pub znear: f32,
    pub zfar: f32,
    /// What [`raw`](Self::raw) returned last, for temporal reprojection.
    prev_view_proj: Option<Mat4>,
}

/// The camera as the shaders read it. Mirrors `CameraUniform` in the WGSL.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraRaw {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    /// Last frame's `view_proj`; this frame's on the first.
    pub prev_view_proj: [[f32; 4]; 4],
}

impl Camera {
    pub fn new(
        eye: Vec3,
        target: Vec3,
        up: Vec3,
        aspect: f32,
        projection: Projection,
        znear: f32,
        zfar: f32,
    ) -> Self {
        Self {
            eye,
            target,
            up,
            aspect,
            projection,
            znear,
            zfar,
            prev_view_proj: None,
        }
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye, self.target, self.up)
    }

    pub fn projection_matrix(&self) -> Mat4 {
        match self.projection {
            Projection::Perspective { fovy } => {
                Mat4::perspective_rh(fovy.to_radians(), self.aspect, self.znear, self.zfar)
            }
            Projection::Orthographic { scale } => {
                let right = scale * self.aspect;
                Mat4::orthographic_rh(-right, right, -scale, scale, self.znear, self.zfar)
            }
        }
    }

    /// Projection times view: world space to clip space.
    pub fn view_proj(&self) -> Mat4 {
        self.projection_matrix() * self.view()
    }

    /// The uniform block for this frame. Call it once per frame: it moves this
    /// frame's matrix into `prev_view_proj` for the next one, so a second call
    /// in the same frame would report no motion at all.
    pub fn advance_frame(&mut self) -> CameraRaw {
        let view = self.view();
        let proj = self.projection_matrix();
        let view_proj = proj * view;
        let prev = self.prev_view_proj.replace(view_proj).unwrap_or(view_proj);
        CameraRaw {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view: view.inverse().to_cols_array_2d(),
            inv_proj: proj.inverse().to_cols_array_2d(),
            prev_view_proj: prev.to_cols_array_2d(),
        }
    }

    pub fn set_position(&mut self, eye: Vec3) {
        self.eye = eye;
    }

    pub fn look_at(&mut self, target: Vec3) {
        self.target = target;
    }

    /// Puts the camera on a circle around `center` and aims it there.
    ///
    /// `radius` is the horizontal distance, `angle` the azimuth in radians
    /// around the Z axis, and `height` the offset along Z. [`up`](Self::up) is
    /// left alone, so set it to Z for a level horizon.
    pub fn orbit(&mut self, center: Vec3, radius: f32, angle: f32, height: f32) {
        self.eye = Vec3::new(
            center.x + radius * angle.cos(),
            center.y + radius * angle.sin(),
            center.z + height,
        );
        self.target = center;
    }
}

/// How a [`CameraController`] moves the camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMode {
    /// Look with the mouse, move along the ground, rise and fall.
    FirstPerson,
    /// No looking; movement pans, and scrolling zooms an orthographic camera.
    Isometric,
}

/// A direction a [`CameraController`] can be told to move in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    Forward,
    Back,
    Left,
    Right,
    Up,
    Down,
}

/// Flies a [`Camera`] from abstract input: directions held, mouse motion and
/// scrolling.
///
/// Which keys mean which [`Move`] is the application's business. Each
/// direction is remembered separately, so holding forward and back together
/// cancels out, and releasing one leaves the other in force.
#[derive(Debug, Clone)]
pub struct CameraController {
    /// Units per second.
    pub speed: f32,
    /// Radians per unit of mouse motion.
    pub sensitivity: f32,
    /// Rotation around the vertical axis, in radians; -π/2 looks down -Z.
    pub yaw: f32,
    /// Rotation above the horizon, in radians.
    pub pitch: f32,
    pub mode: CameraMode,
    held: [bool; 6],
    /// Scale applied to an orthographic camera on the next update.
    zoom: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
            mode: CameraMode::FirstPerson,
            held: [false; 6],
            zoom: 1.0,
        }
    }

    /// Starts or stops moving in `direction`.
    pub fn set(&mut self, direction: Move, held: bool) {
        self.held[direction as usize] = held;
    }

    /// Turns the view by a mouse movement. Ignored in isometric mode.
    pub fn look(&mut self, dx: f32, dy: f32) {
        if self.mode != CameraMode::FirstPerson {
            return;
        }
        self.yaw += dx * self.sensitivity;
        // Short of straight up or down, where the view would flip over.
        let limit = std::f32::consts::FRAC_PI_2 - 0.1;
        self.pitch = (self.pitch - dy * self.sensitivity).clamp(-limit, limit);
    }

    /// Scrolls by `lines`, positive away from the user: faster in
    /// first-person mode, closer in isometric mode.
    pub fn scroll(&mut self, lines: f32) {
        match self.mode {
            CameraMode::FirstPerson => self.speed = (self.speed + lines * 0.5).max(0.1),
            CameraMode::Isometric => {
                if lines > 0.0 {
                    self.zoom = 1.0 / 1.1;
                } else if lines < 0.0 {
                    self.zoom = 1.1;
                }
            }
        }
    }

    /// Moves `camera` for `dt` seconds of whatever is held.
    pub fn update(&mut self, camera: &mut Camera, dt: f32) {
        let forward_amount = self.amount(Move::Forward, Move::Back);
        let right_amount = self.amount(Move::Right, Move::Left);
        let up_amount = self.amount(Move::Up, Move::Down);
        let step = self.speed * dt;

        match self.mode {
            CameraMode::FirstPerson => {
                let forward = Vec3::new(
                    self.yaw.cos() * self.pitch.cos(),
                    self.pitch.sin(),
                    self.yaw.sin() * self.pitch.cos(),
                )
                .normalize();
                let right = forward.cross(Vec3::Y).normalize();
                // Walking stays level whichever way the view is pitched.
                let along_ground = Vec3::new(forward.x, 0.0, forward.z).normalize();

                camera.eye += along_ground * forward_amount * step;
                camera.eye += right * right_amount * step;
                camera.eye += Vec3::Y * up_amount * step;
                camera.target = camera.eye + forward;
                camera.up = Vec3::Y;
            }
            CameraMode::Isometric => {
                let pan = Vec3::new(right_amount * step, 0.0, -forward_amount * step);
                camera.eye += pan;
                camera.target += pan;

                if let Projection::Orthographic { scale } = &mut camera.projection {
                    *scale *= self.zoom;
                }
                self.zoom = 1.0;
            }
        }
    }

    fn amount(&self, positive: Move, negative: Move) -> f32 {
        f32::from(u8::from(self.held[positive as usize]))
            - f32::from(u8::from(self.held[negative as usize]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera(projection: Projection) -> Camera {
        Camera::new(
            Vec3::ZERO,
            Vec3::X,
            Vec3::Y,
            16.0 / 9.0,
            projection,
            0.1,
            100.0,
        )
    }

    #[test]
    fn raw_layout_matches_the_shaders() {
        assert_eq!(size_of::<CameraRaw>(), 256);
    }

    #[test]
    fn matrices_are_projection_times_view() {
        let mut perspective = camera(Projection::Perspective { fovy: 45.0 });
        let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::X, Vec3::Y);
        let expected = Mat4::perspective_rh(45f32.to_radians(), 16.0 / 9.0, 0.1, 100.0) * view;
        assert_eq!(
            perspective.advance_frame().view_proj,
            expected.to_cols_array_2d()
        );

        let mut ortho = camera(Projection::Orthographic { scale: 1.0 });
        let right = 16.0 / 9.0;
        let expected = Mat4::orthographic_rh(-right, right, -1.0, 1.0, 0.1, 100.0) * view;
        assert_eq!(ortho.advance_frame().view_proj, expected.to_cols_array_2d());
    }

    /// What `OPENGL_TO_WGPU_MATRIX` was for in the engine: wgpu's depth runs
    /// from 0 at the near plane to 1 at the far one.
    #[test]
    fn depth_runs_from_zero_to_one() {
        for projection in [
            Projection::Perspective { fovy: 60.0 },
            Projection::Orthographic { scale: 10.0 },
        ] {
            let camera = Camera::new(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y, 1.0, projection, 0.5, 50.0);
            let depth = |z: f32| camera.view_proj().project_point3(Vec3::new(0.0, 0.0, -z)).z;
            assert!(
                depth(0.5).abs() < 1e-5,
                "{projection:?}: near at {}",
                depth(0.5)
            );
            assert!(
                (depth(50.0) - 1.0).abs() < 1e-5,
                "{projection:?}: far at {}",
                depth(50.0)
            );
        }
    }

    #[test]
    fn inverses_undo_their_matrices() {
        let mut camera = Camera::new(
            Vec3::new(3.0, 4.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
            1.5,
            Projection::Perspective { fovy: 50.0 },
            0.1,
            100.0,
        );
        let raw = camera.advance_frame();
        let product = Mat4::from_cols_array_2d(&raw.inv_view) * camera.view();
        assert!(
            product.abs_diff_eq(Mat4::IDENTITY, 1e-5),
            "inv_view is not the view's inverse"
        );
        let product = Mat4::from_cols_array_2d(&raw.inv_proj) * camera.projection_matrix();
        assert!(
            product.abs_diff_eq(Mat4::IDENTITY, 1e-4),
            "inv_proj is not the projection's inverse"
        );
    }

    #[test]
    fn previous_matrix_trails_by_one_frame() {
        let mut camera = camera(Projection::Perspective { fovy: 45.0 });
        let first = camera.advance_frame();
        assert_eq!(first.prev_view_proj, first.view_proj);
        camera.set_position(Vec3::new(0.0, 1.0, 0.0));
        let second = camera.advance_frame();
        assert_eq!(second.prev_view_proj, first.view_proj);
        assert_ne!(second.view_proj, first.view_proj);
    }

    #[test]
    fn orbit_circles_around_z() {
        let mut camera = camera(Projection::Perspective { fovy: 45.0 });
        camera.orbit(
            Vec3::new(1.0, 2.0, 3.0),
            10.0,
            std::f32::consts::FRAC_PI_2,
            5.0,
        );
        assert!(camera.eye.abs_diff_eq(Vec3::new(1.0, 12.0, 8.0), 1e-5));
        assert_eq!(camera.target, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn controller_starts_looking_down_negative_z() {
        let controller = CameraController::new(10.0, 0.5);
        assert_eq!(controller.speed, 10.0);
        assert_eq!(controller.sensitivity, 0.5);
        assert_eq!(controller.mode, CameraMode::FirstPerson);
        assert_eq!(controller.yaw, -std::f32::consts::FRAC_PI_2);
        assert_eq!(controller.pitch, 0.0);
    }

    #[test]
    fn held_directions_combine() {
        let mut controller = CameraController::new(10.0, 0.1);
        for (positive, negative) in [
            (Move::Forward, Move::Back),
            (Move::Right, Move::Left),
            (Move::Up, Move::Down),
        ] {
            controller.set(positive, true);
            assert_eq!(controller.amount(positive, negative), 1.0);
            controller.set(negative, true);
            assert_eq!(
                controller.amount(positive, negative),
                0.0,
                "both held cancel out"
            );
            controller.set(positive, false);
            assert_eq!(
                controller.amount(positive, negative),
                -1.0,
                "the other stays in force"
            );
            controller.set(negative, false);
            assert_eq!(controller.amount(positive, negative), 0.0);
        }
    }

    #[test]
    fn first_person_walks_where_it_looks() {
        let mut camera = camera(Projection::Perspective { fovy: 45.0 });
        let mut controller = CameraController::new(10.0, 0.1);
        controller.set(Move::Forward, true);
        controller.update(&mut camera, 1.0);
        // Yaw -π/2 looks down -Z, so a second at 10 units/s lands at z = -10.
        assert!(
            camera.eye.abs_diff_eq(Vec3::new(0.0, 0.0, -10.0), 1e-4),
            "{}",
            camera.eye
        );
        assert!(camera.target.abs_diff_eq(Vec3::new(0.0, 0.0, -11.0), 1e-4));
        assert_eq!(camera.up, Vec3::Y);
    }

    #[test]
    fn isometric_pans_and_zooms() {
        let mut camera = camera(Projection::Orthographic { scale: 10.0 });
        let mut controller = CameraController::new(10.0, 0.1);
        controller.mode = CameraMode::Isometric;
        controller.set(Move::Right, true);
        controller.scroll(1.0);
        controller.update(&mut camera, 1.0);
        assert!(camera.eye.abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), 1e-5));
        assert!(camera.target.abs_diff_eq(Vec3::new(11.0, 0.0, 0.0), 1e-5));
        assert_eq!(
            camera.projection,
            Projection::Orthographic { scale: 10.0 / 1.1 }
        );
        // The zoom is a one-off, not a rate.
        controller.update(&mut camera, 1.0);
        assert_eq!(
            camera.projection,
            Projection::Orthographic { scale: 10.0 / 1.1 }
        );
    }

    #[test]
    fn mouse_turns_only_in_first_person() {
        let mut controller = CameraController::new(10.0, 0.1);
        controller.look(10.0, 5.0);
        assert!(controller.yaw > -std::f32::consts::FRAC_PI_2);
        assert!(controller.pitch < 0.0);

        controller.mode = CameraMode::Isometric;
        let yaw = controller.yaw;
        controller.look(10.0, 5.0);
        assert_eq!(controller.yaw, yaw);
    }

    #[test]
    fn pitch_stops_short_of_vertical() {
        let mut controller = CameraController::new(10.0, 0.1);
        controller.look(0.0, -1000.0);
        assert!(controller.pitch < std::f32::consts::FRAC_PI_2);
        controller.look(0.0, 1000.0);
        assert!(controller.pitch > -std::f32::consts::FRAC_PI_2);
    }

    #[test]
    fn scrolling_changes_speed_or_zoom() {
        let mut controller = CameraController::new(10.0, 0.1);
        controller.scroll(1.0);
        assert_eq!(controller.speed, 10.5);
        controller.scroll(-1000.0);
        assert_eq!(controller.speed, 0.1, "speed has a floor");

        controller.mode = CameraMode::Isometric;
        controller.scroll(1.0);
        assert!(controller.zoom < 1.0, "zoom in");
        controller.scroll(-1.0);
        assert!(controller.zoom > 1.0, "zoom out");
    }
}
