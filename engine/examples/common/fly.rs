//! A free-flying camera for the demos: `C` captures the mouse to look around,
//! `Escape` releases it, W/A/S/D or the arrows move (by physical key, so
//! Z/Q/S/D on AZERTY), Space and Shift rise and fall, and the wheel sets the
//! speed.

use etib::winit::event::{ElementState, WindowEvent};
use etib::winit::keyboard::PhysicalKey;
use etib::winit::window::{CursorGrabMode, Window};
use etib_input::{Input, KeyCode};
use etib_scene::{Camera, CameraController, Move};

const BINDINGS: &[(Move, &[KeyCode])] = &[
    (Move::Forward, &[KeyCode::KeyW, KeyCode::ArrowUp]),
    (Move::Back, &[KeyCode::KeyS, KeyCode::ArrowDown]),
    (Move::Left, &[KeyCode::KeyA, KeyCode::ArrowLeft]),
    (Move::Right, &[KeyCode::KeyD, KeyCode::ArrowRight]),
    (Move::Up, &[KeyCode::Space]),
    (Move::Down, &[KeyCode::ShiftLeft, KeyCode::ShiftRight]),
];

pub struct FlyCamera {
    controller: CameraController,
    captured: bool,
}

impl FlyCamera {
    /// A controller moving at `speed` units per second, starting the way
    /// `camera` already looks.
    pub fn new(speed: f32, camera: &Camera) -> Self {
        let mut controller = CameraController::new(speed, 0.003);
        let forward = (camera.target - camera.eye).normalize();
        controller.yaw = forward.z.atan2(forward.x);
        controller.pitch = forward.y.asin();
        Self {
            controller,
            captured: false,
        }
    }

    /// Captures or releases the mouse. Returns true for an Escape that only
    /// released it, so the runner does not quit on it.
    pub fn window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        let WindowEvent::KeyboardInput { event, .. } = event else {
            return false;
        };
        if event.state != ElementState::Pressed {
            return false;
        }
        match event.physical_key {
            PhysicalKey::Code(KeyCode::KeyC) => {
                self.capture(window, true);
                true
            }
            PhysicalKey::Code(KeyCode::Escape) if self.captured => {
                self.capture(window, false);
                true
            }
            _ => false,
        }
    }

    /// Moves `camera` by this frame's keys, mouse and wheel.
    pub fn update(&mut self, input: &Input, camera: &mut Camera, dt: f32) {
        for &(direction, keys) in BINDINGS {
            let held = keys.iter().any(|&key| input.is_key_pressed(key));
            self.controller.set(direction, held);
        }
        if self.captured {
            let delta = input.mouse_delta();
            self.controller.look(delta.x, delta.y);
        }
        if input.scroll() != 0.0 {
            self.controller.scroll(input.scroll());
        }
        self.controller.update(camera, dt);
    }

    fn capture(&mut self, window: &Window, capture: bool) {
        self.captured = capture;
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
