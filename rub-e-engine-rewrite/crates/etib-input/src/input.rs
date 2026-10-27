//! Keyboard and mouse state.

use glam::Vec2;
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::held::Held;

/// How many pixels of a touchpad's scroll make one line of a wheel's.
const PIXELS_PER_LINE: f32 = 10.0;

/// The keyboard and the mouse as of this frame.
///
/// The port of the engine's `InputState`, with the mouse added. Keys are
/// physical, so `KeyCode::KeyW` is the key in that place whatever the layout
/// says: Z on an AZERTY keyboard.
///
/// The cycle, once per frame:
///   1. feed it every event: [`window_event`](Self::window_event) from
///      `App::window_event`, [`device_event`](Self::device_event) from
///      `App::device_event`;
///   2. read it in `App::update`;
///   3. call [`clear_frame_state`](Self::clear_frame_state) last in `update`,
///      so the "just" answers and the mouse motion cover exactly one frame.
///
/// It consumes nothing: whether an event was used is for the application to
/// answer from `App::window_event`. When the window loses focus, everything
/// held is released, since the releases would go to another window.
#[derive(Debug, Default)]
pub struct Input {
    keys: Held<KeyCode>,
    buttons: Held<MouseButton>,
    mouse_delta: Vec2,
    scroll: f32,
    cursor: Option<Vec2>,
}

impl Input {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => self.key(event.physical_key, event.state),
            WindowEvent::MouseInput { state, button, .. } => self.button(*button, *state),
            WindowEvent::MouseWheel { delta, .. } => self.wheel(*delta),
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = Some(Vec2::new(position.x as f32, position.y as f32));
            }
            WindowEvent::CursorLeft { .. } => self.cursor = None,
            WindowEvent::Focused(false) => {
                self.keys.release_all();
                self.buttons.release_all();
            }
            _ => {}
        }
    }

    pub fn device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.mouse_delta += Vec2::new(*dx as f32, *dy as f32);
        }
    }

    /// Whether `key` is held down.
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keys.is_down(key)
    }

    /// Whether `key` went down this frame.
    pub fn is_key_just_pressed(&self, key: KeyCode) -> bool {
        self.keys.just_pressed(key)
    }

    /// Whether `key` came up this frame.
    pub fn is_key_just_released(&self, key: KeyCode) -> bool {
        self.keys.just_released(key)
    }

    pub fn is_mouse_pressed(&self, button: MouseButton) -> bool {
        self.buttons.is_down(button)
    }

    pub fn is_mouse_just_pressed(&self, button: MouseButton) -> bool {
        self.buttons.just_pressed(button)
    }

    pub fn is_mouse_just_released(&self, button: MouseButton) -> bool {
        self.buttons.just_released(button)
    }

    /// How far the mouse moved this frame, in the device's raw units, whether
    /// or not the cursor moved with it: what looking around wants.
    pub fn mouse_delta(&self) -> Vec2 {
        self.mouse_delta
    }

    /// How far the wheel turned this frame, in lines, positive away from the
    /// user.
    pub fn scroll(&self) -> f32 {
        self.scroll
    }

    /// Where the cursor is over the window, in physical pixels from the
    /// top-left corner; `None` when it is outside.
    pub fn cursor(&self) -> Option<Vec2> {
        self.cursor
    }

    /// Forgets what happened this frame, keeping what is held: the engine's
    /// `clear_frame_state`. Call it last in `App::update`.
    pub fn clear_frame_state(&mut self) {
        self.keys.clear_frame_state();
        self.buttons.clear_frame_state();
        self.mouse_delta = Vec2::ZERO;
        self.scroll = 0.0;
    }

    fn key(&mut self, key: PhysicalKey, state: ElementState) {
        if let PhysicalKey::Code(code) = key {
            self.keys.set(code, state.is_pressed());
        }
    }

    fn button(&mut self, button: MouseButton, state: ElementState) {
        self.buttons.set(button, state.is_pressed());
    }

    fn wheel(&mut self, delta: MouseScrollDelta) {
        self.scroll += match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(position) => position.y as f32 / PIXELS_PER_LINE,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::dpi::PhysicalPosition;
    use winit::keyboard::NativeKeyCode;

    fn press(input: &mut Input, code: KeyCode) {
        input.key(PhysicalKey::Code(code), ElementState::Pressed);
    }

    fn release(input: &mut Input, code: KeyCode) {
        input.key(PhysicalKey::Code(code), ElementState::Released);
    }

    #[test]
    fn keys_follow_the_engine() {
        let mut input = Input::new();
        press(&mut input, KeyCode::KeyA);
        assert!(input.is_key_pressed(KeyCode::KeyA));
        assert!(input.is_key_just_pressed(KeyCode::KeyA));

        input.clear_frame_state();
        press(&mut input, KeyCode::KeyA); // auto-repeat
        assert!(!input.is_key_just_pressed(KeyCode::KeyA));

        input.clear_frame_state();
        release(&mut input, KeyCode::KeyA);
        assert!(!input.is_key_pressed(KeyCode::KeyA));
        assert!(input.is_key_just_released(KeyCode::KeyA));
    }

    #[test]
    fn losing_focus_releases_everything() {
        let mut input = Input::new();
        press(&mut input, KeyCode::KeyW);
        input.clear_frame_state();
        input.window_event(&WindowEvent::Focused(false));
        assert!(!input.is_key_pressed(KeyCode::KeyW));
        assert!(input.is_key_just_released(KeyCode::KeyW));
    }

    #[test]
    fn unidentified_keys_are_ignored() {
        let mut input = Input::new();
        input.key(
            PhysicalKey::Unidentified(NativeKeyCode::Unidentified),
            ElementState::Pressed,
        );
        assert!(input.keys.is_empty());
    }

    #[test]
    fn mouse_motion_accumulates_until_cleared() {
        let mut input = Input::new();
        input.device_event(&DeviceEvent::MouseMotion { delta: (1.0, 2.0) });
        input.device_event(&DeviceEvent::MouseMotion { delta: (3.0, -1.0) });
        assert_eq!(input.mouse_delta(), Vec2::new(4.0, 1.0));
        input.clear_frame_state();
        assert_eq!(input.mouse_delta(), Vec2::ZERO);
    }

    #[test]
    fn scroll_is_in_lines() {
        let mut input = Input::new();
        input.wheel(MouseScrollDelta::LineDelta(0.0, 1.0));
        input.wheel(MouseScrollDelta::PixelDelta(PhysicalPosition::new(
            0.0, 20.0,
        )));
        assert_eq!(input.scroll(), 3.0);
        input.clear_frame_state();
        assert_eq!(input.scroll(), 0.0);
    }

    #[test]
    fn mouse_buttons() {
        let mut input = Input::new();
        input.button(MouseButton::Left, ElementState::Pressed);
        assert!(input.is_mouse_pressed(MouseButton::Left));
        assert!(input.is_mouse_just_pressed(MouseButton::Left));
        input.clear_frame_state();
        input.button(MouseButton::Left, ElementState::Released);
        assert!(input.is_mouse_just_released(MouseButton::Left));
        assert!(!input.is_mouse_pressed(MouseButton::Left));
    }
}
