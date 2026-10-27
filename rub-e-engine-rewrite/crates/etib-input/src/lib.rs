//! What the keyboard, the mouse and the gamepads are doing, frame by frame.
//!
//! An application owns an [`Input`], feeds it every event it receives, reads
//! it while updating, and clears it once per frame:
//!
//! ```no_run
//! # use etib_input::{Input, KeyCode};
//! # use winit::event::{DeviceEvent, WindowEvent};
//! # use winit::window::Window;
//! struct Game {
//!     input: Input,
//! }
//!
//! impl Game {
//!     // From `App::window_event` and `App::device_event`.
//!     fn window_event(&mut self, _window: &Window, event: &WindowEvent) -> bool {
//!         self.input.window_event(event);
//!         false
//!     }
//!     fn device_event(&mut self, event: &DeviceEvent) {
//!         self.input.device_event(event);
//!     }
//!
//!     // From `App::update`.
//!     fn update(&mut self, _dt: f32) {
//!         if self.input.is_key_just_pressed(KeyCode::Space) {
//!             // jump
//!         }
//!         self.input.clear_frame_state();
//!     }
//! }
//! ```
//!
//! The crate knows winit and nothing else of etib: it draws nothing and needs
//! no device.

mod held;
mod input;

pub use glam::Vec2;
pub use input::Input;
pub use winit::event::MouseButton;
pub use winit::keyboard::KeyCode;
