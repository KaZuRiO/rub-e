//! Gamepads, through gilrs.

use gilrs::{Axis, Button, EventType, GamepadId, Gilrs};
use log::{info, warn};

use crate::held::Held;

/// Stick and trigger values closer to rest than this read as 0, axis by axis,
/// as in the engine's pong.
pub const DEADZONE: f32 = 0.1;

/// The connected gamepads.
///
/// Having none is not an error, and neither is a platform where gamepads
/// cannot be read at all: either way there are simply no gamepads to get.
///
/// Button changes are collected by [`Input::clear_frame_state`], so a press
/// shows in the `update` after it happened, like a key does. Until that has
/// run once, gilrs may not have seen a gamepad that is plugged in.
///
/// [`Input::clear_frame_state`]: crate::Input::clear_frame_state
#[derive(Debug)]
pub struct Gamepads {
    /// `None` where gilrs could not start.
    gilrs: Option<Gilrs>,
    buttons: Held<(GamepadId, Button)>,
}

impl Default for Gamepads {
    fn default() -> Self {
        Self::new()
    }
}

impl Gamepads {
    /// Starts listening for gamepads. A platform where gilrs cannot start
    /// gives an empty set rather than an error.
    pub fn new() -> Self {
        let gilrs = match Gilrs::new() {
            Ok(gilrs) => gilrs,
            // gilrs offers a stand-in context here, but it never sees a
            // gamepad, so there is nothing to keep it for.
            Err(gilrs::Error::NotImplemented(_)) => {
                warn!("gamepads are not supported on this platform");
                return Self::unavailable();
            }
            Err(err) => {
                warn!("gamepads unavailable: {err}");
                return Self::unavailable();
            }
        };
        for (_, pad) in gilrs.gamepads() {
            info!("gamepad connected: {}", pad.name());
        }
        Self {
            gilrs: Some(gilrs),
            buttons: Held::default(),
        }
    }

    /// A set that never has a gamepad, for when gilrs cannot start.
    pub(crate) fn unavailable() -> Self {
        Self {
            gilrs: None,
            buttons: Held::default(),
        }
    }

    /// Whether gamepads can be read at all here.
    pub fn is_available(&self) -> bool {
        self.gilrs.is_some()
    }

    /// How many gamepads are connected.
    pub fn count(&self) -> usize {
        self.gilrs
            .as_ref()
            .map_or(0, |gilrs| gilrs.gamepads().count())
    }

    /// The `index`-th connected gamepad, in the order gilrs lists them.
    pub fn get(&self, index: usize) -> Option<Gamepad<'_>> {
        let (_, pad) = self.gilrs.as_ref()?.gamepads().nth(index)?;
        Some(Gamepad {
            pad,
            buttons: &self.buttons,
        })
    }

    /// Whether `button` went down this frame on any gamepad.
    pub fn any_just_pressed(&self, button: Button) -> bool {
        (0..self.count())
            .filter_map(|index| self.get(index))
            .any(|pad| pad.is_just_pressed(button))
    }

    /// Forgets this frame's changes, then collects those that arrived since.
    pub(crate) fn clear_frame_state(&mut self) {
        self.buttons.clear_frame_state();
        let Some(gilrs) = &mut self.gilrs else {
            return;
        };
        while let Some(event) = gilrs.next_event() {
            match event.event {
                EventType::ButtonPressed(button, _) => self.buttons.set((event.id, button), true),
                EventType::ButtonReleased(button, _) => {
                    self.buttons.set((event.id, button), false);
                }
                EventType::Connected => {
                    info!("gamepad connected: {}", gilrs.gamepad(event.id).name());
                }
                // Its releases will never come.
                EventType::Disconnected => {
                    info!("gamepad disconnected");
                    self.buttons.release_where(|&(id, _)| id == event.id);
                }
                _ => {}
            }
        }
    }
}

/// One connected gamepad, as of this frame.
#[derive(Clone, Copy)]
pub struct Gamepad<'a> {
    pad: gilrs::Gamepad<'a>,
    buttons: &'a Held<(GamepadId, Button)>,
}

impl Gamepad<'_> {
    /// The name the operating system gives the device.
    pub fn name(&self) -> &str {
        self.pad.name()
    }

    /// Whether `button` is held down. `Button::Unknown` never is.
    pub fn is_pressed(&self, button: Button) -> bool {
        // gilrs asserts against being asked about it.
        button != Button::Unknown && self.pad.is_pressed(button)
    }

    /// Whether `button` went down this frame. `Button::Unknown` never does.
    pub fn is_just_pressed(&self, button: Button) -> bool {
        button != Button::Unknown && self.buttons.just_pressed((self.pad.id(), button))
    }

    /// Where `axis` is, in -1..1 for a stick and 0..1 for a trigger, with the
    /// [`DEADZONE`] applied. `Axis::Unknown` reads 0.
    pub fn axis(&self, axis: Axis) -> f32 {
        // gilrs asserts against being asked about it.
        if axis == Axis::Unknown {
            return 0.0;
        }
        deadzone(self.pad.value(axis))
    }
}

fn deadzone(value: f32) -> f32 {
    if value.abs() < DEADZONE { 0.0 } else { value }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadzone_is_per_axis() {
        assert_eq!(deadzone(0.05), 0.0);
        assert_eq!(deadzone(-0.09), 0.0);
        assert_eq!(deadzone(0.5), 0.5);
        assert_eq!(deadzone(-1.0), -1.0);
    }

    #[test]
    fn unavailable_gamepads_are_empty() {
        let mut pads = Gamepads::unavailable();
        assert!(!pads.is_available());
        assert_eq!(pads.count(), 0);
        assert!(pads.get(0).is_none());
        assert!(!pads.any_just_pressed(Button::Start));
        pads.clear_frame_state();
    }

    #[test]
    fn opening_never_panics() {
        // On a machine with no gamepad, and possibly no /dev/input at all.
        let pads = Gamepads::new();
        assert!(pads.get(usize::MAX).is_none());
    }
}
