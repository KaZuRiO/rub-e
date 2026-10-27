//! Keys and gamepads to [`PlayerInput`], with the engine's bindings.
//!
//! Locally, the left player has W/A/S/D, Left Shift (lob), Left Ctrl (boost)
//! and Space (dash), and the first gamepad; the right player has the arrows,
//! Right Shift, Enter and Numpad 0, and the second gamepad. A client has one
//! player, and either set of keys.

use etib_input::{Axis, Button, Input, KeyCode};

use crate::state::PlayerInput;

/// What a gamepad asks for: movement, lob, boost, dash.
struct Pad {
    move_x: f32,
    move_y: f32,
    lob: bool,
    boost: bool,
    dash: bool,
}

/// The `index`-th gamepad's sticks and buttons; the d-pad overrides the stick.
fn gamepad(input: &Input, index: usize) -> Option<Pad> {
    let pad = input.gamepads().get(index)?;
    // `axis` has the dead zone the engine applied already.
    let mut move_y = pad.axis(Axis::LeftStickY);
    let mut move_x = pad.axis(Axis::LeftStickX);
    if pad.is_pressed(Button::DPadUp) {
        move_y = 1.0;
    }
    if pad.is_pressed(Button::DPadDown) {
        move_y = -1.0;
    }
    if pad.is_pressed(Button::DPadRight) {
        move_x = 1.0;
    }
    if pad.is_pressed(Button::DPadLeft) {
        move_x = -1.0;
    }
    Some(Pad {
        move_x,
        move_y,
        lob: pad.is_pressed(Button::RightTrigger2)
            || pad.is_pressed(Button::RightTrigger)
            || pad.is_pressed(Button::East),
        boost: pad.is_pressed(Button::LeftTrigger) || pad.is_pressed(Button::LeftTrigger2),
        dash: pad.is_pressed(Button::South),
    })
}

/// One local player's keys.
struct Keys {
    up: KeyCode,
    down: KeyCode,
    right: KeyCode,
    left: KeyCode,
    lob: KeyCode,
    boost: KeyCode,
    dash: KeyCode,
}

const LEFT_KEYS: Keys = Keys {
    up: KeyCode::KeyW,
    down: KeyCode::KeyS,
    right: KeyCode::KeyD,
    left: KeyCode::KeyA,
    lob: KeyCode::ShiftLeft,
    boost: KeyCode::ControlLeft,
    dash: KeyCode::Space,
};

const RIGHT_KEYS: Keys = Keys {
    up: KeyCode::ArrowUp,
    down: KeyCode::ArrowDown,
    right: KeyCode::ArrowRight,
    left: KeyCode::ArrowLeft,
    lob: KeyCode::ShiftRight,
    boost: KeyCode::Enter,
    dash: KeyCode::Numpad0,
};

/// A local player: the gamepad first, and a held key over the stick.
fn local(input: &Input, pad_index: usize, keys: &Keys) -> PlayerInput {
    let pad = gamepad(input, pad_index).unwrap_or(Pad {
        move_x: 0.0,
        move_y: 0.0,
        lob: false,
        boost: false,
        dash: false,
    });
    let mut player = PlayerInput {
        move_x: pad.move_x,
        move_y: pad.move_y,
        lob_pressed: pad.lob || input.is_key_pressed(keys.lob),
        boost_pressed: pad.boost || input.is_key_just_pressed(keys.boost),
        dash_pressed: pad.dash || input.is_key_just_pressed(keys.dash),
        sequence_number: 0,
    };
    if input.is_key_pressed(keys.up) {
        player.move_y = 1.0;
    } else if input.is_key_pressed(keys.down) {
        player.move_y = -1.0;
    }
    if input.is_key_pressed(keys.right) {
        player.move_x = 1.0;
    } else if input.is_key_pressed(keys.left) {
        player.move_x = -1.0;
    }
    player
}

pub fn left_input(input: &Input) -> PlayerInput {
    local(input, 0, &LEFT_KEYS)
}

pub fn right_input(input: &Input) -> PlayerInput {
    local(input, 1, &RIGHT_KEYS)
}

/// A network client's one player: either set of movement keys, the first
/// gamepad over the keys when it is moved.
pub fn client_input(input: &Input) -> PlayerInput {
    let held = |a: KeyCode, b: KeyCode| input.is_key_pressed(a) || input.is_key_pressed(b);
    let mut player = PlayerInput::default();
    // W/S or Arrow Keys (the server decides whether this is left or right).
    if held(KeyCode::KeyW, KeyCode::ArrowUp) {
        player.move_y = 1.0;
    }
    if held(KeyCode::KeyS, KeyCode::ArrowDown) {
        player.move_y = -1.0;
    }
    if held(KeyCode::KeyD, KeyCode::ArrowRight) {
        player.move_x = 1.0;
    }
    if held(KeyCode::KeyA, KeyCode::ArrowLeft) {
        player.move_x = -1.0;
    }
    player.lob_pressed = held(KeyCode::ShiftLeft, KeyCode::ShiftRight);
    player.boost_pressed = input.is_key_just_pressed(KeyCode::ControlLeft)
        || input.is_key_just_pressed(KeyCode::Enter);
    player.dash_pressed = input.is_key_just_pressed(KeyCode::Space);

    if let Some(pad) = gamepad(input, 0) {
        if pad.move_y != 0.0 {
            player.move_y = pad.move_y;
        }
        if pad.move_x != 0.0 {
            player.move_x = pad.move_x;
        }
        player.lob_pressed |= pad.lob;
        player.boost_pressed |= pad.boost;
        player.dash_pressed |= pad.dash;
    }
    player
}
