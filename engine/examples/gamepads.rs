//! Prints what the connected gamepads do, without opening a window.
//!
//! ```text
//! cargo run --bin gamepads
//! ```
//!
//! Plug a gamepad in, press its buttons and move its left stick; unplug it
//! with a button held to check that the button is released. Stop with Ctrl-C.

use std::thread;
use std::time::Duration;

use etib_input::{Axis, Button, Input};

const BUTTONS: [Button; 19] = [
    Button::South,
    Button::East,
    Button::North,
    Button::West,
    Button::C,
    Button::Z,
    Button::LeftTrigger,
    Button::LeftTrigger2,
    Button::RightTrigger,
    Button::RightTrigger2,
    Button::Select,
    Button::Start,
    Button::Mode,
    Button::LeftThumb,
    Button::RightThumb,
    Button::DPadUp,
    Button::DPadDown,
    Button::DPadLeft,
    Button::DPadRight,
];

/// How far a stick has to move before the change is printed.
const STICK_STEP: f32 = 0.05;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut input = Input::new();
    if !input.gamepads().is_available() {
        println!("gamepad support unavailable");
        return;
    }
    println!("waiting for gamepads; Ctrl-C to stop");

    let mut names: Vec<String> = Vec::new();
    let mut sticks: Vec<(f32, f32)> = Vec::new();
    loop {
        // Gamepad events are collected here, so they show from the next pass.
        input.clear_frame_state();
        let pads = input.gamepads();

        let current: Vec<String> = (0..pads.count())
            .filter_map(|i| pads.get(i))
            .map(|pad| pad.name().to_owned())
            .collect();
        if current != names {
            println!("connected: {current:?}");
            names = current;
            sticks.resize(names.len(), (0.0, 0.0));
        }

        for (i, stick) in sticks.iter_mut().enumerate() {
            let Some(pad) = pads.get(i) else { continue };
            for button in BUTTONS {
                if pad.is_just_pressed(button) {
                    println!("pad {i}: {button:?} pressed");
                }
            }
            let now = (pad.axis(Axis::LeftStickX), pad.axis(Axis::LeftStickY));
            if (now.0 - stick.0).abs() > STICK_STEP || (now.1 - stick.1).abs() > STICK_STEP {
                println!("pad {i}: left stick ({:.2}, {:.2})", now.0, now.1);
                *stick = now;
            }
        }

        thread::sleep(Duration::from_millis(16));
    }
}
