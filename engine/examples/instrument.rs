//! Loading a CLAP instrument and playing it.
//!
//! ```sh
//! # The test instrument built alongside this crate, which needs no licence.
//! cargo build -p etib-examples --example test_instrument
//! cargo run --bin instrument
//!
//! # Or any plugin on this machine.
//! cargo run --bin instrument -- ~/Library/Audio/Plug-Ins/CLAP/Thing.clap
//!
//! # What is inside a file, without instantiating anything.
//! cargo run --bin instrument -- --list Thing.clap
//!
//! # With the plugin's own editor open, which is a debugging aid rather than
//! # something an application would ship.
//! cargo run --bin instrument -- --gui Thing.clap
//! ```

use std::path::PathBuf;
use std::thread::sleep;
use std::time::{Duration, Instant};

use etib_audio::{Audio, Gui, InstrumentConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut gui = false;
    let mut list = false;
    let mut path = None;
    for argument in std::env::args().skip(1) {
        match argument.as_str() {
            "--gui" => gui = true,
            "--list" => list = true,
            other => path = Some(PathBuf::from(other)),
        }
    }

    let path = match path {
        Some(path) => path,
        None => test_instrument().ok_or(
            "pass the path to a .clap file, or build the one that comes with the examples:\n  \
             cargo build -p etib-examples --example test_instrument",
        )?,
    };

    if list {
        // SAFETY: the file is one the person running this chose to open.
        for plugin in unsafe { etib_audio::plugins_in(&path) }? {
            let kind = if plugin.is_instrument {
                "instrument"
            } else {
                "effect"
            };
            println!(
                "{:<40} {} {} ({kind})\n{:<40} {}",
                plugin.name, plugin.version, plugin.vendor, "", plugin.id
            );
        }
        return Ok(());
    }

    let audio = Audio::new()?;
    println!(
        "playing through '{}' at {} Hz, {} frames a block",
        audio.device(),
        audio.sample_rate(),
        audio.block_frames()
    );

    let config = InstrumentConfig {
        // Headless is the default, and what an application ships with. The
        // editor is for looking at what a plugin is doing while working on it.
        gui: if gui { Gui::Debug } else { Gui::Headless },
        gain: 0.7,
        ..InstrumentConfig::default()
    };

    // SAFETY: loading a plugin runs its code. The file is the one the person
    // running this asked for.
    let mut instrument = unsafe { audio.load_instrument(&path, &config) }?;
    println!("\nloaded {:?}", instrument.info());

    let params = instrument.params();
    match params.len() {
        0 => println!("it exposes no parameters"),
        count => {
            println!("{count} parameters:");
            for param in params.iter().take(8) {
                println!(
                    "  {:>4}  {:<28} {} .. {} (default {})",
                    param.id, param.name, param.min, param.max, param.default
                );
            }
        }
    }

    if gui {
        match instrument.show_gui() {
            Ok(()) => println!("\nits editor is open, in a window beside this one"),
            Err(err) => println!("\nno editor: {err:#}"),
        }
    }

    // A tune, so that there is something to listen to: two bars of an arpeggio
    // over a held bass note.
    println!("\nplaying");
    let bass = 36;
    instrument.note_on(0, bass, 0.5);

    let melody = [60, 64, 67, 72, 67, 64, 60, 55];
    for round in 0..2 {
        for (step, key) in melody.iter().enumerate() {
            let key = key + if round == 1 { 3 } else { 0 };
            instrument.note_on(0, key, 0.5 + 0.4 * (step == 0) as u8 as f32);
            hold(&mut instrument, Duration::from_millis(180));
            instrument.note_off(0, key);
        }
    }

    instrument.note_off(0, bass);
    hold(&mut instrument, Duration::from_millis(600));

    if gui && instrument.is_gui_open() {
        // Kept open so that there is something to look at. A plugin that opened
        // its own window tells us when it is closed; one in a window this
        // opened is noticed going away. Neither is guaranteed — a plugin whose
        // editor is a pretence never reports anything — so this gives up after
        // a while rather than hanging.
        println!("\nthe editor is open; closing it, or Ctrl-C, ends this");
        let until = Instant::now() + Duration::from_secs(120);
        while instrument.is_gui_open() && Instant::now() < until {
            instrument.poll();
            pump_the_event_loop();
            sleep(Duration::from_millis(8));
        }
    }

    // Unloading takes the plugin back from the audio thread before destroying
    // it, which is why it can fail and why it is worth saying so.
    println!("unloading");
    instrument.unload()?;
    Ok(())
}

/// Waits, while keeping the instrument polled the way a frame loop would.
fn hold(instrument: &mut etib_audio::Instrument, duration: Duration) {
    let until = Instant::now() + duration;
    while Instant::now() < until {
        instrument.poll();
        pump_the_event_loop();
        sleep(Duration::from_millis(8));
    }
}

/// Draws and clicks whatever windows this process has.
///
/// An application already has this — it is its frame loop — and a plugin's
/// editor is drawn by it like any other window. This example has no window of
/// its own, so it stands in for one, and without this a plugin's editor would
/// open and then sit there frozen.
#[cfg(target_os = "macos")]
fn pump_the_event_loop() {
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSApplication, NSEventMask};
    use objc2_foundation::NSDefaultRunLoopMode;

    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    let app = NSApplication::sharedApplication(mtm);

    // Everything that is waiting, and then straight back to the music: passing
    // no date means do not wait for more.
    while let Some(event) = unsafe {
        app.nextEventMatchingMask_untilDate_inMode_dequeue(
            NSEventMask::Any,
            None,
            NSDefaultRunLoopMode,
            true,
        )
    } {
        app.sendEvent(&event);
    }
}

#[cfg(not(target_os = "macos"))]
fn pump_the_event_loop() {}

/// The test instrument built from the examples package, if it is there.
fn test_instrument() -> Option<PathBuf> {
    let extension = if cfg!(target_os = "windows") {
        "test_instrument.dll"
    } else if cfg!(target_os = "macos") {
        "libtest_instrument.dylib"
    } else {
        "libtest_instrument.so"
    };

    // This program sits in target/<profile>, and cargo leaves the plugin, an
    // example, in target/<profile>/examples.
    let examples = std::env::current_exe()
        .ok()?
        .parent()?
        .join("examples")
        .join(extension);
    examples.exists().then_some(examples)
}
