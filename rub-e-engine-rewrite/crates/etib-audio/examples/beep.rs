//! Everything `etib-audio` does, in the order it is worth hearing.
//!
//! ```sh
//! cargo run -p etib-audio --example beep
//! RUST_LOG=etib_audio=debug cargo run -p etib-audio --example beep
//! ```
//!
//! Pass the path to a WAV file to hear that instead of the generated tones:
//!
//! ```sh
//! cargo run -p etib-audio --example beep -- some-effect.wav
//! ```

use std::thread::sleep;
use std::time::Duration;

use etib_audio::{Audio, Play, Sound, Waveform};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let audio = Audio::new()?;
    println!(
        "playing through '{}' at {} Hz, {} channels",
        audio.device(),
        audio.sample_rate(),
        audio.channels()
    );

    match std::env::args().nth(1) {
        Some(path) => {
            let sound = Sound::from_wav_file(&path)?;
            println!("\n{path}: {sound:?}");
            play_around(&audio, &sound);
        }
        None => tour(&audio),
    }

    // The device keeps playing for as long as this is alive, and stops when it
    // is dropped — so an application holds its `Audio` for as long as it runs.
    Ok(())
}

/// One of each waveform, then what the playback controls do to them.
fn tour(audio: &Audio) {
    let waveforms = [
        (Waveform::Sine, "sine: one frequency, nothing else"),
        (Waveform::Triangle, "triangle: soft and hollow"),
        (Waveform::Saw, "saw: bright and buzzy"),
        (Waveform::Square, "square: hard and hollow"),
        (Waveform::Noise, "noise: no pitch at all"),
    ];

    for (waveform, description) in waveforms {
        println!("{description}");
        let tone = Sound::tone(waveform, 440.0, 0.35);
        audio.play_with(
            &tone,
            Play {
                gain: 0.25,
                ..Play::default()
            },
        );
        sleep(Duration::from_millis(450));
    }

    println!("\na chord: three voices at once, summed");
    for hz in [261.63, 329.63, 392.0] {
        let tone = Sound::tone(Waveform::Sine, hz, 1.0);
        audio.play_with(
            &tone,
            Play {
                gain: 0.2,
                ..Play::default()
            },
        );
    }
    sleep(Duration::from_millis(100));
    println!("  {} voices playing", audio.playing());
    sleep(Duration::from_millis(1100));

    play_around(audio, &Sound::tone(Waveform::Saw, 220.0, 0.6));
}

/// The controls, on whatever sound it is given.
fn play_around(audio: &Audio, sound: &Sound) {
    println!("\npanning: left, centre, right");
    for pan in [-1.0, 0.0, 1.0] {
        audio.play_with(
            sound,
            Play {
                gain: 0.25,
                pan,
                ..Play::default()
            },
        );
        sleep(Duration::from_millis(500));
    }

    println!("\nspeed: half, as recorded, double — the pitch moves with it");
    for speed in [0.5, 1.0, 2.0] {
        audio.play_with(
            sound,
            Play {
                gain: 0.25,
                speed,
                ..Play::default()
            },
        );
        sleep(sound.duration().div_f32(speed) + Duration::from_millis(100));
    }

    println!("\nlooping, with the level and speed changed while it plays");
    let voice = audio.play_with(
        sound,
        Play {
            gain: 0.05,
            looping: true,
            ..Play::default()
        },
    );
    for step in 1..=20 {
        voice.set_gain(0.05 + 0.2 * step as f32 / 20.0);
        voice.set_speed(1.0 + step as f32 / 20.0);
        sleep(Duration::from_millis(100));
    }

    println!("stopping it: faded out over a block, not cut");
    voice.stop();
    sleep(Duration::from_millis(200));
    assert!(!voice.is_playing());

    println!("\ndone; {} voices playing", audio.playing());
    if audio.stolen() > 0 {
        println!("{} voices were cut short to make room", audio.stolen());
    }
}
