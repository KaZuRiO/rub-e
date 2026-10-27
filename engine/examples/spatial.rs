//! A sound circling the listener, then walking away from it.
//!
//! ```text
//! cargo run --bin spatial
//! cargo run --bin spatial -- some.mp3
//! ```
//!
//! With headphones on, the sound goes round: in front, to the right, behind,
//! to the left. Then it moves straight ahead into the distance until it has
//! faded out. Without an argument the sound is a generated hum; with one it
//! is that WAV or MP3 file.

use std::f32::consts::TAU;
use std::thread;
use std::time::{Duration, Instant};

use etib_audio::glam::Vec3;
use etib_audio::{Audio, Listener, Play, Sound, SpatialVoice, Waveform};

/// How often the placement is sent to the mixer, as a game's frames would.
const FRAME: Duration = Duration::from_millis(16);
const CIRCLE_SECONDS: f32 = 6.0;
const RADIUS: f32 = 4.0;
/// Units per second when walking away.
const WALK: f32 = 25.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let sound = match std::env::args().nth(1) {
        Some(path) => load(&path)?,
        None => Sound::tone(Waveform::Triangle, 220.0, 1.0),
    };
    println!("{:.1} s of sound", sound.duration().as_secs_f32());

    let audio = Audio::new_or_silent();
    let listener = Listener::new(Vec3::ZERO, Vec3::NEG_Z);
    let mut voice = SpatialVoice::play(
        &audio,
        &sound,
        Play {
            gain: 0.5,
            looping: true,
            ..Play::default()
        },
        Vec3::new(0.0, 0.0, -RADIUS),
        &listener,
    );

    println!("circling: front, right, behind, left");
    let start = Instant::now();
    while start.elapsed().as_secs_f32() < CIRCLE_SECONDS {
        let angle = start.elapsed().as_secs_f32() / CIRCLE_SECONDS * TAU;
        voice.position = Vec3::new(angle.sin(), 0.0, -angle.cos()) * RADIUS;
        voice.update(&listener);
        thread::sleep(FRAME);
    }

    println!("walking away");
    let start = Instant::now();
    while voice.position.length() < voice.spatial.max_distance {
        voice.position = Vec3::new(0.0, 0.0, -(RADIUS + start.elapsed().as_secs_f32() * WALK));
        voice.update(&listener);
        thread::sleep(FRAME);
    }
    voice.stop();
    thread::sleep(Duration::from_millis(100));
    Ok(())
}

fn load(path: &str) -> etib_audio::Result<Sound> {
    if path.ends_with(".mp3") {
        return Sound::from_mp3_file(path);
    }
    Sound::from_wav_file(path)
}
