//! Sound playback: a mixer over the system's default output device.
//!
//! Enough to give an application the noises it makes — an effect, a blip, a
//! loop underneath everything — and no more than that. A [`Sound`] is samples
//! in memory, [`Audio`] is the device they are mixed onto, and [`play`] starts
//! one and hands back a [`Voice`] to change or stop it while it sounds:
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use etib_audio::{Audio, Play, Sound, Waveform};
//!
//! let audio = Audio::new()?;
//!
//! // Generated, or read from a WAV file on disk or built into the binary.
//! let blip = Sound::tone(Waveform::Square, 660.0, 0.08);
//! let hum = Sound::from_wav_file("hum.wav")?;
//!
//! audio.play_with(&blip, Play { gain: 0.3, pan: -0.5, ..Play::default() });
//!
//! let engine = audio.play_with(&hum, Play { gain: 0.2, looping: true, ..Play::default() });
//! engine.set_speed(1.5);   // and it rises in pitch, as an engine does
//! engine.stop();
//! # Ok(())
//! # }
//! ```
//!
//! [`play`]: Audio::play
//!
//! # Instruments
//!
//! A [`Sound`] is a recording. An [`Instrument`] is a program that makes one on
//! demand: a [CLAP](https://cleveraudio.org) plugin, loaded from a file at
//! runtime, played with notes and mixed in alongside everything else.
//!
//! ```no_run
//! # use etib_audio::{Audio, InstrumentConfig};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let audio = Audio::new()?;
//! // Headless: no editor is created, and a plugin that asks to show one is
//! // refused. `Gui::Debug` is how to see one while working on something.
//! let mut synth = unsafe { audio.load_instrument("Synth.clap", &InstrumentConfig::default()) }?;
//!
//! synth.note_on(0, 60, 0.8);
//! synth.poll();   // once a frame, on this thread
//! # Ok(())
//! # }
//! ```
//!
//! This is behind the `clap` feature, which is on by default. Without it no
//! plugin loader is compiled or linked, and everything above is unchanged. See
//! the [`Instrument`] type for what hosting one involves — chiefly that it is
//! tied to the thread that loaded it, and that it has to be polled.
//!
//! # What it is not
//!
//! There is no streaming, so a long piece of music is decoded in full before
//! the first sample is heard; no compressed format but MP3, behind the `mp3`
//! feature; positioning is a level and a pan worked out on the application's
//! thread (below), not a 3D renderer; and no effects, so nothing is filtered or
//! reverberated on its way out. Each of those is a
//! different library, and this is meant to be the one that is not needed when
//! all an application wants is to make a noise.
//!
//! # Sounds in the world
//!
//! A [`SpatialVoice`] is a [`Voice`] with a position. Once a frame it is told
//! where the [`Listener`] is, and turns distance and direction into the gain
//! and pan it sends the mixer — the same two controls any sound has, so the
//! mixing path is untouched. [`SoundManager`] keeps the listener, looping
//! [`SoundGroup`]s placed by a [`SoundStrategy`], and one-shot effects: the
//! engine's spatial sound, without kira.
//!
//! ```no_run
//! # use etib_audio::{Audio, Listener, SingleCubeStrategy, Sound, SoundManager};
//! # use etib_audio::glam::Vec3;
//! # fn main() -> etib_audio::Result<()> {
//! let mut sounds = SoundManager::new(Audio::new_or_silent());
//! let hum = Sound::from_wav_file("hum.wav")?;
//! sounds.add_group(vec![Vec3::new(10.0, 0.0, 0.0)], &hum, Box::new(SingleCubeStrategy));
//!
//! // Every frame, from the camera.
//! # let (eye, target) = (Vec3::ZERO, Vec3::NEG_Z);
//! sounds.update(Listener::looking_at(eye, target));
//! # Ok(())
//! # }
//! ```
//!
//! # Where it sits
//!
//! This crate stands alone. It shares no types with the rest of the workspace,
//! knows nothing about windows, frames or the GPU, and can be used on its own
//! or with a loop of your own. In the `etib` framework it is optional, behind
//! that crate's default `audio` feature, and reachable as `etib::audio`:
//! nothing in `App`, `Frame` or `run` mentions sound, and an application that
//! makes none never builds this at all.
//!
//! An application owns its `Audio` the same way it owns its pipelines — built
//! once, kept alongside its other state, played from wherever it likes:
//!
//! ```no_run
//! # use etib_audio::{Audio, Sound, Waveform};
//! struct Game {
//!     audio: Audio,
//!     step: Sound,
//! }
//!
//! impl Game {
//!     fn new() -> Self {
//!         Self {
//!             // Falls back to silence rather than refusing to start on a
//!             // machine with no sound device.
//!             audio: Audio::new_or_silent(),
//!             step: Sound::tone(Waveform::Noise, 1.0, 0.05),
//!         }
//!     }
//!
//!     fn on_footstep(&self, pan: f32) {
//!         use etib_audio::Play;
//!         self.audio.play_with(&self.step, Play { gain: 0.2, pan, ..Play::default() });
//!     }
//! }
//! ```
//!
//! # Threads
//!
//! The device calls back on a real-time thread of its own, every few
//! milliseconds, and is given a block of audio in return. Everything in this
//! crate is arranged around that call not being late: the mixing path allocates
//! nothing, locks nothing and logs nothing, sounds reach it over a bounded
//! channel, and voices are sent back to the calling thread to be dropped rather
//! than freed where the deadline is.
//!
//! Everything an application calls is safe to call at any point in its own
//! frame; nothing blocks, and nothing waits for the device.
//!
//! # Logging
//!
//! Through the [`log`] facade, with no logger installed: `info` for the device
//! and for plugins being loaded, `debug` for sounds being read and stopped.
//! Nothing in this crate logs from the audio thread; a plugin's own messages
//! are forwarded wherever it sends them from, which is its choice to make.
//!
//! # Platforms
//!
//! The device is opened through [cpal](https://docs.rs/cpal): CoreAudio on
//! macOS, WASAPI on Windows, and ALSA on Linux — where the ALSA development
//! headers (`libasound2-dev`) have to be installed for this crate to build.

mod engine;
mod error;
#[cfg(feature = "clap")]
mod instrument;
mod mixer;
#[cfg(feature = "mp3")]
mod mp3;
mod sound;
mod spatial;
mod wav;

pub use engine::{Audio, Play, Voice};
pub use error::{Error, Result};
pub use sound::{Sound, Waveform};
pub use spatial::{
    EmitterState, Listener, Placement, SingleCubeStrategy, SoundGroup, SoundManager, SoundStrategy,
    Spatial, SpatialVoice, attenuation, spatialize,
};

#[cfg(feature = "clap")]
pub use instrument::{Gui, Instrument, InstrumentConfig, ParamInfo, PluginInfo, plugins_in};

/// How many sounds may be playing at once. Past this, starting another takes
/// the oldest one; [`Audio::stolen`] counts how often that has happened.
pub use mixer::MAX_VOICES;

// Re-exported so an application can name the backend's types — to enumerate
// devices, say — without a second, mismatched copy of it in its tree.
pub use cpal;
pub use glam;
