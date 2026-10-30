//! A small application framework for windows whose contents are drawn by a
//! compute shader.
//!
//! The GPU work itself belongs to [`etib_core`], which this crate re-exports as
//! [`etib::core`](core). What is added here is everything around it:
//!
//!   * [`run`] opens a window, drives the frame loop, and hands each frame to
//!     an [`App`] you implement.
//!   * [`Gpu`] is the window plus the device and swapchain serving it.
//!   * [`Blit`] presents an offscreen texture on the swapchain image as it is.
//!   * [`PostProcess`] presents one through a tone curve and a [`Grade`],
//!     which is what a scene rendered in HDR needs, and optionally through a
//!     [`Lut`] — a look exported from a grading tool as a `.cube` file.
//!
//! Nothing here assumes what is being drawn, or heard. See `examples/triangle`
//! for a complete application built from these pieces:
//!
//! ```text
//! cargo run --bin triangle
//! ```
//!
//! # Extended-range output
//!
//! [`Config::hdr`] asks for an extended-range surface, and [`Gpu::color_space`]
//! reports what was actually configured — a request the platform cannot meet
//! falls back to sRGB, so both have to be handled either way.
//!
//! On an extended-range surface, 1.0 stops meaning "as bright as it gets" and
//! starts meaning SDR white, with the display's headroom above it.
//! [`Surface::output`](etib_core::Surface::output) works out where white sits
//! and how much headroom there is; hand the result to [`PostProcess::draw`],
//! which tone maps against it, and to the interface layer, which composites at
//! SDR white. Those numbers move with the brightness slider, so re-read them
//! rather than caching them at startup.
//!
//! # Interfaces are optional
//!
//! An egui panel is one thing an application might draw, not something the
//! framework takes a position on. It lives in its own crate, `etib-ui`, behind
//! this crate's default `ui` feature, and is reachable as [`etib::ui`](ui):
//!
//! ```toml
//! # panels included (the default)
//! etib = "0.1"
//! # no interface: egui is not compiled, linked, or in the dependency tree
//! etib = { version = "0.1", default-features = false }
//! ```
//!
//! Nothing in [`App`], [`Frame`] or [`run`] changes between the two. An
//! application without the feature draws its frame and handles its own input;
//! one with it keeps an [`EguiLayer`](etib_ui::EguiLayer) alongside its other
//! resources, offers it each window event, and paints it last.
//!
//! # Sound is optional too
//!
//! So is making a noise. [`etib::audio`](audio) is the `etib-audio` crate,
//! behind this crate's default `audio` feature, and it works the same way: an
//! application builds an [`Audio`](etib_audio::Audio) in its setup, keeps it
//! alongside its pipelines, and plays from wherever it likes.
//!
//! ```no_run
//! # #[cfg(feature = "audio")] {
//! use etib::audio::{Audio, Play, Sound, Waveform};
//!
//! // Falls back to silence rather than refusing to start where there is no
//! // sound device; every call still works and nothing is heard.
//! let audio = Audio::new_or_silent();
//! let blip = Sound::tone(Waveform::Square, 660.0, 0.08);
//!
//! audio.play_with(&blip, Play { gain: 0.3, ..Play::default() });
//! # }
//! ```
//!
//! The frame loop knows nothing about it: sound is paced by the device rather
//! than by frames, so starting one is a call like any other and nothing has to
//! happen in [`App::update`] or [`App::render`].
//!
//! An application that wants more than recordings can load a CLAP instrument
//! and play notes on it, behind the further `clap` feature. Those do need one
//! thing per frame — [`Instrument::poll`](etib_audio::Instrument::poll), which
//! is where a plugin's own callbacks and its window are serviced:
//!
//! ```no_run
//! # #[cfg(feature = "clap")] {
//! # use etib::audio::{Audio, Instrument, InstrumentConfig};
//! # fn go(audio: &Audio) -> Result<(), Box<dyn std::error::Error>> {
//! // Headless by default: no plugin window unless one is asked for.
//! let mut piano = unsafe { audio.load_instrument("Piano.clap", &InstrumentConfig::default()) }?;
//! piano.note_on(0, 60, 0.8);
//!
//! // ... and in App::update, once a frame:
//! piano.poll();
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! # Logging
//!
//! Every crate in the workspace logs through the [`log`] facade and installs no
//! logger. With `env_logger`, `RUST_LOG=etib=debug,etib_core=debug` shows
//! resource allocation and surface changes, and `trace` adds per-frame detail.

mod app;
mod blit;
mod gpu;
mod lut;
mod lut_error;
mod post;

pub use app::{App, Config, Frame, FrameStats, MAX_DT, run};
pub use blit::Blit;
pub use gpu::Gpu;
pub use lut::Lut;
pub use lut_error::LutError;
pub use post::{Grade, PostProcess, Tonemap};

/// The GPU layer this framework is built on.
pub use etib_core as core;

/// egui panels, from the `etib-ui` crate. Requires the `ui` feature.
#[cfg(feature = "ui")]
pub use etib_ui as ui;

/// Sound playback and CLAP instruments, from the `etib-audio` crate. Requires
/// the `audio` feature; instruments need `clap` as well.
#[cfg(feature = "audio")]
pub use etib_audio as audio;

// The types that appear in this crate's own signatures, so an application does
// not have to name `etib_core` to use them.
pub use etib_core::{
    BindGroup, BindLayout, Blend, Color, ColorSpace, ComputePipeline, Device, Display, Encoder,
    GpuTimings, Hdr, Load, Output, PassTime, RenderPipeline, RenderTexture, Shader, Stage,
    StorageFormat, StorageTexture, Surface, TextureFormat, TextureView, UniformBuffer, Vsync,
};

// Re-exported so downstream code cannot accidentally link a second, mismatched
// copy of winit.
pub use winit;
