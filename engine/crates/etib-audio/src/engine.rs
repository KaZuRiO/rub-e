//! The output device, and the handles an application holds on what is playing.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, StreamConfig};
use log::{debug, error, info, warn};

use crate::error::{Error, Result};
use crate::mixer::{
    AtomicF32, COMMAND_QUEUE, Command, Controls, MAX_VOICES, Mixer, Playing, Shared,
};
use crate::sound::Sound;

/// How a sound is played. The sound itself says nothing about this, so the same
/// [`Sound`] can be a footstep in the distance and one right behind you.
///
/// ```
/// # use etib_audio::Play;
/// let quiet = Play { gain: 0.25, ..Play::default() };
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Play {
    /// Level, where 1.0 is the sound as it was recorded. Not in decibels:
    /// halving this is a little under -6 dB, and roughly half as loud.
    pub gain: f32,
    /// Where it sits between the ears: -1 hard left, 0 centred, 1 hard right.
    pub pan: f32,
    /// How fast it is read, where 1.0 is the rate it was recorded at. This
    /// moves the pitch with it, as it does on anything that plays a recording
    /// faster; 0.0 holds it still, which is not the same as stopping it.
    pub speed: f32,
    /// Whether it starts again at the end rather than finishing.
    pub looping: bool,
}

impl Default for Play {
    fn default() -> Self {
        Self {
            gain: 1.0,
            pan: 0.0,
            speed: 1.0,
            looping: false,
        }
    }
}

impl Play {
    /// The fastest a sound may be read. Past this the resampler is reading so
    /// far apart that what comes out is aliasing rather than the sound.
    pub const MAX_SPEED: f32 = 16.0;

    /// The one thing a caller can get wrong that the mixer cannot recover from
    /// is a value that is not a number, so everything is brought into range
    /// here rather than refused.
    fn sane(self) -> Self {
        Self {
            gain: sane(self.gain, 1.0).max(0.0),
            pan: sane(self.pan, 0.0).clamp(-1.0, 1.0),
            speed: sane(self.speed, 1.0).clamp(0.0, Self::MAX_SPEED),
            looping: self.looping,
        }
    }
}

fn sane(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

/// A sound that is playing, and the controls that go with it.
///
/// Dropping this does **not** stop the sound: most effects are started and
/// forgotten, and having them cut off when the handle goes out of scope would
/// make the common case the surprising one. [`stop`](Self::stop) is how a sound
/// ends early.
///
/// Every control may be changed while the sound plays, and takes effect within
/// a block — a millisecond or two. Changes are smoothed across that block, so
/// turning something down mid-note does not click.
#[derive(Clone, Debug)]
pub struct Voice {
    controls: Arc<Controls>,
}

impl Voice {
    /// Sets the level, 1.0 being the sound as recorded.
    pub fn set_gain(&self, gain: f32) {
        self.controls.gain.store(sane(gain, 1.0).max(0.0));
    }

    /// Sets the position between the ears, -1 to 1.
    pub fn set_pan(&self, pan: f32) {
        self.controls.pan.store(sane(pan, 0.0).clamp(-1.0, 1.0));
    }

    /// Sets the playback rate, which moves the pitch with it.
    pub fn set_speed(&self, speed: f32) {
        self.controls
            .speed
            .store(sane(speed, 1.0).clamp(0.0, Play::MAX_SPEED));
    }

    /// Turns looping on or off. Turning it off part-way through lets the sound
    /// finish the pass it is on rather than cutting it.
    pub fn set_looping(&self, looping: bool) {
        self.controls.looping.store(looping, Ordering::Relaxed);
    }

    /// Ends the sound. It is faded out over the next block rather than cut,
    /// because a waveform that stops mid-cycle is a click.
    pub fn stop(&self) {
        self.controls.stopping.store(true, Ordering::Relaxed);
    }

    /// Whether the sound is still sounding. `false` once it has run out, been
    /// stopped, or been dropped to make room for a newer one.
    pub fn is_playing(&self) -> bool {
        self.controls.playing.load(Ordering::Acquire)
    }

    /// A handle to a sound that is already over, for the paths that have to
    /// return one without anything having been started.
    fn finished() -> Self {
        Self {
            controls: Arc::new(Controls {
                gain: AtomicF32::new(0.0),
                pan: AtomicF32::new(0.0),
                speed: AtomicF32::new(1.0),
                looping: AtomicBool::new(false),
                stopping: AtomicBool::new(true),
                playing: AtomicBool::new(false),
            }),
        }
    }
}

/// The sound output: an open device, and a mixer feeding it.
///
/// One of these is enough for an application. Building it opens the system's
/// default output device and starts a stream on it, which then asks for a block
/// of audio every few milliseconds for as long as the `Audio` is alive.
/// Dropping it closes the device and everything playing stops.
///
/// ```no_run
/// # use etib_audio::{Audio, Play, Sound, Waveform};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let audio = Audio::new()?;
/// let blip = Sound::tone(Waveform::Sine, 880.0, 0.1);
///
/// audio.play_with(&blip, Play { gain: 0.25, ..Play::default() });
/// # Ok(())
/// # }
/// ```
///
/// # Failing quietly
///
/// A machine with no sound card, a device already held exclusively by something
/// else, a container with no audio at all: none of these are a reason for an
/// application to refuse to start. [`silent`](Self::silent) is an `Audio` that
/// accepts everything and plays nothing, so that the fallback is one line and
/// no call site has to be written twice.
pub struct Audio {
    shared: Arc<Shared>,
    commands: SyncSender<Command>,
    /// Voices the mixer has finished with, dropped on this thread rather than
    /// on the audio thread. Drained whenever a sound is played.
    retired: Receiver<Playing>,
    /// The stream stops when this is dropped, so it is held for its lifetime
    /// rather than for anything it is asked. `None` in a silent `Audio`.
    _stream: Option<cpal::Stream>,
    sample_rate: u32,
    channels: u16,
    /// The largest block the device says it will ask for.
    block_frames: usize,
    device: String,
}

impl Audio {
    /// Opens the system's default output device and starts mixing.
    ///
    /// The device's own preferred format is taken as it comes — sample rate,
    /// channel count and sample type — and sounds are converted to it as they
    /// are mixed, which is the arrangement least likely to be refused or
    /// resampled again behind our back by the system's own mixer.
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| Error::new("the system reports no default audio output device"))?;
        let name = device.to_string();

        let supported = device.default_output_config().map_err(|err| {
            Error::with_source(format!("'{name}' has no default output configuration"), err)
        })?;
        let sample_format = supported.sample_format();
        let sample_rate = supported.sample_rate();
        let channels = supported.channels();

        // Sized from the largest block the device says it may ask for, so that
        // the mixer never has to grow its buffer on the audio thread. A backend
        // that reports nothing gets one big enough for any sane device.
        let block_frames = match supported.buffer_size() {
            cpal::SupportedBufferSize::Range { max, .. } => (*max as usize).clamp(1024, 16_384),
            cpal::SupportedBufferSize::Unknown => 4096,
        };
        let config = StreamConfig::from(supported);

        info!("opening '{name}' at {sample_rate} Hz, {channels} channels, {sample_format} samples");

        let shared = Arc::new(Shared::new());
        let (commands, command_rx) = sync_channel(COMMAND_QUEUE);
        let (retired_tx, retired) = sync_channel(MAX_VOICES);

        let mixer = Mixer::new(
            command_rx,
            retired_tx,
            shared.clone(),
            sample_rate,
            block_frames,
        );

        let stream = build_stream(&device, &config, sample_format, mixer).map_err(|err| {
            Error::with_source(format!("could not open a stream on '{name}'"), err)
        })?;
        stream.play().map_err(|err| {
            Error::with_source(format!("could not start the stream on '{name}'"), err)
        })?;

        Ok(Self {
            shared,
            commands,
            retired,
            _stream: Some(stream),
            sample_rate,
            channels,
            block_frames,
            device: name,
        })
    }

    /// An `Audio` with no device behind it. Every call works and nothing is
    /// heard, which is what an application wants when the machine has no sound
    /// rather than a second code path without any calls in it.
    pub fn silent() -> Self {
        // No mixer is listening, so a command cannot be sent, and `play` treats
        // that the same as any other full queue: the sound is already over.
        let (commands, _) = sync_channel(COMMAND_QUEUE);
        let (_, retired) = sync_channel(MAX_VOICES);

        Self {
            shared: Arc::new(Shared::new()),
            commands,
            retired,
            _stream: None,
            sample_rate: 0,
            channels: 0,
            block_frames: 0,
            device: "none".to_owned(),
        }
    }

    /// Opens the default device, falling back to [`silent`](Self::silent) and a
    /// logged warning if it cannot be opened.
    ///
    /// This is the constructor an application that would rather run without
    /// sound than not run wants.
    pub fn new_or_silent() -> Self {
        match Self::new() {
            Ok(audio) => audio,
            Err(err) => {
                warn!("no audio output ({err}); continuing without sound");
                Self::silent()
            }
        }
    }

    /// Starts `sound` as it was recorded, and returns a handle to it.
    ///
    /// The handle can be ignored: a sound plays to its end whether or not
    /// anyone is holding it.
    pub fn play(&self, sound: &Sound) -> Voice {
        self.play_with(sound, Play::default())
    }

    /// Starts `sound` with the level, position, rate and looping in `play`.
    pub fn play_with(&self, sound: &Sound, play: Play) -> Voice {
        // Whatever the mixer has finished with is freed here, on this thread,
        // rather than on the one with a deadline.
        while self.retired.try_recv().is_ok() {}

        let play = play.sane();
        let controls = Arc::new(Controls::new(play.gain, play.pan));
        controls.speed.store(play.speed);
        controls.looping.store(play.looping, Ordering::Relaxed);

        let voice = Voice {
            controls: controls.clone(),
        };
        let playing = Playing {
            sound: sound.clone(),
            controls,
            position: 0.0,
            ramp: None,
        };

        match self.commands.try_send(Command::Play(playing)) {
            Ok(()) => voice,
            Err(TrySendError::Full(_)) => {
                warn!("the mixer has not kept up; this sound will not be played");
                Voice::finished()
            }
            // A silent `Audio`, or a stream that has gone away under us.
            Err(TrySendError::Disconnected(_)) => Voice::finished(),
        }
    }

    /// Stops everything that is playing, including sounds whose handles have
    /// been dropped. Each one is faded out rather than cut.
    pub fn stop_all(&self) {
        debug!("stopping all {} voices", self.playing());
        self.shared.stop_generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Sets the level everything is mixed at, where 1.0 is unchanged.
    pub fn set_gain(&self, gain: f32) {
        self.shared.gain.store(sane(gain, 1.0).max(0.0));
    }

    /// The level everything is mixed at.
    pub fn gain(&self) -> f32 {
        self.shared.gain.load()
    }

    /// How many sounds are playing right now.
    pub fn playing(&self) -> usize {
        self.shared.playing.load(Ordering::Relaxed)
    }

    /// How many sounds have been cut short to make room for newer ones since
    /// this `Audio` was opened.
    ///
    /// Anything other than zero means more than [`MAX_VOICES`] sounds were
    /// wanted at once, and the oldest of them were dropped. It is counted
    /// rather than logged because it is noticed on a thread that cannot log.
    ///
    /// [`MAX_VOICES`]: crate::MAX_VOICES
    pub fn stolen(&self) -> u64 {
        self.shared.stolen.load(Ordering::Relaxed)
    }

    /// The rate the device is running at, or 0 when silent. Sounds are
    /// resampled to it, so nothing has to match it.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// How many channels the device takes, or 0 when silent. The mixer works in
    /// stereo and maps onto this.
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// The largest block of frames the device says it will ask for, or 0 when
    /// silent. This is what a plugin is activated for.
    pub fn block_frames(&self) -> usize {
        self.block_frames
    }

    /// A way to reach the mixer, for the parts of this crate that add something
    /// to it.
    #[cfg(feature = "clap")]
    pub(crate) fn commands(&self) -> SyncSender<Command> {
        self.commands.clone()
    }

    /// The device's name, for a log line or an overlay.
    pub fn device(&self) -> &str {
        &self.device
    }

    /// Whether this is a silent `Audio` with no device behind it.
    pub fn is_silent(&self) -> bool {
        self._stream.is_none()
    }
}

impl Drop for Audio {
    fn drop(&mut self) {
        if !self.is_silent() {
            info!("closing the audio device '{}'", self.device);
        }
    }
}

/// Opens the stream in whichever sample type the device asked for.
///
/// Every arm is the same function; only the type the samples are written as
/// differs, and the conversion to it is the backend's, so a device that wants
/// 16-bit integers is served without anything above this line knowing.
fn build_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    format: SampleFormat,
    mixer: Mixer,
) -> std::result::Result<cpal::Stream, cpal::Error> {
    macro_rules! stream {
        ($sample:ty) => {
            open::<$sample>(device, config, mixer)
        };
    }

    match format {
        SampleFormat::F32 => stream!(f32),
        SampleFormat::F64 => stream!(f64),
        SampleFormat::I8 => stream!(i8),
        SampleFormat::I16 => stream!(i16),
        SampleFormat::I32 => stream!(i32),
        SampleFormat::I64 => stream!(i64),
        SampleFormat::U8 => stream!(u8),
        SampleFormat::U16 => stream!(u16),
        SampleFormat::U32 => stream!(u32),
        SampleFormat::U64 => stream!(u64),
        other => Err(cpal::Error::with_message(
            cpal::ErrorKind::UnsupportedConfig,
            format!("the device wants {other} samples, which this cannot write"),
        )),
    }
}

fn open<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    mut mixer: Mixer,
) -> std::result::Result<cpal::Stream, cpal::Error>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    device.build_output_stream(
        *config,
        move |out: &mut [T], _| write(out, channels, &mut mixer),
        on_stream_error,
        None,
    )
}

/// The audio callback: mix a block, then lay it out the way the device wants it.
///
/// The mixer works in stereo whatever the device has, because that is what
/// panning means and what nearly every device is. A mono device is given the
/// sum of both sides, and the channels of a wider one past the first two are
/// left silent — placing a sound in a surround field is a different job, with a
/// listener orientation in it, and guessing at one here would put sounds in the
/// wrong place rather than in no place.
fn write<T: SizedSample + FromSample<f32>>(out: &mut [T], channels: usize, mixer: &mut Mixer) {
    let frames = out.len() / channels;
    let mix = mixer.render(frames);
    let silence = T::from_sample(0.0);

    for (frame, stereo) in out.chunks_mut(channels).zip(mix.as_chunks::<2>().0) {
        match frame {
            [] => {}
            [mono] => *mono = T::from_sample((stereo[0] + stereo[1]) * 0.5),
            [left, right, rest @ ..] => {
                *left = T::from_sample(stereo[0]);
                *right = T::from_sample(stereo[1]);
                rest.fill(silence);
            }
        }
    }
}

/// Reported by the backend from its own thread, so there is nothing to do here
/// but say so. A stream that has failed for good stops calling back, and the
/// application keeps running in silence.
fn on_stream_error(err: cpal::Error) {
    error!("the audio stream reported an error: {err}");
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::sync_channel;

    use super::*;
    use crate::mixer::COMMAND_QUEUE;
    use crate::sound::Waveform;

    /// A mixer with one stereo sound playing: left at full scale, right silent,
    /// which shows where each side of the mix ends up. The ends of the channels
    /// are kept because a mixer whose sender has gone away has no voices.
    struct Harness {
        mixer: Mixer,
        _commands: SyncSender<Command>,
        _retired: Receiver<Playing>,
    }

    fn playing_hard_left() -> Harness {
        let (commands, command_rx) = sync_channel(COMMAND_QUEUE);
        let (retired_tx, retired) = sync_channel(MAX_VOICES);

        let sound = Sound::from_samples(vec![1.0, 0.0, 1.0, 0.0], 2, 48_000).unwrap();
        let controls = Arc::new(Controls {
            gain: AtomicF32::new(1.0),
            pan: AtomicF32::new(0.0),
            speed: AtomicF32::new(1.0),
            looping: AtomicBool::new(true),
            stopping: AtomicBool::new(false),
            playing: AtomicBool::new(true),
        });
        commands
            .try_send(Command::Play(Playing {
                sound,
                controls,
                position: 0.0,
                ramp: None,
            }))
            .unwrap();

        Harness {
            mixer: Mixer::new(command_rx, retired_tx, Arc::new(Shared::new()), 48_000, 512),
            _commands: commands,
            _retired: retired,
        }
    }

    /// Stereo goes out as it was mixed, a mono device is given the sum of both
    /// sides, and the channels of a wider one past the first two are silent.
    #[test]
    fn lays_the_mix_out_the_way_the_device_wants_it() {
        let mut stereo = [0.0f32; 4];
        write(&mut stereo, 2, &mut playing_hard_left().mixer);
        assert_eq!(stereo, [1.0, 0.0, 1.0, 0.0]);

        let mut mono = [0.0f32; 2];
        write(&mut mono, 1, &mut playing_hard_left().mixer);
        assert_eq!(mono, [0.5, 0.5]);

        let mut surround = [0.0f32; 8];
        write(&mut surround, 4, &mut playing_hard_left().mixer);
        assert_eq!(surround, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    }

    /// The device's own sample type is written, not f32, whatever it asks for.
    #[test]
    fn converts_to_the_device_sample_type() {
        let mut out = [0i16; 4];
        write(&mut out, 2, &mut playing_hard_left().mixer);
        assert_eq!(out, [i16::MAX, 0, i16::MAX, 0]);
    }

    /// The fallback has to behave like a working one at every call site.
    #[test]
    fn a_silent_audio_accepts_everything() {
        let audio = Audio::silent();
        let sound = Sound::tone(Waveform::Sine, 440.0, 0.05);

        assert!(audio.is_silent());
        assert_eq!(audio.sample_rate(), 0);

        let voice = audio.play(&sound);
        assert!(
            !voice.is_playing(),
            "nothing can be playing without a device"
        );

        // None of these may panic, and none of them do anything.
        voice.set_gain(0.5);
        voice.set_pan(-1.0);
        voice.set_speed(2.0);
        voice.set_looping(true);
        voice.stop();
        audio.set_gain(0.5);
        audio.stop_all();
        assert_eq!(audio.playing(), 0);
        assert_eq!(audio.stolen(), 0);
    }

    #[test]
    fn play_brings_its_parameters_into_range() {
        let play = Play {
            gain: -3.0,
            pan: 9.0,
            speed: f32::NAN,
            looping: true,
        }
        .sane();

        assert_eq!(play.gain, 0.0);
        assert_eq!(play.pan, 1.0);
        assert_eq!(play.speed, 1.0);
        assert!(play.looping);

        let play = Play {
            speed: 1000.0,
            ..Play::default()
        }
        .sane();
        assert_eq!(play.speed, Play::MAX_SPEED);
    }

    #[test]
    fn a_voice_clamps_what_it_is_set_to() {
        let voice = Voice::finished();

        voice.set_gain(f32::INFINITY);
        assert_eq!(voice.controls.gain.load(), 1.0);
        voice.set_pan(-7.0);
        assert_eq!(voice.controls.pan.load(), -1.0);
        voice.set_speed(-1.0);
        assert_eq!(voice.controls.speed.load(), 0.0);
    }
}
