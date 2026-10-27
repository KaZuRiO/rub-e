//! A sound: samples held in memory, ready to be played any number of times.

use std::fmt;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use log::debug;

use crate::error::{Error, Result};
use crate::wav;

/// Sound effects are short and are played over and over, so one is decoded
/// once, kept in memory as `f32`, and mixed straight from there. A long piece
/// of music is the case this does *not* suit — it would be decoded in full
/// before the first sample is heard.
///
/// A `Sound` is immutable and cheap to clone: the clone shares the samples,
/// which is also how a sound reaches the mixing thread. How it is played —
/// how loud, how fast, where, whether it repeats — is not part of it but of
/// the [`Play`](crate::Play) it is played with, so one sound can be several
/// different noises.
///
/// ```
/// # use etib_audio::{Sound, Waveform};
/// let blip = Sound::tone(Waveform::Sine, 880.0, 0.1);
/// assert_eq!(blip.channels(), 1);
/// ```
#[derive(Clone)]
pub struct Sound {
    data: Arc<Data>,
}

struct Data {
    /// Interleaved, one entry per channel per frame. Never empty.
    samples: Vec<f32>,
    channels: u16,
    sample_rate: u32,
}

/// The shape of a generated [tone](Sound::tone).
///
/// Only the sine is a single frequency. The other three are generated as the
/// plain geometric shape rather than as a sum of harmonics, so they alias above
/// a few kilohertz — which is the sound eight-bit hardware made, and what makes
/// them useful for effects, but it is not what a synthesiser would produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Waveform {
    /// One frequency and nothing else. A clean beep.
    Sine,
    /// Soft and hollow: the odd harmonics, falling away quickly.
    Triangle,
    /// Bright and buzzy: every harmonic. A klaxon, an engine.
    Saw,
    /// Hard and hollow: the odd harmonics, falling away slowly. A retro blip.
    Square,
    /// No pitch at all; `hz` is ignored. Static, wind, an explosion's tail.
    Noise,
}

impl Sound {
    /// The rate a generated tone is written at. High enough that the shapes
    /// with corners in them are not obviously stepped.
    pub const TONE_RATE: u32 = 48_000;

    /// Longest tone this will generate, so a typo cannot ask for an hour of
    /// audio in memory.
    const MAX_TONE_SECONDS: f32 = 60.0;

    /// The fade applied to each end of a generated tone. A waveform that starts
    /// or stops at a non-zero sample is a step, and a step is a click; five
    /// milliseconds is long enough to remove it and short enough not to soften
    /// the attack.
    const TONE_FADE: f32 = 0.005;

    /// Builds a sound from interleaved samples in -1..1.
    ///
    /// `channels` is 1 or 2, and the samples are ordered left, right, left,
    /// right for stereo. Values outside -1..1 are allowed here — they are
    /// clamped at the output, once everything playing has been summed — but a
    /// sound that is already over full scale has nothing left to mix with.
    pub fn from_samples(samples: Vec<f32>, channels: u16, sample_rate: u32) -> Result<Self> {
        if channels == 0 || channels > 2 {
            return Err(Error::new(format!(
                "a sound is mono or stereo, not {channels} channels"
            )));
        }
        if sample_rate == 0 {
            return Err(Error::new("a sound needs a sample rate"));
        }
        if samples.is_empty() {
            return Err(Error::new("a sound needs at least one frame"));
        }
        if !samples.len().is_multiple_of(channels as usize) {
            return Err(Error::new(format!(
                "{} samples is not a whole number of {channels}-channel frames",
                samples.len()
            )));
        }

        Ok(Self {
            data: Arc::new(Data {
                samples,
                channels,
                sample_rate,
            }),
        })
    }

    /// Reads a WAV file from memory: uncompressed PCM at 8, 16, 24 or 32 bits,
    /// or 32 or 64-bit float, mono or stereo.
    ///
    /// This is the one to use with [`include_bytes!`], which builds the sound
    /// into the binary and leaves nothing to find at runtime.
    pub fn from_wav(bytes: &[u8]) -> Result<Self> {
        let wav = wav::parse(bytes)?;
        debug!(
            "read a {}-channel WAV at {} Hz, {} frames",
            wav.channels,
            wav.sample_rate,
            wav.samples.len() / wav.channels as usize
        );
        Self::from_samples(wav.samples, wav.channels, wav.sample_rate)
    }

    /// Reads a WAV file from disk. See [`from_wav`](Self::from_wav) for what
    /// the file may hold.
    pub fn from_wav_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|err| Error::with_source(format!("could not read {}", path.display()), err))?;

        Self::from_wav(&bytes).map_err(|err| {
            // The parser's message says what is wrong with the file; this adds
            // which file it was, which the caller otherwise has to supply.
            Error::with_source(
                format!("{} is not a sound this can play", path.display()),
                err,
            )
        })
    }

    /// Generates a tone: `seconds` of `waveform` at `hz`, mono, at full scale.
    ///
    /// Enough for the noises an interface makes — a confirmation, a warning, a
    /// step in a menu — without a file to ship alongside the binary. The
    /// arguments are clamped rather than refused, since there is no failure
    /// here that is not a typo.
    ///
    /// Full scale is loud, and the shapes other than [`Sine`](Waveform::Sine)
    /// are louder still for the same peak, because so much more of the time is
    /// spent near it. [`Play::gain`](crate::Play::gain) is where that is dealt
    /// with; a quarter is a reasonable place to start.
    pub fn tone(waveform: Waveform, hz: f32, seconds: f32) -> Self {
        let hz = if hz.is_finite() {
            hz.clamp(1.0, 20_000.0)
        } else {
            440.0
        };
        let seconds = if seconds.is_finite() {
            seconds.clamp(0.0, Self::MAX_TONE_SECONDS)
        } else {
            1.0
        };

        let rate = Self::TONE_RATE as f32;
        let frames = ((seconds * rate) as usize).max(1);
        let fade = (Self::TONE_FADE * rate) as usize;

        // A phase step per frame, kept in the cycle so that precision does not
        // drift away over a long tone.
        let step = hz / rate;
        let mut phase = 0.0f32;
        // Whatever this is seeded with, the noise sounds the same; a fixed seed
        // makes the same call produce the same sound every run.
        let mut noise = 0x2545_F491u32;

        let mut samples = Vec::with_capacity(frames);
        for frame in 0..frames {
            let value = match waveform {
                Waveform::Sine => (phase * std::f32::consts::TAU).sin(),
                // -1 at the start of the cycle, 1 halfway, -1 at the end.
                Waveform::Triangle => 4.0 * (phase - (phase + 0.5).floor()).abs() - 1.0,
                Waveform::Saw => 2.0 * phase - 1.0,
                Waveform::Square => {
                    if phase < 0.5 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                Waveform::Noise => {
                    // xorshift32: not a source of randomness anyone should
                    // depend on, but as a source of hiss it is indistinguishable
                    // from one, and it is three instructions.
                    noise ^= noise << 13;
                    noise ^= noise >> 17;
                    noise ^= noise << 5;
                    (noise as f32 / u32::MAX as f32) * 2.0 - 1.0
                }
            };

            let envelope = match fade {
                0 => 1.0,
                fade => ((frame as f32) / fade as f32)
                    .min((frames - 1 - frame) as f32 / fade as f32)
                    .clamp(0.0, 1.0),
            };

            samples.push(value * envelope);
            phase = (phase + step).fract();
        }

        Self {
            data: Arc::new(Data {
                samples,
                channels: 1,
                sample_rate: Self::TONE_RATE,
            }),
        }
    }

    /// 1 for mono, 2 for stereo.
    pub fn channels(&self) -> u16 {
        self.data.channels
    }

    /// The rate the samples were written at. The mixer resamples to whatever
    /// the output device is running at, so this does not have to match it.
    pub fn sample_rate(&self) -> u32 {
        self.data.sample_rate
    }

    /// How many samples there are per channel. Always at least one.
    pub fn frames(&self) -> usize {
        self.data.samples.len() / self.data.channels as usize
    }

    /// How long the sound is at [`speed`](crate::Play::speed) 1.0.
    pub fn duration(&self) -> Duration {
        Duration::from_secs_f64(self.frames() as f64 / self.data.sample_rate as f64)
    }

    /// The interleaved samples, as they will be mixed.
    pub fn samples(&self) -> &[f32] {
        &self.data.samples
    }

    /// The pair for frame `index`, which the mixer reads one frame at a time.
    /// A mono sound gives the same value twice, which is what putting it in
    /// both ears means.
    pub(crate) fn frame(&self, index: usize) -> (f32, f32) {
        let samples = &self.data.samples;
        match self.data.channels {
            1 => (samples[index], samples[index]),
            _ => (samples[index * 2], samples[index * 2 + 1]),
        }
    }
}

impl fmt::Debug for Sound {
    /// Spelling out several hundred thousand samples helps nobody; the shape of
    /// the sound is what a log line or a panic message wants.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sound")
            .field("frames", &self.frames())
            .field("channels", &self.data.channels)
            .field("sample_rate", &self.data.sample_rate)
            .field("duration", &self.duration())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tone_is_the_length_it_was_asked_for() {
        let tone = Sound::tone(Waveform::Sine, 440.0, 0.25);
        assert_eq!(tone.channels(), 1);
        assert_eq!(tone.sample_rate(), Sound::TONE_RATE);
        assert_eq!(tone.frames(), Sound::TONE_RATE as usize / 4);
        assert_eq!(tone.duration(), Duration::from_millis(250));
    }

    /// Both ends have to reach zero, or the tone clicks on and off.
    #[test]
    fn a_tone_fades_in_and_out() {
        let tone = Sound::tone(Waveform::Square, 200.0, 0.2);
        let samples = tone.samples();

        assert_eq!(samples[0], 0.0);
        assert_eq!(samples[samples.len() - 1], 0.0);
        // Past the fade, the square is at full scale again.
        let middle = samples[samples.len() / 2].abs();
        assert!((middle - 1.0).abs() < 1e-6, "{middle}");
    }

    #[test]
    fn every_waveform_stays_inside_full_scale() {
        for waveform in [
            Waveform::Sine,
            Waveform::Triangle,
            Waveform::Saw,
            Waveform::Square,
            Waveform::Noise,
        ] {
            let tone = Sound::tone(waveform, 440.0, 0.05);
            let peak = tone
                .samples()
                .iter()
                .fold(0.0f32, |peak, s| peak.max(s.abs()));
            assert!(peak <= 1.0, "{waveform:?} peaks at {peak}");
            assert!(peak > 0.5, "{waveform:?} is far too quiet at {peak}");
        }
    }

    /// Nothing downstream may be handed an empty or ragged sound.
    #[test]
    fn a_tone_is_never_empty() {
        assert_eq!(Sound::tone(Waveform::Sine, 440.0, 0.0).frames(), 1);
        assert_eq!(
            Sound::tone(Waveform::Sine, f32::NAN, f32::NAN).frames(),
            Sound::TONE_RATE as usize
        );
    }

    #[test]
    fn from_samples_checks_its_shape() {
        let err = |samples: Vec<f32>, channels, rate| {
            Sound::from_samples(samples, channels, rate)
                .unwrap_err()
                .to_string()
        };

        assert!(Sound::from_samples(vec![0.0, 1.0], 2, 48_000).is_ok());
        assert!(err(vec![0.0], 0, 48_000).contains("not 0 channels"));
        assert!(err(vec![0.0], 3, 48_000).contains("not 3 channels"));
        assert!(err(vec![0.0], 1, 0).contains("needs a sample rate"));
        assert!(err(vec![], 1, 48_000).contains("at least one frame"));
        assert!(err(vec![0.0; 3], 2, 48_000).contains("whole number of 2-channel frames"));
    }

    #[test]
    fn a_frame_is_a_pair_either_way() {
        let mono = Sound::from_samples(vec![0.25, 0.5], 1, 48_000).unwrap();
        assert_eq!(mono.frames(), 2);
        assert_eq!(mono.frame(1), (0.5, 0.5));

        let stereo = Sound::from_samples(vec![0.25, 0.5, 0.75, 1.0], 2, 48_000).unwrap();
        assert_eq!(stereo.frames(), 2);
        assert_eq!(stereo.frame(1), (0.75, 1.0));
    }
}
