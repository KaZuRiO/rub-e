//! What runs on the audio thread: summing the sounds that are playing into one
//! stereo buffer, on demand, in time.
//!
//! Nothing in here knows about the audio backend, which is what makes it
//! testable — the mixer is driven by asking it for a block of frames, and a
//! test can ask for one as readily as a device callback can.
//!
//! # The rule this module exists to keep
//!
//! The callback that asks for a block runs on a real-time thread with a
//! deadline of a few milliseconds. Missing it is an audible glitch, so the
//! path from [`Mixer::render`] down **does not allocate, does not lock, does
//! not log, and does not free anything.** That is why:
//!
//!   * the voices and the mix buffer are allocated once, ahead of time;
//!   * commands arrive over a bounded channel, whose storage is allocated when
//!     it is created rather than per message;
//!   * a finished voice is *sent back* to the owning thread rather than dropped
//!     here, since dropping the last handle on a sound would free its samples;
//!   * controls are atomics rather than a mutex.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};

use crate::sound::Sound;

/// How many sounds may be playing at once.
///
/// There is no point mixing more than this: past a couple of dozen, another
/// voice adds to the level rather than to what can be picked out. When they are
/// all in use the oldest is taken, which is what a game engine does, because
/// the sound that started last is the one the listener is waiting for.
pub const MAX_VOICES: usize = 32;

/// How many instruments may be loaded at once.
///
/// The mixer's list of them is allocated at this size up front, so that taking
/// one in never grows it on the audio thread; a load past it is refused.
#[cfg(feature = "clap")]
pub const MAX_INSTRUMENTS: usize = 4;

/// How many `play` calls may be waiting to be picked up. One callback's worth
/// of starts is a handful; this is generous.
pub(crate) const COMMAND_QUEUE: usize = 64;

/// A change of state one thread asks the other for.
pub(crate) enum Command {
    Play(Playing),
    /// A loaded instrument, activated on the other thread and moved here.
    #[cfg(feature = "clap")]
    AddInstrument(Box<crate::instrument::Processor>),
    /// Hand that instrument back, so that it can be destroyed where it was
    /// made. Nothing here drops a plugin.
    #[cfg(feature = "clap")]
    RemoveInstrument(u64),
}

/// Everything the mixer and the handles in front of it share.
pub(crate) struct Shared {
    /// Master level, applied once to the summed mix.
    pub gain: AtomicF32,
    /// How many voices are sounding, published by the mixer each block.
    pub playing: AtomicUsize,
    /// Bumped by `stop_all`; the mixer stops everything when it changes.
    pub stop_generation: AtomicU64,
    /// How many voices have been cut short to make room for a new one. There is
    /// no way to report it from the audio thread, so it is counted here.
    pub stolen: AtomicU64,
    /// How many instruments are loaded, or being loaded. Counted on the
    /// owning thread, which is where loads are refused.
    #[cfg(feature = "clap")]
    pub instruments: AtomicUsize,
}

impl Shared {
    pub fn new() -> Self {
        Self {
            gain: AtomicF32::new(1.0),
            playing: AtomicUsize::new(0),
            stop_generation: AtomicU64::new(0),
            stolen: AtomicU64::new(0),
            #[cfg(feature = "clap")]
            instruments: AtomicUsize::new(0),
        }
    }

    /// Takes one of the [`MAX_INSTRUMENTS`] places, or says why there is none.
    #[cfg(feature = "clap")]
    pub fn reserve_instrument(&self) -> crate::error::Result<()> {
        self.instruments
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |loaded| {
                (loaded < MAX_INSTRUMENTS).then_some(loaded + 1)
            })
            .map(drop)
            .map_err(|_| {
                crate::error::Error::new(format!(
                    "at most {MAX_INSTRUMENTS} instruments can be loaded at once"
                ))
            })
    }

    /// Gives back a place taken by [`reserve_instrument`](Self::reserve_instrument).
    #[cfg(feature = "clap")]
    pub fn release_instrument(&self) {
        self.instruments.fetch_sub(1, Ordering::AcqRel);
    }
}

/// The controls of one playing sound, shared with its [`Voice`](crate::Voice)
/// handle. Read once per block by the mixer, written at any time by the owner.
#[derive(Debug)]
pub(crate) struct Controls {
    pub gain: AtomicF32,
    pub pan: AtomicF32,
    pub speed: AtomicF32,
    pub looping: AtomicBool,
    /// Asked to stop. The mixer fades the voice out over one block and retires
    /// it; cutting a waveform off mid-cycle is a click.
    pub stopping: AtomicBool,
    /// Cleared by the mixer once the voice is done, or by `play` itself if the
    /// sound never reached the mixer.
    pub playing: AtomicBool,
}

impl Controls {
    /// A voice's controls at the level and position it was started with.
    pub fn new(gain: f32, pan: f32) -> Self {
        Self {
            gain: AtomicF32::new(gain),
            pan: AtomicF32::new(pan),
            speed: AtomicF32::new(1.0),
            looping: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
            playing: AtomicBool::new(true),
        }
    }
}

/// One sound, playing.
///
/// Created on the owning thread, mixed on the audio thread, and sent back to
/// the owning thread to be dropped.
pub(crate) struct Playing {
    pub sound: Sound,
    pub controls: Arc<Controls>,
    /// Where playback has reached, in source frames. Fractional because the
    /// source rarely runs at the device's rate.
    pub position: f64,
    /// Where the last block's gain ramp ended, per side. `None` before the
    /// first block, which starts at its target instead of ramping up to it —
    /// a percussive sound must not have its attack softened.
    pub ramp: Option<(f32, f32)>,
}

impl Playing {
    /// Mixes one block into `mix`, which is interleaved stereo. Returns `false`
    /// when the voice is finished and should be retired.
    ///
    /// The controls are read once, here, and interpolated across the block
    /// rather than re-read per frame: a level that jumps between blocks is a
    /// step in the waveform, which is heard as a click, and an atomic load per
    /// sample would be paid for nothing.
    fn render(&mut self, mix: &mut [f32], device_rate: u32) -> bool {
        let frames = mix.len() / 2;
        if frames == 0 {
            return true;
        }

        let stopping = self.controls.stopping.load(Ordering::Relaxed);
        let looping = self.controls.looping.load(Ordering::Relaxed);
        let gain = if stopping {
            0.0
        } else {
            self.controls.gain.load()
        };
        let (pan_l, pan_r) = pan_gains(self.controls.pan.load());
        let (to_l, to_r) = (gain * pan_l, gain * pan_r);
        let (from_l, from_r) = self.ramp.unwrap_or((to_l, to_r));
        self.ramp = Some((to_l, to_r));

        // One source frame per device frame is the rate ratio; speed multiplies
        // it, which moves the pitch with it, as it does on anything that plays
        // a recording faster.
        //
        // Defensive: a handle clamps what it is given, but a speed that is not a
        // number here would put the read position somewhere it cannot come back
        // from. `max` returns the other side of a NaN, so this catches both.
        let speed = self.controls.speed.load().max(0.0);
        let step = (self.sound.sample_rate() as f64 / device_rate as f64) * speed as f64;
        let source_frames = self.sound.frames();

        let mut finished = false;
        for frame in 0..frames {
            if self.position >= source_frames as f64 {
                if looping {
                    // Modulo rather than zero, so a step that overshoots the end
                    // does not shorten every loop by a fraction of a frame.
                    self.position %= source_frames as f64;
                } else {
                    finished = true;
                    break;
                }
            }

            // The ramp reaches its target on the last frame of the block, so the
            // next block starts exactly where this one left off.
            let t = (frame + 1) as f32 / frames as f32;
            let (left, right) = self.sample(looping);

            mix[frame * 2] += left * (from_l + (to_l - from_l) * t);
            mix[frame * 2 + 1] += right * (from_r + (to_r - from_r) * t);
            self.position += step;
        }

        // A stop has had its block to fade out, so it ends here either way.
        !(finished || stopping)
    }

    /// The pair at the current position, interpolated between the frames either
    /// side of it.
    ///
    /// Linear interpolation is what a sound effect wants: it is a gentle
    /// low-pass, inaudible at the small rate changes resampling actually
    /// involves, and it costs two loads and a multiply.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the position is within the sound, and the fraction is meant to lose precision"
    )]
    fn sample(&self, looping: bool) -> (f32, f32) {
        let frames = self.sound.frames();
        let index = self.position as usize;
        let fraction = (self.position - index as f64) as f32;

        let next = match index + 1 {
            // Past the end: a loop continues into its own start, and a one-shot
            // holds its last frame rather than interpolating towards silence.
            next if next >= frames => {
                if looping {
                    0
                } else {
                    frames - 1
                }
            }
            next => next,
        };

        let (a_l, a_r) = self.sound.frame(index);
        let (b_l, b_r) = self.sound.frame(next);
        (a_l + (b_l - a_l) * fraction, a_r + (b_r - a_r) * fraction)
    }
}

/// Sums the voices that are playing. Lives on the audio thread.
pub(crate) struct Mixer {
    commands: Receiver<Command>,
    /// Finished voices go back to the owning thread to be dropped.
    retired: SyncSender<Playing>,
    shared: Arc<Shared>,
    voices: Vec<Playing>,
    /// Loaded instruments, mixed in after the voices. Each one is a plugin
    /// running on this thread.
    ///
    /// Boxed on purpose: an instrument arrives and leaves through the command
    /// queue, whose slots are all the size of its largest message, and it is
    /// moved back out again to be destroyed elsewhere. A pointer is what should
    /// be travelling, not the whole thing.
    #[cfg(feature = "clap")]
    #[allow(clippy::vec_box)]
    instruments: Vec<Box<crate::instrument::Processor>>,
    /// Interleaved stereo, big enough for the largest block asked for so far.
    mix: Vec<f32>,
    device_rate: u32,
    /// The last `stop_all` this mixer has acted on.
    stop_generation: u64,
}

impl Mixer {
    pub fn new(
        commands: Receiver<Command>,
        retired: SyncSender<Playing>,
        shared: Arc<Shared>,
        device_rate: u32,
        block_frames: usize,
    ) -> Self {
        Self {
            commands,
            retired,
            shared,
            voices: Vec::with_capacity(MAX_VOICES),
            #[cfg(feature = "clap")]
            instruments: Vec::with_capacity(MAX_INSTRUMENTS),
            mix: vec![0.0; block_frames * 2],
            device_rate,
            stop_generation: 0,
        }
    }

    /// Mixes `frames` frames and returns them as interleaved stereo, at the
    /// device's sample rate.
    pub fn render(&mut self, frames: usize) -> &[f32] {
        // Says which thread this is, for any plugin that asks. Done here rather
        // than per instrument, so that the answer is right from the first block
        // whether or not anything is loaded yet.
        #[cfg(feature = "clap")]
        crate::instrument::mark_audio_thread();

        self.take_commands();

        let generation = self.shared.stop_generation.load(Ordering::Relaxed);
        if generation != self.stop_generation {
            self.stop_generation = generation;
            for voice in &mut self.voices {
                voice.controls.stopping.store(true, Ordering::Relaxed);
            }
            // An instrument is not stopped — it stays loaded — but everything
            // it is holding down is released.
            #[cfg(feature = "clap")]
            for instrument in &mut self.instruments {
                instrument.all_notes_off();
            }
        }

        // The backend is allowed to ask for a larger block than it advertised.
        // Growing here allocates on the audio thread, which is exactly what this
        // module is trying to avoid, so it is sized generously up front and this
        // is the fallback rather than the plan.
        if self.mix.len() < frames * 2 {
            self.mix.resize(frames * 2, 0.0);
        }
        let mix = &mut self.mix[..frames * 2];
        mix.fill(0.0);

        let device_rate = self.device_rate;
        let retired = &self.retired;
        self.voices.retain_mut(|voice| {
            if voice.render(mix, device_rate) {
                return true;
            }
            voice.controls.playing.store(false, Ordering::Release);
            retire(retired, voice);
            false
        });

        #[cfg(feature = "clap")]
        for instrument in &mut self.instruments {
            instrument.render(mix);
        }

        self.shared
            .playing
            .store(self.voices.len(), Ordering::Relaxed);

        // The sum of several sounds can leave the range a device takes. Clamping
        // is the honest answer: a limiter would duck everything else whenever
        // one sound peaked, which is a decision a mixer this size has no
        // business making.
        let master = self.shared.gain.load();
        for sample in mix.iter_mut() {
            *sample = (*sample * master).clamp(-1.0, 1.0);
        }

        &self.mix[..frames * 2]
    }

    /// Picks up everything asked for since the last block.
    fn take_commands(&mut self) {
        while let Ok(command) = self.commands.try_recv() {
            match command {
                Command::Play(voice) => {
                    if self.voices.len() == MAX_VOICES {
                        // Take the oldest, which is the one most likely to have
                        // been heard already.
                        let mut oldest = self.voices.remove(0);
                        oldest.controls.playing.store(false, Ordering::Release);
                        retire(&self.retired, &mut oldest);
                        self.shared.stolen.fetch_add(1, Ordering::Relaxed);
                    }
                    self.voices.push(voice);
                }
                #[cfg(feature = "clap")]
                Command::AddInstrument(instrument) => self.instruments.push(instrument),
                #[cfg(feature = "clap")]
                Command::RemoveInstrument(id) => {
                    if let Some(index) = self.instruments.iter().position(|i| i.id == id) {
                        let mut instrument = self.instruments.remove(index);
                        // Stopped here, on the thread CLAP says it belongs on,
                        // and only then handed back to be destroyed.
                        instrument.stop();
                        instrument.retire();
                    }
                }
            }
        }
    }
}

/// Hands a finished voice back to the owning thread, leaving a stand-in behind
/// so that nothing is dropped here.
///
/// If the queue is full the voice is dropped on this thread after all, which
/// may free its samples and cost a few microseconds. That only happens when the
/// owner has not called anything for a long time, and a glitch then is better
/// than growing the queue without a bound.
fn retire(retired: &SyncSender<Playing>, voice: &mut Playing) {
    let stand_in = Playing {
        sound: voice.sound.clone(),
        controls: voice.controls.clone(),
        position: 0.0,
        ramp: None,
    };
    let going = std::mem::replace(voice, stand_in);
    if let Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) = retired.try_send(going) {
        // Dropped here; see above.
    }
}

/// Where a voice sits between the ears.
///
/// Panning takes the far side away rather than moving a fixed amount of energy
/// across, so a sound played at `pan` 0 and `gain` 1 comes out as exactly the
/// sound that was recorded. The alternative — constant power, where the two
/// gains square to 1 — keeps the loudness even across a sweep, but at the price
/// of every centred sound being 3 dB down, which is a strange thing for a sound
/// effect that is never panned at all to pay.
pub(crate) fn pan_gains(pan: f32) -> (f32, f32) {
    let pan = pan.clamp(-1.0, 1.0);
    ((1.0 - pan).min(1.0), (1.0 + pan).min(1.0))
}

/// An `f32` that can be written from one thread and read from another.
///
/// There is no `AtomicF32` in the standard library, and there does not need to
/// be: the bits go through an `AtomicU32` unchanged.
#[derive(Debug)]
pub(crate) struct AtomicF32(AtomicU32);

impl AtomicF32 {
    pub fn new(value: f32) -> Self {
        Self(AtomicU32::new(value.to_bits()))
    }

    pub fn load(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }

    pub fn store(&self, value: f32) {
        self.0.store(value.to_bits(), Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::sync_channel;

    #[cfg(feature = "clap")]
    #[test]
    fn instrument_slots_are_bounded() {
        let shared = Shared::new();
        for _ in 0..MAX_INSTRUMENTS {
            shared.reserve_instrument().unwrap();
        }
        let err = shared.reserve_instrument().unwrap_err();
        assert_eq!(
            err.to_string(),
            "at most 4 instruments can be loaded at once"
        );
        shared.release_instrument();
        assert!(shared.reserve_instrument().is_ok());
    }

    /// A mixer and the ends of the channels a caller would hold.
    struct Harness {
        mixer: Mixer,
        commands: SyncSender<Command>,
        retired: Receiver<Playing>,
        shared: Arc<Shared>,
    }

    fn harness(device_rate: u32) -> Harness {
        let (commands, command_rx) = sync_channel(COMMAND_QUEUE);
        let (retired_tx, retired) = sync_channel(MAX_VOICES);
        let shared = Arc::new(Shared::new());
        Harness {
            mixer: Mixer::new(command_rx, retired_tx, shared.clone(), device_rate, 512),
            commands,
            retired,
            shared,
        }
    }

    fn controls() -> Arc<Controls> {
        Arc::new(Controls {
            gain: AtomicF32::new(1.0),
            pan: AtomicF32::new(0.0),
            speed: AtomicF32::new(1.0),
            looping: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
            playing: AtomicBool::new(true),
        })
    }

    fn voice(sound: Sound, controls: &Arc<Controls>) -> Playing {
        Playing {
            sound,
            controls: controls.clone(),
            position: 0.0,
            ramp: None,
        }
    }

    /// A constant sound at the device's own rate should come out unchanged, and
    /// stop when it runs out.
    #[test]
    fn plays_a_sound_through_and_stops() {
        let mut h = harness(48_000);
        let sound = Sound::from_samples(vec![0.5; 4], 1, 48_000).unwrap();
        let controls = controls();
        h.commands
            .try_send(Command::Play(voice(sound, &controls)))
            .unwrap();

        let block = h.mixer.render(8);
        assert_eq!(&block[..8], &[0.5; 8]);
        assert_eq!(&block[8..], &[0.0; 8]);

        assert!(!controls.playing.load(Ordering::Acquire));
        assert_eq!(h.shared.playing.load(Ordering::Relaxed), 0);
        // The samples went back to this thread rather than being freed there.
        assert!(h.retired.try_recv().is_ok());
    }

    /// Two sounds at once sum, and the master gain applies to the sum.
    #[test]
    fn sums_voices_under_the_master_gain() {
        let mut h = harness(48_000);
        let sound = Sound::from_samples(vec![0.25; 64], 1, 48_000).unwrap();
        for _ in 0..2 {
            h.commands
                .try_send(Command::Play(voice(sound.clone(), &controls())))
                .unwrap();
        }
        h.shared.gain.store(0.5);

        let block = h.mixer.render(4);
        assert!((block[0] - 0.25).abs() < 1e-6, "{}", block[0]);
        assert_eq!(h.shared.playing.load(Ordering::Relaxed), 2);
    }

    /// Half the device's rate means two device frames per source frame, with
    /// the value in between interpolated.
    #[test]
    fn resamples_to_the_device_rate() {
        let mut h = harness(48_000);
        let sound = Sound::from_samples(vec![0.0, 1.0, 0.0, 1.0], 1, 24_000).unwrap();
        h.commands
            .try_send(Command::Play(voice(sound, &controls())))
            .unwrap();

        let block = h.mixer.render(4);
        let left: Vec<f32> = block.chunks(2).map(|frame| frame[0]).collect();
        assert_eq!(left, vec![0.0, 0.5, 1.0, 0.5]);
    }

    /// A loop runs past the end of its samples and back into the start.
    #[test]
    fn loops_without_running_out() {
        let mut h = harness(48_000);
        let sound = Sound::from_samples(vec![1.0, 1.0], 1, 48_000).unwrap();
        let controls = controls();
        controls.looping.store(true, Ordering::Relaxed);
        h.commands
            .try_send(Command::Play(voice(sound, &controls)))
            .unwrap();

        let block = h.mixer.render(8);
        assert!(block.iter().all(|s| *s == 1.0), "{block:?}");
        assert!(controls.playing.load(Ordering::Acquire));
    }

    /// Stopping fades over the block it was asked in, rather than cutting.
    #[test]
    fn a_stop_fades_out_within_one_block() {
        let mut h = harness(48_000);
        let sound = Sound::from_samples(vec![1.0; 64], 1, 48_000).unwrap();
        let controls = controls();
        h.commands
            .try_send(Command::Play(voice(sound, &controls)))
            .unwrap();
        h.mixer.render(4);

        controls.stopping.store(true, Ordering::Relaxed);
        let block = h.mixer.render(4);
        let left: Vec<f32> = block.chunks(2).map(|frame| frame[0]).collect();

        assert_eq!(left, vec![0.75, 0.5, 0.25, 0.0]);
        assert!(!controls.playing.load(Ordering::Acquire));
    }

    /// `stop_all` reaches voices the caller no longer holds a handle to.
    #[test]
    fn stop_all_reaches_every_voice() {
        let mut h = harness(48_000);
        let sound = Sound::from_samples(vec![1.0; 64], 1, 48_000).unwrap();
        h.commands
            .try_send(Command::Play(voice(sound, &controls())))
            .unwrap();
        h.mixer.render(4);

        h.shared.stop_generation.fetch_add(1, Ordering::Relaxed);
        h.mixer.render(4);
        assert_eq!(h.shared.playing.load(Ordering::Relaxed), 0);
    }

    /// Past the limit the oldest voice makes way, and is counted.
    #[test]
    fn steals_the_oldest_voice_when_full() {
        let mut h = harness(48_000);
        let sound = Sound::from_samples(vec![1.0; 1024], 1, 48_000).unwrap();
        let first = controls();
        h.commands
            .try_send(Command::Play(voice(sound.clone(), &first)))
            .unwrap();
        for _ in 0..MAX_VOICES {
            h.commands
                .try_send(Command::Play(voice(sound.clone(), &controls())))
                .unwrap();
        }

        h.mixer.render(4);
        assert_eq!(h.shared.playing.load(Ordering::Relaxed), MAX_VOICES);
        assert_eq!(h.shared.stolen.load(Ordering::Relaxed), 1);
        assert!(
            !first.playing.load(Ordering::Acquire),
            "the oldest should be the one taken"
        );
    }

    /// Hard over is silent on the far side, and centred is the sound itself.
    #[test]
    fn pans_by_taking_the_far_side_away() {
        assert_eq!(pan_gains(-1.0), (1.0, 0.0));
        assert_eq!(pan_gains(1.0), (0.0, 1.0));
        assert_eq!(pan_gains(0.0), (1.0, 1.0));
        assert_eq!(pan_gains(0.5), (0.5, 1.0));

        // Out of range rather than wrong: clamped to the ends.
        assert_eq!(pan_gains(-4.0), pan_gains(-1.0));
    }
}
