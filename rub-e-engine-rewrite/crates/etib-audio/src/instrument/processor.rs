//! The half of an instrument that lives on the audio thread.
//!
//! Everything here is created on the loading thread — buffers, event storage,
//! the plugin's activated audio processor — and then moved across, because the
//! rules this crate keeps for its own mixing path (no allocation, no lock, no
//! freeing) apply just as much to a plugin's block as to a sound's.
//!
//! What a plugin does inside its own `process` is its own business, and a badly
//! behaved one can still glitch the output. That is the one part of the
//! deadline this crate cannot keep on its own.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};

use clack_host::events::Match;
use clack_host::events::event_types::{NoteOffEvent, NoteOnEvent, ParamValueEvent};
use clack_host::prelude::*;

use crate::instrument::host::Host;
use crate::mixer::Controls;

/// Something to be handed to the plugin at the start of the next block.
///
/// Events are applied at a block boundary rather than at the sample they were
/// sent for: an application calling from its frame loop cannot say when within
/// the next few milliseconds it meant, and pretending otherwise would add a
/// timestamp that means nothing.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Event {
    NoteOn {
        channel: u16,
        key: u16,
        velocity: f64,
    },
    NoteOff {
        channel: u16,
        key: u16,
        velocity: f64,
    },
    /// Every note on every channel, released rather than cut.
    AllNotesOff,
    Param {
        id: ClapId,
        value: f64,
    },
}

/// One loaded instrument, playing.
pub(crate) struct Processor {
    pub id: u64,
    plugin: PluginAudioProcessor<Host>,
    controls: Arc<Controls>,
    events: Receiver<Event>,

    /// Written each block from [`events`](Self::events); never grows past the
    /// capacity reserved for it on the loading thread.
    input_events: EventBuffer,
    /// What the plugin sends back — parameter changes it made itself, note ends.
    /// Read for nothing yet, but a plugin has to be given somewhere to write.
    output_events: EventBuffer,

    input_ports: AudioPorts,
    output_ports: AudioPorts,
    /// Silence, handed to the plugin's input port if it has one.
    input_buffers: Vec<Vec<f32>>,
    output_buffers: Vec<Vec<f32>>,
    /// The largest block the plugin was activated for. A device that asks for
    /// more than this is served in several passes.
    max_frames: usize,

    /// Frames since the instrument was loaded, which is what a plugin uses to
    /// keep an LFO or a delay line in step.
    steady_time: u64,
    /// Where the last block's gain ramp ended, per side.
    ramp: Option<(f32, f32)>,

    /// Set by the mixer when everything is asked to stop; released at the top
    /// of the next block, with the notes the application itself sent.
    all_off_pending: bool,

    /// Raised when the plugin's own processing fails, and reported by
    /// [`Instrument::poll`](crate::Instrument::poll) on the other thread.
    failed: Arc<AtomicBool>,
    /// Where this goes to be dropped when it is taken out of the mixer.
    return_to: SyncSender<Box<Processor>>,
}

impl Processor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u64,
        plugin: PluginAudioProcessor<Host>,
        controls: Arc<Controls>,
        events: Receiver<Event>,
        input_channels: usize,
        output_channels: usize,
        max_frames: usize,
        failed: Arc<AtomicBool>,
        return_to: SyncSender<Box<Processor>>,
    ) -> Self {
        Self {
            id,
            plugin,
            controls,
            events,
            // A block's worth of notes with room to spare; the queue feeding
            // this is bounded well below it.
            input_events: EventBuffer::with_capacity(64),
            output_events: EventBuffer::with_capacity(256),
            input_ports: AudioPorts::with_capacity(input_channels.max(1), 1),
            output_ports: AudioPorts::with_capacity(output_channels.max(1), 1),
            input_buffers: vec![vec![0.0; max_frames]; input_channels],
            output_buffers: vec![vec![0.0; max_frames]; output_channels],
            max_frames,
            steady_time: 0,
            ramp: None,
            all_off_pending: false,
            failed,
            return_to,
        }
    }

    /// Mixes one block of the instrument into `mix`, which is interleaved
    /// stereo at the device's rate.
    pub fn render(&mut self, mix: &mut [f32]) {
        let frames = mix.len() / 2;
        if frames == 0 || self.output_buffers.is_empty() {
            return;
        }

        let gain = self.controls.gain.load();
        let (pan_l, pan_r) = crate::mixer::pan_gains(self.controls.pan.load());
        let (to_l, to_r) = (gain * pan_l, gain * pan_r);
        let (from_l, from_r) = self.ramp.unwrap_or((to_l, to_r));
        self.ramp = Some((to_l, to_r));

        self.take_events();

        let mut done = 0;
        while done < frames {
            // The plugin was activated for a maximum block size and may have
            // sized its own buffers to it, so a larger request is split.
            let count = (frames - done).min(self.max_frames);
            if !self.process(count) {
                self.failed.store(true, Ordering::Release);
                return;
            }

            for frame in 0..count {
                // The ramp runs over the whole block, not over each pass of
                // this loop, or a split block would be heard as two.
                let t = (done + frame + 1) as f32 / frames as f32;
                let (left, right) = self.output_frame(frame);

                mix[(done + frame) * 2] += left * (from_l + (to_l - from_l) * t);
                mix[(done + frame) * 2 + 1] += right * (from_r + (to_r - from_r) * t);
            }

            // Only the first pass carries this block's events.
            self.input_events.clear();
            self.steady_time += count as u64;
            done += count;
        }
    }

    /// Everything asked for since the last block, in the order it was asked.
    fn take_events(&mut self) {
        self.input_events.clear();

        if std::mem::take(&mut self.all_off_pending) {
            self.input_events.push(&NoteOffEvent::new(
                0,
                Pckn::new(0u16, Match::All, Match::All, Match::All),
                0.0,
            ));
        }

        while let Ok(event) = self.events.try_recv() {
            // Note ids are left as wildcards: an application that plays a key
            // and later releases it expects that to reach whichever voice took
            // the key, which is what matching on the key alone means.
            match event {
                Event::NoteOn {
                    channel,
                    key,
                    velocity,
                } => self.input_events.push(&NoteOnEvent::new(
                    0,
                    Pckn::new(0u16, channel, key, Match::All),
                    velocity,
                )),
                Event::NoteOff {
                    channel,
                    key,
                    velocity,
                } => self.input_events.push(&NoteOffEvent::new(
                    0,
                    Pckn::new(0u16, channel, key, Match::All),
                    velocity,
                )),
                Event::AllNotesOff => self.input_events.push(&NoteOffEvent::new(
                    0,
                    Pckn::new(0u16, Match::All, Match::All, Match::All),
                    0.0,
                )),
                Event::Param { id, value } => {
                    self.input_events
                        .push(&ParamValueEvent::new(0, id, Pckn::match_all(), value))
                }
            }
        }
    }

    /// Runs the plugin over `count` frames, leaving the result in
    /// [`output_buffers`](Self::output_buffers). `false` means it failed.
    fn process(&mut self, count: usize) -> bool {
        for buffer in &mut self.output_buffers {
            buffer[..count].fill(0.0);
        }
        self.output_events.clear();

        let Ok(plugin) = self.plugin.ensure_processing_started() else {
            return false;
        };

        // Borrowed field by field, so that the plugin's buffers and its
        // processor are held at the same time.
        let mut inputs = Vec::with_capacity(1);
        if !self.input_buffers.is_empty() {
            inputs.push(AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    self.input_buffers
                        .iter_mut()
                        .map(|buffer| InputChannel::constant(&mut buffer[..count])),
                ),
            });
        }
        let input_audio = self.input_ports.with_input_buffers(inputs);

        let mut output_audio = self.output_ports.with_output_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_output_only(
                self.output_buffers
                    .iter_mut()
                    .map(|buffer| &mut buffer[..count]),
            ),
        }]);

        plugin
            .process(
                &input_audio,
                &mut output_audio,
                &self.input_events.as_input(),
                &mut self.output_events.as_output(),
                Some(self.steady_time),
                None,
            )
            .is_ok()
    }

    /// The plugin's output for one frame, as a pair.
    ///
    /// A mono instrument is put in both ears; anything wider than stereo is
    /// taken from its first two channels, since placing a surround field needs
    /// a listener orientation this does not have.
    fn output_frame(&self, frame: usize) -> (f32, f32) {
        match self.output_buffers.as_slice() {
            [mono] => (mono[frame], mono[frame]),
            [left, right, ..] => (left[frame], right[frame]),
            [] => (0.0, 0.0),
        }
    }

    /// Releases every note, for when the mixer is asked to stop everything.
    ///
    /// Taken up at the start of the next block, along with whatever the
    /// application asked for in the meantime.
    pub fn all_notes_off(&mut self) {
        self.all_off_pending = true;
    }

    /// Tells the plugin that continuous processing has ended.
    ///
    /// CLAP puts this on the audio thread, which is the whole reason it happens
    /// here rather than where the plugin is destroyed: a plugin that checks —
    /// and several do — reports a host that stops it anywhere else as
    /// misbehaving, because its own `process` state is only safe to unwind from
    /// the thread that was running it.
    pub fn stop(&mut self) {
        self.plugin.ensure_processing_stopped();
    }

    /// Takes the plugin's audio processor back out, to be deactivated on the
    /// thread that loaded it. Already stopped by then.
    pub fn into_plugin(self: Box<Self>) -> PluginAudioProcessor<Host> {
        self.plugin
    }

    /// Hands this back to the thread that loaded the plugin, which is the one
    /// allowed to deactivate and destroy it.
    ///
    /// If nobody is waiting for it — the handle was dropped without unloading —
    /// it is dropped here instead, which may free the plugin's buffers on the
    /// audio thread.
    pub fn retire(self: Box<Self>) {
        let return_to = self.return_to.clone();
        let _ = return_to.try_send(self);
    }
}
