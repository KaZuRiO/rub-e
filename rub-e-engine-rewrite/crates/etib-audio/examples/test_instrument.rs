//! A CLAP instrument, built here so that the host in this crate has something
//! to load that is not a commercial plugin.
//!
//! Eight voices of sine with an attack and a release, a stereo output port and
//! one note input. That is enough to exercise the whole path: loading,
//! activation, notes reaching the audio thread, audio coming back, and
//! unloading.
//!
//! Its editor is a pretence: it accepts the whole protocol and prints what it
//! was asked, but there is no window. Drawing one would mean a fixture with a
//! platform toolkit in it, and what is worth testing here is the host's side of
//! the conversation — that it negotiates an API, creates before showing, and
//! destroys what it created.
//!
//! It builds as a `cdylib`, so cargo leaves a loadable library behind:
//!
//! ```sh
//! cargo build -p etib-audio --example test_instrument
//! cargo run -p etib-audio --example instrument -- \
//!     target/debug/examples/libtest_instrument.dylib
//! ```

use clack_extensions::audio_ports::{
    AudioPortFlags, AudioPortInfo, AudioPortInfoWriter, AudioPortType, PluginAudioPorts,
    PluginAudioPortsImpl,
};
use clack_extensions::gui::{GuiConfiguration, GuiSize, PluginGui, PluginGuiImpl, Window};
use clack_extensions::note_ports::{
    NoteDialect, NoteDialects, NotePortInfo, NotePortInfoWriter, PluginNotePorts,
    PluginNotePortsImpl,
};
use clack_plugin::events::event_types::{NoteOffEvent, NoteOnEvent};
use clack_plugin::prelude::*;

/// The most voices that can sound at once. Past this the oldest is taken.
const VOICES: usize = 8;

pub struct TestInstrument;

impl Plugin for TestInstrument {
    type AudioProcessor<'a> = Synth;
    type Shared<'a> = ();
    type MainThread<'a> = Ports;

    fn declare_extensions(builder: &mut PluginExtensions<Self>, _shared: Option<&()>) {
        builder
            .register::<PluginAudioPorts>()
            .register::<PluginNotePorts>()
            .register::<PluginGui>();
    }
}

impl DefaultPluginFactory for TestInstrument {
    fn get_descriptor() -> PluginDescriptor {
        PluginDescriptor::new("dev.etib.test-instrument", "etib Test Instrument")
            .with_vendor("etib")
            .with_version("0.1.0")
            .with_features([c"instrument", c"synthesizer", c"stereo"])
    }

    fn new_shared(_host: HostSharedHandle<'_>) -> Result<(), PluginError> {
        Ok(())
    }

    fn new_main_thread<'a>(
        _host: HostMainThreadHandle<'a>,
        _shared: &'a (),
    ) -> Result<Ports, PluginError> {
        Ok(Ports::default())
    }
}

/// The main-thread half: what the ports are, and a window that is not there.
#[derive(Default)]
pub struct Ports {
    gui_created: std::cell::Cell<bool>,
    gui_shown: std::cell::Cell<bool>,
}

impl PluginMainThread<'_, ()> for Ports {}

impl PluginAudioPortsImpl for Ports {
    fn count(&self, is_input: bool) -> u32 {
        // An instrument: audio out, nothing in.
        u32::from(!is_input)
    }

    fn get(&self, index: u32, is_input: bool, writer: &mut AudioPortInfoWriter) {
        if is_input || index > 0 {
            return;
        }
        writer.set(&AudioPortInfo {
            id: ClapId::new(0),
            name: b"main",
            channel_count: 2,
            flags: AudioPortFlags::IS_MAIN,
            port_type: Some(AudioPortType::STEREO),
            in_place_pair: None,
        });
    }
}

impl PluginNotePortsImpl for Ports {
    fn count(&self, is_input: bool) -> u32 {
        u32::from(is_input)
    }

    fn get(&self, index: u32, is_input: bool, writer: &mut NotePortInfoWriter) {
        if !is_input || index > 0 {
            return;
        }
        writer.set(&NotePortInfo {
            id: ClapId::new(0),
            name: b"notes",
            supported_dialects: NoteDialects::CLAP,
            preferred_dialect: Some(NoteDialect::Clap),
        });
    }
}

/// A window this plugin does not have.
///
/// Everything is accepted and reported so that a host can be watched doing the
/// right things in the right order. `get_size` answers with something plausible
/// because a host is allowed to ask before anything is on screen.
impl PluginGuiImpl for Ports {
    fn is_api_supported(&self, configuration: GuiConfiguration) -> bool {
        eprintln!(
            "[test instrument] is_api_supported({:?}, floating: {})",
            configuration.api_type, configuration.is_floating
        );
        // Floating only: this pretend window cannot be put inside another.
        configuration.is_floating
    }

    fn get_preferred_api(&self) -> Option<GuiConfiguration<'_>> {
        None
    }

    fn create(&self, configuration: GuiConfiguration) -> Result<(), PluginError> {
        eprintln!(
            "[test instrument] create(floating: {})",
            configuration.is_floating
        );
        self.gui_created.set(true);
        Ok(())
    }

    fn destroy(&self) {
        eprintln!("[test instrument] destroy");
        self.gui_created.set(false);
        self.gui_shown.set(false);
    }

    fn set_scale(&self, _scale: f64) -> Result<(), PluginError> {
        Err(PluginError::Message("this window has no scale"))
    }

    fn get_size(&self) -> Option<GuiSize> {
        Some(GuiSize {
            width: 640,
            height: 360,
        })
    }

    fn set_size(&self, _size: GuiSize) -> Result<(), PluginError> {
        Err(PluginError::Message("this window cannot be resized"))
    }

    fn set_parent(&self, _window: Window) -> Result<(), PluginError> {
        Err(PluginError::Message("this window cannot be embedded"))
    }

    fn set_transient(&self, _window: Window) -> Result<(), PluginError> {
        Ok(())
    }

    fn suggest_title(&self, title: &str) {
        eprintln!("[test instrument] suggest_title({title:?})");
    }

    fn show(&self) -> Result<(), PluginError> {
        eprintln!("[test instrument] show");
        if !self.gui_created.get() {
            return Err(PluginError::Message("shown before being created"));
        }
        self.gui_shown.set(true);
        Ok(())
    }

    fn hide(&self) -> Result<(), PluginError> {
        eprintln!("[test instrument] hide");
        self.gui_shown.set(false);
        Ok(())
    }
}

/// One key, sounding.
#[derive(Clone, Copy, Default)]
struct Voice {
    key: u16,
    phase: f32,
    step: f32,
    /// Peak level this voice is heading for, which is its velocity.
    peak: f32,
    /// Where the envelope is, 0..=peak.
    level: f32,
    held: bool,
    active: bool,
}

pub struct Synth {
    voices: [Voice; VOICES],
    sample_rate: f32,
}

impl Synth {
    fn note_on(&mut self, key: u16, velocity: f32) {
        // A free voice, or the quietest one if they are all busy.
        let slot = self
            .voices
            .iter()
            .position(|voice| !voice.active)
            .unwrap_or_else(|| {
                let mut quietest = 0;
                for (index, voice) in self.voices.iter().enumerate() {
                    if voice.level < self.voices[quietest].level {
                        quietest = index;
                    }
                }
                quietest
            });

        // The equal-tempered scale, with key 69 at 440 Hz, as MIDI has it.
        let hz = 440.0 * 2f32.powf((key as f32 - 69.0) / 12.0);
        self.voices[slot] = Voice {
            key,
            phase: 0.0,
            step: hz / self.sample_rate,
            peak: velocity.clamp(0.0, 1.0),
            level: 0.0,
            held: true,
            active: true,
        };
    }

    fn note_off(&mut self, key: Option<u16>) {
        for voice in &mut self.voices {
            if key.is_none_or(|key| voice.key == key) {
                voice.held = false;
            }
        }
    }
}

impl<'a> PluginAudioProcessor<'a, (), Ports> for Synth {
    fn activate(
        _host: HostAudioProcessorHandle<'a>,
        _main_thread: &Ports,
        _shared: &'a (),
        config: PluginAudioConfiguration,
    ) -> Result<Self, PluginError> {
        Ok(Self {
            voices: [Voice::default(); VOICES],
            sample_rate: config.sample_rate as f32,
        })
    }

    fn process(
        &mut self,
        _process: Process,
        mut audio: Audio,
        events: Events,
    ) -> Result<ProcessStatus, PluginError> {
        for event in events.input {
            if let Some(note) = event.as_event::<NoteOnEvent>() {
                let key = note.pckn().key.into_specific().unwrap_or(60);
                self.note_on(key, note.velocity() as f32);
            } else if let Some(note) = event.as_event::<NoteOffEvent>() {
                self.note_off(note.pckn().key.into_specific());
            }
        }

        // Roughly 5 ms to reach full level and 200 ms to fall away, which is
        // long enough to hear and short enough not to hide a stuck note.
        let attack = 1.0 / (0.005 * self.sample_rate);
        let release = 1.0 / (0.2 * self.sample_rate);

        for mut port in &mut audio {
            let Some(channels) = port.channels()?.into_f32() else {
                continue;
            };

            for pair in channels {
                let out = match pair {
                    ChannelPair::OutputOnly(out) => out,
                    ChannelPair::InPlace(out) => out,
                    _ => continue,
                };

                let mut voices = self.voices;
                for sample in out.iter_mut() {
                    let mut sum = 0.0;
                    for voice in voices.iter_mut().filter(|voice| voice.active) {
                        voice.level = match voice.held {
                            true => (voice.level + attack * voice.peak).min(voice.peak),
                            false => voice.level - release * voice.peak,
                        };
                        if voice.level <= 0.0 {
                            voice.active = false;
                            continue;
                        }

                        sum += (voice.phase * std::f32::consts::TAU).sin() * voice.level;
                        voice.phase = (voice.phase + voice.step).fract();
                    }
                    // A quarter, so that a handful of voices at once still fits.
                    *sample = sum * 0.25;
                }
                // Both channels are the same, so the envelope is advanced once
                // per channel and the result of the last one is what is kept.
                self.voices = voices;
            }
        }

        Ok(ProcessStatus::Continue)
    }
}

clack_export_entry!(SinglePluginEntry<TestInstrument>);
