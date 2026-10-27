//! Playing a CLAP instrument: loading one, sending it notes, and — when
//! something needs looking at — opening its own window.
//!
//! A [`Sound`](crate::Sound) is a recording; an instrument is a program that
//! makes one on demand. [CLAP](https://cleveraudio.org) is the plugin format
//! for those: a file the application does not ship, loaded at runtime, that
//! turns notes into audio.
//!
//! ```no_run
//! # use etib_audio::{Audio, Instrument, InstrumentConfig};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let audio = Audio::new()?;
//!
//! // Loading a plugin runs its code; see the safety note on `load_instrument`.
//! let mut piano = unsafe {
//!     audio.load_instrument("/Library/Audio/Plug-Ins/CLAP/Piano.clap", &InstrumentConfig::default())
//! }?;
//!
//! piano.note_on(0, 60, 0.8);   // middle C
//! piano.note_off(0, 60);
//!
//! // Once a frame, on the thread that loaded it.
//! piano.poll();
//! # Ok(())
//! # }
//! ```
//!
//! # Headless by default
//!
//! An instrument plays without a window, and nothing here creates one unless it
//! is asked to. [`InstrumentConfig::gui`] is [`Gui::Headless`] by default: the
//! plugin's editor is never created, and a plugin that would like to show it is
//! told no.
//!
//! [`Gui::Debug`] is the other setting, and it is what the name says — a way to
//! see what a plugin is doing while working on the application, not something
//! to ship. Under it, [`Instrument::show_gui`] opens the editor in a window
//! alongside the application's rather than inside it.
//!
//! Where that window comes from is the plugin's choice more than this crate's.
//! CLAP has two arrangements: *floating*, where the plugin opens a window
//! itself, and *embedded*, where the host hands it one to draw into. Only
//! embedding is required of a plugin, and plenty of real ones — Surge XT among
//! them — do not float at all, so `show_gui` asks for floating first and opens
//! a plain window itself when that is refused. Opening one is implemented on
//! macOS; elsewhere a plugin that will not float has to be given a window by
//! the application, with [`Instrument::show_gui_in`], which takes anything with
//! a window handle — a winit window, for instance.
//!
//! Either way the window needs an event loop to be drawn and clicked, which the
//! platform runs on the thread the instrument was loaded on. In an `etib`
//! application that is the frame loop, and the window behaves like any other;
//! a program with no event loop of its own has to pump one, as
//! `examples/instrument` does.
//!
//! # Threads
//!
//! CLAP says which thread each call belongs on, and that shape is visible here.
//! [`Instrument`] is the main-thread half: it owns the plugin, and it is not
//! [`Send`]. The audio half is moved to the device's thread when the instrument
//! is loaded, and comes back to be destroyed when it is unloaded.
//!
//! Two things follow for an application:
//!
//! * **[`poll`](Instrument::poll) has to be called**, once a frame or
//!   thereabouts, from the thread that loaded the instrument. It is where a
//!   plugin's own callbacks, its timers and its window are serviced. Notes do
//!   not depend on it; an unpolled instrument still plays, but its editor
//!   freezes and any work it deferred never happens.
//! * **Notes are applied at a block boundary**, a millisecond or two after they
//!   are sent, rather than at an exact sample. An application calling from a
//!   frame loop cannot say when within the next block it meant.
//!
//! # What is not here
//!
//! No presets or saved state, no MIDI beyond notes, no plugin-supplied effects
//! chain, and no audio into the plugin — an instrument's input port, if it has
//! one, is fed silence. The editor is a window of its own: putting one inside
//! the application's is [`Instrument::show_gui_in`]'s business, and what it is
//! given is the application's to lay out.

mod host;
mod processor;
mod window;

use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::time::{Duration, Instant};

use clack_extensions::audio_ports::{AudioPortInfoBuffer, PluginAudioPorts};
use clack_extensions::gui::{GuiApiType, GuiConfiguration, GuiSize, PluginGui, Window};
use clack_extensions::params::{ParamInfoBuffer, PluginParams};
use clack_extensions::timer::PluginTimer;
use clack_host::factory::plugin::PluginFactory;
use clack_host::plugin::PluginDescriptor;
use clack_host::prelude::*;
use log::{debug, error, info, warn};

use crate::engine::Audio;
use crate::error::{Error, Result};
use crate::instrument::window::DebugWindow;
use crate::mixer::{self, Command, Controls};
use raw_window_handle::HasWindowHandle;

pub(crate) use host::mark_audio_thread;
pub(crate) use processor::Processor;

use host::{Host, MainThread, Shared};
use processor::Event;

/// Instruments are told apart by a number the mixer can match on, since it has
/// no other way to name one. Unique within the process, and never reused.
static NEXT_INSTRUMENT: AtomicU64 = AtomicU64::new(0);

/// How many notes may be waiting to reach the plugin. A block's worth of
/// playing is a handful of notes; this is generous.
const EVENT_QUEUE: usize = 64;

/// How long [`Instrument::unload`] waits for the audio thread to hand the
/// plugin back. One block is a few milliseconds; this is long enough that only
/// a stopped device runs it out.
const TEARDOWN_TIMEOUT: Duration = Duration::from_millis(250);

/// Whether an instrument may open its editor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Gui {
    /// No window, ever. The plugin's editor is not created, and a plugin asking
    /// to show it is refused.
    ///
    /// This is the default, and what an application ships with: an instrument
    /// is something it plays, not something it puts on the screen.
    #[default]
    Headless,
    /// The editor may be opened, as a floating window of the plugin's own.
    ///
    /// A debugging aid — for hearing what a patch is doing while building
    /// something against it. The window belongs to the plugin, opens alongside
    /// the application's own rather than inside it, and nothing is created
    /// until [`Instrument::show_gui`] is called.
    Debug,
}

/// How an instrument is loaded and how it is first heard.
#[derive(Clone, Debug)]
pub struct InstrumentConfig {
    /// Which plugin to take from the file, by its CLAP id.
    ///
    /// A `.clap` file may hold several. `None` takes the first one that says it
    /// is an instrument, or the first one at all if none of them do.
    pub plugin_id: Option<String>,
    /// Whether the editor may be opened. [`Gui::Headless`] by default.
    pub gui: Gui,
    /// Level, where 1.0 is the plugin's output as it comes.
    pub gain: f32,
    /// Where it sits between the ears: -1 hard left, 0 centred, 1 hard right.
    pub pan: f32,
}

impl Default for InstrumentConfig {
    fn default() -> Self {
        Self {
            plugin_id: None,
            gui: Gui::Headless,
            gain: 1.0,
            pan: 0.0,
        }
    }
}

/// What a `.clap` file says about one of the plugins inside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PluginInfo {
    /// The id to pass as [`InstrumentConfig::plugin_id`].
    pub id: String,
    pub name: String,
    pub vendor: String,
    pub version: String,
    /// Whether the plugin declares itself an instrument. Anything else is an
    /// effect, which this can load but will hear nothing from, since it feeds
    /// its input silence.
    pub is_instrument: bool,
}

/// One of a plugin's parameters, as it describes itself.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamInfo {
    /// The id to pass to [`Instrument::set_param`].
    pub id: u32,
    pub name: String,
    pub min: f64,
    pub max: f64,
    pub default: f64,
}

/// Everything a `.clap` file holds, without instantiating anything.
///
/// # Safety
///
/// This loads the file as a dynamic library and runs its entry point, which is
/// to say it runs code that is not part of this program. Nothing can check that
/// what is at `path` is a well-formed CLAP plugin rather than something that
/// will corrupt this process, so the caller is the one deciding to trust it.
pub unsafe fn plugins_in(path: impl AsRef<Path>) -> Result<Vec<PluginInfo>> {
    let path = path.as_ref();
    // SAFETY: the caller has taken on the trust decision; passed straight down.
    let entry = unsafe { load_entry(path) }?;
    let factory = plugin_factory(&entry, path)?;

    Ok(factory.plugin_descriptors().map(describe).collect())
}

impl Audio {
    /// Loads a CLAP instrument and starts mixing it in.
    ///
    /// The plugin is activated for this device's sample rate and block size, so
    /// it has to be loaded against a real device: a [silent](Audio::silent)
    /// `Audio` has neither, and this returns an error rather than pretending.
    ///
    /// The returned [`Instrument`] plays for as long as it is held. Dropping it
    /// unloads the plugin, as [`unload`](Instrument::unload) does, but without
    /// a way to report what went wrong.
    ///
    /// # Safety
    ///
    /// This loads `path` as a dynamic library and runs code from it — the
    /// plugin's entry point, and then the plugin itself, on this thread and on
    /// the audio thread. Nothing here can check that the file is a well-formed
    /// CLAP plugin rather than something that will corrupt this process, and
    /// even a well-formed one is another program's code running inside this
    /// one. The caller is the one deciding to trust the file.
    pub unsafe fn load_instrument(
        &self,
        path: impl AsRef<Path>,
        config: &InstrumentConfig,
    ) -> Result<Instrument> {
        let path = path.as_ref();
        if self.is_silent() {
            return Err(Error::new(format!(
                "cannot load {} because there is no audio device to play it through",
                path.display()
            )));
        }

        // Taken first and given back if anything below fails, so that the
        // mixer never holds more instruments than it made room for.
        let slot = InstrumentSlot::reserve(self.shared())?;

        // SAFETY: the caller has taken on the trust decision; passed straight down.
        let entry = unsafe { load_entry(path) }?;
        let factory = plugin_factory(&entry, path)?;
        let descriptor = choose(&factory, config.plugin_id.as_deref(), path)?;
        let info = describe(descriptor);
        let plugin_id = descriptor
            .id()
            .ok_or_else(|| Error::new(format!("a plugin in {} has no id", path.display())))?
            .to_owned();

        info!(
            "loading '{}' {} by {} from {}",
            info.name,
            info.version,
            info.vendor,
            path.display()
        );
        if !info.is_instrument {
            warn!(
                "'{}' does not declare itself an instrument; it will be fed silence",
                info.name
            );
        }

        let host_info = HostInfo::new(
            "etib",
            "etib",
            "https://github.com/",
            env!("CARGO_PKG_VERSION"),
        )
        .map_err(|err| Error::with_source("could not describe this host to the plugin", err))?;

        let name = info.name.clone();
        let mut instance = PluginInstance::<Host>::new(
            move |_| Shared::new(name),
            |_| MainThread::new(),
            &entry,
            &plugin_id,
            &host_info,
        )
        .map_err(|err| Error::with_source(format!("could not instantiate '{}'", info.name), err))?;

        let (input_channels, output_channels) = port_layout(&mut instance);
        if output_channels == 0 {
            return Err(Error::new(format!(
                "'{}' has no audio output port, so there is nothing to play",
                info.name
            )));
        }
        debug!(
            "'{}': {input_channels} in, {output_channels} out, activating at {} Hz",
            info.name,
            self.sample_rate()
        );

        let plugin = instance
            .activate(
                |_, _| (),
                PluginAudioConfiguration {
                    sample_rate: f64::from(self.sample_rate()),
                    min_frames_count: 1,
                    max_frames_count: u32::try_from(self.block_frames())
                        .expect("block sizes are clamped to 16384 frames"),
                },
            )
            .map_err(|err| {
                Error::with_source(format!("could not activate '{}'", info.name), err)
            })?;

        let controls = Arc::new(Controls::new(config.gain, config.pan));
        let failed = Arc::new(AtomicBool::new(false));
        let (events, event_rx) = sync_channel(EVENT_QUEUE);
        // One slot: the processor comes back exactly once, when it is removed.
        let (return_tx, returning) = sync_channel(1);

        let id = NEXT_INSTRUMENT.fetch_add(1, Ordering::Relaxed);
        let processor = Processor::new(
            id,
            plugin.into(),
            controls.clone(),
            event_rx,
            input_channels,
            output_channels,
            self.block_frames(),
            failed.clone(),
            return_tx,
        );

        let instrument = Instrument {
            extensions: Extensions::of(&mut instance),
            instance,
            id,
            info,
            controls,
            events,
            commands: self.commands(),
            returning,
            failed,
            gui: config.gui,
            gui_state: GuiState::None,
            gui_kind: GuiKind::Floating,
            host_window: None,
            output_channels,
            reported_failure: false,
            unloaded: false,
            slot,
        };

        // Handing it over is the last thing, so that nothing is half-loaded if
        // any of the above failed.
        if self
            .commands()
            .try_send(Command::AddInstrument(Box::new(processor)))
            .is_err()
        {
            return Err(Error::new(
                "the mixer is not taking anything; the audio device may have stopped",
            ));
        }

        Ok(instrument)
    }
}

/// A loaded CLAP instrument.
///
/// Not [`Send`]: the plugin belongs to the thread that loaded it, which is the
/// only one that may call [`poll`](Self::poll), open its window, or unload it.
/// Playing notes is the exception — that reaches the audio thread through a
/// queue — but the handle itself stays where it was made.
pub struct Instrument {
    instance: PluginInstance<Host>,
    extensions: Extensions,
    id: u64,
    info: PluginInfo,
    controls: Arc<Controls>,
    events: SyncSender<Event>,
    commands: SyncSender<Command>,
    returning: Receiver<Box<Processor>>,
    failed: Arc<AtomicBool>,
    gui: Gui,
    gui_state: GuiState,
    gui_kind: GuiKind,
    /// The window this crate opened for the editor, if it opened one. Dropped
    /// only after the plugin has been told to destroy what is inside it.
    host_window: Option<DebugWindow>,
    output_channels: usize,
    reported_failure: bool,
    unloaded: bool,
    /// This instrument's place in the mixer, given back when it is dropped.
    slot: InstrumentSlot,
}

/// One of the mixer's [`MAX_INSTRUMENTS`](mixer::MAX_INSTRUMENTS) places,
/// given back when dropped.
struct InstrumentSlot(Option<Arc<mixer::Shared>>);

impl InstrumentSlot {
    fn reserve(shared: &Arc<mixer::Shared>) -> Result<Self> {
        shared.reserve_instrument()?;
        Ok(Self(Some(shared.clone())))
    }

    /// Keeps the place taken for good: for a plugin the audio thread never
    /// handed back, which is still in the mixer's list.
    fn keep(&mut self) {
        self.0 = None;
    }
}

impl Drop for InstrumentSlot {
    fn drop(&mut self) {
        if let Some(shared) = self.0.take() {
            shared.release_instrument();
        }
    }
}

/// The extensions this host uses, looked up once rather than per call.
struct Extensions {
    gui: Option<PluginGui>,
    timer: Option<PluginTimer>,
    params: Option<PluginParams>,
}

impl Extensions {
    fn of(instance: &mut PluginInstance<Host>) -> Self {
        let handle = instance.plugin_handle();
        Self {
            gui: handle.get_extension(),
            timer: handle.get_extension(),
            params: handle.get_extension(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GuiState {
    /// Never created, or destroyed again.
    None,
    Shown,
    Hidden,
}

/// How a plugin's editor ended up on the screen, which decides what the host
/// still has to do for it afterwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GuiKind {
    /// The plugin opened its own window. Nothing more is owed to it.
    Floating,
    /// It is drawing into a window, and whether that window is this crate's is
    /// what [`Instrument::host_window`] says.
    Embedded,
}

impl Instrument {
    /// What the plugin says about itself.
    pub fn info(&self) -> &PluginInfo {
        &self.info
    }

    /// How many channels the plugin's main output port has. One is played into
    /// both ears; more than two is taken from the first two.
    pub fn channels(&self) -> usize {
        self.output_channels
    }

    /// Starts a note. `key` is a MIDI key number — 60 is middle C — and
    /// `velocity` runs from 0 to 1.
    ///
    /// A note is taken up at the start of the next block, a millisecond or two
    /// from now.
    pub fn note_on(&self, channel: u8, key: u8, velocity: f32) {
        self.send(Event::NoteOn {
            channel: u16::from(channel),
            key: u16::from(key),
            velocity: f64::from(velocity.clamp(0.0, 1.0)),
        });
    }

    /// Releases a note, letting it ring out the way the patch says it should.
    pub fn note_off(&self, channel: u8, key: u8) {
        self.send(Event::NoteOff {
            channel: u16::from(channel),
            key: u16::from(key),
            velocity: 0.0,
        });
    }

    /// Releases every note on every channel.
    pub fn all_notes_off(&self) {
        self.send(Event::AllNotesOff);
    }

    /// Sets a parameter, by the id from [`params`](Self::params).
    ///
    /// The value is what the plugin says it means: no range checking is done
    /// here, because a plugin is free to clamp or wrap as it sees fit, and it
    /// is the one that knows.
    pub fn set_param(&self, id: u32, value: f64) {
        match ClapId::from_raw(id) {
            Some(id) => self.send(Event::Param { id, value }),
            None => warn!("{id} is not a usable parameter id"),
        }
    }

    /// Everything the plugin exposes as a parameter.
    ///
    /// Read from the plugin each time it is called, on this thread. It is not
    /// something to call per frame.
    pub fn params(&mut self) -> Vec<ParamInfo> {
        let Some(params) = self.extensions.params else {
            return Vec::new();
        };

        let handle = self.instance.plugin_handle();
        let count = params.count(&handle);
        let mut buffer = ParamInfoBuffer::new();
        let mut out = Vec::with_capacity(count as usize);

        for index in 0..count {
            let Some(info) = params.get_info(&handle, index, &mut buffer) else {
                continue;
            };
            out.push(ParamInfo {
                id: info.id.into(),
                name: String::from_utf8_lossy(info.name).into_owned(),
                min: info.min_value,
                max: info.max_value,
                default: info.default_value,
            });
        }
        out
    }

    /// Sets the level this instrument is mixed at, 1.0 being its output as it
    /// comes. Smoothed over a block, like everything else.
    pub fn set_gain(&self, gain: f32) {
        self.controls
            .gain
            .store(if gain.is_finite() { gain.max(0.0) } else { 1.0 });
    }

    /// Sets where it sits between the ears, -1 to 1.
    pub fn set_pan(&self, pan: f32) {
        self.controls.pan.store(if pan.is_finite() {
            pan.clamp(-1.0, 1.0)
        } else {
            0.0
        });
    }

    /// Services the plugin's main-thread work: its own callbacks, its timers,
    /// and its window.
    ///
    /// Call this about once a frame, from the thread that loaded the
    /// instrument. Nothing here blocks, and a plugin with nothing to do costs
    /// a handful of atomic reads.
    pub fn poll(&mut self) {
        if self.unloaded {
            return;
        }

        let (callback, restart) = self.instance.access_shared_handler(|shared| {
            (
                Shared::take(&shared.callback_requested),
                Shared::take(&shared.restart_requested),
            )
        });

        if callback {
            self.instance.call_on_main_thread_callback();
        }
        if restart {
            // Restarting means deactivating, rebuilding the audio half and
            // putting it back in the mixer. A plugin asks for it when its
            // latency or port layout changed, neither of which this host
            // follows, so it is reported rather than done.
            warn!(
                "'{}' asked to be restarted, which this host does not do",
                self.info.name
            );
        }
        if self.failed.load(Ordering::Acquire) && !self.reported_failure {
            self.reported_failure = true;
            error!(
                "'{}' failed while processing; it will be silent from here",
                self.info.name
            );
        }

        self.tick_timers();
        self.service_gui();
    }

    /// Fires whichever of the plugin's timers are due.
    fn tick_timers(&mut self) {
        let Some(timer) = self.extensions.timer else {
            return;
        };
        if !self.instance.access_handler(MainThread::has_timers) {
            return;
        }

        let now = Instant::now();
        let due = self
            .instance
            .access_handler(|main: &MainThread| main.due_timers(now));

        let handle = self.instance.plugin_handle();
        for id in due {
            timer.on_timer(&handle, id);
        }
    }

    /// Whether the plugin's editor is on screen.
    pub fn is_gui_open(&self) -> bool {
        self.gui_state == GuiState::Shown
    }

    /// Whether this instrument was loaded with its editor allowed.
    pub fn gui(&self) -> Gui {
        self.gui
    }

    /// Opens the plugin's editor.
    ///
    /// Only under [`Gui::Debug`]; a headless instrument returns an error saying
    /// so, rather than quietly doing nothing.
    ///
    /// Where the editor ends up is the plugin's choice more than this crate's.
    /// A plugin that can open a window of its own is asked to — that is the
    /// arrangement that costs nothing and behaves like any other window on the
    /// desktop. Most cannot: CLAP only requires a plugin to *embed*, drawing
    /// into a window the host provides, so for those a plain window is opened
    /// here, sized to what the plugin asks for, and the editor is put inside
    /// it. Either way it is a window alongside the application's rather than
    /// part of it.
    ///
    /// Opening a window is only implemented on macOS. Elsewhere, a plugin that
    /// will not float has to be given a window by the application, with
    /// [`show_gui_in`](Self::show_gui_in).
    pub fn show_gui(&mut self) -> Result<()> {
        let gui = self.editor()?;
        // Taken before the plugin's handle, which borrows this exclusively.
        let name = self.info.name.clone();
        let title = self.window_title();
        let api = platform_api()?;

        if self.gui_state == GuiState::None {
            let floating = GuiConfiguration {
                api_type: api,
                is_floating: true,
            };
            let embedded = GuiConfiguration {
                api_type: api,
                is_floating: false,
            };

            let handle = self.instance.plugin_handle();
            let kind = if gui.is_api_supported(&handle, floating) {
                gui.create(&handle, floating)
                    .map_err(|err| created(&name, err))?;
                gui.suggest_title(&handle, &title);
                debug!("'{name}' will open its editor itself");
                GuiKind::Floating
            } else if gui.is_api_supported(&handle, embedded) {
                if !window::is_supported() {
                    return Err(Error::new(format!(
                        "'{name}' only draws its editor into a window it is given, and opening \
                         one is not implemented on this platform; pass a window of your own to \
                         show_gui_in"
                    )));
                }

                gui.create(&handle, embedded)
                    .map_err(|err| created(&name, err))?;
                // Asked after `create`, because that is when a plugin works out
                // how big its editor is.
                let size = gui.get_size(&handle).unwrap_or(GuiSize {
                    width: 800,
                    height: 600,
                });

                let window =
                    match DebugWindow::open(&title.to_string_lossy(), size.width, size.height) {
                        Ok(window) => window,
                        Err(err) => {
                            gui.destroy(&handle);
                            return Err(err);
                        }
                    };

                // SAFETY: the window is kept in this `Instrument`, and is only
                // closed and dropped after the plugin has been told to destroy
                // the editor inside it — see `destroy_gui`.
                let parented = Window::from_window(window.handle())
                    .ok_or_else(|| Error::new("that window is not one a plugin can draw into"))
                    .and_then(|parent| unsafe {
                        gui.set_parent(&handle, parent).map_err(|err| {
                            Error::with_source(format!("'{name}' would not use that window"), err)
                        })
                    });
                if let Err(err) = parented {
                    gui.destroy(&handle);
                    return Err(err);
                }

                debug!(
                    "opened a {}x{} window for '{name}'",
                    size.width, size.height
                );
                self.host_window = Some(window);
                GuiKind::Embedded
            } else {
                return Err(Error::new(format!(
                    "'{name}' cannot show its editor on this platform, floating or embedded"
                )));
            };

            self.gui_kind = kind;
            self.gui_state = GuiState::Hidden;
            let owns_window = self.host_window.is_some();
            self.instance.access_shared_handler(|shared| {
                shared.owns_window.store(owns_window, Ordering::Release)
            });
        }

        let handle = self.instance.plugin_handle();
        gui.show(&handle).map_err(|err| {
            Error::with_source(format!("'{name}' could not show its editor"), err)
        })?;
        self.gui_state = GuiState::Shown;
        info!("opened the editor for '{name}'");
        Ok(())
    }

    /// Puts the plugin's editor inside a window the application provides.
    ///
    /// The portable way to see a plugin: anything implementing
    /// [`HasWindowHandle`] will do, and a winit window does, so an application
    /// that already has a windowing layer can open a second window and hand it
    /// over rather than leaving this crate to find one. It is also the only way
    /// on a platform where [`show_gui`](Self::show_gui) has no window to open.
    ///
    /// # Safety
    ///
    /// The window must stay alive, and stay a window, until this instrument's
    /// editor is destroyed — which is when it is hidden, unloaded, or dropped.
    /// A window closed before then leaves the plugin drawing into something
    /// that is no longer there.
    pub unsafe fn show_gui_in(&mut self, window: &impl HasWindowHandle) -> Result<()> {
        let gui = self.editor()?;
        let name = self.info.name.clone();
        let configuration = GuiConfiguration {
            api_type: platform_api()?,
            is_floating: false,
        };

        if self.gui_state != GuiState::None {
            return Err(Error::new(format!(
                "'{name}' already has its editor open somewhere"
            )));
        }

        let handle = self.instance.plugin_handle();
        if !gui.is_api_supported(&handle, configuration) {
            return Err(Error::new(format!(
                "'{name}' will not draw its editor into another window on this platform"
            )));
        }
        gui.create(&handle, configuration)
            .map_err(|err| created(&name, err))?;

        let parented = Window::from_window(window)
            .ok_or_else(|| Error::new("that window is not one a plugin can draw into"))
            .and_then(|parent| {
                // SAFETY: the caller has undertaken to keep the window alive for
                // as long as the editor is in it.
                unsafe { gui.set_parent(&handle, parent) }.map_err(|err| {
                    Error::with_source(format!("'{name}' would not use that window"), err)
                })
            });
        if let Err(err) = parented {
            gui.destroy(&handle);
            return Err(err);
        }

        gui.show(&handle).map_err(|err| {
            Error::with_source(format!("'{name}' could not show its editor"), err)
        })?;
        self.gui_kind = GuiKind::Embedded;
        self.gui_state = GuiState::Shown;
        info!("opened the editor for '{name}' in a window it was given");
        Ok(())
    }

    /// The size the plugin's editor would like to be, once it has one.
    pub fn gui_size(&mut self) -> Option<(u32, u32)> {
        let gui = self.extensions.gui?;
        if self.gui_state == GuiState::None {
            return None;
        }

        let handle = self.instance.plugin_handle();
        gui.get_size(&handle).map(|size| (size.width, size.height))
    }

    /// The editor extension, if this instrument is allowed one and the plugin
    /// has one.
    fn editor(&self) -> Result<PluginGui> {
        if self.gui != Gui::Debug {
            return Err(Error::new(format!(
                "'{}' was loaded headless; load it with Gui::Debug to open its window",
                self.info.name
            )));
        }
        self.extensions
            .gui
            .ok_or_else(|| Error::new(format!("'{}' has no editor to show", self.info.name)))
    }

    fn window_title(&self) -> CString {
        CString::new(format!("{} — etib", self.info.name)).unwrap_or_else(|_| c"etib".to_owned())
    }

    /// Takes the editor off the screen, keeping it built so that showing it
    /// again is immediate.
    pub fn hide_gui(&mut self) -> Result<()> {
        let Some(gui) = self.extensions.gui else {
            return Ok(());
        };
        if self.gui_state != GuiState::Shown {
            return Ok(());
        }

        let handle = self.instance.plugin_handle();
        gui.hide(&handle).map_err(|err| {
            Error::with_source(
                format!("'{}' could not hide its editor", self.info.name),
                err,
            )
        })?;
        self.gui_state = GuiState::Hidden;

        // A window of ours has nothing in it worth looking at once the editor
        // is hidden. One the application gave us is the application's to hide.
        if let Some(window) = &self.host_window {
            window.close();
        }
        Ok(())
    }

    /// Acts on what the plugin, or the user, did to the editor since the last
    /// poll.
    fn service_gui(&mut self) {
        let (show, hide, closed, destroyed, size) = self.instance.access_shared_handler(|shared| {
            (
                Shared::take(&shared.gui_show_requested),
                Shared::take(&shared.gui_hide_requested),
                Shared::take(&shared.gui_closed),
                shared.gui_was_destroyed.load(Ordering::Acquire),
                shared.gui_requested_size.swap(0, Ordering::AcqRel),
            )
        });

        // A window of ours that is no longer on screen is the user having
        // closed it. There is no callback for that, so it is noticed here.
        if self.gui_state == GuiState::Shown
            && self
                .host_window
                .as_ref()
                .is_some_and(|window| !window.is_open())
        {
            debug!("the editor window for '{}' was closed", self.info.name);
            self.destroy_gui();
            return;
        }

        if let (true, Some(window)) = (size != 0, &self.host_window) {
            window.resize(size >> 16, size & 0xFFFF);
        }

        if closed {
            debug!("'{}' closed its own editor", self.info.name);
            if destroyed {
                // The plugin tore the window down itself and is waiting for the
                // host to say it noticed.
                self.destroy_gui();
            } else {
                self.gui_state = GuiState::Hidden;
            }
            return;
        }

        if hide {
            let _ = self.hide_gui();
        }
        if show {
            if self.gui == Gui::Debug {
                if let Err(err) = self.show_gui() {
                    warn!("{err}");
                }
            } else {
                // Headless is headless, whoever asks.
                debug!(
                    "'{}' asked to show its editor; it was loaded headless",
                    self.info.name
                );
            }
        }
    }

    /// Frees the editor's resources. Safe to call when there is none.
    ///
    /// The order matters: the plugin is told to take its editor apart first,
    /// and only then does the window it was drawing into go away. The other way
    /// round leaves it drawing into something that has been freed.
    fn destroy_gui(&mut self) {
        let Some(gui) = self.extensions.gui else {
            return;
        };
        if self.gui_state == GuiState::None {
            return;
        }

        let handle = self.instance.plugin_handle();
        gui.destroy(&handle);
        self.gui_state = GuiState::None;

        self.instance
            .access_shared_handler(|shared| shared.owns_window.store(false, Ordering::Release));
        if let Some(window) = self.host_window.take() {
            window.close();
        }
    }

    /// Stops the instrument and unloads the plugin.
    ///
    /// This takes the plugin's audio half back from the audio thread before
    /// destroying it, which is what CLAP asks for and what keeps the plugin's
    /// own deallocation off the thread with the deadline. That means waiting
    /// for one block — a few milliseconds — which is why this is a call rather
    /// than something that happens quietly in a destructor.
    ///
    /// Dropping an `Instrument` does the same thing, minus the error.
    pub fn unload(mut self) -> Result<()> {
        self.teardown()
    }

    fn teardown(&mut self) -> Result<()> {
        if std::mem::replace(&mut self.unloaded, true) {
            return Ok(());
        }

        self.all_notes_off();
        self.destroy_gui();

        if self
            .commands
            .try_send(Command::RemoveInstrument(self.id))
            .is_err()
        {
            return Err(Error::new(format!(
                "could not ask the mixer to release '{}'",
                self.info.name
            )));
        }

        match self.returning.recv_timeout(TEARDOWN_TIMEOUT) {
            Ok(processor) => {
                // Already stopped, on the audio thread, before it was handed
                // back; this only unwraps what that left behind.
                let stopped = processor.into_plugin().into_stopped();
                self.instance.deactivate(stopped);
                info!("unloaded '{}'", self.info.name);
                Ok(())
            }
            Err(_) => {
                self.slot.keep();
                Err(Error::new(format!(
                    "the audio thread did not hand '{}' back within {TEARDOWN_TIMEOUT:?}; \
                     it is still loaded",
                    self.info.name
                )))
            }
        }
    }

    /// Queues one event for the audio thread.
    fn send(&self, event: Event) {
        if self.unloaded {
            return;
        }
        if self.events.try_send(event).is_err() {
            warn!(
                "'{}' is not taking events fast enough; dropped {event:?}",
                self.info.name
            );
        }
    }
}

impl Drop for Instrument {
    fn drop(&mut self) {
        if let Err(err) = self.teardown() {
            warn!("{err}");
        }
    }
}

impl std::fmt::Debug for Instrument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instrument")
            .field("name", &self.info.name)
            .field("id", &self.info.id)
            .field("channels", &self.output_channels)
            .field("gui", &self.gui)
            .finish()
    }
}

/// What to say when a plugin will not build its editor.
fn created(name: &str, err: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::with_source(format!("'{name}' could not create its editor"), err)
}

/// The windowing API a plugin's editor is put in on this platform.
fn platform_api() -> Result<GuiApiType<'static>> {
    GuiApiType::default_for_current_platform()
        .ok_or_else(|| Error::new("this platform has no windowing API a plugin could use"))
}

/// Loads a `.clap` file's entry point.
///
/// # Safety
///
/// Runs code from `path`. See [`plugins_in`].
unsafe fn load_entry(path: &Path) -> Result<PluginEntry> {
    // SAFETY: the caller has taken on the trust decision.
    unsafe { PluginEntry::load(path) }
        .map_err(|err| Error::with_source(format!("could not load {}", path.display()), err))
}

fn plugin_factory<'a>(entry: &'a PluginEntry, path: &Path) -> Result<PluginFactory<'a>> {
    entry.get_plugin_factory().ok_or_else(|| {
        Error::new(format!(
            "{} is a CLAP file with no plugins in it",
            path.display()
        ))
    })
}

/// Picks the plugin to instantiate: the one that was asked for, or the first
/// instrument in the file.
fn choose<'a>(
    factory: &PluginFactory<'a>,
    wanted: Option<&str>,
    path: &Path,
) -> Result<&'a PluginDescriptor> {
    let matches = |descriptor: &PluginDescriptor| match wanted {
        Some(wanted) => descriptor
            .id()
            .is_some_and(|id| id.to_bytes() == wanted.as_bytes()),
        None => is_instrument(descriptor),
    };

    if let Some(descriptor) = factory.plugin_descriptors().find(|d| matches(d)) {
        return Ok(descriptor);
    }
    if wanted.is_none() {
        // Nothing in the file admits to being an instrument; take the first and
        // let the warning in `load_instrument` say so.
        if let Some(descriptor) = factory.plugin_descriptors().next() {
            return Ok(descriptor);
        }
    }

    Err(match wanted {
        Some(wanted) => Error::new(format!(
            "{} holds no plugin with the id '{wanted}'; it has {}",
            path.display(),
            listing(factory)
        )),
        None => Error::new(format!("{} holds no plugins at all", path.display())),
    })
}

fn listing(factory: &PluginFactory) -> String {
    let ids: Vec<String> = factory
        .plugin_descriptors()
        .filter_map(|descriptor| Some(text(descriptor.id()?)))
        .collect();
    match ids.is_empty() {
        true => "none".to_owned(),
        false => ids.join(", "),
    }
}

fn is_instrument(descriptor: &PluginDescriptor) -> bool {
    descriptor
        .features()
        .any(|feature| feature.to_bytes() == b"instrument")
}

fn describe(descriptor: &PluginDescriptor) -> PluginInfo {
    PluginInfo {
        id: descriptor.id().map(text).unwrap_or_default(),
        name: descriptor.name().map(text).unwrap_or_default(),
        vendor: descriptor.vendor().map(text).unwrap_or_default(),
        version: descriptor.version().map(text).unwrap_or_default(),
        is_instrument: is_instrument(descriptor),
    }
}

fn text(value: &CStr) -> String {
    value.to_string_lossy().into_owned()
}

/// How many channels the plugin's main input and output ports have.
///
/// Only one port each way is connected — an instrument has one — so this looks
/// for the port flagged as the main one and falls back to the first.
fn port_layout(instance: &mut PluginInstance<Host>) -> (usize, usize) {
    let handle = instance.plugin_handle();
    let Some(ports) = handle.get_extension::<PluginAudioPorts>() else {
        // A plugin that declares no ports at all is taken to be stereo out,
        // which is what the extension's absence has always meant in practice.
        return (0, 2);
    };

    let mut buffer = AudioPortInfoBuffer::new();
    let mut first_port = |is_input: bool| -> usize {
        for index in 0..ports.count(&handle, is_input) {
            if let Some(info) = ports.get(&handle, index, is_input, &mut buffer) {
                return info.channel_count as usize;
            }
        }
        0
    };

    (first_port(true), first_port(false))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A path that holds no plugin is an error with the loader's own complaint
    /// kept underneath it, not a panic.
    #[test]
    fn a_file_that_is_not_there_is_reported() {
        // SAFETY: there is nothing at this path to run.
        let err = unsafe { plugins_in("/nowhere/at/all/nothing.clap") }.unwrap_err();

        assert!(err.to_string().contains("could not load"), "{err}");
        assert!(
            std::error::Error::source(&err).is_some(),
            "the loader's own message should be kept as the source"
        );
    }

    /// A plugin is activated for a device's sample rate and block size, and a
    /// silent `Audio` has neither.
    #[test]
    fn a_silent_audio_cannot_load_one() {
        let audio = crate::Audio::silent();
        // SAFETY: it fails before anything is loaded.
        let err = unsafe { audio.load_instrument("anything.clap", &InstrumentConfig::default()) }
            .unwrap_err();

        assert!(err.to_string().contains("no audio device"), "{err}");
    }

    /// The thing this is easiest to get wrong: an instrument that opens a
    /// window nobody asked for.
    #[test]
    fn the_default_is_headless() {
        assert_eq!(Gui::default(), Gui::Headless);
        assert_eq!(InstrumentConfig::default().gui, Gui::Headless);
    }
}
