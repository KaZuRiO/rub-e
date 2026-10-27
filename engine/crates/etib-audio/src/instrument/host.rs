//! The host side of the CLAP contract: what a plugin is allowed to ask of us.
//!
//! A CLAP plugin talks back to its host through a set of callbacks, split by
//! which thread each one may be called from. That split is the whole reason
//! this file is shaped the way it is:
//!
//! | Handler | Called from | Holds |
//! |---|---|---|
//! | [`Shared`] | any thread | atomics only — a request is recorded here and acted on later, on the right thread |
//! | [`MainThread`] | the thread that loaded the plugin | the timers it registered |
//! | audio processor | the device's thread | nothing; this host asks nothing of it |
//!
//! Nothing a plugin asks for is done where it is asked. A request to show a
//! window, to restart, or to be called back arrives as a flag, and
//! [`Instrument::poll`](crate::Instrument::poll) acts on it from the thread
//! that is allowed to.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use clack_extensions::audio_ports::{AudioPortRescanFlags, HostAudioPorts, HostAudioPortsImpl};
use clack_extensions::gui::{GuiSize, HostGui, HostGuiImpl};
use clack_extensions::log::{HostLog, HostLogImpl, LogSeverity};
use clack_extensions::note_ports::{
    HostNotePorts, HostNotePortsImpl, NoteDialects, NotePortRescanFlags,
};
use clack_extensions::params::{
    HostParams, HostParamsImplMainThread, HostParamsImplShared, ParamClearFlags, ParamRescanFlags,
};
use clack_extensions::thread_check::{HostThreadCheck, HostThreadCheckImpl};
use clack_extensions::timer::{HostTimer, HostTimerImpl, TimerId};
use clack_host::prelude::*;
use log::{debug, error, info, trace, warn};

thread_local! {
    /// Set by the mixer on the thread the device calls it on, so that
    /// [`is_audio_thread`](HostThreadCheckImpl::is_audio_thread) can answer
    /// exactly rather than by elimination.
    static ON_AUDIO_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// Called once per block by the mixer, on the device's thread. Reading and
/// writing a thread-local `Cell` is a register operation with no lock and no
/// allocation behind it.
pub(crate) fn mark_audio_thread() {
    ON_AUDIO_THREAD.with(|flag| flag.set(true));
}

/// The set of handlers, tying the three thread contexts together.
pub(crate) struct Host;

impl HostHandlers for Host {
    type Shared<'a> = Shared;
    type MainThread<'a> = MainThread;
    /// This host asks nothing of the plugin from the audio thread beyond
    /// `process`, which is not an extension.
    type AudioProcessor<'a> = ();

    fn declare_extensions(builder: &mut HostExtensions<Self>, _shared: &Self::Shared<'_>) {
        builder
            .register::<HostLog>()
            .register::<HostThreadCheck>()
            .register::<HostAudioPorts>()
            .register::<HostNotePorts>()
            .register::<HostParams>()
            .register::<HostTimer>()
            .register::<HostGui>();
    }
}

/// What the plugin has asked for, from whichever thread it asked.
///
/// Every field is an atomic and every method returns immediately: these are
/// called from the plugin's own code, sometimes from inside `process`, and a
/// host that took a lock in one of them would stall the audio thread.
pub(crate) struct Shared {
    /// Named in log lines, since several plugins may be loaded at once.
    name: String,
    /// The thread the plugin was loaded on, which is the only one allowed to
    /// call most of its functions.
    main_thread: ThreadId,

    pub restart_requested: AtomicBool,
    pub callback_requested: AtomicBool,

    pub gui_show_requested: AtomicBool,
    pub gui_hide_requested: AtomicBool,
    /// The plugin's window has gone away by itself — the user closed it.
    pub gui_closed: AtomicBool,
    /// Set with [`gui_closed`](Self::gui_closed) when the plugin destroyed the
    /// window on its way out, which the host has to acknowledge.
    pub gui_was_destroyed: AtomicBool,
    /// A size the plugin would like its window to be, packed as width and
    /// height. Acted on in `poll`, and only when the window is one this host
    /// opened; a window the application provided is the application's to size.
    pub gui_requested_size: AtomicU32,
    /// Whether the editor is in a window this host opened and can resize.
    pub owns_window: AtomicBool,
}

impl Shared {
    pub fn new(name: String) -> Self {
        Self {
            name,
            main_thread: std::thread::current().id(),
            restart_requested: AtomicBool::new(false),
            callback_requested: AtomicBool::new(false),
            gui_show_requested: AtomicBool::new(false),
            gui_hide_requested: AtomicBool::new(false),
            gui_closed: AtomicBool::new(false),
            gui_was_destroyed: AtomicBool::new(false),
            gui_requested_size: AtomicU32::new(0),
            owns_window: AtomicBool::new(false),
        }
    }

    /// Reads a flag and clears it, so that each request is acted on once.
    pub fn take(flag: &AtomicBool) -> bool {
        flag.swap(false, Ordering::AcqRel)
    }
}

impl SharedHandler<'_> for Shared {
    fn request_restart(&self) {
        self.restart_requested.store(true, Ordering::Release);
    }

    /// The plugin wants to be processing. It already is, for as long as it is
    /// loaded — this host has no concept of a plugin that is loaded and idle.
    fn request_process(&self) {}

    fn request_callback(&self) {
        self.callback_requested.store(true, Ordering::Release);
    }
}

impl HostLogImpl for Shared {
    /// Forwards the plugin's own diagnostics to the `log` facade.
    ///
    /// CLAP allows this to be called from any thread, including from inside
    /// `process`, where going through `log` may take a lock — the one place in
    /// this crate where something on the audio thread might. It is forwarded
    /// anyway: a plugin that logs from its process callback has already made
    /// that choice, and swallowing what it says would leave whoever is
    /// debugging it with nothing.
    fn log(&self, severity: LogSeverity, message: &str) {
        let name = &self.name;
        match severity {
            LogSeverity::Debug => debug!("{name}: {message}"),
            LogSeverity::Info => info!("{name}: {message}"),
            LogSeverity::Warning => warn!("{name}: {message}"),
            LogSeverity::Error | LogSeverity::Fatal => error!("{name}: {message}"),
            LogSeverity::HostMisbehaving => {
                error!("{name} reports that this host misbehaved: {message}")
            }
            LogSeverity::PluginMisbehaving => warn!("{name} misbehaved: {message}"),
        }
    }
}

impl HostThreadCheckImpl for Shared {
    fn is_main_thread(&self) -> bool {
        std::thread::current().id() == self.main_thread
    }

    fn is_audio_thread(&self) -> bool {
        ON_AUDIO_THREAD.with(Cell::get)
    }
}

impl HostGuiImpl for Shared {
    fn resize_hints_changed(&self) {
        trace!("{}: resize hints changed", self.name);
    }

    /// Only an embedded window can be resized by the host, and this host only
    /// offers floating ones, so the request is recorded for a log line and
    /// declined. A plugin whose own window is floating resizes it itself.
    fn request_resize(&self, new_size: GuiSize) -> Result<(), HostError> {
        let packed = (new_size.width.min(0xFFFF) << 16) | new_size.height.min(0xFFFF);
        self.gui_requested_size.store(packed, Ordering::Relaxed);
        Err(HostError::Message(
            "this host does not resize plugin windows",
        ))
    }

    fn request_show(&self) -> Result<(), HostError> {
        self.gui_show_requested.store(true, Ordering::Release);
        Ok(())
    }

    fn request_hide(&self) -> Result<(), HostError> {
        self.gui_hide_requested.store(true, Ordering::Release);
        Ok(())
    }

    fn closed(&self, was_destroyed: bool) {
        self.gui_was_destroyed
            .store(was_destroyed, Ordering::Release);
        self.gui_closed.store(true, Ordering::Release);
    }
}

impl HostParamsImplShared for Shared {
    /// The plugin wants its parameter values read back. They are carried by
    /// `process`, which runs continuously here, so the next block is the flush.
    fn request_flush(&self) {}
}

/// What may only be asked from the thread that loaded the plugin.
pub(crate) struct MainThread {
    timers: RefCell<Vec<Timer>>,
    next_timer: Cell<u32>,
}

/// A timer the plugin registered, which the host is expected to tick.
struct Timer {
    id: TimerId,
    period: Duration,
    due: Instant,
}

impl MainThread {
    pub fn new() -> Self {
        Self {
            timers: RefCell::new(Vec::new()),
            next_timer: Cell::new(0),
        }
    }

    /// The timers that are due now, and when each is next due.
    ///
    /// Returned as a list rather than ticked here, because ticking one means
    /// calling into the plugin, and this is called while the plugin's own
    /// handle is borrowed.
    pub fn due_timers(&self, now: Instant) -> Vec<TimerId> {
        let mut timers = self.timers.borrow_mut();
        let mut due = Vec::new();
        for timer in timers.iter_mut() {
            if now >= timer.due {
                due.push(timer.id);
                // From now, not from when it was due: a host that fell behind
                // should not then fire the same timer several times in a row.
                timer.due = now + timer.period;
            }
        }
        due
    }

    pub fn has_timers(&self) -> bool {
        !self.timers.borrow().is_empty()
    }
}

impl MainThreadHandler<'_> for MainThread {}

impl HostTimerImpl for MainThread {
    fn register_timer(&self, period_ms: u32) -> Result<TimerId, HostError> {
        // A plugin asking for a 1 ms timer gets one tick per frame, because
        // that is how often this host is polled. The floor keeps a plugin from
        // asking for something that only looks like it was granted.
        let period = Duration::from_millis(period_ms.max(15) as u64);

        let id = TimerId(self.next_timer.get());
        self.next_timer.set(id.0 + 1);
        self.timers.borrow_mut().push(Timer {
            id,
            period,
            due: Instant::now() + period,
        });

        debug!("plugin registered timer {id} every {period:?}");
        Ok(id)
    }

    fn unregister_timer(&self, timer_id: TimerId) -> Result<(), HostError> {
        let mut timers = self.timers.borrow_mut();
        match timers.iter().position(|timer| timer.id == timer_id) {
            Some(index) => {
                timers.remove(index);
                debug!("plugin unregistered timer {timer_id}");
                Ok(())
            }
            None => Err(HostError::Message("no such timer")),
        }
    }
}

impl HostAudioPortsImpl for MainThread {
    /// Changing the port layout means deactivating and reactivating the plugin,
    /// which would mean taking it out of the mixer and putting it back. An
    /// instrument's layout does not change in practice, so this host says no
    /// rather than implementing a path it cannot test.
    fn is_rescan_flag_supported(&self, _flag: AudioPortRescanFlags) -> bool {
        false
    }

    fn rescan(&self, flags: AudioPortRescanFlags) {
        warn!("plugin asked for an audio port rescan ({flags:?}); this host cannot do that");
    }
}

impl HostNotePortsImpl for MainThread {
    /// CLAP's own note events are what this host sends. MIDI is declared too,
    /// because a plugin that only accepts MIDI would otherwise be told it has
    /// no way to be played at all — it is up to the plugin to pick.
    fn supported_dialects(&self) -> NoteDialects {
        NoteDialects::CLAP | NoteDialects::MIDI
    }

    fn rescan(&self, _flags: NotePortRescanFlags) {
        trace!("plugin asked for a note port rescan");
    }
}

impl HostParamsImplMainThread for MainThread {
    fn rescan(&self, flags: ParamRescanFlags) {
        trace!("plugin rescanned its parameters ({flags:?})");
    }

    fn clear(&self, _param_id: ClapId, _flags: ParamClearFlags) {}
}
