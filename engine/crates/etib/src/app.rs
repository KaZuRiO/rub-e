//! Window lifecycle, event routing and the frame loop.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use etib_core::{Acquired, Encoder, Hdr, Instance, TextureView, Vsync};
use log::{debug, error, info, trace};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Fullscreen, Window, WindowId};

use crate::gpu::Gpu;

/// How the window is created. Everything else about the application lives in
/// the [`App`] implementation.
pub struct Config {
    pub title: String,
    /// Initial inner size, in logical pixels.
    pub size: (u32, u32),
    /// Exit when Escape is pressed and the [`App`] did not consume the key.
    pub exit_on_escape: bool,
    /// Whether to ask for an extended-range (HDR) surface.
    ///
    /// What was actually configured is [`Gpu::color_space`], which an
    /// application has to respect: on an extended-range surface, 1.0 means SDR
    /// white rather than the brightest thing the display can show.
    pub hdr: Hdr,
    /// Open borderless fullscreen on the current monitor, as the engine always
    /// did, rather than as a window of [`size`](Self::size).
    pub fullscreen: bool,
    /// Whether presentation waits for the display's refresh: the engine's
    /// `vsync` setting. Reading it from a settings file is the application's
    /// business.
    pub vsync: Vsync,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            title: "etib".to_owned(),
            size: (1280, 800),
            exit_on_escape: true,
            hdr: Hdr::Allow,
            fullscreen: false,
            vsync: Vsync::On,
        }
    }
}

/// Everything the runner hands to [`App::render`] for one frame.
pub struct Frame<'a> {
    pub gpu: &'a Gpu,
    /// The frame's encoder. The runner submits it once `render` returns.
    pub encoder: &'a mut Encoder,
    /// View of the swapchain image this frame draws into.
    pub view: &'a TextureView,
    pub stats: FrameStats<'a>,
}

/// Read-only measurements of the frame in flight, useful for an overlay.
#[derive(Clone, Debug)]
pub struct FrameStats<'a> {
    /// Smoothed frame rate; a raw `1 / dt` reads as noise.
    pub fps: f32,
    pub frame_index: u64,
    pub surface_size: (u32, u32),
    /// The graphics backend in use, formatted for display.
    pub backend: &'a str,
}

/// The longest step [`App::update`] is ever handed, in seconds.
///
/// A frame that took longer — the window was dragged, the process stopped at a
/// breakpoint, a level was loading — is reported as this much time instead, so
/// that a simulation advanced by `dt` does not jump through walls and floors to
/// catch up. Time is lost rather than compressed.
pub const MAX_DT: f32 = 0.1;

/// The `dt` [`App::update`] is given for a frame that took `elapsed`.
fn frame_dt(elapsed: Duration) -> f32 {
    elapsed.as_secs_f32().min(MAX_DT)
}

/// The application behind the frame loop.
///
/// The runner owns the window, the surface and the frame pacing; an
/// implementation owns its pipelines and its state.
pub trait App {
    /// Offered every window event before the runner acts on it. Return `true`
    /// when the event was consumed, e.g. because egui used it, which suppresses
    /// the runner's own key handling for that event.
    fn window_event(&mut self, _window: &Window, _event: &WindowEvent) -> bool {
        false
    }

    /// Offered every raw device event, such as relative mouse motion — what
    /// a first-person camera wants, since it keeps coming once the cursor is
    /// captured and has stopped moving. These are not tied to the window and
    /// arrive whether or not it has focus, so an application usually only
    /// acts on them while it has captured the cursor.
    fn device_event(&mut self, _event: &DeviceEvent) {}

    /// The surface has been reconfigured to a new, non-degenerate size.
    fn resize(&mut self, _gpu: &Gpu, _width: u32, _height: u32) {}

    /// Advances state by `dt` seconds, before the frame is recorded. `dt` is
    /// never more than [`MAX_DT`].
    fn update(&mut self, _dt: f32) {}

    /// Whether the application wants the frame loop to end. Asked after every
    /// [`update`](Self::update) and every window event; returning `true` exits
    /// the loop, drops the application where it stands, and returns from
    /// [`run`].
    fn exit_requested(&self) -> bool {
        false
    }

    /// Records this frame's GPU work into `frame.encoder`.
    fn render(&mut self, frame: Frame<'_>);
}

/// Opens a window and runs the frame loop until the application exits.
///
/// `build` is called once, when the window has been created and the device
/// and surface exist. A later `resumed` keeps the application it built.
///
/// # Errors
///
/// If the event loop cannot be created or exits with an error, and if opening
/// the window, the graphics device or `build` fails — the error that stopped
/// startup, with what was being attempted as its context.
pub fn run<A, F>(config: Config, build: F) -> Result<()>
where
    A: App,
    F: FnMut(&Gpu) -> Result<A>,
{
    let event_loop = EventLoop::new().context("failed to create the winit event loop")?;

    // The graphics backend wants the platform display connection up front.
    // winit's owned handle is the right thing to hand it, and it outlives the
    // event loop.
    let instance = Instance::new(event_loop.owned_display_handle());

    let mut runner = Runner {
        instance,
        config,
        build,
        state: None,
        last_frame: Instant::now(),
        frame_index: 0,
        fps: 0.0,
        last_fps_log: Instant::now(),
        failure: None,
    };

    event_loop
        .run_app(&mut runner)
        .context("the event loop exited with an error")?;

    if let Some(err) = runner.failure {
        return Err(err);
    }

    info!("clean shutdown");
    Ok(())
}

/// Everything that only exists once a window has been created. On some
/// platforms that happens after startup, hence the Option in Runner.
struct State<A> {
    gpu: Gpu,
    app: A,
}

struct Runner<A, F> {
    instance: Instance,
    config: Config,
    build: F,
    state: Option<State<A>>,
    last_frame: Instant,
    frame_index: u64,
    fps: f32,
    last_fps_log: Instant,
    /// Why startup stopped, for `run` to return once the loop has exited.
    failure: Option<anyhow::Error>,
}

impl<A, F> Runner<A, F>
where
    A: App,
    F: FnMut(&Gpu) -> Result<A>,
{
    /// Runs one frame: acquire, let the app record its passes, submit, present.
    fn render(&mut self, event_loop: &ActiveEventLoop) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        let now = Instant::now();
        let elapsed = now.duration_since(self.last_frame);
        let dt = elapsed.as_secs_f32();
        self.last_frame = now;

        // Exponential moving average; dt can spike badly on the first frame or
        // after the window has been dragged.
        if dt > 0.0 {
            let instant_fps = 1.0 / dt;
            self.fps = if self.fps == 0.0 {
                instant_fps
            } else {
                self.fps * 0.9 + instant_fps * 0.1
            };
        }

        state.app.update(frame_dt(elapsed));
        if state.app.exit_requested() {
            info!("the application asked to exit");
            event_loop.exit();
            return;
        }

        self.frame_index += 1;
        if now.duration_since(self.last_fps_log).as_secs_f32() >= 1.0 {
            info!("frame {} | {:.1} fps", self.frame_index, self.fps);
            self.last_fps_log = now;
        }
        trace!(
            "frame {} begin (dt {:.2} ms)",
            self.frame_index,
            dt * 1000.0
        );

        let frame = match state.gpu.surface.acquire() {
            Acquired::Frame(frame) => frame,
            // The surface has already recovered as far as it can; try again.
            Acquired::Skip => {
                state.gpu.window.request_redraw();
                return;
            }
            Acquired::Fatal => {
                error!("the surface is unusable; shutting down");
                event_loop.exit();
                return;
            }
        };

        let stats = FrameStats {
            fps: self.fps,
            frame_index: self.frame_index,
            surface_size: state.gpu.size(),
            backend: state.gpu.device.backend(),
        };

        let mut encoder = state.gpu.device.encoder("frame encoder");
        state.app.render(Frame {
            gpu: &state.gpu,
            encoder: &mut encoder,
            view: frame.view(),
            stats,
        });

        state.gpu.device.submit(encoder);
        frame.present(&state.gpu.device);
        trace!("frame {} presented", self.frame_index);

        // Animation is continuous, so keep asking for the next frame.
        state.gpu.window.request_redraw();
    }
}

impl<A, F> ApplicationHandler for Runner<A, F>
where
    A: App,
    F: FnMut(&Gpu) -> Result<A>,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            debug!("resumed with an existing renderer; nothing to rebuild");
            return;
        }

        info!("creating window");
        let attributes = Window::default_attributes()
            .with_title(self.config.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.config.size.0,
                self.config.size.1,
            ))
            .with_fullscreen(
                self.config
                    .fullscreen
                    .then_some(Fullscreen::Borderless(None)),
            );

        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                self.failure = Some(anyhow::Error::new(err).context("failed to create the window"));
                event_loop.exit();
                return;
            }
        };

        let gpu = match Gpu::new(
            &self.instance,
            window.clone(),
            self.config.hdr,
            self.config.vsync,
        ) {
            Ok(gpu) => gpu,
            Err(err) => {
                self.failure = Some(err.context("graphics initialisation failed"));
                event_loop.exit();
                return;
            }
        };

        let app = match (self.build)(&gpu) {
            Ok(app) => app,
            Err(err) => {
                self.failure = Some(err.context("application setup failed"));
                event_loop.exit();
                return;
            }
        };

        self.state = Some(State { gpu, app });

        self.last_frame = Instant::now();
        self.last_fps_log = Instant::now();
        info!("pipeline ready; entering the frame loop");
        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        // The app gets first refusal on every event, so its UI can capture
        // clicks and keystrokes before the runner reacts to them.
        let window = state.gpu.window.clone();
        let consumed = state.app.window_event(&window, &event);
        if state.app.exit_requested() {
            info!("the application asked to exit");
            event_loop.exit();
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                info!("close requested; shutting down");
                event_loop.exit();
            }
            WindowEvent::Resized(size) if state.gpu.surface.resize(size.width, size.height) => {
                state.app.resize(&state.gpu, size.width, size.height);
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                info!("scale factor changed to {scale_factor:.2}");
            }
            WindowEvent::RedrawRequested => {
                self.render(event_loop);
            }
            WindowEvent::KeyboardInput { event, .. } if !consumed && self.config.exit_on_escape => {
                use winit::keyboard::{Key, NamedKey};
                if event.state.is_pressed() && event.logical_key == Key::Named(NamedKey::Escape) {
                    info!("escape pressed; shutting down");
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(state) = self.state.as_mut() {
            state.app.device_event(&event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Animate continuously rather than only on input.
        event_loop.set_control_flow(ControlFlow::Poll);
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        info!(
            "event loop exiting after {} frames; releasing GPU resources",
            self.frame_index
        );
        self.state = None;
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn frame_dt_is_clamped() {
        assert_eq!(frame_dt(Duration::ZERO), 0.0);
        assert_eq!(frame_dt(Duration::from_millis(16)), 0.016);
        assert_eq!(frame_dt(Duration::from_secs(3)), MAX_DT);
    }

    #[test]
    fn an_app_does_not_ask_to_exit_by_default() {
        struct Nothing;
        impl App for Nothing {
            fn render(&mut self, _: Frame<'_>) {}
        }
        assert!(!Nothing.exit_requested());
    }
}
