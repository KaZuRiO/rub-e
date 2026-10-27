//! A window for a plugin's editor to live in, for the debugging case where the
//! plugin will not open one of its own.
//!
//! CLAP has two ways for a plugin to show its editor. In a *floating* window it
//! opens one itself and the host has nothing to provide; *embedded*, the host
//! hands it a window to draw into. Floating is the one that costs this crate
//! nothing, but it is optional for a plugin and plenty of real ones — Surge XT
//! among them — only embed. So there has to be a window, and since the whole
//! point of [`Gui::Debug`](super::Gui) is to see a plugin while working on
//! something, waiting for the application to provide one would defeat it.
//!
//! This is therefore the one corner of the crate that touches the platform's
//! windowing, and it is kept to exactly that: open a window of a given size,
//! hand out its handle, resize it, ask whether it is still on screen, close it.
//! Anything more — moving it, keyboard focus, a place in the application's own
//! window — is [`Instrument::show_gui_in`](super::Instrument::show_gui_in),
//! where the application provides the window and this knows nothing about it.
//!
//! Only macOS is implemented. On the rest, `show_gui` says so and points at
//! `show_gui_in`, rather than carrying platform code that has never been run.

use crate::error::{Error, Result};

#[cfg(target_os = "macos")]
mod platform {
    use objc2::rc::Retained;
    use objc2::{MainThreadMarker, MainThreadOnly};
    use objc2_app_kit::{
        NSApplication, NSApplicationActivationPolicy, NSBackingStoreType, NSWindow,
        NSWindowStyleMask,
    };
    use objc2_foundation::{NSPoint, NSRect, NSSize, NSString};
    use raw_window_handle::{
        AppKitWindowHandle, HandleError, HasWindowHandle, RawWindowHandle, WindowHandle,
    };
    use std::cell::Cell;
    use std::ptr::NonNull;

    use crate::error::{Error, Result};

    pub(super) struct Window {
        window: Retained<NSWindow>,
        /// Whether the window has ever actually been on screen.
        ///
        /// Ordering a window to the front is a request to the platform, not
        /// something that has happened by the time the call returns, so for the
        /// first moments of its life a window that is perfectly fine reports
        /// that it is not visible. Without this, the first poll would read that
        /// as the user having closed it.
        seen: Cell<bool>,
        /// Proof that this was made on the main thread, which is where every
        /// one of these calls has to happen.
        mtm: MainThreadMarker,
    }

    impl Window {
        pub fn open(title: &str, width: u32, height: u32) -> Result<Self> {
            let mtm = MainThreadMarker::new().ok_or_else(|| {
                Error::new("a plugin's window can only be opened on the main thread")
            })?;

            // A program that has not been through a windowing framework is not
            // yet an application as far as the platform is concerned, and
            // nothing it opens would be shown. Saying so here is what makes
            // this work in a program with no interface of its own; in one that
            // has been through winit this has already happened and is a no-op.
            let app = NSApplication::sharedApplication(mtm);
            if app.activationPolicy() == NSApplicationActivationPolicy::Prohibited {
                app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
                // Without this, a program that has never been through a
                // windowing framework never finishes starting up, and anything
                // it orders onto the screen is held back. A framework that has
                // already done this leaves the policy alone, so this only runs
                // where nothing else has taken care of it.
                app.finishLaunching();
            }

            let frame = NSRect::new(
                NSPoint::new(0.0, 0.0),
                NSSize::new(f64::from(width), f64::from(height)),
            );
            let style = NSWindowStyleMask::Titled
                | NSWindowStyleMask::Closable
                | NSWindowStyleMask::Miniaturizable
                | NSWindowStyleMask::Resizable;

            // SAFETY: a plain window with a standard style and a backing store,
            // created on the main thread as AppKit requires.
            let window = unsafe {
                NSWindow::initWithContentRect_styleMask_backing_defer(
                    NSWindow::alloc(mtm),
                    frame,
                    style,
                    NSBackingStoreType::Buffered,
                    false,
                )
            };

            window.setTitle(&NSString::from_str(title));
            // Closing the window must not release it: this holds a reference to
            // it for as long as the plugin's editor is inside, and a window
            // that released itself on close would leave that dangling.
            //
            // SAFETY: turning this off is what makes the reference above sound;
            // it is the platform's own advice for a window a program keeps.
            unsafe { window.setReleasedWhenClosed(false) };
            window.center();
            window.makeKeyAndOrderFront(None);
            app.activate();

            Ok(Self {
                window,
                seen: Cell::new(false),
                mtm,
            })
        }

        pub fn resize(&self, width: u32, height: u32) {
            self.window
                .setContentSize(NSSize::new(f64::from(width), f64::from(height)));
        }

        /// Whether the window is still on screen. This is how a user closing it
        /// is noticed, without a delegate to be called back by.
        pub fn is_open(&self) -> bool {
            if self.window.isVisible() {
                self.seen.set(true);
                return true;
            }
            // Not visible and never has been: it is still on its way up.
            !self.seen.get()
        }

        pub fn close(&self) {
            let _ = self.mtm;
            self.window.orderOut(None);
        }
    }

    impl HasWindowHandle for Window {
        fn window_handle(&self) -> std::result::Result<WindowHandle<'_>, HandleError> {
            let view = self.window.contentView().ok_or(HandleError::Unavailable)?;

            let view = NonNull::new(Retained::as_ptr(&view).cast_mut().cast())
                .ok_or(HandleError::Unavailable)?;

            // SAFETY: the view belongs to the window this borrows from, and
            // lives for as long as it does.
            unsafe {
                Ok(WindowHandle::borrow_raw(RawWindowHandle::AppKit(
                    AppKitWindowHandle::new(view),
                )))
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use crate::error::{Error, Result};
    use raw_window_handle::{HandleError, HasWindowHandle, WindowHandle};

    /// A window this platform's implementation does not exist for.
    pub(super) struct Window(std::convert::Infallible);

    impl Window {
        pub fn open(_title: &str, _width: u32, _height: u32) -> Result<Self> {
            Err(Error::new(
                "opening a window for a plugin's editor is only implemented on macOS; \
                 pass one of your own to show_gui_in instead",
            ))
        }

        pub fn resize(&self, _width: u32, _height: u32) {
            match self.0 {}
        }

        pub fn is_open(&self) -> bool {
            match self.0 {}
        }

        pub fn close(&self) {
            match self.0 {}
        }
    }

    impl HasWindowHandle for Window {
        fn window_handle(&self) -> std::result::Result<WindowHandle<'_>, HandleError> {
            match self.0 {}
        }
    }
}

/// A window opened by this crate for a plugin's editor.
pub(crate) struct DebugWindow {
    window: platform::Window,
}

impl DebugWindow {
    /// Opens a window of `width` by `height` points, centred and in front.
    pub fn open(title: &str, width: u32, height: u32) -> Result<Self> {
        // A plugin that cannot say how big its editor is gets something it can
        // be resized away from.
        let width = width.clamp(160, 4096);
        let height = height.clamp(120, 4096);

        Ok(Self {
            window: platform::Window::open(title, width, height)?,
        })
    }

    pub fn resize(&self, width: u32, height: u32) {
        self.window
            .resize(width.clamp(160, 4096), height.clamp(120, 4096));
    }

    /// Whether it is still on screen, or the user has closed it.
    pub fn is_open(&self) -> bool {
        self.window.is_open()
    }

    pub fn close(&self) {
        self.window.close();
    }

    /// The handle a plugin is given to draw into.
    pub fn handle(&self) -> &impl raw_window_handle::HasWindowHandle {
        &self.window
    }
}

/// Whether this platform has a window implementation at all, for the message
/// `show_gui` gives when it does not.
pub(crate) fn is_supported() -> bool {
    cfg!(target_os = "macos")
}

// Keeps the unused-import warning away on the platforms where `Error` is only
// used by the other branch.
const _: fn() -> Result<()> = || Err(Error::new(""));
