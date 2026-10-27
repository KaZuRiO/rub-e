//! Entry point: the backend connection, and opening a device for a window.

use std::fmt::Debug;
use std::sync::Arc;

use log::info;
use wgpu::rwh::{HasDisplayHandle, HasWindowHandle};

use crate::device::Device;
use crate::display::Hdr;
use crate::error::{Error, Result};
use crate::surface::Surface;

/// The graphics backend connection. One per process is plenty.
pub struct Instance {
    inner: wgpu::Instance,
}

impl Instance {
    /// Connects to the platform's graphics backends.
    ///
    /// `display` is the windowing system's display connection, which the
    /// backend wants before any window exists and which must outlive it — with
    /// winit, that is `event_loop.owned_display_handle()`.
    pub fn new(display: impl HasDisplayHandle + Debug + Send + Sync + 'static) -> Self {
        let inner = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(display)),
        );
        info!(
            "graphics instance created; backends available: {:?}",
            wgpu::Instance::enabled_backend_features()
        );
        Self { inner }
    }

    /// Creates a surface for `window`, picks an adapter that can present to it,
    /// opens a device, and configures the swapchain at `width` x `height`.
    ///
    /// `hdr` says whether to take an extended-range surface when the platform
    /// offers one; [`Surface::color_space`] reports what was actually
    /// configured, which is what the application has to encode for.
    ///
    /// The window is kept alive by the returned [`Surface`] for as long as it
    /// exists, so the caller may drop its own handle.
    ///
    /// This blocks while the adapter and device are requested. On native
    /// backends that resolves immediately; it is not suitable for the web.
    pub fn open<W>(
        &self,
        window: Arc<W>,
        width: u32,
        height: u32,
        hdr: Hdr,
    ) -> Result<(Device, Surface)>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        let surface = self
            .inner
            .create_surface(window)
            .map_err(|err| Error::new("failed to create a surface for the window", err))?;

        // Device creation is async but resolves immediately on native backends,
        // so blocking here costs nothing and keeps callers synchronous.
        let adapter = pollster::block_on(self.inner.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            // Exact adapter limits: this is a trusted native app, not a sandbox
            // that needs fingerprinting resistance.
            apply_limit_buckets: false,
        }))
        .map_err(|err| Error::new("no graphics adapter supports this surface", err))?;

        // Ask for nothing beyond the downlevel defaults and stay portable; a
        // compute pass and a fullscreen blit need no optional features.
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("etib device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            ..Default::default()
        }))
        .map_err(|err| Error::new("failed to open a graphics device", err))?;
        info!("device opened");

        let device = Device::new(device, queue, &adapter);
        let surface = Surface::new(surface, &adapter, device.clone(), width, height, hdr);

        Ok((device, surface))
    }
}
