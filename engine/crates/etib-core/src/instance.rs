//! Entry point: the backend connection, and opening a device for a window.

use std::fmt::Debug;
use std::sync::Arc;

use log::info;
use wgpu::rwh::{HasDisplayHandle, HasWindowHandle};

use crate::device::Device;
use crate::display::{Hdr, Vsync};
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
    /// configured, which is what the application has to encode for. `vsync`
    /// picks the presentation mode.
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
        vsync: Vsync,
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
        let adapter =
            pollster::block_on(self.inner.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                // Exact adapter limits: this is a trusted native app, not a sandbox
                // that needs fingerprinting resistance.
                apply_limit_buckets: false,
            }))
            .map_err(|err| Error::new("no graphics adapter supports this surface", err))?;

        let device = open_device(&adapter)?;
        let surface = Surface::new(surface, &adapter, device.clone(), width, height, hdr, vsync);

        Ok((device, surface))
    }
}

/// Storage buffers a single shader stage may bind. The downlevel default is 4,
/// and GPU culling binds 5 in one compute stage; 8 is what every desktop
/// adapter offers.
pub(crate) const STORAGE_BUFFERS_PER_STAGE: u32 = 8;

/// The device everything in this crate runs on: no required features beyond
/// `TIMESTAMP_QUERY`, taken when the adapter offers it, and the downlevel
/// default limits except for the storage buffers per stage.
///
/// Asking for the minimum rather than the adapter's own limits keeps the same
/// binary working across backends, and a machine that cannot meet it is told
/// so at startup rather than when a pipeline fails to validate.
pub(crate) fn open_device(adapter: &wgpu::Adapter) -> Result<Device> {
    let offered = adapter.limits();
    if offered.max_storage_buffers_per_shader_stage < STORAGE_BUFFERS_PER_STAGE {
        return Err(Error::new(
            "the graphics adapter cannot bind enough storage buffers per shader stage",
            LimitTooLow {
                limit: "max_storage_buffers_per_shader_stage",
                needed: STORAGE_BUFFERS_PER_STAGE,
                offered: offered.max_storage_buffers_per_shader_stage,
            },
        ));
    }

    let required_limits = wgpu::Limits {
        max_storage_buffers_per_shader_stage: STORAGE_BUFFERS_PER_STAGE,
        ..wgpu::Limits::downlevel_defaults()
    }
    .using_resolution(offered);

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("etib device"),
        required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
        required_limits,
        ..Default::default()
    }))
    .map_err(|err| Error::new("failed to open a graphics device", err))?;
    info!("device opened");

    Ok(Device::new(device, queue, adapter))
}

/// The detail behind a refused adapter: which limit, and by how much.
#[derive(Debug)]
struct LimitTooLow {
    limit: &'static str,
    needed: u32,
    offered: u32,
}

impl std::fmt::Display for LimitTooLow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} is {} but at least {} is needed",
            self.limit, self.offered, self.needed
        )
    }
}

impl std::error::Error for LimitTooLow {}
