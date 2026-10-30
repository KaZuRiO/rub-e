//! The open GPU connection: what resources are made from, and where work is sent.

use std::sync::{Arc, Mutex, PoisonError};

use log::{debug, info};

use crate::encoder::Encoder;
use crate::error::{Error, Result};
use crate::instance::open_device;
use crate::timing::{GpuTimings, Timer};

/// A device and its queue, cheap to clone and shared by every resource made
/// from it.
///
/// This is the handle you pass to every constructor in this crate. Cloning it
/// clones a reference, not the device.
#[derive(Clone)]
pub struct Device {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: Arc<DeviceInfo>,
    /// Present only when the device was opened with `TIMESTAMP_QUERY`.
    timer: Option<Arc<Mutex<Timer>>>,
}

struct DeviceInfo {
    backend: String,
    adapter: String,
    max_texture_size: u32,
}

impl Device {
    pub(crate) fn new(device: wgpu::Device, queue: wgpu::Queue, adapter: &wgpu::Adapter) -> Self {
        let adapter_info = adapter.get_info();
        info!(
            "adapter: {:?} {} ({:?}), driver {} {}",
            adapter_info.device_type,
            adapter_info.name,
            adapter_info.backend,
            adapter_info.driver,
            adapter_info.driver_info
        );
        debug!("enabled device features: {:?}", device.features());

        device.on_uncaptured_error(Arc::new(|error| {
            log::error!("uncaptured GPU error: {error}");
        }));

        let info = DeviceInfo {
            backend: format!("{:?}", adapter_info.backend),
            adapter: adapter_info.name,
            max_texture_size: device.limits().max_texture_dimension_2d,
        };

        let timer = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| Arc::new(Mutex::new(Timer::new(&device, &queue))));

        Self {
            device,
            queue,
            info: Arc::new(info),
            timer,
        }
    }

    /// Opens a device with no window: on whatever adapter is there, preferring
    /// the low-power one, with the same features and limits
    /// [`Instance::open`](crate::Instance::open) asks for.
    ///
    /// For tests and offscreen tools. A machine with no usable GPU gets an
    /// `Err`, which a test can take as its cue to skip.
    pub fn open_headless() -> Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(|err| Error::new("no graphics adapter is available", err))?;
        open_device(&adapter)
    }

    /// The graphics backend in use, formatted for display, e.g. `"Vulkan"`.
    pub fn backend(&self) -> &str {
        &self.info.backend
    }

    /// The adapter's reported name, e.g. `"NVIDIA GeForce RTX 4070 SUPER"`.
    pub fn adapter_name(&self) -> &str {
        &self.info.adapter
    }

    /// Largest 2D texture this device will allocate, in texels per side.
    pub fn max_texture_size(&self) -> u32 {
        self.info.max_texture_size
    }

    /// Opens a command encoder to record a frame's work into.
    pub fn encoder(&self, label: &str) -> Encoder {
        Encoder::new(self, label, None)
    }

    /// Whether this device can measure passes on the GPU. When it cannot,
    /// [`timed_encoder`](Self::timed_encoder) gives an ordinary encoder.
    pub fn gpu_timing_available(&self) -> bool {
        self.timer.is_some()
    }

    /// Opens a command encoder whose passes are timed on the GPU, and whose
    /// results [`poll_timings`](Self::poll_timings) returns under `tag`.
    ///
    /// It is an ordinary encoder when the device cannot time passes, or when
    /// every timing slot is still waiting for its readback. At most 32 passes
    /// of one encoder are timed; the rest run untimed.
    pub fn timed_encoder(&self, label: &str, tag: u64) -> Encoder {
        let lease = self
            .timer
            .as_ref()
            .and_then(|timer| Timer::lease(timer, tag));
        Encoder::new(self, label, lease)
    }

    /// Finishes an encoder and submits its work to the queue. For a timed
    /// encoder this also starts the readback of its timestamps, without
    /// waiting for it.
    pub fn submit(&self, encoder: Encoder) {
        let (commands, lease) = encoder.finish();
        self.queue.submit(Some(commands));
        if let Some(lease) = lease {
            lease.launch();
        }
    }

    /// The measurements that have arrived since the last call, oldest first.
    /// Never waits: a submission the GPU has not finished is simply not there
    /// yet. Empty on a device that cannot time passes.
    pub fn poll_timings(&self) -> Vec<GpuTimings> {
        let Some(timer) = &self.timer else {
            return Vec::new();
        };
        if let Err(err) = self.device.poll(wgpu::PollType::Poll) {
            debug!("polling for GPU timings failed: {err}");
        }
        timer
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .collect()
    }

    /// Runs `create` inside a validation error scope, and panics naming the
    /// object if the backend refused it.
    ///
    /// Without the scope a validation error goes to the uncaptured-error
    /// handler, which only logs: an invalid shader would leave a running
    /// program drawing nothing. Shaders, pipelines and bind groups are fixed
    /// by the program rather than by its input, so a refused one is a bug.
    pub(crate) fn checked<T>(
        &self,
        what: &str,
        label: &str,
        create: impl FnOnce(&wgpu::Device) -> T,
    ) -> T {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let created = create(&self.device);
        if let Some(error) = pollster::block_on(scope.pop()) {
            panic!("{what} '{label}' is invalid: {error}");
        }
        created
    }

    // -- Interop ------------------------------------------------------------
    //
    // The wrappers above cover everything this workspace needs. These hand out
    // the underlying wgpu objects for code that has to speak wgpu directly,
    // such as the egui paint backend.

    /// The underlying wgpu device.
    pub fn raw(&self) -> &wgpu::Device {
        &self.device
    }

    /// The underlying wgpu queue.
    pub fn raw_queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("adapter", &self.info.adapter)
            .field("backend", &self.info.backend)
            .finish()
    }
}
