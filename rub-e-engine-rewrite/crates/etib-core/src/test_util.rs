//! A device for tests, opened without a window.

use crate::device::Device;
use crate::instance::open_device;

/// A device on whatever adapter is there, with the same limits
/// [`Instance::open`](crate::Instance::open) asks for, or `None` on a machine
/// with no usable GPU — in which case the test says so and passes, as there is
/// nothing it could check.
pub(crate) fn device() -> Option<Device> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    })) {
        Ok(adapter) => adapter,
        Err(err) => {
            eprintln!("skipping: no graphics adapter ({err})");
            return None;
        }
    };
    match open_device(&adapter) {
        Ok(device) => Some(device),
        Err(err) => {
            let detail = std::error::Error::source(&err).map(ToString::to_string);
            eprintln!("skipping: {err} ({})", detail.unwrap_or_default());
            None
        }
    }
}
