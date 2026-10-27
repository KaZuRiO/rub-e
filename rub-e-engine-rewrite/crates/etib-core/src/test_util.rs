//! A device for tests, opened without a window.

use crate::device::Device;

/// A device on whatever adapter is there, or `None` on a machine with no
/// usable GPU — in which case the test says so and passes, as there is nothing
/// it could check.
pub(crate) fn device() -> Option<Device> {
    match Device::open_headless() {
        Ok(device) => Some(device),
        Err(err) => {
            let detail = std::error::Error::source(&err).map(ToString::to_string);
            eprintln!("skipping: {err} ({})", detail.unwrap_or_default());
            None
        }
    }
}
