//! A device for tests, opened without a window.
//!
//! Compiled for this crate's own tests, and for other crates' behind the
//! `test-util` feature.

use crate::device::Device;

/// A device on whatever adapter is there, or `None` on a machine with no
/// usable GPU — in which case the test says so and passes, as there is nothing
/// it could check.
///
/// With `ETIB_REQUIRE_GPU=1` in the environment a missing GPU is a failure
/// instead, so that a machine expected to have one cannot pass by skipping.
pub fn device() -> Option<Device> {
    match Device::open_headless() {
        Ok(device) => Some(device),
        Err(err) => {
            let detail = std::error::Error::source(&err).map(ToString::to_string);
            let err = format!("{err} ({})", detail.unwrap_or_default());
            if std::env::var("ETIB_REQUIRE_GPU").as_deref() == Ok("1") {
                panic!("ETIB_REQUIRE_GPU is set but no GPU is available: {err}");
            }
            eprintln!("skipping: {err}");
            None
        }
    }
}
