//! The window and the GPU objects tied to it.

use std::sync::Arc;

use anyhow::{Context as _, Result};
use etib_core::{ColorSpace, Device, Hdr, Instance, Surface, TextureFormat};
use log::info;
use winit::window::Window;

/// A window, the device drawing to it, and its swapchain.
///
/// Handed to [`App`](crate::App) on every callback; the fields are public
/// because that is all this type is for.
pub struct Gpu {
    pub device: Device,
    pub surface: Surface,
    /// Kept alive here because the surface presents to it.
    pub window: Arc<Window>,
}

impl Gpu {
    pub(crate) fn new(instance: &Instance, window: Arc<Window>, hdr: Hdr) -> Result<Self> {
        let size = window.inner_size();
        info!(
            "opening a device for the window ({}x{}, scale factor {:.2})",
            size.width,
            size.height,
            window.scale_factor()
        );

        let (device, surface) = instance
            .open(window.clone(), size.width, size.height, hdr)
            .context("failed to open a graphics device for the window")?;

        Ok(Self {
            device,
            surface,
            window,
        })
    }

    /// The swapchain size in pixels.
    pub fn size(&self) -> (u32, u32) {
        self.surface.size()
    }

    /// The format any pipeline drawing to the window has to target.
    pub fn format(&self) -> TextureFormat {
        self.surface.format()
    }

    /// How the values written to the window are interpreted. An application
    /// that writes anything brighter than SDR white has to check this.
    pub fn color_space(&self) -> ColorSpace {
        self.surface.color_space()
    }
}
