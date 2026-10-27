//! The swapchain: configuration, and acquiring an image to draw into.

use log::{debug, error, info, trace, warn};

use crate::device::Device;
use crate::display::{ColorSpace, Display, Hdr, Output};
use crate::texture::{TextureFormat, TextureView};

/// What came back from [`Surface::acquire`].
///
/// The recoverable cases are already handled: the surface reconfigures itself
/// where that is the fix, so a caller only has to decide between drawing,
/// trying again next frame, and giving up.
pub enum Acquired {
    /// Draw into this image, then [present](Frame::present) it.
    Frame(Frame),
    /// Nothing to draw into this time (the window is occluded, or the swapchain
    /// was just rebuilt). Ask for another frame.
    Skip,
    /// The surface is unusable and will not recover. Shut down.
    Fatal,
}

/// A swapchain image, borrowed for the duration of one frame.
pub struct Frame {
    texture: wgpu::SurfaceTexture,
    view: TextureView,
}

impl Frame {
    /// The view to draw into.
    pub fn view(&self) -> &TextureView {
        &self.view
    }

    /// Hands the finished image to the compositor. Call after submitting the
    /// frame's work.
    pub fn present(self, device: &Device) {
        device.raw_queue().present(self.texture);
    }
}

/// A window's swapchain, kept in step with the window's size.
pub struct Surface {
    inner: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: Device,
    color_space: ColorSpace,
    /// Kept so the display's luminance can be re-read while running: those
    /// numbers move with the brightness slider.
    adapter: wgpu::Adapter,
}

impl Surface {
    pub(crate) fn new(
        inner: wgpu::Surface<'static>,
        adapter: &wgpu::Adapter,
        device: Device,
        width: u32,
        height: u32,
        hdr: Hdr,
    ) -> Self {
        let caps = inner.get_capabilities(adapter);
        debug!(
            "surface capabilities: formats={:?} present_modes={:?} alpha_modes={:?}",
            caps.formats, caps.present_modes, caps.alpha_modes
        );
        debug!("surface format capabilities: {:?}", caps.format_capabilities);

        // What the display is currently doing, which is a different question
        // from what the surface could be configured with.
        let display = Display::from_wgpu(&inner.display_hdr_info(adapter));
        info!("display reports {display:?}");

        let (format, color_space) = select_output(&caps, hdr, &display);

        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Fifo
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: match color_space {
                ColorSpace::Srgb => wgpu::SurfaceColorSpace::Srgb,
                ColorSpace::ExtendedLinear => wgpu::SurfaceColorSpace::ExtendedSrgbLinear,
            },
            width: width.max(1),
            height: height.max(1),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };

        inner.configure(device.raw(), &config);
        info!(
            "surface configured: {}x{} {:?} {:?} {:?}",
            config.width, config.height, config.format, color_space, config.present_mode
        );

        Self {
            inner,
            config,
            device,
            color_space,
            adapter: adapter.clone(),
        }
    }

    /// How the values written to this surface are interpreted.
    pub fn color_space(&self) -> ColorSpace {
        self.color_space
    }

    /// Whether this surface takes values above 1.0 as brighter-than-SDR output.
    pub fn is_hdr(&self) -> bool {
        self.color_space.is_hdr()
    }

    /// What the platform currently reports about the display being presented
    /// to. Worth re-reading rather than caching: the numbers move with the
    /// brightness slider, and when the window is dragged to another monitor.
    ///
    /// Call it from the main thread. On macOS the answer lives on
    /// main-thread-only AppKit objects, and anywhere else it comes back empty.
    pub fn display(&self) -> Display {
        Display::from_wgpu(&self.inner.display_hdr_info(&self.adapter))
    }

    /// How a finished, display-referred image should be written to this
    /// surface right now. Re-read it as [`Surface::display`] changes.
    pub fn output(&self) -> Output {
        Output::detect(self.color_space, &self.display())
    }

    /// The format of the swapchain images, which any pipeline drawing into them
    /// has to target.
    pub fn format(&self) -> TextureFormat {
        TextureFormat::from_raw(self.config.format)
    }

    /// The current swapchain size in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Rebuilds the swapchain at a new size. Returns `false`, changing nothing,
    /// for an unchanged or degenerate (minimised) size.
    pub fn resize(&mut self, width: u32, height: u32) -> bool {
        if width == 0 || height == 0 {
            debug!("ignoring resize to {width}x{height} (window is minimised)");
            return false;
        }
        if width == self.config.width && height == self.config.height {
            return false;
        }

        info!(
            "resizing surface {}x{} -> {}x{}",
            self.config.width, self.config.height, width, height
        );
        self.config.width = width;
        self.config.height = height;
        self.configure();
        true
    }

    /// Takes the next image to draw into, recovering from a stale swapchain on
    /// the way. See [`Acquired`].
    pub fn acquire(&self) -> Acquired {
        use wgpu::CurrentSurfaceTexture as Current;

        let texture = match self.inner.get_current_texture() {
            Current::Success(texture) => texture,
            // Still presentable, but the swapchain no longer matches the
            // surface; draw this frame, then rebuild for the next one.
            Current::Suboptimal(texture) => {
                debug!("surface texture is suboptimal; reconfiguring after this frame");
                self.configure();
                texture
            }
            // Usually a resize or a display change we have not seen yet.
            Current::Outdated | Current::Lost => {
                self.configure();
                return Acquired::Skip;
            }
            Current::Timeout => {
                warn!("surface timed out; skipping this frame");
                return Acquired::Skip;
            }
            Current::Occluded => {
                trace!("window is occluded; skipping this frame");
                return Acquired::Skip;
            }
            Current::Validation => {
                // The uncaptured-error handler installed with the device logs
                // the underlying validation error.
                error!("validation error while acquiring the surface texture");
                return Acquired::Fatal;
            }
        };

        let view = TextureView::from_raw(
            texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
        );
        Acquired::Frame(Frame { texture, view })
    }

    /// Re-applies the current configuration, to recover from a lost swapchain.
    fn configure(&self) {
        self.inner.configure(self.device.raw(), &self.config);
    }
}

/// Picks the format and colour space to configure.
///
/// An extended-range surface is only taken when it was asked for, the platform
/// offers `Rgba16Float` in the extended linear colour space (the combination
/// every native backend agrees on), and, unless forced, the display reports
/// headroom to use. Wide-gamut and PQ/HLG spaces are deliberately left alone:
/// they change how output has to be encoded, and there is nothing here that
/// would encode it.
fn select_output(
    caps: &wgpu::SurfaceCapabilities,
    hdr: Hdr,
    display: &Display,
) -> (wgpu::TextureFormat, ColorSpace) {
    const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

    if hdr != Hdr::Off {
        let offered = caps
            .color_spaces(HDR_FORMAT)
            .contains(wgpu::SurfaceColorSpaces::EXTENDED_SRGB_LINEAR);

        if offered && (hdr == Hdr::Force || display.has_headroom()) {
            info!("configuring extended-range output ({HDR_FORMAT:?}, scRGB)");
            return (HDR_FORMAT, ColorSpace::ExtendedLinear);
        }

        if offered {
            info!("extended-range surface offered, but the display reports no headroom to use it; using sRGB output");
        } else {
            info!("no extended-range surface offered; using sRGB output");
        }
    }

    // Shaders write linear values and rely on the hardware to encode sRGB, so
    // an sRGB format is what standard-range output wants.
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or_else(|| {
            warn!(
                "no sRGB surface format available; colours will look washed out (falling back to {:?})",
                caps.formats[0]
            );
            caps.formats[0]
        });

    (format, ColorSpace::Srgb)
}
