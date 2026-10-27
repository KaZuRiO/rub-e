//! What the swapchain expects, and what the display behind it can show.

/// How the values written to a surface are interpreted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColorSpace {
    /// sRGB, standard dynamic range. The format applies the transfer function
    /// on write, so shaders write linear values, and anything above 1.0 is
    /// clamped by the display pipeline.
    #[default]
    Srgb,
    /// Extended linear sRGB, also known as scRGB: linear values, where 1.0 is
    /// the reference SDR white and larger values drive the display brighter,
    /// as far as its headroom allows.
    ///
    /// The format does *not* encode, so shaders write linear values here too —
    /// the difference is that they are no longer limited to 0..1.
    ExtendedLinear,
}

impl ColorSpace {
    /// Whether values above 1.0 mean anything on this surface.
    pub fn is_hdr(self) -> bool {
        matches!(self, Self::ExtendedLinear)
    }
}

/// Whether to configure an extended-range surface when one is available.
///
/// What was actually configured is [`Surface::color_space`], which is what an
/// application has to encode for either way.
///
/// [`Surface::color_space`]: crate::Surface::color_space
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Hdr {
    /// Take an extended-range surface when the platform offers one *and* the
    /// display reports headroom to use it.
    ///
    /// Both halves matter, because they are separate questions: a surface
    /// reports the colour spaces it can be configured with, not the state the
    /// display is in. Metal offers extended range for a float surface on any
    /// display, and DXGI keeps offering it while HDR is switched off. Taking it
    /// regardless would cost a float swapchain for nothing, and would place
    /// white by a convention the platform is no longer honouring: an image at
    /// 80 nits against a 280-nit desktop.
    Allow,
    /// Take an extended-range surface whenever the platform offers one, even
    /// when it reports nothing about the display.
    ///
    /// For the platforms that expose no luminance at all (Vulkan outside
    /// Windows, GLES), where [`Allow`](Self::Allow) can never be satisfied, and
    /// for exercising the path deliberately.
    Force,
    /// Always configure a standard-range sRGB surface.
    #[default]
    Off,
}

/// Whether presentation waits for the display's refresh.
///
/// The surface log line reports the mode that was actually configured.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Vsync {
    /// One image per refresh, never torn. Always available, and what the
    /// engine's `vsync: true` meant; the frame rate is capped at the refresh
    /// rate.
    #[default]
    On,
    /// As many images as the GPU can render: the newest finished image is
    /// shown at each refresh where the platform can do that without tearing,
    /// otherwise images are shown as soon as they are ready and may tear.
    /// Falls back to [`On`](Self::On) where neither is offered.
    Off,
}

/// What the platform reports about the display a surface is presenting to.
///
/// Every field is optional because every platform reports a different subset,
/// and all of it is advisory: the figures are often EDID-sourced and run
/// optimistic. Re-read it rather than caching — the brightness slider moves
/// these numbers while the application runs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Display {
    /// How far above SDR white the display can be driven before it clips, as a
    /// multiplier. `1.0` means no headroom at all; `None` means the platform
    /// did not report enough to say.
    ///
    /// This is the number a tone mapper wants. Apple reports it directly, and
    /// on Windows it is derived from the peak and SDR white levels below.
    pub headroom: Option<f32>,
    /// The luminance the OS maps SDR white to, in nits. Windows only, and it
    /// moves with the SDR brightness slider.
    pub sdr_white_nits: Option<f32>,
    /// Peak luminance of a small highlight, in nits. Windows only.
    pub peak_nits: Option<f32>,
    /// Whether the platform says the display is currently in HDR mode. This is
    /// about the display's state, not about what a surface can be configured
    /// with — a surface may offer an HDR colour space either way.
    pub hdr_mode: Option<bool>,
}

impl Display {
    pub(crate) fn from_wgpu(info: &wgpu::DisplayHdrInfo) -> Self {
        Self {
            headroom: info.tone_map_headroom(),
            sdr_white_nits: info.luminance.and_then(|l| l.sdr_white_nits),
            peak_nits: info.luminance.and_then(|l| l.max_nits),
            hdr_mode: info.coarse.and_then(|c| c.high_dynamic_range),
        }
    }

    /// The headroom to tone map against, falling back to `1.0` (no headroom)
    /// when the platform reported nothing usable.
    pub fn headroom_or_sdr(&self) -> f32 {
        self.headroom.filter(|h| *h >= 1.0).unwrap_or(1.0)
    }

    /// Whether the display says it can currently show anything brighter than
    /// SDR white. `false` also covers "the platform did not say", which is why
    /// [`Hdr::Force`] exists.
    pub fn has_headroom(&self) -> bool {
        // A headroom above 1.0 is the direct answer; Windows reports HDR mode
        // separately, and can have it on while the nit figures are missing.
        self.headroom.is_some_and(|h| h > 1.0) || self.hdr_mode == Some(true)
    }
}

/// The reference white of extended linear sRGB (scRGB), in nits.
///
/// scRGB fixes 1.0 at this luminance, which is dimmer than a modern desktop
/// runs SDR white at. Matching the rest of the desktop therefore means scaling
/// by `sdr_white_nits / 80`, which is what [`Output::detect`] does.
pub const SCRGB_REFERENCE_NITS: f32 = 80.0;

/// How display-referred values map onto the units a surface actually wants.
///
/// Rendering produces an image where `1.0` is white and anything above it is a
/// highlight. What that has to become depends on the surface: on an sRGB one,
/// `1.0` is simply the maximum, while on an extended-range one it is a specific
/// luminance, and highlights may go above it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Output {
    /// The value to write for SDR white.
    pub sdr_white: f32,
    /// How far above SDR white output may go, as a multiplier. `1.0` means
    /// highlights have to be compressed to white.
    pub headroom: f32,
}

impl Output {
    /// Standard range: white is 1.0 and there is nothing above it.
    pub const SDR: Self = Self {
        sdr_white: 1.0,
        headroom: 1.0,
    };

    /// Works out the mapping for a surface and the display behind it.
    ///
    /// For an extended-range surface this converts scRGB's fixed 80-nit
    /// reference white to whatever the OS currently calls SDR white, so the
    /// image sits at the same brightness as every other window. When the
    /// platform reports no white level (everywhere except Windows) the scale
    /// stays 1.0, which is already correct on Apple EDR.
    pub fn detect(color_space: ColorSpace, display: &Display) -> Self {
        match color_space {
            ColorSpace::Srgb => Self::SDR,
            ColorSpace::ExtendedLinear => Self {
                sdr_white: display
                    .sdr_white_nits
                    .filter(|nits| *nits > 0.0)
                    .map(|nits| nits / SCRGB_REFERENCE_NITS)
                    .unwrap_or(1.0),
                headroom: display.headroom_or_sdr(),
            },
        }
    }

    /// The largest value that still means anything on this surface.
    pub fn peak(&self) -> f32 {
        self.sdr_white * self.headroom
    }
}
