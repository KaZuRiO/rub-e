//! Tone mapping and colour grading, on the way to the swapchain image.

use bytemuck::{Pod, Zeroable};
use etib_core::{
    BindGroup, BindLayout, Blend, Color, Device, Encoder, Filter, Load, Output, RenderPipeline,
    Sampler, Shader, Stage, Texture3d, TextureFormat, TextureView, UniformBuffer,
};
use log::debug;

use crate::lut::Lut;

/// The curve that maps scene-referred values onto what a display can show.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tonemap {
    /// No curve at all: anything above 1.0 clips. Useful as a reference, and
    /// correct for a source that is already display-referred.
    None,
    /// Reinhard, applied to luminance. Cheap, neutral, and desaturates little,
    /// but highlights flatten out early.
    ///
    /// The only curve here that uses an HDR display's headroom: it rolls off
    /// towards whatever peak the display reports rather than towards white.
    Reinhard,
    /// A curve fit to the ACES filmic tone mapper: contrastier, with a long
    /// highlight roll-off. The usual default.
    ///
    /// The fit targets standard range, so it saturates at SDR white even on an
    /// HDR surface, where it gives the reference SDR look at the right
    /// brightness rather than a brighter image.
    #[default]
    Aces,
}

impl Tonemap {
    /// Every variant, in the order they are worth trying.
    pub const ALL: [Self; 3] = [Self::None, Self::Reinhard, Self::Aces];

    /// A short name, for a menu.
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Reinhard => "Reinhard",
            Self::Aces => "ACES",
        }
    }

    /// Must match the TONEMAP_* constants in shaders/post.wgsl.
    fn index(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Reinhard => 1,
            Self::Aces => 2,
        }
    }
}

/// The look [`PostProcess`] applies.
///
/// Exposure and tint act on the scene before the tone curve, which decides what
/// the curve is given to compress; contrast and saturation act on the mapped
/// image, which is where a grade normally sits. A lookup table, if one is set,
/// comes last of all — see [`PostProcess::set_lut`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grade {
    /// Exposure in stops. Each `+1.0` doubles the light reaching the curve.
    pub exposure_ev: f32,
    /// The curve that brings scene-referred light into display range.
    pub tonemap: Tonemap,
    /// `1.0` leaves contrast alone; higher pivots around mid-grey.
    pub contrast: f32,
    /// `0.0` is greyscale, `1.0` untouched, above that oversaturated.
    pub saturation: f32,
    /// Colour filter in linear RGB, multiplied in before the curve. White is
    /// neutral.
    pub tint: [f32; 3],
    /// How much of the lookup table set with [`PostProcess::set_lut`] to mix
    /// in: `0.0` bypasses it, `1.0` applies it as it is, and in between dials
    /// the look back towards the ungraded image.
    ///
    /// Ignored when no table is set.
    pub lut_strength: f32,
}

impl Default for Grade {
    fn default() -> Self {
        Self {
            exposure_ev: 0.0,
            tonemap: Tonemap::default(),
            contrast: 1.0,
            saturation: 1.0,
            tint: [1.0, 1.0, 1.0],
            lut_strength: 1.0,
        }
    }
}

/// Mirror of the Grade struct in shaders/post.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Params {
    tint: [f32; 4],
    exposure: f32,
    contrast: f32,
    saturation: f32,
    tonemap: u32,
    headroom: f32,
    sdr_white: f32,
    /// Entries per side of the bound lookup table, or 0.0 when there is none,
    /// which is how the shader knows to skip the stage.
    lut_size: f32,
    lut_strength: f32,
}

/// The final pass of a frame: reads the rendered scene, applies a tone curve,
/// a grade and an optional lookup table, and writes the result to the image
/// that gets presented.
///
/// The source is expected to be scene-referred — a float [`StorageTexture`] the
/// compute pass wrote, with values above 1.0 — and the target display-referred,
/// typically the swapchain view. An interface should be drawn *after* this
/// pass, not through it: its colours are already display-referred.
///
/// The [`Output`] handed to [`PostProcess::draw`] says what the target expects,
/// which is what makes the same pass correct for an sRGB and an extended-range
/// surface: on the former it is `Output::SDR` and everything clamps at white,
/// on the latter white sits wherever the OS puts it and highlights may go
/// above. [`Surface::output`](etib_core::Surface::output) works it out.
///
/// The lookup table is optional and off until [`PostProcess::set_lut`] is
/// called; see [`Lut`] for where one comes from.
///
/// [`StorageTexture`]: etib_core::StorageTexture
pub struct PostProcess {
    pipeline: RenderPipeline,
    layout: BindLayout,
    sampler: Sampler,
    uniforms: UniformBuffer<Params>,
    /// The lookup table in the bind group. Always a real texture — a shader
    /// cannot leave a binding empty — so with no table set this is a neutral
    /// one and `lut_size` below is 0.0, which turns the stage off.
    lut: Texture3d,
    lut_size: f32,
    bindings: BindGroup,
}

impl PostProcess {
    /// A pass tone mapping `source` into targets of `target`'s format, with no
    /// lookup table. Call [`rebind`](Self::rebind) when the source is recreated.
    pub fn new(device: &Device, target: TextureFormat, source: &TextureView) -> Self {
        let shader = Shader::wgsl(device, "post", include_str!("shaders/post.wgsl"));

        let layout = BindLayout::build(device, "post")
            .texture(Stage::Fragment)
            .sampler(Stage::Fragment)
            .uniform::<Params>(Stage::Fragment)
            .texture_3d(Stage::Fragment)
            .finish();

        let pipeline = RenderPipeline::screen(
            device,
            "post",
            &shader,
            ("vs_main", "fs_main"),
            target,
            Blend::Replace,
            &layout,
        );

        // The source is always the same size as the target, so that read is 1:1
        // and its filter mode never actually kicks in. The lookup table shares
        // this sampler and does depend on it: interpolating between its entries
        // is what makes a cube of a few dozen colours a continuous grade, and
        // clamping is what keeps the edges of the cube from wrapping around.
        let sampler = Sampler::new(device, "post", Filter::Linear);
        let uniforms = UniformBuffer::new(device, "post grade");
        let lut = neutral_lut(device);
        let bindings = layout.group(device, "post", &[source, &sampler, &uniforms, lut.view()]);

        Self {
            pipeline,
            layout,
            sampler,
            uniforms,
            lut,
            lut_size: 0.0,
            bindings,
        }
    }

    /// Rebinds after the source texture has been recreated on resize.
    pub fn rebind(&mut self, device: &Device, source: &TextureView) {
        debug!("rebinding post-processing to the new source texture");
        self.bindings = self.bind(device, source);
    }

    /// Sets the lookup table applied after the tone curve and the grade, or
    /// clears it with `None`.
    ///
    /// The table is indexed by the gamma-encoded colour, which is the space a
    /// grading tool authors one in; the pass encodes before the lookup and
    /// decodes after, so a `.cube` file behaves as its author intended.
    ///
    /// The table is uploaded here, so this is not something to call per frame;
    /// [`Grade::lut_strength`] is the per-frame control, and a strength of 0.0
    /// bypasses a table that stays uploaded. The `source` is wanted for the
    /// same reason [`rebind`](Self::rebind) wants it: one bind group holds the
    /// source and the table together.
    pub fn set_lut(&mut self, device: &Device, source: &TextureView, lut: Option<&Lut>) {
        (self.lut, self.lut_size) = match lut {
            Some(lut) => {
                debug!("uploading a {}³ lookup table", lut.size());
                (
                    Texture3d::cube(device, "post lut", lut.size(), lut.rgba()),
                    lut.size() as f32,
                )
            }
            None => {
                debug!("clearing the lookup table");
                (neutral_lut(device), 0.0)
            }
        };
        self.bindings = self.bind(device, source);
    }

    /// Whether a lookup table is currently bound, which is what makes
    /// [`Grade::lut_strength`] mean anything.
    pub fn has_lut(&self) -> bool {
        self.lut_size > 0.0
    }

    fn bind(&self, device: &Device, source: &TextureView) -> BindGroup {
        self.layout.group(
            device,
            "post",
            &[source, &self.sampler, &self.uniforms, self.lut.view()],
        )
    }

    /// Records the pass. This clears the target, so it must run before anything
    /// that paints on top of it.
    pub fn draw(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        target: &TextureView,
        grade: &Grade,
        output: Output,
    ) {
        self.uniforms.write(
            device,
            &Params {
                tint: [grade.tint[0], grade.tint[1], grade.tint[2], 1.0],
                exposure: grade.exposure_ev.exp2(),
                contrast: grade.contrast,
                saturation: grade.saturation,
                tonemap: grade.tonemap.index(),
                headroom: output.headroom.max(1.0),
                sdr_white: output.sdr_white.max(0.0),
                lut_size: self.lut_size,
                lut_strength: grade.lut_strength.clamp(0.0, 1.0),
            },
        );

        encoder
            .render_pass("post", target, Load::Clear(Color::BLACK))
            .draw(&self.pipeline, &self.bindings, 0..3);
    }
}

/// The table bound when there is none: the identity, at its smallest.
///
/// Nothing samples it while `lut_size` is 0.0, but a binding has to point at
/// something, and a neutral table is the right thing for it to point at.
fn neutral_lut(device: &Device) -> Texture3d {
    let identity = Lut::identity(Lut::MIN_SIZE);
    Texture3d::cube(
        device,
        "post lut (neutral)",
        identity.size(),
        identity.rgba(),
    )
}
