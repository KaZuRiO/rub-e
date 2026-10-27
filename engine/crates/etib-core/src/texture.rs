//! Texture formats, offscreen textures, views and samplers.

use log::debug;

use crate::device::Device;

/// The pixel format of a texture, as an opaque token.
///
/// Formats come from the surface or from a [`StorageFormat`]; this type only
/// exists to hand one to the pipeline that has to match it.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureFormat(wgpu::TextureFormat);

impl TextureFormat {
    /// Whether the hardware applies sRGB encoding when writing to this format.
    pub fn is_srgb(self) -> bool {
        self.0.is_srgb()
    }

    /// Eight-bit sRGB. The hardware encodes on write and decodes on read, so
    /// shaders work in linear on both sides; the format a standard-range
    /// intermediate wants.
    pub const RGBA8_SRGB: Self = Self(wgpu::TextureFormat::Rgba8UnormSrgb);

    pub(crate) fn from_raw(format: wgpu::TextureFormat) -> Self {
        Self(format)
    }

    /// Interop: the underlying wgpu format.
    pub fn raw(self) -> wgpu::TextureFormat {
        self.0
    }
}

impl std::fmt::Debug for TextureFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

/// Formats a compute shader can write to with `textureStore`.
///
/// All of them are linear rather than sRGB, which is the whole list's defining
/// property: storage writes are not defined for sRGB formats, so a shader that
/// wants sRGB output has to encode it itself or leave it to the blit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum StorageFormat {
    /// Eight bits per channel, the cheapest format that holds a colour image.
    #[default]
    Rgba8Unorm,
    /// Half floats, for values that need range or precision beyond 0..1.
    Rgba16Float,
    /// Full floats.
    Rgba32Float,
}

impl StorageFormat {
    fn raw(self) -> wgpu::TextureFormat {
        match self {
            Self::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            Self::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
            Self::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
        }
    }

    /// The matching [`TextureFormat`], for pipelines that have to declare it.
    pub fn format(self) -> TextureFormat {
        TextureFormat::from_raw(self.raw())
    }
}

/// A view of a texture, which is what gets bound and drawn into.
pub struct TextureView(wgpu::TextureView);

impl TextureView {
    pub(crate) fn from_raw(view: wgpu::TextureView) -> Self {
        Self(view)
    }

    /// Interop: the underlying wgpu view.
    pub fn raw(&self) -> &wgpu::TextureView {
        &self.0
    }
}

/// What the three window-sized textures share: the texture, its view, and the
/// size and label it was last created with.
struct Target {
    /// Held so the view stays valid.
    _texture: wgpu::Texture,
    view: TextureView,
    size: (u32, u32),
    label: String,
}

impl Target {
    fn new(
        texture: wgpu::Texture,
        view: TextureView,
        label: &str,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            _texture: texture,
            view,
            size: (width, height),
            label: label.to_owned(),
        }
    }

    /// Recreates the texture through `create` at a new size. Returns `true`
    /// when it actually changed, which is the caller's cue to rebuild its bind
    /// groups; an unchanged or degenerate size is ignored.
    fn resize(
        &mut self,
        device: &Device,
        kind: &str,
        width: u32,
        height: u32,
        create: impl Fn(&Device, &str, u32, u32) -> (wgpu::Texture, TextureView),
    ) -> bool {
        if width == 0 || height == 0 || (width, height) == self.size {
            return false;
        }

        debug!(
            "recreating {kind} texture '{}' {}x{} -> {width}x{height}",
            self.label, self.size.0, self.size.1
        );
        let (texture, view) = create(device, &self.label, width, height);
        self._texture = texture;
        self.view = view;
        self.size = (width, height);
        true
    }
}

/// An offscreen texture a compute shader writes and a later pass samples.
///
/// Resizing recreates the texture and its view, so anything holding a bind
/// group over the view has to rebuild it; [`StorageTexture::resize`] reports
/// when that is needed.
pub struct StorageTexture {
    target: Target,
    format: StorageFormat,
}

impl StorageTexture {
    pub fn new(
        device: &Device,
        label: &str,
        width: u32,
        height: u32,
        format: StorageFormat,
    ) -> Self {
        let (texture, view) = create(device, label, width, height, format);
        Self {
            target: Target::new(texture, view, label, width, height),
            format,
        }
    }

    /// The view to bind, both for the compute write and the sampled read.
    pub fn view(&self) -> &TextureView {
        &self.target.view
    }

    pub fn format(&self) -> StorageFormat {
        self.format
    }

    pub fn size(&self) -> (u32, u32) {
        self.target.size
    }

    /// Recreates the texture at a new size. Returns `true` when it actually
    /// changed, which is the caller's cue to rebuild its bind groups; an
    /// unchanged or degenerate size is ignored.
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) -> bool {
        let format = self.format;
        self.target.resize(
            device,
            "storage",
            width,
            height,
            |device, label, width, height| create(device, label, width, height, format),
        )
    }
}

/// How a sampler interpolates between texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Filter {
    /// Nearest texel; what you want for a pixel-exact source.
    Nearest,
    /// Bilinear.
    #[default]
    Linear,
}

impl Filter {
    fn raw(self) -> wgpu::FilterMode {
        match self {
            Self::Nearest => wgpu::FilterMode::Nearest,
            Self::Linear => wgpu::FilterMode::Linear,
        }
    }
}

/// How a shader reads a texture. Addressing is always clamp-to-edge, which is
/// what full-target passes want.
pub struct Sampler(wgpu::Sampler);

impl Sampler {
    pub fn new(device: &Device, label: &str, filter: Filter) -> Self {
        Self(device.raw().create_sampler(&wgpu::SamplerDescriptor {
            label: Some(label),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: filter.raw(),
            min_filter: filter.raw(),
            ..Default::default()
        }))
    }

    pub(crate) fn raw(&self) -> &wgpu::Sampler {
        &self.0
    }
}

fn create(
    device: &Device,
    label: &str,
    width: u32,
    height: u32,
    format: StorageFormat,
) -> (wgpu::Texture, TextureView) {
    debug!("allocating {width}x{height} {format:?} storage texture '{label}'");
    let texture = device.raw().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: format.raw(),
        // STORAGE_BINDING for the compute write, TEXTURE_BINDING for the read.
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = TextureView::from_raw(texture.create_view(&wgpu::TextureViewDescriptor::default()));
    (texture, view)
}

/// A texture a render pass draws into and a later pass samples.
///
/// The counterpart to [`StorageTexture`], for work that comes out of a render
/// pass rather than a compute one — an interface drawn separately and
/// composited, say, or an intermediate in a chain of screen passes.
pub struct RenderTexture {
    target: Target,
    format: TextureFormat,
}

impl RenderTexture {
    pub fn new(
        device: &Device,
        label: &str,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Self {
        let (texture, view) = create_render_target(device, label, width, height, format);
        Self {
            target: Target::new(texture, view, label, width, height),
            format,
        }
    }

    /// The view, both as a pass target and as a binding to sample.
    pub fn view(&self) -> &TextureView {
        &self.target.view
    }

    pub fn format(&self) -> TextureFormat {
        self.format
    }

    pub fn size(&self) -> (u32, u32) {
        self.target.size
    }

    /// Recreates the texture at a new size, returning `true` when it actually
    /// changed and bind groups over the view have to be rebuilt.
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) -> bool {
        let format = self.format;
        self.target.resize(
            device,
            "render",
            width,
            height,
            |device, label, width, height| {
                create_render_target(device, label, width, height, format)
            },
        )
    }
}

fn create_render_target(
    device: &Device,
    label: &str,
    width: u32,
    height: u32,
    format: TextureFormat,
) -> (wgpu::Texture, TextureView) {
    debug!("allocating {width}x{height} {format:?} render texture '{label}'");
    let texture = device.raw().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: format.raw(),
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = TextureView::from_raw(texture.create_view(&wgpu::TextureViewDescriptor::default()));
    (texture, view)
}

/// A cube of values the CPU fills once and a shader reads, such as a colour
/// lookup table.
///
/// The counterpart to the two textures above: those are written by the GPU and
/// resized with the surface, this one is handed its contents at creation and
/// never changes size.
///
/// The format is half-float RGBA, which is filterable on every backend — the
/// hardware interpolating between entries is what lets a small cube stand in
/// for a continuous function — and finer than the 8-bit quantisation an image
/// would otherwise pick up passing through one.
pub struct Texture3d {
    /// Held so the view stays valid.
    _texture: wgpu::Texture,
    view: TextureView,
    size: u32,
}

impl Texture3d {
    /// Creates a `size`×`size`×`size` texture from `rgba`, four floats per
    /// texel, with x varying fastest and z slowest.
    ///
    /// # Panics
    ///
    /// If `size` is zero, or `rgba` is not exactly `size³ × 4` floats long.
    pub fn cube(device: &Device, label: &str, size: u32, rgba: &[f32]) -> Self {
        assert!(size > 0, "3D texture '{label}' cannot be empty");
        let expected = (size as usize).pow(3) * 4;
        assert_eq!(
            rgba.len(),
            expected,
            "3D texture '{label}' is {size}³ texels, so it needs {expected} floats, not {}",
            rgba.len()
        );

        debug!("allocating {size}³ Rgba16Float 3D texture '{label}'");
        let texture = device.raw().create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: size,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let halves: Vec<u16> = rgba.iter().copied().map(to_f16).collect();
        device.raw_queue().write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(&halves),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // Four channels of two bytes each.
                bytes_per_row: Some(size * 8),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: size,
            },
        );

        let view =
            TextureView::from_raw(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        Self {
            _texture: texture,
            view,
            size,
        }
    }

    /// The view to bind.
    pub fn view(&self) -> &TextureView {
        &self.view
    }

    /// The length of one side, in texels.
    pub fn size(&self) -> u32 {
        self.size
    }
}

/// A depth buffer, for a render pass that has to know what is in front.
///
/// Always `Depth32Float`, which every backend supports as an attachment. It
/// resizes with the colour target it is paired with, and like the other
/// window-sized textures reports when it actually did.
pub struct DepthTexture {
    target: Target,
}

impl DepthTexture {
    pub(crate) const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn new(device: &Device, label: &str, width: u32, height: u32) -> Self {
        let (texture, view) = create_depth(device, label, width, height);
        Self {
            target: Target::new(texture, view, label, width, height),
        }
    }

    /// The view, as a depth attachment or as a binding to sample.
    pub fn view(&self) -> &TextureView {
        &self.target.view
    }

    pub fn size(&self) -> (u32, u32) {
        self.target.size
    }

    /// Recreates the texture at a new size, returning `true` when it actually
    /// changed; an unchanged or degenerate size is ignored.
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) -> bool {
        self.target
            .resize(device, "depth", width, height, create_depth)
    }
}

fn create_depth(
    device: &Device,
    label: &str,
    width: u32,
    height: u32,
) -> (wgpu::Texture, TextureView) {
    debug!("allocating {width}x{height} depth texture '{label}'");
    let texture = device.raw().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DepthTexture::FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = TextureView::from_raw(texture.create_view(&wgpu::TextureViewDescriptor::default()));
    (texture, view)
}

/// Six square faces sampled by direction, such as a skybox.
///
/// A compute shader fills it — typically by projecting an equirectangular
/// image — through [`TextureCube::storage_view`], which sees the faces as a
/// six-layer array (`texture_storage_2d_array`). Everything else samples it
/// through [`TextureCube::view`] as a `texture_cube`.
pub struct TextureCube {
    /// Held so the views stay valid.
    _texture: wgpu::Texture,
    view: TextureView,
    storage_view: TextureView,
    format: StorageFormat,
    size: u32,
}

impl TextureCube {
    /// # Panics
    ///
    /// If `size` is zero.
    pub fn new(device: &Device, label: &str, size: u32, format: StorageFormat) -> Self {
        assert!(size > 0, "cube texture '{label}' cannot be empty");
        debug!("allocating {size}x{size}x6 {format:?} cube texture '{label}'");
        let texture = device.raw().create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: format.raw(),
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = TextureView::from_raw(texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(label),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        }));
        let storage_view =
            TextureView::from_raw(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(label),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            }));
        Self {
            _texture: texture,
            view,
            storage_view,
            format,
            size,
        }
    }

    /// The view to sample, bound with
    /// [`BindLayoutBuilder::texture_cube`](crate::BindLayoutBuilder::texture_cube).
    pub fn view(&self) -> &TextureView {
        &self.view
    }

    /// The faces as a six-layer array for a compute shader to write, bound with
    /// [`BindLayoutBuilder::storage_texture_array`](crate::BindLayoutBuilder::storage_texture_array).
    pub fn storage_view(&self) -> &TextureView {
        &self.storage_view
    }

    pub fn format(&self) -> StorageFormat {
        self.format
    }

    /// The length of one face's side, in texels.
    pub fn size(&self) -> u32 {
        self.size
    }
}

/// An image the CPU uploads once and a shader samples, such as a decoded HDR
/// environment map.
///
/// Half-float RGBA, like [`Texture3d`], so values above 1.0 survive and the
/// texture is filterable everywhere.
pub struct Texture2d {
    /// Held so the view stays valid.
    _texture: wgpu::Texture,
    view: TextureView,
    size: (u32, u32),
}

impl Texture2d {
    /// Creates a `width`×`height` texture from `rgba`, four floats per texel,
    /// row by row from the top.
    ///
    /// # Panics
    ///
    /// If either side is zero, or `rgba` is not exactly `width × height × 4`
    /// floats long.
    pub fn from_rgba_f32(
        device: &Device,
        label: &str,
        width: u32,
        height: u32,
        rgba: &[f32],
    ) -> Self {
        assert!(
            width > 0 && height > 0,
            "2D texture '{label}' cannot be empty"
        );
        let expected = width as usize * height as usize * 4;
        assert_eq!(
            rgba.len(),
            expected,
            "2D texture '{label}' is {width}x{height} texels, so it needs {expected} floats, not {}",
            rgba.len()
        );

        debug!("allocating {width}x{height} Rgba16Float texture '{label}'");
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.raw().create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let halves: Vec<u16> = rgba.iter().copied().map(to_f16).collect();
        device.raw_queue().write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(&halves),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 8),
                rows_per_image: Some(height),
            },
            extent,
        );

        let view =
            TextureView::from_raw(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        Self {
            _texture: texture,
            view,
            size: (width, height),
        }
    }

    /// The view to bind.
    pub fn view(&self) -> &TextureView {
        &self.view
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }
}

/// Encodes a float as a half, rounding to nearest.
///
/// Values past half's range (65504) become infinity, as they would under any
/// other rounding of a number that large, and NaN stays NaN.
fn to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let magnitude = bits & 0x7fff_ffff;

    // Infinity and NaN, which carry across as themselves. The mantissa bit
    // keeps a NaN a NaN: dropping the low 13 bits could otherwise leave a
    // payload of zero, which reads as infinity.
    if magnitude >= 0x7f80_0000 {
        let nan = u16::from(magnitude > 0x7f80_0000) << 9;
        return sign | 0x7c00 | nan;
    }

    let exponent = ((magnitude >> 23) as i32) - 127 + 15;
    let mantissa = magnitude & 0x007f_ffff;

    if exponent >= 0x1f {
        // Too large for a half: infinity, with the sign kept.
        return sign | 0x7c00;
    }

    if exponent <= 0 {
        // Below half's smallest normal. Anything under half a subnormal step
        // rounds to zero; the rest shifts the implicit leading 1 back in and
        // becomes a subnormal.
        if exponent < -10 {
            return sign;
        }
        let with_implicit = mantissa | 0x0080_0000;
        let shift = (14 - exponent) as u32;
        let rounded = (with_implicit >> shift) + ((with_implicit >> (shift - 1)) & 1);
        return sign | rounded as u16;
    }

    // A carry out of the mantissa lands in the exponent, which is what rounding
    // up across a power of two should do.
    let half = ((exponent as u32) << 10 | (mantissa >> 13)) as u16;
    sign | (half + ((mantissa >> 12) & 1) as u16)
}

#[cfg(test)]
mod tests {
    use super::to_f16;

    /// Decodes a half, so the round trip can be checked against the original.
    fn from_f16(half: u16) -> f32 {
        let sign = f32::from_bits(u32::from(half & 0x8000) << 16);
        let exponent = i32::from((half >> 10) & 0x1f);
        let mantissa = f32::from(half & 0x03ff);
        let magnitude = match exponent {
            0 => mantissa * 2.0f32.powi(-24),
            0x1f if mantissa == 0.0 => f32::INFINITY,
            0x1f => f32::NAN,
            _ => (1.0 + mantissa / 1024.0) * 2.0f32.powi(exponent - 15),
        };
        magnitude.copysign(if sign.is_sign_negative() { -1.0 } else { 1.0 })
    }

    #[test]
    fn exact_values_survive() {
        for value in [0.0, 1.0, 0.5, -1.0, -0.25, 2.0, 1024.0, 65504.0] {
            assert_eq!(from_f16(to_f16(value)), value, "{value}");
        }
    }

    /// The precision a lookup table actually relies on: a half has 11 bits of
    /// mantissa, so anything in 0..1 comes back within about 1/2048 of itself.
    #[test]
    fn unit_range_keeps_its_precision() {
        for step in 0..=1000 {
            let value = step as f32 / 1000.0;
            let error = (from_f16(to_f16(value)) - value).abs();
            assert!(error <= 1.0 / 2048.0, "{value} came back off by {error}");
        }
    }

    #[test]
    fn rounds_to_nearest() {
        // Halfway between the two halves either side of 1.0 (1.0 + 1/2048),
        // which rounds up to 1.0 + 1/1024.
        assert_eq!(from_f16(to_f16(1.0 + 1.0 / 2048.0)), 1.0 + 1.0 / 1024.0);
        // A quarter of the way, which rounds back down to 1.0.
        assert_eq!(from_f16(to_f16(1.0 + 1.0 / 4096.0)), 1.0);
    }

    #[test]
    fn extremes_are_handled() {
        assert_eq!(from_f16(to_f16(f32::INFINITY)), f32::INFINITY);
        assert_eq!(from_f16(to_f16(-f32::INFINITY)), f32::NEG_INFINITY);
        assert!(from_f16(to_f16(f32::NAN)).is_nan());
        // Past half's range.
        assert_eq!(from_f16(to_f16(1.0e9)), f32::INFINITY);
        // Under its smallest normal, where subnormals take over: the value is
        // kept to within one subnormal step of 2^-24.
        let subnormal_step = 2.0f32.powi(-24);
        assert!((from_f16(to_f16(1.0e-7)) - 1.0e-7).abs() <= subnormal_step);
        // And under those, where there is nothing left to represent.
        assert_eq!(from_f16(to_f16(1.0e-30)), 0.0);
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use crate::test_util;

    #[test]
    fn depth_resizes_only_when_it_has_to() {
        let Some(device) = test_util::device() else {
            return;
        };
        let mut depth = DepthTexture::new(&device, "test depth", 64, 32);
        assert!(!depth.resize(&device, 64, 32), "same size");
        assert!(!depth.resize(&device, 0, 32), "degenerate size");
        assert!(depth.resize(&device, 128, 64));
        assert_eq!(depth.size(), (128, 64));
    }

    #[test]
    fn storage_and_render_textures_resize_only_when_they_have_to() {
        let Some(device) = test_util::device() else {
            return;
        };
        let mut storage =
            StorageTexture::new(&device, "test storage", 64, 32, StorageFormat::Rgba16Float);
        assert!(!storage.resize(&device, 64, 32), "same size");
        assert!(!storage.resize(&device, 0, 32), "degenerate size");
        assert!(storage.resize(&device, 128, 64));
        assert_eq!(storage.size(), (128, 64));

        let mut render =
            RenderTexture::new(&device, "test render", 64, 32, TextureFormat::RGBA8_SRGB);
        assert!(!render.resize(&device, 64, 32), "same size");
        assert!(!render.resize(&device, 64, 0), "degenerate size");
        assert!(render.resize(&device, 128, 64));
        assert_eq!(render.size(), (128, 64));
    }

    #[test]
    fn cube_and_image_textures_are_created() {
        let Some(device) = test_util::device() else {
            return;
        };
        let cube = TextureCube::new(&device, "test cube", 16, StorageFormat::Rgba16Float);
        assert_eq!(cube.size(), 16);
        let image = Texture2d::from_rgba_f32(&device, "test image", 4, 2, &[0.5; 32]);
        assert_eq!(image.size(), (4, 2));
    }
}
