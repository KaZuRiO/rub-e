//! An environment drawn behind everything, from an equirectangular HDR image.

use etib_core::{
    BindGroup, BindLayout, Blend, Compare, ComputePipeline, Cull, Depth, Device, Filter,
    RasterDesc, RenderPass, RenderPipeline, Sampler, Shader, Stage, StorageFormat, Texture2d,
    TextureCube, TextureFormat, UniformBuffer,
};
use glam::Mat4;
use log::{debug, info};

use crate::error::{Error, Result};

/// The equirectangular-to-cube shader's workgroup side.
const WORKGROUP: u32 = 16;

/// A cube map and the pipeline that draws it.
///
/// It is drawn last, as one triangle at the far plane, so the depth test lets
/// it through only where nothing else was drawn.
pub(crate) struct Skybox {
    pipeline: RenderPipeline,
    bindings: BindGroup,
    /// Bound in `bindings`; held so they live as long.
    _cube: TextureCube,
    _sampler: Sampler,
    _rotation: UniformBuffer<[[f32; 4]; 4]>,
}

impl Skybox {
    /// Decodes a Radiance `.hdr` image and projects it onto a cube with faces
    /// of `face_size` texels, turned by `rotation`.
    ///
    /// The projection runs on the GPU in a submission of its own, so the cube
    /// is ready by the time the next frame samples it.
    pub fn from_hdr_bytes(
        device: &Device,
        camera_layout: &BindLayout,
        target: TextureFormat,
        bytes: &[u8],
        face_size: u32,
        rotation: Mat4,
    ) -> Result<Self> {
        let image = image::load_from_memory_with_format(bytes, image::ImageFormat::Hdr)
            .map_err(|err| Error::Image {
                source: Box::new(err),
            })?
            .to_rgba32f();
        let (width, height) = image.dimensions();
        info!("projecting a {width}x{height} sky onto a {face_size}² cube");

        let source =
            Texture2d::from_rgba_f32(device, "sky equirectangular", width, height, image.as_raw());
        let cube = TextureCube::new(device, "sky", face_size, StorageFormat::Rgba16Float);
        project(device, &source, &cube);

        let layout = BindLayout::build(device, "skybox")
            .texture_cube(Stage::Fragment)
            .sampler(Stage::Fragment)
            .uniform::<[[f32; 4]; 4]>(Stage::Fragment)
            .finish();
        let sampler = Sampler::new(device, "skybox", Filter::Linear);
        let rotation_buffer = UniformBuffer::new(device, "skybox rotation");
        rotation_buffer.write(device, &rotation.to_cols_array_2d());
        let bindings = layout.group(device, "skybox", &[cube.view(), &sampler, &rotation_buffer]);

        let shader = Shader::wgsl(device, "skybox", include_str!("shaders/skybox.wgsl"));
        let pipeline = RenderPipeline::raster(
            device,
            &RasterDesc {
                label: "skybox",
                shader: &shader,
                entry_points: ("vs_main", "fs_main"),
                layouts: &[camera_layout, &layout],
                vertex: &[],
                target,
                // At the far plane, where a cleared depth of 1.0 still lets it
                // through, and recording nothing.
                depth: Some(Depth {
                    write: false,
                    compare: Compare::LessEqual,
                }),
                blend: Blend::Replace,
                cull: Cull::None,
            },
        );

        Ok(Self {
            pipeline,
            bindings,
            _cube: cube,
            _sampler: sampler,
            _rotation: rotation_buffer,
        })
    }

    pub fn draw(&self, pass: &mut RenderPass<'_>, camera: &BindGroup) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, camera);
        pass.set_bind_group(1, &self.bindings);
        pass.draw_vertices(0..3);
    }
}

/// Fills the six faces of `cube` from an equirectangular `source`.
fn project(device: &Device, source: &Texture2d, cube: &TextureCube) {
    let layout = BindLayout::build(device, "equirectangular")
        .texture(Stage::Compute)
        .storage_texture_array(Stage::Compute, cube.format())
        .finish();
    let shader = Shader::wgsl(
        device,
        "equirectangular",
        include_str!("shaders/equirectangular.wgsl"),
    );
    let pipeline = ComputePipeline::new(
        device,
        "equirectangular",
        &shader,
        "compute_equirect_to_cubemap",
        &layout,
    );
    let bindings = layout.group(
        device,
        "equirectangular",
        &[source.view(), cube.storage_view()],
    );

    let groups = cube.size().div_ceil(WORKGROUP);
    let mut encoder = device.encoder("equirectangular");
    encoder
        .compute_pass("equirectangular")
        .dispatch(&pipeline, &bindings, (groups, groups, 6));
    device.submit(encoder);
    debug!("sky projected in {groups}x{groups}x6 workgroups");
}
