//! The scene: static and dynamic cubes, culled and drawn into a
//! scene-referred target, with a skybox behind them.

use etib_core::{
    BindGroup, BindLayout, Blend, Color, Cull, Depth, DepthLoad, DepthTexture, Device, Encoder,
    IndexBuffer, Load, RasterDesc, RenderPipeline, RenderTexture, Shader, Stage, StorageBuffer,
    StorageFormat, TextureView, UniformBuffer, VertexBuffer,
};
use glam::Mat4;

use crate::camera::{Camera, CameraRaw};
use crate::cube::{CubeRaw, INDICES, VERTICES, Vertex};
use crate::culling::{CullingPasses, StaticCubes};
use crate::dynamic::{DynamicId, DynamicModel, DynamicModels};
use crate::error::Result;
use crate::model::ModelCube;
use crate::skybox::Skybox;

/// What the scene is cleared to where neither a cube nor a skybox is drawn.
const BACKGROUND: Color = Color::BLACK;

/// Cubes, culled on the GPU and drawn with depth into an `Rgba16Float`
/// texture, with an optional skybox behind them.
///
/// The output is **scene-referred**: nothing here tone maps. The application
/// hands [`target_view`](Self::target_view) to `etib::PostProcess`, which does,
/// and can paint an interface over the result:
///
/// ```text
/// scene.render(device, encoder, &mut camera);           // cull + draw
/// post.draw(device, encoder, frame.view, &grade, out);  // tone map
/// ```
///
/// Static cubes are fixed at construction and culled two-pass on the GPU.
/// Dynamic models are added, moved and removed at runtime; they are few, so
/// they are drawn without culling.
pub struct Scene {
    pipeline: RenderPipeline,
    camera_layout: BindLayout,
    camera: UniformBuffer<CameraRaw>,
    camera_group: BindGroup,
    vertices: VertexBuffer<Vertex>,
    indices: IndexBuffer,

    culling: CullingPasses,
    statics: Option<StaticCubes>,

    dynamic: DynamicModels,
    dynamic_buffer: StorageBuffer<CubeRaw>,
    /// How many dynamic instances the buffer holds as of the last upload.
    dynamic_uploaded: u32,

    skybox: Option<Skybox>,

    target: RenderTexture,
    depth: DepthTexture,
}

impl Scene {
    /// The format of [`target_view`](Self::target_view).
    pub const TARGET: StorageFormat = StorageFormat::Rgba16Float;

    /// Builds a scene of `static_cubes`, with room for `dynamic_capacity`
    /// cubes across all the dynamic models, drawing at `width` x `height`.
    pub fn new(
        device: &Device,
        static_cubes: &[ModelCube],
        dynamic_capacity: usize,
        width: u32,
        height: u32,
    ) -> Self {
        let camera_layout = BindLayout::build(device, "camera")
            .uniform::<CameraRaw>(Stage::All)
            .finish();
        let camera = UniformBuffer::new(device, "camera");
        let camera_group = camera_layout.group(device, "camera", &[&camera]);

        let shader = Shader::wgsl(device, "cubes", include_str!("shaders/cube.wgsl"));
        let pipeline = RenderPipeline::raster(
            device,
            &RasterDesc {
                label: "cubes",
                shader: &shader,
                entry_points: ("vs_main", "fs_main"),
                layouts: &[&camera_layout],
                vertex: &[Vertex::LAYOUT, CubeRaw::LAYOUT],
                target: Self::TARGET.format(),
                depth: Some(Depth::default()),
                blend: Blend::Replace,
                cull: Cull::Back,
            },
        );

        let culling = CullingPasses::new(device, &camera_layout);
        let statics = StaticCubes::new(device, &culling, static_cubes);

        let dynamic_capacity = dynamic_capacity.max(1);
        let dynamic_buffer =
            StorageBuffer::new(device, "dynamic instances", dynamic_capacity as u32);

        Self {
            pipeline,
            camera_layout,
            camera,
            camera_group,
            vertices: VertexBuffer::new(device, "cube vertices", VERTICES),
            indices: IndexBuffer::u16(device, "cube indices", INDICES),
            culling,
            statics,
            dynamic: DynamicModels::new(dynamic_capacity),
            dynamic_buffer,
            dynamic_uploaded: 0,
            skybox: None,
            target: RenderTexture::new(device, "scene", width, height, Self::TARGET.format()),
            depth: DepthTexture::new(device, "scene depth", width, height),
        }
    }

    /// Sets the skybox from a Radiance `.hdr` equirectangular image, projected
    /// onto a cube with faces of `face_size` texels and turned by `rotation`.
    /// Replaces any previous one.
    pub fn set_skybox_from_hdr(
        &mut self,
        device: &Device,
        bytes: &[u8],
        face_size: u32,
        rotation: Mat4,
    ) -> Result<()> {
        self.skybox = Some(Skybox::from_hdr_bytes(
            device,
            &self.camera_layout,
            Self::TARGET.format(),
            bytes,
            face_size,
            rotation,
        )?);
        Ok(())
    }

    pub fn clear_skybox(&mut self) {
        self.skybox = None;
    }

    /// Adds a dynamic model and returns the id that moves or removes it.
    ///
    /// # Errors
    ///
    /// [`Error::Capacity`](crate::Error::Capacity) if the dynamic models would
    /// hold more cubes than the scene was built with room for; nothing is
    /// added then.
    pub fn add_dynamic(&mut self, model: DynamicModel) -> Result<DynamicId> {
        self.dynamic.add(model)
    }

    /// Removes a dynamic model; a stale id is ignored.
    pub fn remove_dynamic(&mut self, id: DynamicId) {
        self.dynamic.remove(id);
    }

    pub fn dynamic(&self, id: DynamicId) -> Option<&DynamicModel> {
        self.dynamic.get(id)
    }

    /// The model, to move or recolour. Changes are uploaded by the next
    /// [`render`](Self::render).
    pub fn dynamic_mut(&mut self, id: DynamicId) -> Option<&mut DynamicModel> {
        self.dynamic.get_mut(id)
    }

    pub fn static_count(&self) -> u32 {
        self.statics.as_ref().map_or(0, |statics| statics.count)
    }

    /// Cubes across all the live dynamic models.
    pub fn dynamic_count(&self) -> u32 {
        self.dynamic.cube_count() as u32
    }

    /// The scene-referred image, for the presentation pass to sample.
    pub fn target_view(&self) -> &TextureView {
        self.target.view()
    }

    pub fn size(&self) -> (u32, u32) {
        self.target.size()
    }

    /// Resizes the target and its depth. Returns `true` when the target was
    /// recreated, which is the cue to rebind whatever samples it.
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) -> bool {
        self.depth.resize(device, width, height);
        self.target.resize(device, width, height)
    }

    /// Records the frame: culling, then one depth-tested pass of static
    /// cubes, dynamic cubes and the skybox, into [`target_view`](Self::target_view).
    ///
    /// `camera.aspect` is set to the scene's own, so a resize needs nothing
    /// else to keep the image in proportion.
    pub fn render(&mut self, device: &Device, encoder: &mut Encoder, camera: &mut Camera) {
        let (width, height) = self.target.size();
        camera.aspect = width as f32 / height as f32;
        self.camera.write(device, &camera.raw());

        if let Some(instances) = self.dynamic.take_changes() {
            if !instances.is_empty() {
                self.dynamic_buffer.write(device, 0, instances);
            }
            self.dynamic_uploaded = instances.len() as u32;
        }

        if let Some(statics) = &self.statics {
            statics.reset(device);
            statics.cull(
                &mut encoder.compute_pass("cull"),
                &self.culling,
                &self.camera_group,
            );
        }

        let mut pass = encoder.render_pass_with_depth(
            "scene",
            self.target.view(),
            Load::Clear(BACKGROUND),
            &self.depth,
            DepthLoad::Clear(1.0),
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.camera_group);
        pass.set_vertex_buffer(0, &self.vertices);
        pass.set_index_buffer(&self.indices);

        if let Some(statics) = &self.statics {
            pass.set_vertex_buffer(1, &statics.visible);
            pass.draw_indexed_indirect(&statics.draw, 0);
        }
        if self.dynamic_uploaded > 0 {
            pass.set_vertex_buffer(1, &self.dynamic_buffer);
            pass.draw_indexed(0..self.indices.len(), 0, 0..self.dynamic_uploaded);
        }
        if let Some(skybox) = &self.skybox {
            skybox.draw(&mut pass, &self.camera_group);
        }
    }
}

#[cfg(test)]
mod tests {
    use etib_core::test_util::device;
    use etib_core::wgpu;
    use glam::Vec3;

    use super::*;
    use crate::camera::Projection;

    fn white_cube(x: f32, y: f32, z: f32) -> ModelCube {
        ModelCube {
            position: Vec3::new(x, y, z),
            color: Vec3::ONE,
        }
    }

    fn camera() -> Camera {
        Camera::new(
            Vec3::new(0.0, 5.0, 10.0),
            Vec3::ZERO,
            Vec3::Y,
            1.0,
            Projection::Perspective { fovy: 60.0 },
            0.1,
            1000.0,
        )
    }

    /// A 2x1 Radiance image, as the engine's tests used.
    fn tiny_hdr() -> Vec<u8> {
        use image::codecs::hdr::HdrEncoder;
        let pixels = [image::Rgb([1.0f32, 1.0, 1.0]), image::Rgb([0.5, 0.5, 0.5])];
        let mut bytes = Vec::new();
        HdrEncoder::new(&mut bytes)
            .encode(&pixels, 2, 1)
            .expect("encoding a test image");
        bytes
    }

    fn finish(device: &Device, encoder: Encoder) {
        device.submit(encoder);
        device
            .raw()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device lost");
    }

    #[test]
    fn empty_scene_has_no_static_geometry() {
        let Some(device) = device() else { return };
        let mut scene = Scene::new(&device, &[], 10, 800, 600);
        assert_eq!(scene.static_count(), 0);
        assert_eq!(scene.dynamic_count(), 0);

        let mut encoder = device.encoder("test");
        scene.render(&device, &mut encoder, &mut camera());
        finish(&device, encoder);
    }

    #[test]
    fn static_cubes_are_split_into_chunks() {
        let Some(device) = device() else { return };
        let cubes = [
            white_cube(0.0, 0.0, 0.0),
            white_cube(33.0, 0.0, 0.0),
            white_cube(66.0, 0.0, 0.0),
        ];
        let scene = Scene::new(&device, &cubes, 10, 800, 600);
        assert_eq!(scene.static_count(), 3);
        assert_eq!(scene.statics.as_ref().unwrap().chunk_count, 3);
    }

    #[test]
    fn dynamic_models_are_added_moved_and_removed() {
        let Some(device) = device() else { return };
        let mut scene = Scene::new(&device, &[], 10, 800, 600);
        let id = scene
            .add_dynamic(DynamicModel::from_cubes(vec![white_cube(0.0, 0.0, 0.0)]))
            .unwrap();
        assert_eq!(scene.dynamic_count(), 1);

        scene.dynamic_mut(id).unwrap().position = Vec3::new(5.0, 0.0, 0.0);
        assert_eq!(
            scene.dynamic(id).unwrap().position,
            Vec3::new(5.0, 0.0, 0.0)
        );

        let mut encoder = device.encoder("test");
        scene.render(&device, &mut encoder, &mut camera());
        finish(&device, encoder);
        assert_eq!(scene.dynamic_uploaded, 1);

        scene.remove_dynamic(id);
        assert!(scene.dynamic(id).is_none());
        scene.remove_dynamic(id);
    }

    #[test]
    fn a_full_frame_renders() {
        let Some(device) = device() else { return };
        let cubes: Vec<_> = (0..100)
            .map(|i| white_cube((i % 10) as f32 * 4.0, 0.0, (i / 10) as f32 * 4.0))
            .collect();
        let mut scene = Scene::new(&device, &cubes, 4, 64, 64);
        scene
            .add_dynamic(DynamicModel::from_cubes(vec![white_cube(0.0, 2.0, 0.0); 4]))
            .unwrap();
        scene
            .set_skybox_from_hdr(&device, &tiny_hdr(), 16, Mat4::IDENTITY)
            .expect("the skybox loads");

        let mut camera = camera();
        for _ in 0..2 {
            let mut encoder = device.encoder("test");
            scene.render(&device, &mut encoder, &mut camera);
            finish(&device, encoder);
        }
    }

    #[test]
    fn resizing_reports_a_new_target() {
        let Some(device) = device() else { return };
        let mut scene = Scene::new(&device, &[], 10, 800, 600);
        assert!(!scene.resize(&device, 800, 600));
        assert!(scene.resize(&device, 1280, 720));
        assert_eq!(scene.size(), (1280, 720));

        let mut camera = camera();
        let mut encoder = device.encoder("test");
        scene.render(&device, &mut encoder, &mut camera);
        finish(&device, encoder);
        assert_eq!(camera.aspect, 1280.0 / 720.0);
    }

    #[test]
    fn a_broken_skybox_is_an_error() {
        let Some(device) = device() else { return };
        let mut scene = Scene::new(&device, &[], 10, 64, 64);
        let err = scene
            .set_skybox_from_hdr(&device, b"not an image", 16, Mat4::IDENTITY)
            .unwrap_err();
        assert!(matches!(err, crate::Error::Image { .. }), "{err:?}");
    }
}
