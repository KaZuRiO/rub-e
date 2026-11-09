//! The scene: static and dynamic cubes, culled and drawn into a
//! scene-referred target, with a skybox behind them.

use etib_core::{
    BindGroup, BindLayout, Blend, Color, Cull, Depth, DepthLoad, DepthTexture, Device, Encoder,
    IndexBuffer, Load, RasterDesc, RenderPipeline, RenderTexture, Shader, Stage, StorageBuffer,
    StorageFormat, TextureView, UniformBuffer, VertexBuffer,
};
use glam::{IVec3, Mat4, Vec3};

use crate::camera::{Camera, CameraRaw};
use crate::cube::{CubeRaw, INDICES, VERTICES, Vertex};
use crate::culling::{CHUNK_SIZE, CullingPasses, StaticCubes};
use crate::dynamic::{DynamicId, DynamicModel, DynamicModels};
use crate::error::Result;
use crate::model::ModelCube;
use crate::rebuild::{Candidate, DEFAULT_REBUILD_BUDGET, plan_rebuilds};
use crate::skybox::Skybox;
use crate::world::{BlockEdit, World};

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
/// Static cubes are the scenery: blocks on a grid of cells, set and removed
/// one by one with [`set_block`](Self::set_block) or a whole box at a time with
/// [`edit_region`](Self::edit_region), within the static capacity given at
/// construction, and culled two-pass on the GPU. Only the chunks an edit
/// touched are sent again. Dynamic models are added, moved and removed at runtime; they are few, so
/// they are drawn without culling.
pub struct Scene {
    pipeline: RenderPipeline,
    camera_layout: BindLayout,
    camera: UniformBuffer<CameraRaw>,
    camera_group: BindGroup,
    vertices: VertexBuffer<Vertex>,
    indices: IndexBuffer,

    culling: CullingPasses,
    /// The static cubes as the game sees them.
    world: World,
    /// Their copy on the GPU; `None` for a static capacity of 0.
    statics: Option<StaticCubes>,
    /// Cubes per frame rebuilt for region edits.
    rebuild_budget: u32,

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

    /// Builds a scene of `static_cubes`, with room for `static_capacity`
    /// static cubes and `dynamic_capacity` cubes across all the dynamic
    /// models, drawing at `width` x `height`.
    ///
    /// The GPU memory of the static cubes follows `static_capacity`, at about
    /// 220 bytes a cube.
    ///
    /// # Panics
    ///
    /// If `static_capacity` is below `static_cubes.len()`.
    pub fn new(
        device: &Device,
        static_cubes: &[ModelCube],
        static_capacity: usize,
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

        assert!(
            static_cubes.len() <= static_capacity,
            "{} static cubes do not fit a static capacity of {static_capacity}",
            static_cubes.len()
        );
        let world = World::new(static_cubes, static_capacity);
        let culling = CullingPasses::new(device, &camera_layout);
        let statics = StaticCubes::new(device, &culling, &world);

        let dynamic_capacity = dynamic_capacity.max(1);
        let dynamic_buffer =
            StorageBuffer::new(device, "dynamic instances", dynamic_slots(dynamic_capacity));

        Self {
            pipeline,
            camera_layout,
            camera,
            camera_group,
            vertices: VertexBuffer::new(device, "cube vertices", VERTICES),
            indices: IndexBuffer::u16(device, "cube indices", INDICES),
            culling,
            world,
            statics,
            rebuild_budget: DEFAULT_REBUILD_BUDGET,
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

    /// Removes the skybox; the background is cleared to black again.
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

    /// The model `id` names, or `None` once it has been removed.
    pub fn dynamic(&self, id: DynamicId) -> Option<&DynamicModel> {
        self.dynamic.get(id)
    }

    /// The model, to move or recolour. Changes are uploaded by the next
    /// [`render`](Self::render), which panics if a model replaced or grown
    /// through here takes the scene past its dynamic capacity.
    pub fn dynamic_mut(&mut self, id: DynamicId) -> Option<&mut DynamicModel> {
        self.dynamic.get_mut(id)
    }

    /// Puts a block of linear RGB `color` in `cell`, replacing any. It is on
    /// screen from the next frame: its chunk is rebuilt within that frame,
    /// whatever the budget. That suits a player's few blocks a frame; to
    /// change many at once, [`edit_region`](Self::edit_region) spreads the
    /// work over frames instead.
    ///
    /// # Errors
    ///
    /// [`Error::Capacity`](crate::Error::Capacity) if the scene would hold
    /// more static cubes than it was built with room for; nothing changes
    /// then.
    pub fn set_block(&mut self, cell: IVec3, color: Vec3) -> Result<()> {
        self.world.set_block(cell, color)
    }

    /// Empties `cell`, on screen from the next frame. An empty cell is left
    /// alone, and so are off-grid cubes.
    pub fn remove_block(&mut self, cell: IVec3) {
        self.world.remove_block(cell);
    }

    /// The colour of the block in `cell`, as of the last edit, even while the
    /// screen still shows its chunk as it was.
    pub fn block(&self, cell: IVec3) -> Option<Vec3> {
        self.world.block(cell)
    }

    /// Calls `edit` for each cell of the box from `min` to `max`, both
    /// included, and applies the answers as one edit. An off-grid cube
    /// follows the answer for the cell its centre rounds to. A box with `min`
    /// above `max` on any axis does nothing.
    ///
    /// The changed chunks reach the screen over the next frames, nearest the
    /// camera first, within the [rebuild budget](Self::set_rebuild_budget):
    /// an explosion never freezes the image, and until a chunk's turn comes
    /// the screen shows it as it was. [`block`](Self::block) and
    /// [`static_count`](Self::static_count) see the change at once.
    ///
    /// # Errors
    ///
    /// [`Error::Capacity`](crate::Error::Capacity) if the scene would hold
    /// more static cubes than it was built with room for; nothing changes
    /// then.
    pub fn edit_region(
        &mut self,
        min: IVec3,
        max: IVec3,
        edit: impl FnMut(IVec3) -> BlockEdit,
    ) -> Result<()> {
        self.world.edit_region(min, max, edit)
    }

    /// The cubes rebuilt per frame for [`edit_region`](Self::edit_region),
    /// [`DEFAULT_REBUILD_BUDGET`](crate::DEFAULT_REBUILD_BUDGET) to begin
    /// with. Counted in cubes, not time, so the same edits always take the
    /// same frames. One chunk a frame goes through even past it, so a budget
    /// of 0 still gets there.
    pub fn set_rebuild_budget(&mut self, cubes: u32) {
        self.rebuild_budget = cubes;
    }

    /// The chunks whose new version is not on screen yet.
    pub fn pending_rebuilds(&self) -> usize {
        self.world.dirty().count()
    }

    /// How many static cubes the scene holds, before culling.
    pub fn static_count(&self) -> u32 {
        u32::try_from(self.world.cube_count()).expect("a static capacity under u32::MAX")
    }

    /// Cubes across all the live dynamic models.
    pub fn dynamic_count(&self) -> u32 {
        dynamic_slots(self.dynamic.cube_count())
    }

    /// The scene-referred image, for the presentation pass to sample.
    pub fn target_view(&self) -> &TextureView {
        self.target.view()
    }

    /// The size of the target, in pixels.
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
        self.camera.write(device, &camera.advance_frame());

        if let Some(instances) = self.dynamic.take_changes() {
            if !instances.is_empty() {
                self.dynamic_buffer.write(device, 0, instances);
            }
            self.dynamic_uploaded = dynamic_slots(instances.len());
        }

        self.rebuild_chunks(device, camera.eye);

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

impl Scene {
    /// Writes to the GPU the changed chunks this frame has room and budget
    /// for. Nothing at all happens in a frame with no edits waiting.
    fn rebuild_chunks(&mut self, device: &Device, eye: Vec3) {
        let Some(statics) = &mut self.statics else {
            return;
        };
        if self.world.dirty().next().is_none() {
            return;
        }
        let candidates: Vec<Candidate> = self
            .world
            .dirty()
            .map(|(key, priority)| {
                let centre = (IVec3::from_array(key).as_vec3() + 0.5) * CHUNK_SIZE;
                Candidate {
                    key,
                    priority,
                    distance: centre.distance(eye),
                    old_len: statics.gpu_len(key),
                    new_len: self.world.chunk_len(key),
                }
            })
            .collect();
        for key in plan_rebuilds(&candidates, statics.free_slots(), self.rebuild_budget) {
            statics.rebuild(device, &self.world, key);
            self.world.clean(key);
        }
    }
}

/// A number of dynamic cubes as the `u32` instance count a draw takes. Every
/// count it is given is at most the capacity, which went through it first.
///
/// # Panics
///
/// For a capacity past `u32::MAX` cubes.
fn dynamic_slots(cubes: usize) -> u32 {
    u32::try_from(cubes).expect("a dynamic capacity of more than u32::MAX cubes")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use etib_core::test_util::device;
    use etib_core::wgpu;
    use glam::{IVec3, Vec3};

    use super::*;
    use crate::camera::Projection;
    use crate::world::BlockEdit;

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
        let mut scene = Scene::new(&device, &[], 0, 10, 800, 600);
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
        let scene = Scene::new(&device, &cubes, cubes.len(), 10, 800, 600);
        assert_eq!(scene.static_count(), 3);
        assert_eq!(scene.statics.as_ref().unwrap().chunk_count(), 3);
    }

    #[test]
    fn dynamic_models_are_added_moved_and_removed() {
        let Some(device) = device() else { return };
        let mut scene = Scene::new(&device, &[], 0, 10, 800, 600);
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
        let mut scene = Scene::new(&device, &cubes, cubes.len(), 4, 64, 64);
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

    /// How many static cubes survived culling in the frame `camera` sees.
    fn cubes_kept(device: &Device, cubes: &[ModelCube], camera: &mut Camera) -> u32 {
        let mut scene = Scene::new(device, cubes, cubes.len(), 1, 64, 64);
        let mut encoder = device.encoder("test");
        scene.render(device, &mut encoder, camera);
        finish(device, encoder);
        let statics = scene.statics.as_ref().expect("the scene has static cubes");
        statics.draw.read_back(device)[0].instance_count
    }

    #[test]
    fn culling_keeps_only_cubes_in_view() {
        let Some(device) = device() else { return };
        let cubes = [
            white_cube(0.0, 0.0, 0.0),     // in view
            white_cube(1.0, 0.0, 0.0),     // in view
            white_cube(0.0, 1.0, -3.0),    // in view
            white_cube(25.0, 0.0, 0.0),    // right of the view, same chunk as those
            white_cube(0.0, 5.0, 40.0),    // behind the eye
            white_cube(-500.0, 0.0, 0.0),  // far to the left, in its own chunk
            white_cube(0.0, 0.0, -2000.0), // past the far plane
        ];
        assert_eq!(cubes_kept(&device, &cubes, &mut camera()), 3);
    }

    #[test]
    fn ortho_culling_keeps_only_cubes_in_the_box() {
        let Some(device) = device() else { return };
        let mut camera = Camera::new(
            Vec3::new(0.0, 5.0, 10.0),
            Vec3::ZERO,
            Vec3::Y,
            1.0,
            Projection::Orthographic { scale: 5.0 },
            0.1,
            1000.0,
        );
        let cubes = [
            white_cube(0.0, 0.0, 0.0),   // in the box
            white_cube(1.0, 0.0, 0.0),   // in the box
            white_cube(-50.0, 0.0, 0.0), // left of it, in its own chunk
            white_cube(0.0, 60.0, 0.0),  // above it
        ];
        assert_eq!(cubes_kept(&device, &cubes, &mut camera), 2);
    }

    /// The planes follow the scene's own aspect from the first frame: a cube
    /// only a wide view takes in is kept by a wide scene, even with a camera
    /// built square.
    #[test]
    fn culling_follows_the_scene_aspect() {
        let Some(device) = device() else { return };
        // About 6.5 units off-axis is the edge of a square view at this
        // distance; a 4:1 view reaches about 26.
        let cubes = [white_cube(15.0, 0.0, 0.0)];
        let mut scene = Scene::new(&device, &cubes, cubes.len(), 1, 128, 32);
        let mut camera = camera();
        assert_eq!(camera.aspect, 1.0);

        let mut encoder = device.encoder("test");
        scene.render(&device, &mut encoder, &mut camera);
        finish(&device, encoder);
        let statics = scene.statics.as_ref().expect("the scene has static cubes");
        assert_eq!(statics.draw.read_back(&device)[0].instance_count, 1);
    }

    #[test]
    fn resizing_reports_a_new_target() {
        let Some(device) = device() else { return };
        let mut scene = Scene::new(&device, &[], 0, 10, 800, 600);
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
    fn an_empty_chunk_row_is_never_visible() {
        let Some(device) = device() else { return };
        let mut scene = Scene::new(&device, &[white_cube(0.0, 0.0, 0.0)], 8, 1, 64, 64);
        let mut encoder = device.encoder("test");
        scene.render(&device, &mut encoder, &mut camera());
        finish(&device, encoder);
        let statics = scene.statics.as_ref().unwrap();
        assert_eq!(statics.draw.read_back(&device)[0].instance_count, 1);
    }

    #[test]
    fn a_capacity_of_zero_makes_no_static_buffers() {
        let Some(device) = device() else { return };
        assert!(Scene::new(&device, &[], 0, 1, 64, 64).statics.is_none());
    }

    #[test]
    #[should_panic(expected = "static capacity")]
    fn more_cubes_than_capacity_panics() {
        let Some(device) = device() else {
            panic!("static capacity")
        };
        Scene::new(&device, &[white_cube(0.0, 0.0, 0.0)], 0, 1, 64, 64);
    }

    fn render_once(device: &Device, scene: &mut Scene) {
        let mut encoder = device.encoder("test");
        scene.render(device, &mut encoder, &mut camera());
        finish(device, encoder);
    }

    fn colour(i: i32) -> Vec3 {
        Vec3::new(
            (i % 7) as f32 / 7.0,
            (i % 5) as f32 / 5.0,
            (i % 3) as f32 / 3.0,
        )
    }

    fn block_cubes(blocks: &BTreeMap<[i32; 3], Vec3>) -> Vec<ModelCube> {
        blocks
            .iter()
            .map(|(cell, color)| ModelCube {
                position: IVec3::from_array(*cell).as_vec3(),
                color: *color,
            })
            .collect()
    }

    /// The GPU holds what a scene built from `cubes` would.
    fn same_as_fresh(device: &Device, scene: &Scene, cubes: &[ModelCube]) {
        let mut fresh = Scene::new(device, cubes, cubes.len().max(1), 1, 64, 64);
        render_once(device, &mut fresh);
        let bytes = |s: &Scene| -> Vec<u8> {
            bytemuck::cast_slice(&s.statics.as_ref().unwrap().live_instances(device)).to_vec()
        };
        assert_eq!(bytes(scene), bytes(&fresh));
    }

    fn visible_count(device: &Device, scene: &Scene) -> u32 {
        scene.statics.as_ref().unwrap().draw.read_back(device)[0].instance_count
    }

    #[test]
    fn a_block_is_on_the_gpu_at_the_next_frame() {
        let Some(device) = device() else { return };
        let mut scene = Scene::new(&device, &[white_cube(0.0, 0.0, 0.0)], 4, 1, 64, 64);
        render_once(&device, &mut scene);
        scene.set_block(IVec3::new(1, 0, 0), colour(1)).unwrap();
        render_once(&device, &mut scene);
        assert_eq!(scene.pending_rebuilds(), 0);
        same_as_fresh(
            &device,
            &scene,
            &[
                white_cube(0.0, 0.0, 0.0),
                ModelCube {
                    position: Vec3::X,
                    color: colour(1),
                },
            ],
        );
    }

    #[test]
    fn a_rebuilt_chunk_matches_a_fresh_scene_after_many_edits() {
        let Some(device) = device() else { return };
        let mut scene = Scene::new(&device, &[], 32, 1, 64, 64);
        let mut blocks = BTreeMap::new();
        for i in 0..20 {
            // Three chunks along X.
            let cell = IVec3::new(i * 5 - 20, i % 3, -(i % 4));
            scene.set_block(cell, colour(i)).unwrap();
            blocks.insert(cell.to_array(), colour(i));
        }
        render_once(&device, &mut scene);
        for i in (0..20).step_by(3) {
            let cell = IVec3::new(i * 5 - 20, i % 3, -(i % 4));
            scene.set_block(cell, colour(i + 1)).unwrap();
            blocks.insert(cell.to_array(), colour(i + 1));
        }
        for i in (1..20).step_by(4) {
            let cell = IVec3::new(i * 5 - 20, i % 3, -(i % 4));
            scene.remove_block(cell);
            blocks.remove(&cell.to_array());
        }
        render_once(&device, &mut scene);
        assert_eq!(scene.static_count() as usize, blocks.len());
        same_as_fresh(&device, &scene, &block_cubes(&blocks));
    }

    #[test]
    fn a_waiting_chunk_keeps_its_old_contents() {
        let Some(device) = device() else { return };
        let mut scene = Scene::new(
            &device,
            &[white_cube(0.0, 0.0, 0.0), white_cube(40.0, 0.0, 0.0)],
            8,
            1,
            64,
            64,
        );
        scene.set_rebuild_budget(1);
        render_once(&device, &mut scene);
        let before = scene.statics.as_ref().unwrap().live_instances(&device);
        scene
            .edit_region(IVec3::ZERO, IVec3::new(40, 0, 0), |_| BlockEdit::Remove)
            .unwrap();
        assert_eq!(scene.block(IVec3::ZERO), None);
        assert_eq!(scene.static_count(), 0);
        assert_eq!(
            scene.statics.as_ref().unwrap().live_instances(&device),
            before
        );
        assert_eq!(scene.pending_rebuilds(), 2);
        render_once(&device, &mut scene);
        assert_eq!(scene.pending_rebuilds(), 1);
        render_once(&device, &mut scene);
        assert_eq!(scene.pending_rebuilds(), 0);
    }

    #[test]
    fn past_capacity_a_scene_reports_it_and_changes_nothing() {
        let Some(device) = device() else { return };
        let mut scene = Scene::new(&device, &[white_cube(0.0, 0.0, 0.0)], 1, 1, 64, 64);
        let err = scene.set_block(IVec3::X, Vec3::ONE).unwrap_err();
        assert!(
            matches!(
                err,
                crate::Error::Capacity {
                    needed: 2,
                    capacity: 1
                }
            ),
            "{err:?}"
        );
        let err = scene
            .edit_region(IVec3::X, IVec3::new(3, 0, 0), |_| BlockEdit::Set(Vec3::ONE))
            .unwrap_err();
        assert!(matches!(err, crate::Error::Capacity { .. }), "{err:?}");
        assert_eq!(scene.static_count(), 1);
        assert_eq!(scene.pending_rebuilds(), 0);
    }

    #[test]
    fn a_capacity_of_zero_refuses_blocks() {
        let Some(device) = device() else { return };
        let mut scene = Scene::new(&device, &[], 0, 1, 64, 64);
        assert!(scene.set_block(IVec3::ZERO, Vec3::ONE).is_err());
        render_once(&device, &mut scene);
    }

    #[test]
    fn filling_emptying_and_refilling_in_another_order_never_fails() {
        let Some(device) = device() else { return };
        let mut scene = Scene::new(&device, &[], 64, 1, 64, 64);
        // 64 blocks, 16 in each of four chunks along X.
        let cells: Vec<IVec3> = (0..64)
            .map(|i| IVec3::new((i / 16) * 32 + i % 16, 0, 0))
            .collect();
        let mut blocks = BTreeMap::new();
        for (i, cell) in (0..).zip(&cells) {
            scene.set_block(*cell, colour(i)).unwrap();
            blocks.insert(cell.to_array(), colour(i));
        }
        render_once(&device, &mut scene);
        for round in 0..3i32 {
            for cell in cells.iter().skip(usize::from(round % 2 == 1)).step_by(2) {
                scene.remove_block(*cell);
                blocks.remove(&cell.to_array());
            }
            render_once(&device, &mut scene);
            for (i, cell) in (0..64).zip(&cells).rev() {
                if scene.block(*cell).is_none() {
                    scene.set_block(*cell, colour(i + round)).unwrap();
                    blocks.insert(cell.to_array(), colour(i + round));
                }
            }
            render_once(&device, &mut scene);
        }
        assert_eq!(scene.static_count(), 64);
        same_as_fresh(&device, &scene, &block_cubes(&blocks));
    }

    #[test]
    fn a_full_scene_can_swap_a_far_zone_for_a_near_one_and_take_a_block() {
        let Some(device) = device() else { return };
        let far: Vec<_> = (0..8)
            .map(|i| white_cube(500.0 + i as f32, 0.0, 0.0))
            .collect();
        let mut scene = Scene::new(&device, &far, 8, 1, 64, 64);
        scene.set_rebuild_budget(1);
        render_once(&device, &mut scene);

        scene
            .edit_region(IVec3::new(500, 0, 0), IVec3::new(507, 0, 0), |_| {
                BlockEdit::Remove
            })
            .unwrap();
        scene
            .edit_region(IVec3::ZERO, IVec3::new(6, 0, 0), |_| {
                BlockEdit::Set(Vec3::ONE)
            })
            .unwrap();
        scene.set_block(IVec3::new(7, 0, 0), Vec3::ONE).unwrap();
        render_once(&device, &mut scene);

        let near: Vec<_> = (0..8).map(|i| white_cube(i as f32, 0.0, 0.0)).collect();
        same_as_fresh(&device, &scene, &near);
    }

    #[test]
    fn a_scene_emptied_and_refilled_matches_a_fresh_one() {
        let Some(device) = device() else { return };
        let cubes = [white_cube(0.0, 0.0, 0.0), white_cube(1.0, 0.0, 0.0)];
        let mut scene = Scene::new(&device, &cubes, 4, 1, 64, 64);
        render_once(&device, &mut scene);
        scene
            .edit_region(IVec3::ZERO, IVec3::X, |_| BlockEdit::Remove)
            .unwrap();
        render_once(&device, &mut scene);
        assert_eq!(visible_count(&device, &scene), 0);

        scene.set_block(IVec3::new(0, 1, 0), colour(2)).unwrap();
        scene.set_block(IVec3::new(2, 0, -1), colour(3)).unwrap();
        render_once(&device, &mut scene);
        same_as_fresh(
            &device,
            &scene,
            &[
                ModelCube {
                    position: Vec3::Y,
                    color: colour(2),
                },
                ModelCube {
                    position: Vec3::new(2.0, 0.0, -1.0),
                    color: colour(3),
                },
            ],
        );
        assert_eq!(visible_count(&device, &scene), 2);
    }

    #[test]
    fn a_scene_built_empty_can_receive_blocks() {
        let Some(device) = device() else { return };
        let mut scene = Scene::new(&device, &[], 4, 1, 64, 64);
        render_once(&device, &mut scene);
        scene.set_block(IVec3::ZERO, Vec3::ONE).unwrap();
        render_once(&device, &mut scene);
        assert_eq!(visible_count(&device, &scene), 1);
        same_as_fresh(&device, &scene, &[white_cube(0.0, 0.0, 0.0)]);
    }

    #[test]
    fn a_broken_skybox_is_an_error() {
        let Some(device) = device() else { return };
        let mut scene = Scene::new(&device, &[], 0, 10, 64, 64);
        let err = scene
            .set_skybox_from_hdr(&device, b"not an image", 16, Mat4::IDENTITY)
            .unwrap_err();
        assert!(matches!(err, crate::Error::Image { .. }), "{err:?}");
    }
}
