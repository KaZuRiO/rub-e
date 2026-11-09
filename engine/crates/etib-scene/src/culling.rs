//! Frustum culling on the GPU, in two compute passes.
//!
//! The static cubes are grouped into chunks — cubes of 32×32×32 world units.
//! Each frame the first pass tests every chunk's bounding box against the
//! camera and writes one flag per chunk; the second runs once per cube slot,
//! skips those whose chunk is out of view, tests the rest one by one, and
//! appends the survivors to `visible` while counting them into the indirect
//! draw. Nothing comes back to the CPU.
//!
//! The buffers are sized once, for the scene's static capacity. A chunk the
//! [`World`] changed is rewritten whole by [`StaticCubes::rebuild`], into any
//! free slots: its cubes may be spread over several pieces, since the cube
//! pass finds a slot's chunk through `chunk_ids` alone. Row 0 of the chunk
//! table is kept empty, with a count of 0, which the chunk pass always calls
//! invisible; every free slot points at it.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use bytemuck::Zeroable as _;
use etib_core::{
    Access, BindGroup, BindLayout, ComputePass, ComputePipeline, Device, DrawIndexedArgs,
    IndirectBuffer, Shader, Stage, StorageBuffer,
};
use glam::Vec3;

use crate::cube::{CubeRaw, INDICES};
use crate::slots::SlotAllocator;
use crate::world::World;

/// The side of a chunk, in world units, on all three axes.
pub(crate) const CHUNK_SIZE: f32 = 32.0;

/// Threads per workgroup in both culling shaders.
const WORKGROUP: u32 = 64;

/// One chunk as `chunk_cull.wgsl` reads it.
///
/// Flat rather than built from `vec3`s, so the layout is the same on both
/// sides without WGSL's alignment rules getting a say.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ChunkRaw {
    pub aabb_min: [f32; 3],
    pub _pad0: f32,
    pub aabb_max: [f32; 3],
    /// Where the chunk's cubes start in the static instances.
    pub start_idx: u32,
    pub count: u32,
    pub _pad1: [u32; 3],
}

/// A chunk, named by its position on the grid of chunks: the chunk `[1, 0, 0]`
/// covers `32.0..64.0` in X and `0.0..32.0` in Y and Z.
pub(crate) type ChunkKey = [i32; 3];

/// The chunk a position falls in. Past `i32` chunks, which a world would need
/// to be 2^36 units wide to reach, the key saturates.
pub(crate) fn chunk_of(position: Vec3) -> ChunkKey {
    (position / CHUNK_SIZE).floor().as_ivec3().to_array()
}

/// A number of cubes, chunks or indices as the `u32` the shaders count in.
///
/// # Panics
///
/// Past `u32::MAX`, which the GPU buffers holding them could not index.
fn count(len: usize) -> u32 {
    u32::try_from(len).expect("more than u32::MAX cubes, chunks or indices")
}

/// The two culling pipelines and their layouts. Group 0 of both is the
/// camera, the same bind group the draws use.
pub(crate) struct CullingPasses {
    chunk_pipeline: ComputePipeline,
    chunk_layout: BindLayout,
    cube_pipeline: ComputePipeline,
    cube_layout: BindLayout,
}

impl CullingPasses {
    pub fn new(device: &Device, camera_layout: &BindLayout) -> Self {
        let chunk_layout = BindLayout::build(device, "chunk cull")
            .storage_buffer::<ChunkRaw>(Stage::Compute, Access::Read)
            .storage_buffer::<u32>(Stage::Compute, Access::ReadWrite)
            .finish();
        let shader = Shader::wgsl(
            device,
            "chunk cull",
            include_str!("shaders/chunk_cull.wgsl"),
        );
        let chunk_pipeline = ComputePipeline::with_layouts(
            device,
            "chunk cull",
            &shader,
            "main",
            &[camera_layout, &chunk_layout],
        );

        // Five storage buffers in one stage, which is what the device's raised
        // limit is for.
        let cube_layout = BindLayout::build(device, "cube cull")
            .storage_buffer::<CubeRaw>(Stage::Compute, Access::Read)
            .storage_buffer::<CubeRaw>(Stage::Compute, Access::ReadWrite)
            .indirect_buffer(Stage::Compute, Access::ReadWrite)
            .storage_buffer::<u32>(Stage::Compute, Access::Read)
            .storage_buffer::<u32>(Stage::Compute, Access::Read)
            .finish();
        let shader = Shader::wgsl(device, "cube cull", include_str!("shaders/cull.wgsl"));
        let cube_pipeline = ComputePipeline::with_layouts(
            device,
            "cube cull",
            &shader,
            "main",
            &[camera_layout, &cube_layout],
        );

        Self {
            chunk_pipeline,
            chunk_layout,
            cube_pipeline,
            cube_layout,
        }
    }
}

/// The static cubes on the GPU, and everything culling them needs.
pub(crate) struct StaticCubes {
    instances: StorageBuffer<CubeRaw>,
    chunks: StorageBuffer<ChunkRaw>,
    /// Bound through the cull groups only, but held so it lives as long.
    _chunk_visible: StorageBuffer<u32>,
    chunk_ids: StorageBuffer<u32>,
    /// The cubes that survived this frame, drawn as instances.
    pub visible: StorageBuffer<CubeRaw>,
    pub draw: IndirectBuffer,
    chunk_group: BindGroup,
    cube_group: BindGroup,
    slots: SlotAllocator,
    /// Where each chunk on the GPU sits: its row and its slots.
    placed: BTreeMap<ChunkKey, Placed>,
    /// Rows given back, lowest reused first.
    free_rows: BTreeSet<u32>,
    /// One past the highest row handed out; row 0 is the empty one.
    rows_end: u32,
}

/// A chunk's place on the GPU.
struct Placed {
    row: u32,
    pieces: Vec<Range<u32>>,
}

impl StaticCubes {
    /// The buffers for `world.capacity()` cubes, with every chunk of `world`
    /// written in key order, so a scene that never changes has its chunks
    /// packed. `None` for a capacity of 0: a buffer cannot be empty, and there
    /// would be nothing to cull or draw.
    pub fn new(device: &Device, passes: &CullingPasses, world: &World) -> Option<Self> {
        if world.capacity() == 0 {
            return None;
        }
        let capacity = count(world.capacity());
        // A chunk can hold a single cube, so as many rows as cubes, plus the
        // empty one.
        let rows = capacity
            .checked_add(1)
            .expect("a static capacity under u32::MAX");

        let instances = StorageBuffer::new(device, "static instances", capacity);
        let visible = StorageBuffer::new(device, "visible instances", capacity);
        let draw = IndirectBuffer::new(device, "static draw", 1);
        let chunks = StorageBuffer::new(device, "chunks", rows);
        let chunk_visible = StorageBuffer::new(device, "chunk visibility", rows);
        let chunk_ids = StorageBuffer::new(device, "cube chunk ids", capacity);

        let chunk_group =
            passes
                .chunk_layout
                .group(device, "chunk cull", &[&chunks, &chunk_visible]);
        let cube_group = passes.cube_layout.group(
            device,
            "cube cull",
            &[&instances, &visible, &draw, &chunk_ids, &chunk_visible],
        );

        let mut statics = Self {
            instances,
            chunks,
            _chunk_visible: chunk_visible,
            chunk_ids,
            visible,
            draw,
            chunk_group,
            cube_group,
            slots: SlotAllocator::new(capacity),
            placed: BTreeMap::new(),
            free_rows: BTreeSet::new(),
            rows_end: 1,
        };
        for key in world.chunk_keys() {
            statics.rebuild(device, world, key);
        }
        log::info!(
            "{} static cubes in {} chunk(s), room for {capacity}",
            world.cube_count(),
            statics.chunk_count()
        );
        Some(statics)
    }

    /// Replaces what the GPU holds for chunk `key` with what `world` holds,
    /// all in one go: the writes are queued, so they land together before the
    /// culling recorded after them.
    ///
    /// # Panics
    ///
    /// If the free slots cannot hold the chunk, which the rebuild planner
    /// rules out.
    pub fn rebuild(&mut self, device: &Device, world: &World, key: ChunkKey) {
        let old = self.placed.remove(&key);
        if let Some(old) = &old {
            for piece in &old.pieces {
                self.chunk_ids
                    .write(device, piece.start, &vec![0; piece.len()]);
            }
            self.slots.free(&old.pieces);
        }

        let instances = world.instances(key);
        if instances.is_empty() {
            if let Some(old) = old {
                self.chunks.write(device, old.row, &[ChunkRaw::zeroed()]);
                self.free_rows.insert(old.row);
            }
            return;
        }

        let row = match old {
            Some(old) => old.row,
            None => self.free_rows.pop_first().unwrap_or_else(|| {
                self.rows_end += 1;
                self.rows_end - 1
            }),
        };
        let pieces = self
            .slots
            .alloc(count(instances.len()))
            .expect("the rebuild planner leaves room for every chunk it writes");

        let mut rest = instances.as_slice();
        for piece in &pieces {
            let (these, others) = rest.split_at(piece.len());
            self.instances.write(device, piece.start, these);
            self.chunk_ids
                .write(device, piece.start, &vec![row; piece.len()]);
            rest = others;
        }

        // The bounds of the cubes' faces, not of their centres.
        let (min, max) = instances.iter().fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(min, max), cube| {
                let centre = Vec3::from_slice(&cube.model[3][..3]);
                (min.min(centre - 0.5), max.max(centre + 0.5))
            },
        );
        self.chunks.write(
            device,
            row,
            &[ChunkRaw {
                aabb_min: min.to_array(),
                _pad0: 0.0,
                aabb_max: max.to_array(),
                start_idx: pieces[0].start,
                count: count(instances.len()),
                _pad1: [0; 3],
            }],
        );
        self.placed.insert(key, Placed { row, pieces });
    }

    /// Slots no chunk holds, counting none still taken by a chunk's old
    /// version.
    pub fn free_slots(&self) -> u32 {
        self.slots.free_count()
    }

    /// The cubes chunk `key` takes on the GPU, 0 if it is not there.
    pub fn gpu_len(&self, key: ChunkKey) -> u32 {
        self.placed.get(&key).map_or(0, |placed| {
            placed
                .pieces
                .iter()
                .map(|piece| piece.end - piece.start)
                .sum()
        })
    }

    /// The chunks on the GPU.
    pub fn chunk_count(&self) -> usize {
        self.placed.len()
    }

    /// Every instance a chunk holds on the GPU, sorted by their bytes so two
    /// scenes compare whatever slots they landed in.
    #[cfg(test)]
    pub fn live_instances(&self, device: &Device) -> Vec<CubeRaw> {
        let ids = self.chunk_ids.read_back(device);
        let mut live: Vec<CubeRaw> = self
            .instances
            .read_back(device)
            .into_iter()
            .zip(ids)
            .filter(|(_, id)| *id != 0)
            .map(|(cube, _)| cube)
            .collect();
        live.sort_by(|a, b| bytemuck::bytes_of(a).cmp(bytemuck::bytes_of(b)));
        live
    }

    /// Zeroes this frame's count. Queued, so it lands before the culling
    /// recorded after it in the same submission.
    pub fn reset(&self, device: &Device) {
        self.draw.write(
            device,
            0,
            &DrawIndexedArgs {
                index_count: count(INDICES.len()),
                ..Default::default()
            },
        );
    }

    /// Records both culling passes, over the rows and slots in use only: a
    /// scene packed at load costs what it always did.
    pub fn cull(&self, pass: &mut ComputePass<'_>, passes: &CullingPasses, camera: &BindGroup) {
        pass.set_pipeline(&passes.chunk_pipeline);
        pass.set_bind_group(0, camera);
        pass.set_bind_group(1, &self.chunk_group);
        pass.dispatch_workgroups(self.rows_end.div_ceil(WORKGROUP), 1, 1);

        pass.set_pipeline(&passes.cube_pipeline);
        pass.set_bind_group(1, &self.cube_group);
        pass.dispatch_workgroups(self.slots.end().div_ceil(WORKGROUP), 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use etib_core::test_util::device;

    use super::*;
    use crate::camera::CameraRaw;
    use crate::model::ModelCube;

    fn cube(x: f32, y: f32, z: f32) -> ModelCube {
        ModelCube {
            position: Vec3::new(x, y, z),
            color: Vec3::new(0.5, 0.25, 1.0),
        }
    }

    fn statics(device: &Device, world: &World) -> StaticCubes {
        let camera = BindLayout::build(device, "camera")
            .uniform::<CameraRaw>(Stage::All)
            .finish();
        let passes = CullingPasses::new(device, &camera);
        StaticCubes::new(device, &passes, world).expect("a capacity above 0")
    }

    #[test]
    fn chunk_layout_matches_the_shader() {
        assert_eq!(size_of::<ChunkRaw>(), 48);
        assert_eq!(std::mem::offset_of!(ChunkRaw, aabb_max), 16);
        assert_eq!(std::mem::offset_of!(ChunkRaw, start_idx), 28);
        assert_eq!(std::mem::offset_of!(ChunkRaw, count), 32);
    }

    #[test]
    fn a_tall_column_spans_several_chunks() {
        assert_eq!(chunk_of(Vec3::new(0.0, 33.0, 0.0)), [0, 1, 0]);
        assert_eq!(chunk_of(Vec3::new(0.0, -1.0, 0.0)), [0, -1, 0]);
    }

    #[test]
    fn chunks_are_packed_at_load_after_the_empty_row() {
        let Some(device) = device() else { return };
        let world = World::new(
            &[
                cube(66.0, 0.0, 0.0),
                cube(0.0, 0.0, 0.0),
                cube(33.0, 0.0, 0.0),
                cube(1.0, 5.0, 31.0),
                cube(-1.0, 0.0, 0.0),
            ],
            5,
        );
        let statics = statics(&device, &world);
        let rows = statics.chunks.read_back(&device);
        // Chunks [-1,0,0], [0,0,0], [1,0,0], [2,0,0], in that order, from row 1.
        assert_eq!(rows[0], ChunkRaw::zeroed());
        assert_eq!(
            rows[1..5]
                .iter()
                .map(|c| (c.start_idx, c.count))
                .collect::<Vec<_>>(),
            [(0, 1), (1, 2), (3, 1), (4, 1)]
        );
        assert_eq!(statics.chunk_ids.read_back(&device), [1, 2, 2, 3, 4]);
        let instances = statics.instances.read_back(&device);
        assert_eq!(instances[0].model[3], [-1.0, 0.0, 0.0, 1.0]);
        assert_eq!(instances[0].color, [0.5, 0.25, 1.0, 1.0]);
    }

    #[test]
    fn chunk_bounds_cover_whole_cubes() {
        let Some(device) = device() else { return };
        let world = World::new(&[cube(0.0, 0.0, 0.0), cube(1.0, 5.0, 31.0)], 2);
        let chunk = statics(&device, &world).chunks.read_back(&device)[1];
        assert_eq!(chunk.aabb_min, [-0.5, -0.5, -0.5]);
        assert_eq!(chunk.aabb_max, [1.5, 5.5, 31.5]);
    }
}
