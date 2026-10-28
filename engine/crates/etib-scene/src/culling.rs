//! Frustum culling on the GPU, in two compute passes.
//!
//! The static cubes are grouped into chunks — 32×32 tiles in the XZ plane, of
//! any height — once, on the CPU. Each frame the first pass tests every
//! chunk's bounding box against the camera and writes one flag per chunk; the
//! second runs once per cube, skips those whose chunk is out of view, tests
//! the rest one by one, and appends the survivors to `visible` while counting
//! them into the indirect draw. Nothing comes back to the CPU.

use std::collections::HashMap;

use etib_core::{
    Access, BindGroup, BindLayout, ComputePass, ComputePipeline, Device, DrawIndexedArgs,
    IndirectBuffer, Shader, Stage, StorageBuffer,
};
use glam::{Mat4, Vec3};

use crate::cube::{CubeRaw, INDICES};
use crate::model::ModelCube;

/// The side of a chunk in the XZ plane, in world units.
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

/// The static cubes as the culling passes want them: sorted by chunk, with
/// the chunk each one belongs to alongside.
#[derive(Debug)]
pub(crate) struct StaticLayout {
    pub instances: Vec<CubeRaw>,
    /// For each instance, the index of its chunk.
    pub chunk_ids: Vec<u32>,
    pub chunks: Vec<ChunkRaw>,
}

/// Sorts cubes into chunks and works out each chunk's bounds.
///
/// Chunks come out in a fixed order, by tile, so the same cubes always give
/// the same buffers.
pub(crate) fn build_chunks(cubes: &[ModelCube]) -> StaticLayout {
    let mut tiles: HashMap<(i32, i32), Vec<&ModelCube>> = HashMap::new();
    for cube in cubes {
        let key = tile_of(cube.position);
        tiles.entry(key).or_default().push(cube);
    }
    let mut keys: Vec<_> = tiles.keys().copied().collect();
    keys.sort_unstable();

    let mut layout = StaticLayout {
        instances: Vec::with_capacity(cubes.len()),
        chunk_ids: Vec::with_capacity(cubes.len()),
        chunks: Vec::with_capacity(keys.len()),
    };
    for (index, key) in keys.iter().enumerate() {
        let members = &tiles[key];
        // The bounds of the cubes' faces, not of their centres.
        let (min, max) = members.iter().fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(min, max), cube| (min.min(cube.position - 0.5), max.max(cube.position + 0.5)),
        );

        layout.chunks.push(ChunkRaw {
            aabb_min: min.to_array(),
            _pad0: 0.0,
            aabb_max: max.to_array(),
            start_idx: count(layout.instances.len()),
            count: count(members.len()),
            _pad1: [0; 3],
        });
        for cube in members {
            layout.instances.push(CubeRaw::new(
                Mat4::from_translation(cube.position),
                cube.color.extend(1.0),
            ));
            layout.chunk_ids.push(count(index));
        }
    }
    layout
}

/// The XZ tile a position falls in.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a world 32 units a tile would need to be 2^36 units wide to overflow, and `as` saturates"
)]
fn tile_of(position: Vec3) -> (i32, i32) {
    (
        (position.x / CHUNK_SIZE).floor() as i32,
        (position.z / CHUNK_SIZE).floor() as i32,
    )
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
    /// Bound through the cull groups only, but held so they live as long.
    _instances: StorageBuffer<CubeRaw>,
    _chunks: StorageBuffer<ChunkRaw>,
    _chunk_visible: StorageBuffer<u32>,
    _chunk_ids: StorageBuffer<u32>,
    /// The cubes that survived this frame, drawn as instances.
    pub visible: StorageBuffer<CubeRaw>,
    pub draw: IndirectBuffer,
    chunk_group: BindGroup,
    cube_group: BindGroup,
    pub count: u32,
    pub chunk_count: u32,
}

impl StaticCubes {
    /// `None` for no cubes: a buffer cannot be empty, and there would be
    /// nothing to cull or draw.
    pub fn new(device: &Device, passes: &CullingPasses, cubes: &[ModelCube]) -> Option<Self> {
        if cubes.is_empty() {
            return None;
        }
        let layout = build_chunks(cubes);
        let count = count(layout.instances.len());
        let chunk_count = self::count(layout.chunks.len());

        let instances = StorageBuffer::from_slice(device, "static instances", &layout.instances);
        let visible = StorageBuffer::new(device, "visible instances", count);
        let draw = IndirectBuffer::new(device, "static draw", 1);
        let chunks = StorageBuffer::from_slice(device, "chunks", &layout.chunks);
        let chunk_visible = StorageBuffer::new(device, "chunk visibility", chunk_count);
        let chunk_ids = StorageBuffer::from_slice(device, "cube chunk ids", &layout.chunk_ids);

        let chunk_group =
            passes
                .chunk_layout
                .group(device, "chunk cull", &[&chunks, &chunk_visible]);
        let cube_group = passes.cube_layout.group(
            device,
            "cube cull",
            &[&instances, &visible, &draw, &chunk_ids, &chunk_visible],
        );
        log::info!("{count} static cubes in {chunk_count} chunk(s)");

        Some(Self {
            _instances: instances,
            _chunks: chunks,
            _chunk_visible: chunk_visible,
            _chunk_ids: chunk_ids,
            visible,
            draw,
            chunk_group,
            cube_group,
            count,
            chunk_count,
        })
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

    /// Records both culling passes.
    pub fn cull(&self, pass: &mut ComputePass<'_>, passes: &CullingPasses, camera: &BindGroup) {
        pass.set_pipeline(&passes.chunk_pipeline);
        pass.set_bind_group(0, camera);
        pass.set_bind_group(1, &self.chunk_group);
        pass.dispatch_workgroups(self.chunk_count.div_ceil(WORKGROUP), 1, 1);

        pass.set_pipeline(&passes.cube_pipeline);
        pass.set_bind_group(1, &self.cube_group);
        pass.dispatch_workgroups(self.count.div_ceil(WORKGROUP), 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube(x: f32, y: f32, z: f32) -> ModelCube {
        ModelCube {
            position: Vec3::new(x, y, z),
            color: Vec3::new(0.5, 0.25, 1.0),
        }
    }

    #[test]
    fn chunk_layout_matches_the_shader() {
        assert_eq!(size_of::<ChunkRaw>(), 48);
        assert_eq!(std::mem::offset_of!(ChunkRaw, aabb_max), 16);
        assert_eq!(std::mem::offset_of!(ChunkRaw, start_idx), 28);
        assert_eq!(std::mem::offset_of!(ChunkRaw, count), 32);
    }

    #[test]
    fn cubes_are_grouped_by_tile() {
        let layout = build_chunks(&[
            cube(66.0, 0.0, 0.0),
            cube(0.0, 0.0, 0.0),
            cube(33.0, 0.0, 0.0),
            cube(1.0, 5.0, 31.0),
            cube(-1.0, 0.0, 0.0),
        ]);
        // Tiles (-1,0), (0,0), (1,0), (2,0), in that order.
        assert_eq!(layout.chunks.len(), 4);
        assert_eq!(
            layout
                .chunks
                .iter()
                .map(|c| (c.start_idx, c.count))
                .collect::<Vec<_>>(),
            [(0, 1), (1, 2), (3, 1), (4, 1)]
        );
        assert_eq!(layout.chunk_ids, [0, 1, 1, 2, 3]);
        assert_eq!(layout.instances.len(), 5);
        assert_eq!(layout.instances[0].model[3], [-1.0, 0.0, 0.0, 1.0]);
        assert_eq!(layout.instances[0].color, [0.5, 0.25, 1.0, 1.0]);
    }

    #[test]
    fn chunk_bounds_cover_whole_cubes() {
        let layout = build_chunks(&[cube(0.0, 0.0, 0.0), cube(1.0, 5.0, 31.0)]);
        let chunk = layout.chunks[0];
        assert_eq!(chunk.aabb_min, [-0.5, -0.5, -0.5]);
        assert_eq!(chunk.aabb_max, [1.5, 5.5, 31.5]);
    }

    #[test]
    fn the_same_cubes_give_the_same_layout() {
        let cubes: Vec<_> = (0..200)
            .map(|i| cube((i * 7 % 150) as f32, 0.0, (i * 13 % 90) as f32))
            .collect();
        let (a, b) = (build_chunks(&cubes), build_chunks(&cubes));
        assert_eq!(a.chunks, b.chunks);
        assert_eq!(a.chunk_ids, b.chunk_ids);
        assert_eq!(a.instances, b.instances);
    }
}
