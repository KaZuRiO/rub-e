//! The static scenery as the CPU knows it: blocks grouped by chunk, and which
//! chunks the GPU has yet to catch up with.
//!
//! This is the truth the game reads back. The GPU copy of a chunk is rebuilt
//! from it, a few chunks a frame, by [`instances`](World::instances) — the one
//! place a chunk turns into what gets drawn.

use std::collections::{BTreeMap, BTreeSet};

use glam::{IVec3, Mat4, Vec3};

use crate::cube::CubeRaw;
use crate::culling::{CHUNK_SIZE, ChunkKey, chunk_of};
use crate::error::{Error, Result};
use crate::model::ModelCube;

/// What [`Scene::edit_region`](crate::Scene::edit_region) does to one cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockEdit {
    /// Leaves the cell as it is.
    Keep,
    /// Puts a block of this linear RGB colour in the cell, replacing any.
    Set(Vec3),
    /// Empties the cell, along with any off-grid cube whose centre rounds to it.
    Remove,
}

/// How soon a changed chunk has to reach the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Priority {
    /// When the frame's rebuild budget gets to it.
    Budget,
    /// In the frame being recorded, whatever the budget.
    Immediate,
}

/// The cubes of one chunk.
#[derive(Default)]
struct Chunk {
    /// The blocks, by cell. Ordered, so the instances come out the same way
    /// whatever order the blocks were placed in.
    blocks: BTreeMap<[i32; 3], Vec3>,
    /// Cubes off the grid, in the order they were given.
    off_grid: Vec<ModelCube>,
}

impl Chunk {
    fn len(&self) -> usize {
        self.blocks.len() + self.off_grid.len()
    }
}

/// Every static cube, by chunk, and the chunks changed since the GPU last saw
/// them.
pub(crate) struct World {
    chunks: BTreeMap<ChunkKey, Chunk>,
    dirty: BTreeMap<ChunkKey, Priority>,
    capacity: usize,
    count: usize,
}

/// The chunk a cell is in.
fn chunk_of_cell(cell: IVec3) -> ChunkKey {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "CHUNK_SIZE is 32, a whole number"
    )]
    let side = CHUNK_SIZE as i32;
    cell.div_euclid(IVec3::splat(side)).to_array()
}

/// Whether a position is the centre of a cell.
fn on_grid(position: Vec3) -> bool {
    position == position.round()
}

impl World {
    /// The world of `cubes`, with room for `capacity` cubes. A block given
    /// twice for the same cell keeps the last one. Nothing is marked to
    /// rebuild: the caller uploads every chunk once.
    pub fn new(cubes: &[ModelCube], capacity: usize) -> Self {
        let mut world = Self {
            chunks: BTreeMap::new(),
            dirty: BTreeMap::new(),
            capacity,
            count: 0,
        };
        for cube in cubes {
            if on_grid(cube.position) {
                let cell = cube.position.as_ivec3();
                world.put(cell, cube.color);
            } else {
                world
                    .chunks
                    .entry(chunk_of(cube.position))
                    .or_default()
                    .off_grid
                    .push(*cube);
                world.count += 1;
            }
        }
        world
    }

    /// The cubes the world has room for.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The cubes the world holds now.
    pub fn cube_count(&self) -> usize {
        self.count
    }

    /// Puts or replaces the block of `cell`; the chunk is rebuilt at once.
    pub fn set_block(&mut self, cell: IVec3, color: Vec3) -> Result<()> {
        match self.block(cell) {
            Some(old) if old == color => return Ok(()),
            Some(_) => {}
            None => self.check(self.count + 1)?,
        }
        self.put(cell, color);
        self.mark(chunk_of_cell(cell), Priority::Immediate);
        Ok(())
    }

    /// Empties `cell`; the chunk is rebuilt at once. An empty cell is left
    /// alone, and so are off-grid cubes.
    pub fn remove_block(&mut self, cell: IVec3) {
        if self.take(cell) {
            self.mark(chunk_of_cell(cell), Priority::Immediate);
        }
    }

    /// The colour of the block in `cell`.
    pub fn block(&self, cell: IVec3) -> Option<Vec3> {
        self.chunks
            .get(&chunk_of_cell(cell))?
            .blocks
            .get(&cell.to_array())
            .copied()
    }

    /// Calls `edit` once for each cell of the box, `min` and `max` included,
    /// in increasing X, then Y, then Z, and applies the answers all together,
    /// or not at all if they would take the world past its capacity. An
    /// off-grid cube follows the answer for the cell its centre rounds to.
    /// The chunks changed are rebuilt within the frame's budget.
    pub fn edit_region(
        &mut self,
        min: IVec3,
        max: IVec3,
        mut edit: impl FnMut(IVec3) -> BlockEdit,
    ) -> Result<()> {
        if min.cmpgt(max).any() {
            return Ok(());
        }
        let mut sets = Vec::new();
        let mut removes = BTreeSet::new();
        for x in min.x..=max.x {
            for y in min.y..=max.y {
                for z in min.z..=max.z {
                    let cell = IVec3::new(x, y, z);
                    match edit(cell) {
                        BlockEdit::Keep => {}
                        BlockEdit::Set(color) => {
                            if self.block(cell) != Some(color) {
                                sets.push((cell, color));
                            }
                        }
                        BlockEdit::Remove => {
                            removes.insert(cell.to_array());
                        }
                    }
                }
            }
        }

        // Off-grid cubes are stored by their centre, which is never more than
        // half a cell from the cell it rounds to: one chunk around the box
        // covers them all.
        let (low, high) = (
            chunk_of_cell(min.saturating_sub(IVec3::ONE)),
            chunk_of_cell(max.saturating_add(IVec3::ONE)),
        );
        let mut off_grid_removed = Vec::new();
        if !removes.is_empty() {
            for (&key, chunk) in &self.chunks {
                let inside = (0..3).all(|axis| (low[axis]..=high[axis]).contains(&key[axis]));
                if inside
                    && chunk
                        .off_grid
                        .iter()
                        .any(|cube| removes.contains(&cube.position.round().as_ivec3().to_array()))
                {
                    off_grid_removed.push(key);
                }
            }
        }
        let removed_off_grid: usize = off_grid_removed
            .iter()
            .map(|key| {
                self.chunks[key]
                    .off_grid
                    .iter()
                    .filter(|cube| removes.contains(&cube.position.round().as_ivec3().to_array()))
                    .count()
            })
            .sum();

        let added = sets
            .iter()
            .filter(|(cell, _)| self.block(*cell).is_none())
            .count();
        let removed_blocks = removes
            .iter()
            .filter(|cell| self.block(IVec3::from_array(**cell)).is_some())
            .count();
        let needed = self.count + added - removed_blocks - removed_off_grid;
        self.check(needed)?;

        for cell in &removes {
            let cell = IVec3::from_array(*cell);
            if self.take(cell) {
                self.mark(chunk_of_cell(cell), Priority::Budget);
            }
        }
        for key in off_grid_removed {
            let chunk = self.chunks.get_mut(&key).expect("found above");
            let before = chunk.off_grid.len();
            chunk
                .off_grid
                .retain(|cube| !removes.contains(&cube.position.round().as_ivec3().to_array()));
            self.count -= before - chunk.off_grid.len();
            if chunk.len() == 0 {
                self.chunks.remove(&key);
            }
            self.mark(key, Priority::Budget);
        }
        for (cell, color) in sets {
            self.put(cell, color);
            self.mark(chunk_of_cell(cell), Priority::Budget);
        }
        Ok(())
    }

    /// The keys of every chunk holding a cube, in order.
    pub fn chunk_keys(&self) -> impl Iterator<Item = ChunkKey> + '_ {
        self.chunks.keys().copied()
    }

    /// What the GPU draws for chunk `key`: its blocks by cell, then its
    /// off-grid cubes in the order they came. Empty for a chunk that no
    /// longer holds anything. The same cubes always give the same instances.
    pub fn instances(&self, key: ChunkKey) -> Vec<CubeRaw> {
        let Some(chunk) = self.chunks.get(&key) else {
            return Vec::new();
        };
        let block = |position: Vec3, color: Vec3| {
            CubeRaw::new(Mat4::from_translation(position), color.extend(1.0))
        };
        chunk
            .blocks
            .iter()
            .map(|(cell, color)| block(IVec3::from_array(*cell).as_vec3(), *color))
            .chain(
                chunk
                    .off_grid
                    .iter()
                    .map(|cube| block(cube.position, cube.color)),
            )
            .collect()
    }

    /// How many cubes chunk `key` holds.
    pub fn chunk_len(&self, key: ChunkKey) -> u32 {
        self.chunks.get(&key).map_or(0, |chunk| {
            u32::try_from(chunk.len())
                .expect("a chunk holds at most 32^3 blocks and the off-grid cubes given")
        })
    }

    /// The chunks changed since they were last [`clean`](Self::clean)ed, in
    /// key order, with how soon they are wanted.
    pub fn dirty(&self) -> impl Iterator<Item = (ChunkKey, Priority)> + '_ {
        self.dirty.iter().map(|(key, priority)| (*key, *priority))
    }

    /// Records that the GPU has caught up with chunk `key`.
    pub fn clean(&mut self, key: ChunkKey) {
        self.dirty.remove(&key);
    }

    fn check(&self, needed: usize) -> Result<()> {
        if needed > self.capacity {
            return Err(Error::Capacity {
                needed,
                capacity: self.capacity,
            });
        }
        Ok(())
    }

    /// Puts a block without marking anything.
    fn put(&mut self, cell: IVec3, color: Vec3) {
        let chunk = self.chunks.entry(chunk_of_cell(cell)).or_default();
        if chunk.blocks.insert(cell.to_array(), color).is_none() {
            self.count += 1;
        }
    }

    /// Takes a block out without marking anything; `false` if there was none.
    fn take(&mut self, cell: IVec3) -> bool {
        let key = chunk_of_cell(cell);
        let Some(chunk) = self.chunks.get_mut(&key) else {
            return false;
        };
        if chunk.blocks.remove(&cell.to_array()).is_none() {
            return false;
        }
        self.count -= 1;
        if chunk.len() == 0 {
            self.chunks.remove(&key);
        }
        true
    }

    /// Marks a chunk to rebuild. A chunk wanted at once stays so.
    fn mark(&mut self, key: ChunkKey, priority: Priority) {
        let entry = self.dirty.entry(key).or_insert(priority);
        if priority == Priority::Immediate {
            *entry = Priority::Immediate;
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::{IVec3, Vec3};

    use super::*;
    use crate::cube::CubeRaw;
    use crate::error::Error;
    use crate::model::ModelCube;

    fn world(capacity: usize) -> World {
        World::new(&[], capacity)
    }
    const RED: Vec3 = Vec3::new(1.0, 0.0, 0.0);
    const BLUE: Vec3 = Vec3::new(0.0, 0.0, 1.0);

    #[test]
    fn a_set_block_is_read_back_at_once() {
        let mut w = world(4);
        w.set_block(IVec3::new(1, 2, 3), RED).unwrap();
        assert_eq!(w.block(IVec3::new(1, 2, 3)), Some(RED));
        assert_eq!(w.cube_count(), 1);
    }

    #[test]
    fn removing_an_empty_cell_marks_nothing() {
        let mut w = world(4);
        w.remove_block(IVec3::ZERO);
        assert_eq!(w.dirty().count(), 0);
    }

    #[test]
    fn the_last_of_two_blocks_in_a_cell_wins() {
        let c = |color| ModelCube {
            position: Vec3::ZERO,
            color,
        };
        let w = World::new(&[c(RED), c(BLUE)], 2);
        assert_eq!(w.block(IVec3::ZERO), Some(BLUE));
        assert_eq!(w.cube_count(), 1);
    }

    #[test]
    fn going_past_capacity_fails_and_changes_nothing() {
        let mut w = world(1);
        w.set_block(IVec3::ZERO, RED).unwrap();
        w.clean([0, 0, 0]);
        assert!(matches!(
            w.set_block(IVec3::X, RED),
            Err(Error::Capacity {
                needed: 2,
                capacity: 1
            })
        ));
        assert_eq!(w.block(IVec3::X), None);
        assert_eq!(w.dirty().count(), 0);
    }

    #[test]
    fn replacing_a_block_at_capacity_succeeds() {
        let mut w = world(1);
        w.set_block(IVec3::ZERO, RED).unwrap();
        w.set_block(IVec3::ZERO, BLUE).unwrap();
        assert_eq!(w.block(IVec3::ZERO), Some(BLUE));
    }

    #[test]
    fn negative_cells_fall_in_negative_chunks() {
        let mut w = world(1);
        w.set_block(IVec3::new(-1, 0, 0), RED).unwrap();
        assert_eq!(w.chunk_keys().collect::<Vec<_>>(), [[-1, 0, 0]]);
    }

    #[test]
    fn a_block_on_a_border_rebuilds_only_its_own_chunk() {
        let mut w = world(4);
        w.set_block(IVec3::new(32, 0, 0), RED).unwrap();
        w.clean([1, 0, 0]);
        w.set_block(IVec3::new(31, 0, 0), RED).unwrap();
        assert_eq!(
            w.dirty().collect::<Vec<_>>(),
            [([0, 0, 0], Priority::Immediate)]
        );
    }

    #[test]
    fn a_region_marks_each_chunk_once_for_the_budget() {
        let mut w = world(100);
        w.edit_region(IVec3::ZERO, IVec3::new(40, 0, 0), |_| BlockEdit::Set(RED))
            .unwrap();
        assert_eq!(
            w.dirty().collect::<Vec<_>>(),
            [([0, 0, 0], Priority::Budget), ([1, 0, 0], Priority::Budget)]
        );
    }

    #[test]
    fn a_block_makes_a_waiting_chunk_immediate() {
        let mut w = world(4);
        w.edit_region(IVec3::ZERO, IVec3::ZERO, |_| BlockEdit::Set(RED))
            .unwrap();
        w.set_block(IVec3::X, BLUE).unwrap();
        assert_eq!(
            w.dirty().collect::<Vec<_>>(),
            [([0, 0, 0], Priority::Immediate)]
        );
    }

    #[test]
    fn a_region_past_capacity_changes_nothing() {
        let mut w = world(1);
        let result = w.edit_region(IVec3::ZERO, IVec3::new(1, 0, 0), |_| BlockEdit::Set(RED));
        assert!(matches!(
            result,
            Err(Error::Capacity {
                needed: 2,
                capacity: 1
            })
        ));
        assert_eq!(w.cube_count(), 0);
        assert_eq!(w.dirty().count(), 0);
    }

    #[test]
    fn a_region_counts_what_it_removes_before_checking_capacity() {
        let mut w = world(1);
        w.set_block(IVec3::ZERO, RED).unwrap();
        w.edit_region(IVec3::ZERO, IVec3::X, |c| {
            if c == IVec3::ZERO {
                BlockEdit::Remove
            } else {
                BlockEdit::Set(BLUE)
            }
        })
        .unwrap();
        assert_eq!(w.block(IVec3::X), Some(BLUE));
        assert_eq!(w.cube_count(), 1);
    }

    #[test]
    fn an_inverted_box_changes_nothing() {
        let mut w = world(4);
        let mut calls = 0;
        w.edit_region(IVec3::ONE, IVec3::ZERO, |_| {
            calls += 1;
            BlockEdit::Set(RED)
        })
        .unwrap();
        assert_eq!(calls, 0);
    }

    #[test]
    fn off_grid_cubes_follow_the_answer_for_their_rounded_cell() {
        let off = |x| ModelCube {
            position: Vec3::new(x, 0.25, 0.0),
            color: RED,
        };
        let mut w = World::new(&[off(0.25), off(3.25)], 4);
        w.edit_region(IVec3::ZERO, IVec3::new(3, 0, 0), |c| {
            if c.x == 0 {
                BlockEdit::Remove
            } else {
                BlockEdit::Keep
            }
        })
        .unwrap();
        assert_eq!(w.cube_count(), 1);
        let kept = w.instances([0, 0, 0]);
        assert_eq!(kept[0].model[3][0], 3.25);
    }

    #[test]
    fn the_same_blocks_give_the_same_instances() {
        let cells = [
            IVec3::new(5, 1, 2),
            IVec3::new(1, 0, 0),
            IVec3::new(9, 9, 9),
        ];
        let build = |order: &[IVec3]| {
            let mut w = world(4);
            for &c in order {
                w.set_block(c, RED).unwrap();
            }
            bytemuck::cast_slice::<CubeRaw, u8>(&w.instances([0, 0, 0])).to_vec()
        };
        assert_eq!(build(&cells), build(&[cells[2], cells[0], cells[1]]));
    }
}
