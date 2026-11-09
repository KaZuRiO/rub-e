//! The static scenery as the CPU knows it: blocks grouped by chunk, and which
//! chunks the GPU has yet to catch up with.
//!
//! This is the truth the game reads back. The GPU copy of a chunk is rebuilt
//! from it, a few chunks a frame, by [`instances`](World::instances) — the one
//! place a chunk turns into what gets drawn.

use std::collections::{BTreeMap, BTreeSet};

use glam::{IVec3, Mat4, Vec3};

use crate::cube::{CubeRaw, pack_srgb8, unpack_srgb8};
use crate::culling::{CHUNK_SIZE, ChunkKey, chunk_of};
use crate::error::{Error, Result};
use crate::model::ModelCube;

/// What [`Scene::edit_region`](crate::Scene::edit_region) does to one cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockEdit {
    /// Leaves the cell as it is.
    Keep,
    /// Puts a block of this linear RGB colour in the cell, replacing any. The
    /// block keeps the colour as drawn, in sRGB with 8 bits per channel
    /// clamped to 0..1; a colour that encodes like the one in place changes
    /// nothing.
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

/// Cells along one side of a chunk.
const SIDE: i32 = 32;
/// Cells in a chunk.
const CELLS: usize = 32 * 32 * 32;

/// Where a cell sits in its chunk: `x + 32·y + 1024·z` on its position within
/// the chunk, which is never negative.
fn local_index(cell: IVec3) -> usize {
    let local = cell.rem_euclid(IVec3::splat(SIDE)).as_uvec3();
    usize::try_from(local.x + 32 * local.y + 1024 * local.z).expect("below 32^3")
}

/// The cell at `index` in chunk `key`, the inverse of [`local_index`].
fn cell_at(key: ChunkKey, index: usize) -> IVec3 {
    let index = i32::try_from(index).expect("below 32^3");
    let local = IVec3::new(index % 32, index / 32 % 32, index / 1024);
    IVec3::from_array(key) * SIDE + local
}

/// The cubes of one chunk: its blocks as a mask of the cells that hold one
/// and a colour for each, so that the instances come out in cell order
/// whatever order the blocks were placed in.
struct Chunk {
    /// One bit per cell, set where there is a block.
    occupied: Box<[u64; CELLS / 64]>,
    /// The colour of each block as drawn (see [`pack_srgb8`]); meaningless for
    /// an empty cell.
    colors: Box<[u32]>,
    /// How many bits of `occupied` are set.
    blocks: u32,
    /// Cubes off the grid, in the order they were given.
    off_grid: Vec<ModelCube>,
}

impl Default for Chunk {
    fn default() -> Self {
        Self {
            occupied: Box::new([0; CELLS / 64]),
            colors: vec![0; CELLS].into_boxed_slice(),
            blocks: 0,
            off_grid: Vec::new(),
        }
    }
}

impl Chunk {
    fn len(&self) -> usize {
        self.blocks as usize + self.off_grid.len()
    }

    fn color(&self, index: usize) -> Option<u32> {
        (self.occupied[index / 64] >> (index % 64) & 1 == 1).then(|| self.colors[index])
    }

    /// Puts a colour in a cell; `true` if the cell was empty.
    fn insert(&mut self, index: usize, packed: u32) -> bool {
        let word = &mut self.occupied[index / 64];
        let bit = 1 << (index % 64);
        let was_empty = *word & bit == 0;
        *word |= bit;
        self.colors[index] = packed;
        self.blocks += u32::from(was_empty);
        was_empty
    }

    /// Empties a cell; `true` if it held a block.
    fn remove(&mut self, index: usize) -> bool {
        let word = &mut self.occupied[index / 64];
        let bit = 1 << (index % 64);
        let was_full = *word & bit != 0;
        *word &= !bit;
        self.blocks -= u32::from(was_full);
        was_full
    }

    /// The cells holding a block, in increasing order.
    fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.occupied.iter().enumerate().flat_map(|(word, &bits)| {
            let mut bits = bits;
            std::iter::from_fn(move || {
                if bits == 0 {
                    return None;
                }
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                Some(word * 64 + bit)
            })
        })
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

/// The cell a position rounds to.
fn rounded_cell(position: Vec3) -> [i32; 3] {
    position.round().as_ivec3().to_array()
}

/// The off-grid cubes of the chunks from `low` to `high`, as (chunk, place in
/// its list, cell they round to), chunk by chunk.
fn off_grid_near(
    chunks: &BTreeMap<ChunkKey, Chunk>,
    low: ChunkKey,
    high: ChunkKey,
) -> Vec<(ChunkKey, usize, [i32; 3])> {
    let mut near = Vec::new();
    for (&key, chunk) in chunks.range([low[0], i32::MIN, i32::MIN]..=[high[0], i32::MAX, i32::MAX])
    {
        if (0..3).all(|axis| (low[axis]..=high[axis]).contains(&key[axis])) {
            near.extend(
                chunk
                    .off_grid
                    .iter()
                    .enumerate()
                    .map(|(place, cube)| (key, place, rounded_cell(cube.position))),
            );
        }
    }
    near
}

/// Looks up the colour of cells one after the other, remembering the last
/// chunk it found: neighbouring cells share one, and a tree walk is the cost
/// to avoid.
struct Lookup<'a> {
    chunks: &'a BTreeMap<ChunkKey, Chunk>,
    last: Option<(ChunkKey, Option<&'a Chunk>)>,
}

impl Lookup<'_> {
    fn color(&mut self, cell: IVec3) -> Option<u32> {
        let key = chunk_of_cell(cell);
        let chunk = match self.last {
            Some((last, chunk)) if last == key => chunk,
            _ => {
                let chunk = self.chunks.get(&key);
                self.last = Some((key, chunk));
                chunk
            }
        };
        chunk?.color(local_index(cell))
    }
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
                world.put(cell, pack_srgb8(cube.color));
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

    /// Puts or replaces the block of `cell`; the chunk is rebuilt at once. A
    /// colour that encodes like the one in place changes nothing.
    pub fn set_block(&mut self, cell: IVec3, color: Vec3) -> Result<()> {
        let packed = pack_srgb8(color);
        match self.packed(cell) {
            Some(old) if old == packed => return Ok(()),
            Some(_) => {}
            None => self.check(self.count + 1)?,
        }
        self.put(cell, packed);
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

    /// The colour of the block in `cell`, as drawn.
    pub fn block(&self, cell: IVec3) -> Option<Vec3> {
        self.packed(cell).map(unpack_srgb8)
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
        // Off-grid cubes are stored by their centre, which is never more than
        // half a cell from the cell it rounds to: one chunk around the box
        // covers them all.
        let (low, high) = (
            chunk_of_cell(min.saturating_sub(IVec3::ONE)),
            chunk_of_cell(max.saturating_add(IVec3::ONE)),
        );
        // The off-grid cubes near the box, as (chunk, place in its list, cell
        // they round to), found at the first removal: a box that only sets
        // blocks never needs them. With the blocks, their cells are the only
        // ones a removal can empty: most of a blast is air, and skipping it
        // keeps a large removal from stalling the frame. Usually there are
        // none.
        let mut near: Option<(Vec<_>, BTreeSet<[i32; 3]>)> = None;
        let mut sets = Vec::new();
        let mut removes = Vec::new();
        // The cells where a removal meets an off-grid cube.
        let mut off_grid_hits = BTreeSet::new();
        // Each cell is visited once and the world is not changed until the
        // scan ends, so these counts are exact.
        let mut added = 0;
        let mut removed_blocks = 0;
        let mut lookup = Lookup {
            chunks: &self.chunks,
            last: None,
        };
        for x in min.x..=max.x {
            for y in min.y..=max.y {
                for z in min.z..=max.z {
                    let cell = IVec3::new(x, y, z);
                    match edit(cell) {
                        BlockEdit::Keep => {}
                        BlockEdit::Set(color) => {
                            let packed = pack_srgb8(color);
                            let old = lookup.color(cell);
                            if old != Some(packed) {
                                added += usize::from(old.is_none());
                                sets.push((cell, packed));
                            }
                        }
                        BlockEdit::Remove => {
                            if lookup.color(cell).is_some() {
                                removed_blocks += 1;
                                removes.push(cell);
                            }
                            let (_, near_cells) = near.get_or_insert_with(|| {
                                let near = off_grid_near(&self.chunks, low, high);
                                let cells = near.iter().map(|&(_, _, cell)| cell).collect();
                                (near, cells)
                            });
                            if near_cells.contains(&cell.to_array()) {
                                off_grid_hits.insert(cell.to_array());
                            }
                        }
                    }
                }
            }
        }

        // The off-grid cubes to drop, grouped by chunk, each group by place.
        let mut doomed: Vec<(ChunkKey, Vec<usize>)> = Vec::new();
        for (key, place, cell) in near.map(|(near, _)| near).unwrap_or_default() {
            if !off_grid_hits.contains(&cell) {
                continue;
            }
            match doomed.last_mut() {
                Some((last, places)) if *last == key => places.push(place),
                _ => doomed.push((key, vec![place])),
            }
        }
        let removed_off_grid: usize = doomed.iter().map(|(_, places)| places.len()).sum();

        let needed = self.count + added - removed_blocks - removed_off_grid;
        self.check(needed)?;

        // Cells come in scan order, so runs of one chunk are long: one lookup
        // and one mark for each, not for each cell.
        for run in removes.chunk_by(|a, b| chunk_of_cell(*a) == chunk_of_cell(*b)) {
            let key = chunk_of_cell(run[0]);
            let chunk = self.chunks.get_mut(&key).expect("it held these blocks");
            for &cell in run {
                chunk.remove(local_index(cell));
            }
            self.count -= run.len();
            if chunk.len() == 0 {
                self.chunks.remove(&key);
            }
            self.mark(key, Priority::Budget);
        }
        for (key, places) in doomed {
            let chunk = self.chunks.get_mut(&key).expect("it held these cubes");
            let mut places = places.into_iter().peekable();
            let mut place = 0;
            chunk.off_grid.retain(|_| {
                let gone = places.next_if_eq(&place).is_some();
                place += 1;
                !gone
            });
            if chunk.len() == 0 {
                self.chunks.remove(&key);
            }
            self.mark(key, Priority::Budget);
        }
        self.count -= removed_off_grid;
        for run in sets.chunk_by(|a, b| chunk_of_cell(a.0) == chunk_of_cell(b.0)) {
            let key = chunk_of_cell(run[0].0);
            let chunk = self.chunks.entry(key).or_default();
            for &(cell, packed) in run {
                self.count += usize::from(chunk.insert(local_index(cell), packed));
            }
            self.mark(key, Priority::Budget);
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
        let mut instances = Vec::with_capacity(chunk.len());
        instances.extend(chunk.indices().map(|index| {
            block(
                cell_at(key, index).as_vec3(),
                unpack_srgb8(chunk.colors[index]),
            )
        }));
        instances.extend(
            chunk
                .off_grid
                .iter()
                .map(|cube| block(cube.position, cube.color)),
        );
        instances
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

    /// The colour of the block in `cell`, packed.
    fn packed(&self, cell: IVec3) -> Option<u32> {
        self.chunks
            .get(&chunk_of_cell(cell))?
            .color(local_index(cell))
    }

    /// Puts a block without marking anything.
    fn put(&mut self, cell: IVec3, packed: u32) {
        let chunk = self.chunks.entry(chunk_of_cell(cell)).or_default();
        self.count += usize::from(chunk.insert(local_index(cell), packed));
    }

    /// Takes a block out without marking anything; `false` if there was none.
    fn take(&mut self, cell: IVec3) -> bool {
        let key = chunk_of_cell(cell);
        let Some(chunk) = self.chunks.get_mut(&key) else {
            return false;
        };
        if !chunk.remove(local_index(cell)) {
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
    fn an_off_grid_cube_is_removed_from_the_chunk_next_to_its_cell() {
        // Stored in chunk 0 by its centre, it rounds to cell 32, in chunk 1.
        let cube = ModelCube {
            position: Vec3::new(31.75, 0.25, 0.0),
            color: RED,
        };
        let mut w = World::new(&[cube], 1);
        let cell = IVec3::new(32, 0, 0);
        w.edit_region(cell, cell, |_| BlockEdit::Remove).unwrap();
        assert_eq!(w.cube_count(), 0);
        assert!(w.instances([0, 0, 0]).is_empty());
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

    #[test]
    fn negative_cells_land_at_the_far_end_of_their_chunk() {
        assert_eq!(local_index(IVec3::splat(-1)), CELLS - 1);
        assert_eq!(cell_at([-1, -1, -1], CELLS - 1), IVec3::splat(-1));
        assert_eq!(
            cell_at([2, 0, -3], local_index(IVec3::new(70, 5, -90))),
            IVec3::new(70, 5, -90)
        );
    }

    #[test]
    fn a_block_reads_back_its_colour_as_drawn() {
        let mut w = world(1);
        let c = Vec3::new(0.3, 0.6, 0.9);
        w.set_block(IVec3::ZERO, c).unwrap();
        assert_eq!(w.block(IVec3::ZERO), Some(unpack_srgb8(pack_srgb8(c))));
    }

    #[test]
    fn a_colour_that_packs_the_same_marks_nothing() {
        let mut w = world(1);
        let c = Vec3::new(0.3, 0.6, 0.9);
        assert_eq!(pack_srgb8(c + 1e-5), pack_srgb8(c));
        w.set_block(IVec3::ZERO, c).unwrap();
        w.clean([0, 0, 0]);
        w.set_block(IVec3::ZERO, c + 1e-5).unwrap();
        w.edit_region(IVec3::ZERO, IVec3::ZERO, |_| BlockEdit::Set(c + 1e-5))
            .unwrap();
        assert_eq!(w.dirty().count(), 0);
    }
}
