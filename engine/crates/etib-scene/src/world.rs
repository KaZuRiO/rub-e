//! The static scenery as the CPU knows it: blocks grouped by chunk, and which
//! chunks the GPU has yet to catch up with.
//!
//! This is the truth the game reads back. The GPU copy of a chunk is rebuilt
//! from it, a few chunks a frame, by [`instances`](World::instances) — the one
//! place a chunk turns into what gets drawn.

use std::collections::{BTreeMap, BTreeSet};

use glam::{IVec3, Vec3};

use crate::cube::{StaticCubeRaw, pack_srgb8, unpack_srgb8};
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

    /// The occupancy of row `row` (`y + 32·z`), bit `x` for cell `x`. A
    /// word holds two rows: `y` even in its low half, odd in its high half.
    #[expect(clippy::cast_possible_truncation, reason = "one half of the word")]
    fn row(&self, row: usize) -> u32 {
        (self.occupied[row / 2] >> (32 * (row % 2))) as u32
    }
}

/// Rows of cells along X in a chunk, one for each (Y, Z).
const ROWS: usize = 32 * 32;

/// The occupancy of row `row` in a chunk that may not exist, empty if not.
fn row_of(chunk: Option<&Chunk>, row: usize) -> u32 {
    chunk.map_or(0, |chunk| chunk.row(row))
}

/// A chunk with the six chunks across its faces, looked up once, to tell its
/// exposed blocks from the buried ones: a block is exposed when one of the
/// six cells it shares a face with is empty, a missing chunk counting as
/// empty. Off-grid cubes are always exposed and bury nothing.
struct Exposure<'a> {
    chunk: &'a Chunk,
    /// The chunks across the faces at -X, +X, -Y, +Y, -Z and +Z.
    neighbours: [Option<&'a Chunk>; 6],
}

impl<'a> Exposure<'a> {
    fn new(chunks: &'a BTreeMap<ChunkKey, Chunk>, key: ChunkKey) -> Option<Self> {
        let chunk = chunks.get(&key)?;
        let neighbours = FACES.map(|face| chunks.get(&face_neighbour(key, face)?));
        Some(Self { chunk, neighbours })
    }

    /// The exposed blocks of row `row` (`y + 32·z`), bit `x` for cell `x`.
    fn row(&self, row: usize) -> u32 {
        let [west, east, down, up, back, front] = self.neighbours;
        let (y, z) = (row % 32, row / 32);
        let here = self.chunk.row(row);
        // Each neighbour's occupancy brought to the bit of the cell beside it.
        let minus_x = here << 1 | row_of(west, row) >> 31;
        let plus_x = here >> 1 | (row_of(east, row) & 1) << 31;
        let minus_y = if y > 0 {
            self.chunk.row(row - 1)
        } else {
            row_of(down, row + 31)
        };
        let plus_y = if y < 31 {
            self.chunk.row(row + 1)
        } else {
            row_of(up, row - 31)
        };
        let minus_z = if z > 0 {
            self.chunk.row(row - 32)
        } else {
            row_of(back, row + 31 * 32)
        };
        let plus_z = if z < 31 {
            self.chunk.row(row + 32)
        } else {
            row_of(front, row - 31 * 32)
        };
        here & !(minus_x & plus_x & minus_y & plus_y & minus_z & plus_z)
    }
}

/// The faces of a chunk, as the axis across them and the side.
const FACES: [(usize, i32); 6] = [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)];

/// The chunk across `face` of chunk `key`, or `None` past the end of the key
/// range, where an off-grid cube's chunk saturates and there is no chunk to
/// look at.
fn face_neighbour(key: ChunkKey, (axis, side): (usize, i32)) -> Option<ChunkKey> {
    let mut neighbour = key;
    neighbour[axis] = neighbour[axis].checked_add(side)?;
    Some(neighbour)
}

/// The faces of its chunk `cell` touches, as a bit for each of [`FACES`].
fn faces_touched(cell: IVec3) -> u8 {
    let local = cell.rem_euclid(IVec3::splat(SIDE)).to_array();
    let mut faces = 0;
    for (bit, &(axis, side)) in FACES.iter().enumerate() {
        let edge = if side < 0 { 0 } else { SIDE - 1 };
        if local[axis] == edge {
            faces |= 1 << bit;
        }
    }
    faces
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

/// How far from an integer a coordinate may be and still count as on it.
const GRID_TOLERANCE: f32 = 1e-4;

/// Past this, in either direction, an f32 no longer holds every integer, so
/// there is no grid to sit on.
const GRID_LIMIT: f32 = 16_777_216.0; // 2^24

/// The cell `p` is the centre of, if every coordinate is finite, within
/// [`GRID_LIMIT`], and within [`GRID_TOLERANCE`] of an integer. A NaN or
/// infinite coordinate is never on the grid, so such a cube neither occupies
/// a cell nor buries another.
fn grid_position(p: Vec3) -> Option<IVec3> {
    if !p.is_finite() || p.abs().max_element() >= GRID_LIMIT {
        return None;
    }
    let rounded = p.round();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "rounded already, and within 2^24, so it fits an i32"
    )]
    let on_grid = ((p - rounded).abs().max_element() <= GRID_TOLERANCE)
        .then(|| IVec3::new(rounded.x as i32, rounded.y as i32, rounded.z as i32));
    on_grid
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
            if let Some(cell) = grid_position(cube.position) {
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
    /// colour that encodes like the one in place changes nothing. Cells past
    /// ±2^24 on any axis, where an f32 no longer holds every integer, are not
    /// supported.
    pub fn set_block(&mut self, cell: IVec3, color: Vec3) -> Result<()> {
        let packed = pack_srgb8(color);
        match self.packed(cell) {
            Some(old) if old == packed => return Ok(()),
            Some(_) => {}
            None => self.check(self.count + 1)?,
        }
        let filled = self.put(cell, packed);
        self.mark_cell(cell, Priority::Immediate, filled);
        Ok(())
    }

    /// Empties `cell`; the chunk is rebuilt at once. An empty cell is left
    /// alone, and so are off-grid cubes.
    pub fn remove_block(&mut self, cell: IVec3) {
        if self.take(cell) {
            self.mark_cell(cell, Priority::Immediate, true);
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

        // Cells come in scan order, Z innermost, so a run of one chunk is at
        // most 32 cells: one lookup and one mark for each, not for each cell.
        // The chunks across the faces the filled or emptied cells touch are
        // marked too, once a run.
        for run in removes.chunk_by(|a, b| chunk_of_cell(*a) == chunk_of_cell(*b)) {
            let key = chunk_of_cell(run[0]);
            let chunk = self.chunks.get_mut(&key).expect("it held these blocks");
            let mut faces = 0;
            for &cell in run {
                chunk.remove(local_index(cell));
                faces |= faces_touched(cell);
            }
            self.count -= run.len();
            if chunk.len() == 0 {
                self.chunks.remove(&key);
            }
            self.mark(key, Priority::Budget);
            self.mark_across(key, faces, Priority::Budget);
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
            let mut faces = 0;
            for &(cell, packed) in run {
                if chunk.insert(local_index(cell), packed) {
                    self.count += 1;
                    faces |= faces_touched(cell);
                }
            }
            self.mark(key, Priority::Budget);
            self.mark_across(key, faces, Priority::Budget);
        }
        Ok(())
    }

    /// The keys of every chunk holding a cube, in order.
    pub fn chunk_keys(&self) -> impl Iterator<Item = ChunkKey> + '_ {
        self.chunks.keys().copied()
    }

    /// What the GPU draws for chunk `key`: its exposed blocks by cell, then
    /// its off-grid cubes in the order they came. A block is exposed when one
    /// of the six cells it shares a face with is empty, a missing chunk
    /// counting as empty; off-grid cubes are always drawn and bury nothing.
    /// Empty for a chunk that no longer holds anything. The same cubes always
    /// give the same instances.
    pub fn instances(&self, key: ChunkKey) -> Vec<StaticCubeRaw> {
        let Some(exposure) = Exposure::new(&self.chunks, key) else {
            return Vec::new();
        };
        let chunk = exposure.chunk;
        let mut instances = Vec::new();
        for row in 0..ROWS {
            let mut bits = exposure.row(row);
            while bits != 0 {
                let index = row * 32 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                instances.push(StaticCubeRaw {
                    position: cell_at(key, index).as_vec3().to_array(),
                    color: chunk.colors[index],
                });
            }
        }
        instances.extend(
            chunk
                .off_grid
                .iter()
                .map(|cube| StaticCubeRaw::new(cube.position, cube.color)),
        );
        instances
    }

    /// How many instances [`instances`](Self::instances) gives for chunk
    /// `key`, exactly.
    pub fn sent_len(&self, key: ChunkKey) -> u32 {
        Exposure::new(&self.chunks, key).map_or(0, |exposure| {
            let blocks: u32 = (0..ROWS).map(|row| exposure.row(row).count_ones()).sum();
            let off_grid = u32::try_from(exposure.chunk.off_grid.len())
                .expect("a chunk holds fewer off-grid cubes than u32::MAX");
            blocks + off_grid
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

    /// Puts a block without marking anything; `true` if the cell was empty.
    fn put(&mut self, cell: IVec3, packed: u32) -> bool {
        let chunk = self.chunks.entry(chunk_of_cell(cell)).or_default();
        let filled = chunk.insert(local_index(cell), packed);
        self.count += usize::from(filled);
        filled
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

    /// Marks the chunk of `cell` to rebuild and, if the cell was filled or
    /// emptied, every existing chunk across a face it touches: what those
    /// show depends on it.
    fn mark_cell(&mut self, cell: IVec3, priority: Priority, occupancy_changed: bool) {
        let key = chunk_of_cell(cell);
        self.mark(key, priority);
        if occupancy_changed {
            self.mark_across(key, faces_touched(cell), priority);
        }
    }

    /// Marks every existing chunk across the `faces` of chunk `key` (a bit
    /// for each of [`FACES`]). A missing chunk shows nothing to update and is
    /// never created for it.
    fn mark_across(&mut self, key: ChunkKey, faces: u8, priority: Priority) {
        for (bit, &face) in FACES.iter().enumerate() {
            let Some(neighbour) = face_neighbour(key, face) else {
                continue;
            };
            if faces & 1 << bit != 0 && self.chunks.contains_key(&neighbour) {
                self.mark(neighbour, priority);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::{IVec3, Vec3};

    use super::*;
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
        assert_eq!(kept[0].position[0], 3.25);
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
            bytemuck::cast_slice::<StaticCubeRaw, u8>(&w.instances([0, 0, 0])).to_vec()
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

    fn cube_at(p: Vec3) -> ModelCube {
        ModelCube {
            position: p,
            color: Vec3::new(0.5, 0.25, 1.0),
        }
    }

    /// A solid `size`³ block with its lowest corner at `min`.
    fn block(min: IVec3, size: i32) -> Vec<ModelCube> {
        let mut cubes = Vec::new();
        for x in 0..size {
            for y in 0..size {
                for z in 0..size {
                    cubes.push(cube_at((min + IVec3::new(x, y, z)).as_vec3()));
                }
            }
        }
        cubes
    }

    fn from_cubes(cubes: &[ModelCube]) -> World {
        World::new(cubes, cubes.len())
    }

    /// Where an instance is drawn.
    fn position(cube: &StaticCubeRaw) -> Vec3 {
        Vec3::from(cube.position)
    }

    /// Every instance sent, across all chunks, by position.
    fn exposed_positions(world: &World) -> Vec<Vec3> {
        world
            .chunk_keys()
            .flat_map(|key| world.instances(key))
            .map(|cube| position(&cube))
            .collect()
    }

    fn exposed_set(world: &World) -> BTreeSet<[i32; 3]> {
        exposed_positions(world)
            .into_iter()
            .map(|p| p.round().as_ivec3().to_array())
            .collect()
    }

    #[test]
    fn a_lone_cube_is_exposed() {
        let world = from_cubes(&[cube_at(Vec3::new(3.0, 4.0, 5.0))]);
        assert_eq!(exposed_positions(&world), [Vec3::new(3.0, 4.0, 5.0)]);
        assert_eq!(world.cube_count(), 1);
    }

    #[test]
    fn the_centre_of_a_3x3x3_block_is_buried() {
        let world = from_cubes(&block(IVec3::ZERO, 3));
        let exposed = exposed_set(&world);
        assert_eq!(exposed.len(), 26);
        assert!(!exposed.contains(&[1, 1, 1]));
        assert_eq!(world.cube_count(), 27);
    }

    #[test]
    fn neighbours_are_found_across_chunks() {
        // x = 30, 31 in chunk 0 and x = 32 in chunk 1.
        let world = from_cubes(&block(IVec3::new(30, 0, 0), 3));
        assert_eq!(world.chunk_keys().count(), 2);
        let exposed = exposed_set(&world);
        assert_eq!(exposed.len(), 26);
        assert!(!exposed.contains(&[31, 1, 1]));
    }

    #[test]
    fn neighbours_are_found_across_chunks_on_y_and_z() {
        for (axis, centre) in [(1, [1, 31, 1]), (2, [1, 1, 31])] {
            // 31 and 32 on `axis` are in chunks 0 and 1.
            let mut min = IVec3::ZERO;
            min[axis] = 30;
            let world = from_cubes(&block(min, 3));
            assert_eq!(world.chunk_keys().count(), 2, "axis {axis}");
            let exposed = exposed_set(&world);
            assert_eq!(exposed.len(), 26, "axis {axis}");
            assert!(!exposed.contains(&centre), "axis {axis}");
            let mut key = [0, 0, 0];
            key[axis] = 1;
            for key in [[0, 0, 0], key] {
                assert_eq!(world.sent_len(key) as usize, world.instances(key).len());
            }
        }
    }

    #[test]
    fn negative_coordinates_land_in_negative_chunks() {
        // -3..=-1 on every axis: all in chunk (-1, -1, -1).
        let world = from_cubes(&block(IVec3::splat(-3), 3));
        assert_eq!(world.chunk_keys().collect::<Vec<_>>(), [[-1, -1, -1]]);
        let exposed = exposed_set(&world);
        assert_eq!(exposed.len(), 26);
        assert!(!exposed.contains(&[-2, -2, -2]));
    }

    #[test]
    fn off_grid_cubes_are_never_buried_and_bury_nothing() {
        let mut cubes = block(IVec3::ZERO, 3);
        // The centre, moved half a cube along X.
        cubes.retain(|c| c.position != Vec3::ONE);
        cubes.push(cube_at(Vec3::new(1.5, 1.0, 1.0)));
        let world = from_cubes(&cubes);
        let exposed = exposed_positions(&world);
        assert_eq!(exposed.len(), 27);
        assert!(exposed.contains(&Vec3::new(1.5, 1.0, 1.0)));
    }

    #[test]
    fn a_closed_cavity_keeps_its_walls() {
        // A 5³ block with only its centre removed: the 98 cubes of the shell
        // and the 6 that touch the hole are exposed.
        let mut cubes = block(IVec3::ZERO, 5);
        cubes.retain(|c| c.position != Vec3::splat(2.0));
        let world = from_cubes(&cubes);
        let exposed = exposed_set(&world);
        assert_eq!(exposed.len(), 98 + 6);
        assert!(exposed.contains(&[2, 2, 1]));
        assert!(!exposed.contains(&[1, 1, 1]));
    }

    #[test]
    fn duplicates_count_once() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let world = from_cubes(&[cube_at(p), cube_at(p)]);
        assert_eq!(exposed_positions(&world), [p]);
        assert_eq!(world.cube_count(), 1);
    }

    #[test]
    fn colours_are_kept() {
        let world = from_cubes(&[cube_at(Vec3::ZERO)]);
        let raw = world.instances([0, 0, 0])[0];
        assert_eq!(raw.color, pack_srgb8(Vec3::new(0.5, 0.25, 1.0)));
    }

    #[test]
    fn only_finite_coordinates_within_float_precision_are_on_the_grid() {
        assert_eq!(
            grid_position(Vec3::new(1.0, -2.0, 3.0)),
            Some(IVec3::new(1, -2, 3))
        );
        for p in [
            Vec3::new(f32::NAN, 5.0, 5.0),
            Vec3::new(5.0, f32::NAN, 5.0),
            Vec3::new(5.0, 5.0, f32::NAN),
            Vec3::new(f32::INFINITY, 5.0, 5.0),
            Vec3::new(5.0, f32::NEG_INFINITY, 5.0),
            // Past 2^24 an f32 no longer holds every integer.
            Vec3::new(1.0e10, 5.0, 5.0),
            Vec3::new(5.0, 5.0, -3.0e7),
        ] {
            assert_eq!(grid_position(p), None, "{p}");
        }
    }

    #[test]
    fn a_cube_with_a_bad_coordinate_neither_occupies_nor_buries() {
        // The NaN cube would land on (0, 1, 1) if it were put on the grid,
        // and the 1e10 one next to i32::MAX, where looking at its neighbour
        // would overflow.
        let mut cubes = block(IVec3::ZERO, 3);
        cubes.retain(|c| c.position != Vec3::new(0.0, 1.0, 1.0));
        cubes.push(cube_at(Vec3::new(f32::NAN, 1.0, 1.0)));
        cubes.push(cube_at(Vec3::new(1.0e10, 0.0, 0.0)));
        let world = from_cubes(&cubes);
        assert_eq!(world.block(IVec3::new(0, 1, 1)), None);
        // The 26 placed cubes, now all exposed through the hole, plus the two
        // off the grid.
        assert_eq!(exposed_positions(&world).len(), 26 + 2);
    }

    #[test]
    fn a_cube_in_a_saturated_chunk_has_no_neighbour_to_overflow_into() {
        // Off the grid, `chunk_of` saturates their chunk key to i32::MIN or
        // i32::MAX, where there is no chunk across the face to look at.
        let far = [
            Vec3::new(f32::INFINITY, 0.0, 0.0),
            Vec3::new(f32::NEG_INFINITY, 0.0, 0.0),
            Vec3::new(-1.0e12, 0.0, 0.0),
        ];
        let mut cubes = block(IVec3::ZERO, 3);
        cubes.extend(far.map(cube_at));
        let world = from_cubes(&cubes);
        let mut sent = Vec::new();
        for key in world.chunk_keys() {
            assert_eq!(world.sent_len(key) as usize, world.instances(key).len());
            sent.extend(world.instances(key).iter().map(position));
        }
        for p in far {
            assert!(sent.contains(&p), "{p}");
        }
        // The block next to the origin is untouched: the shell is exposed.
        assert_eq!(sent.len(), 26 + far.len());
    }

    #[test]
    fn float_noise_snaps_to_the_grid() {
        let noisy: Vec<_> = block(IVec3::ZERO, 3)
            .into_iter()
            .map(|c| cube_at(c.position + Vec3::splat(0.000_01)))
            .collect();
        let world = from_cubes(&noisy);
        assert_eq!(exposed_set(&world).len(), 26);
        assert!(world.block(IVec3::ONE).is_some());
    }

    #[test]
    fn a_cube_just_short_of_a_chunk_edge_snaps_into_the_next_chunk() {
        // 31.99999 is 32 on the grid, which is chunk 1, not chunk 0.
        let world = from_cubes(&[cube_at(Vec3::new(31.999_99, 0.0, 0.0))]);
        assert_eq!(world.chunk_keys().collect::<Vec<_>>(), [[1, 0, 0]]);
        assert_eq!(exposed_positions(&world), [Vec3::new(32.0, 0.0, 0.0)]);
    }

    // Cells 30..=33 on X, 0..=2 on Y and Z: two chunks along X, (31, 1, 1) and
    // (32, 1, 1) buried.
    fn slab() -> World {
        let mut cubes = Vec::new();
        for x in 30..=33 {
            for y in 0..=2 {
                for z in 0..=2 {
                    cubes.push(cube_at(IVec3::new(x, y, z).as_vec3()));
                }
            }
        }
        let mut w = from_cubes(&cubes);
        assert_eq!(w.cube_count(), 36);
        let keys: Vec<_> = w.chunk_keys().collect();
        for key in keys {
            w.clean(key);
        }
        w
    }

    #[test]
    fn removing_a_face_cell_marks_the_neighbour_behind_it() {
        let mut w = slab();
        assert!(!exposed_set(&w).contains(&[32, 1, 1]));
        w.remove_block(IVec3::new(31, 1, 1));
        assert_eq!(
            w.dirty().map(|(k, _)| k).collect::<Vec<_>>(),
            [[0, 0, 0], [1, 0, 0]]
        );
        assert!(
            w.instances([1, 0, 0])
                .iter()
                .any(|c| c.position == [32.0, 1.0, 1.0])
        );
    }

    #[test]
    fn recolouring_a_face_cell_rebuilds_only_its_chunk() {
        let mut w = slab();
        w.set_block(IVec3::new(31, 1, 1), BLUE).unwrap();
        assert_eq!(w.dirty().map(|(k, _)| k).collect::<Vec<_>>(), [[0, 0, 0]]);
    }

    #[test]
    fn a_corner_cell_marks_its_three_face_neighbours() {
        // Blocks at (31, 31, 31) and in the three chunks across its faces.
        let corner = IVec3::splat(31);
        let mut w = from_cubes(&[
            cube_at(corner.as_vec3()),
            cube_at((corner + IVec3::X).as_vec3()),
            cube_at((corner + IVec3::Y).as_vec3()),
            cube_at((corner + IVec3::Z).as_vec3()),
        ]);
        w.remove_block(corner);
        assert_eq!(
            w.dirty().collect::<Vec<_>>(),
            [
                ([0, 0, 0], Priority::Immediate),
                ([0, 0, 1], Priority::Immediate),
                ([0, 1, 0], Priority::Immediate),
                ([1, 0, 0], Priority::Immediate),
            ]
        );
    }

    #[test]
    fn a_face_cell_with_no_neighbour_chunk_marks_only_its_own() {
        let mut w = world(1);
        w.set_block(IVec3::new(31, 0, 0), RED).unwrap();
        assert_eq!(w.dirty().map(|(k, _)| k).collect::<Vec<_>>(), [[0, 0, 0]]);
        assert_eq!(w.chunk_keys().count(), 1);
    }

    #[test]
    fn a_region_marks_neighbours_for_the_budget() {
        let mut w = slab();
        let cell = IVec3::new(31, 1, 1);
        w.edit_region(cell, cell, |_| BlockEdit::Remove).unwrap();
        assert_eq!(
            w.dirty().collect::<Vec<_>>(),
            [([0, 0, 0], Priority::Budget), ([1, 0, 0], Priority::Budget)]
        );
    }

    #[test]
    fn sent_len_counts_what_instances_returns() {
        let mut opened = slab();
        opened.remove_block(IVec3::new(31, 1, 1));
        let mut off_grid = block(IVec3::ZERO, 3);
        off_grid.push(cube_at(Vec3::new(1.5, 3.0, 1.0)));
        for w in [slab(), opened, from_cubes(&off_grid)] {
            for key in w.chunk_keys() {
                assert_eq!(w.sent_len(key) as usize, w.instances(key).len());
            }
        }
    }
}
