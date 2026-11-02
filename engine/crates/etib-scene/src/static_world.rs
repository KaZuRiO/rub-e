//! The static cubes on the CPU, by chunk: which places are occupied, and so
//! which cubes have a face that can be seen.
//!
//! A cube whose six neighbours are all occupied is buried: no ray reaches any
//! of its faces, so it is never sent to the GPU. The map is kept per chunk
//! (the culling's 32×32 tiles in XZ) so that a later change to the world only
//! has to look again at one chunk and those it borders.

use std::collections::{BTreeMap, HashMap};

use glam::{IVec3, Vec3};

use crate::cube::{StaticCubeRaw, pack_srgb8};
use crate::culling::tile_of;
use crate::model::ModelCube;

/// A chunk's place among the tiles, as [`tile_of`] gives it.
pub(crate) type ChunkKey = (i32, i32);

/// How far from an integer a coordinate may be and still count as on it.
const GRID_TOLERANCE: f32 = 1e-4;

/// The six neighbours of a place on the grid.
const NEIGHBOURS: [IVec3; 6] = [
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Y,
    IVec3::NEG_Y,
    IVec3::Z,
    IVec3::NEG_Z,
];

/// Past this, in either direction, an f32 no longer holds every integer, so
/// there is no grid to sit on.
const GRID_LIMIT: f32 = 16_777_216.0; // 2^24

/// The place on the integer grid `p` sits on, if every coordinate is finite,
/// within [`GRID_LIMIT`], and within [`GRID_TOLERANCE`] of an integer. A NaN
/// or infinite coordinate is never on it, so such a cube neither occupies a
/// place nor buries another.
pub(crate) fn grid_position(p: Vec3) -> Option<IVec3> {
    if !p.is_finite() || p.abs().max_element() >= GRID_LIMIT {
        return None;
    }
    let rounded = p.round();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "rounded already, and a world past i32 on any axis is not one the culling could tile"
    )]
    let on_grid = ((p - rounded).abs().max_element() <= GRID_TOLERANCE)
        .then(|| IVec3::new(rounded.x as i32, rounded.y as i32, rounded.z as i32));
    on_grid
}

/// The cubes of one chunk.
#[derive(Debug, Default)]
struct ChunkCubes {
    /// Cubes on the grid, by place, with their packed colour. Two cubes at one
    /// place are one: the later wins.
    grid: HashMap<IVec3, u32>,
    /// Cubes off the grid: never buried, and never burying another.
    loose: Vec<StaticCubeRaw>,
}

/// Every static cube, by chunk.
#[derive(Debug, Default)]
pub(crate) struct StaticWorld {
    chunks: BTreeMap<ChunkKey, ChunkCubes>,
    /// Cubes given, duplicates included.
    total: u32,
}

impl StaticWorld {
    /// Sorts `cubes` into chunks. A cube on the grid goes to the chunk of its
    /// place on it, so one a hair short of a tile edge joins its neighbours.
    pub fn from_cubes(cubes: &[ModelCube]) -> Self {
        let mut world = Self {
            total: u32::try_from(cubes.len()).expect("more than u32::MAX static cubes"),
            ..Self::default()
        };
        for cube in cubes {
            match grid_position(cube.position) {
                Some(place) => {
                    world
                        .chunks
                        .entry(tile_of(place.as_vec3()))
                        .or_default()
                        .grid
                        .insert(place, pack_srgb8(cube.color));
                }
                None => world
                    .chunks
                    .entry(tile_of(cube.position))
                    .or_default()
                    .loose
                    .push(StaticCubeRaw::new(cube.position, cube.color)),
            }
        }
        world
    }

    /// Whether a cube on the grid occupies `place`.
    pub fn is_occupied(&self, place: IVec3) -> bool {
        self.chunks
            .get(&tile_of(place.as_vec3()))
            .is_some_and(|chunk| chunk.grid.contains_key(&place))
    }

    /// The chunks that hold at least one cube, in a fixed order.
    pub fn chunk_keys(&self) -> impl Iterator<Item = ChunkKey> + '_ {
        self.chunks.keys().copied()
    }

    /// The cubes of chunk `key` with at least one face free: those on the
    /// grid sorted by `(y, z, x)`, then those off it in the order given.
    pub fn exposed(&self, key: ChunkKey) -> Vec<StaticCubeRaw> {
        let Some(chunk) = self.chunks.get(&key) else {
            return Vec::new();
        };
        let mut places: Vec<(IVec3, u32)> = chunk
            .grid
            .iter()
            .filter(|(place, _)| {
                NEIGHBOURS
                    .iter()
                    .any(|step| !self.is_occupied(**place + *step))
            })
            .map(|(place, color)| (*place, *color))
            .collect();
        places.sort_unstable_by_key(|(place, _)| (place.y, place.z, place.x));
        places
            .into_iter()
            .map(|(place, color)| StaticCubeRaw {
                position: place.as_vec3().to_array(),
                color,
            })
            .chain(chunk.loose.iter().copied())
            .collect()
    }

    /// How many cubes the world was built from, duplicates included.
    pub fn total(&self) -> u32 {
        self.total
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use glam::{IVec3, Vec3};

    use super::*;
    use crate::cube::pack_srgb8;
    use crate::model::ModelCube;

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

    /// Every exposed cube, across all chunks, by position.
    fn exposed_positions(world: &StaticWorld) -> Vec<Vec3> {
        world
            .chunk_keys()
            .flat_map(|key| world.exposed(key))
            .map(|raw| Vec3::from(raw.position))
            .collect()
    }

    fn exposed_set(world: &StaticWorld) -> HashSet<IVec3> {
        exposed_positions(world)
            .into_iter()
            .map(|p| p.round().as_ivec3())
            .collect()
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
        let world = StaticWorld::from_cubes(&cubes);
        assert!(!world.is_occupied(IVec3::new(0, 1, 1)));
        // The 26 placed cubes, now all exposed through the hole, plus the two
        // loose ones.
        assert_eq!(exposed_positions(&world).len(), 26 + 2);
    }

    #[test]
    fn a_lone_cube_is_exposed() {
        let world = StaticWorld::from_cubes(&[cube_at(Vec3::new(3.0, 4.0, 5.0))]);
        assert_eq!(exposed_positions(&world), [Vec3::new(3.0, 4.0, 5.0)]);
        assert_eq!(world.total(), 1);
    }

    #[test]
    fn the_centre_of_a_3x3x3_block_is_buried() {
        let world = StaticWorld::from_cubes(&block(IVec3::ZERO, 3));
        let exposed = exposed_set(&world);
        assert_eq!(exposed.len(), 26);
        assert!(!exposed.contains(&IVec3::ONE));
        assert_eq!(world.total(), 27);
    }

    #[test]
    fn neighbours_are_found_across_chunks() {
        // x = 30, 31 in tile 0 and x = 32 in tile 1.
        let world = StaticWorld::from_cubes(&block(IVec3::new(30, 0, 0), 3));
        assert_eq!(world.chunk_keys().count(), 2);
        let exposed = exposed_set(&world);
        assert_eq!(exposed.len(), 26);
        assert!(!exposed.contains(&IVec3::new(31, 1, 1)));
    }

    #[test]
    fn negative_coordinates_land_in_negative_tiles() {
        // -3..=-1 on every axis: all in tile (-1, -1).
        let world = StaticWorld::from_cubes(&block(IVec3::splat(-3), 3));
        assert_eq!(world.chunk_keys().collect::<Vec<_>>(), [(-1, -1)]);
        let exposed = exposed_set(&world);
        assert_eq!(exposed.len(), 26);
        assert!(!exposed.contains(&IVec3::splat(-2)));
    }

    #[test]
    fn float_noise_snaps_to_the_grid() {
        let noisy: Vec<_> = block(IVec3::ZERO, 3)
            .into_iter()
            .map(|c| cube_at(c.position + Vec3::splat(0.000_01)))
            .collect();
        let world = StaticWorld::from_cubes(&noisy);
        assert_eq!(exposed_set(&world).len(), 26);
    }

    #[test]
    fn a_cube_just_short_of_a_tile_edge_snaps_into_the_next_tile() {
        // 31.99999 is 32 on the grid, which is tile 1, not tile 0.
        let world = StaticWorld::from_cubes(&[cube_at(Vec3::new(31.999_99, 0.0, 0.0))]);
        assert_eq!(world.chunk_keys().collect::<Vec<_>>(), [(1, 0)]);
    }

    #[test]
    fn off_grid_cubes_are_never_buried_and_bury_nothing() {
        let mut cubes = block(IVec3::ZERO, 3);
        // The centre, moved half a cube along X.
        cubes.retain(|c| c.position != Vec3::ONE);
        cubes.push(cube_at(Vec3::new(1.5, 1.0, 1.0)));
        let world = StaticWorld::from_cubes(&cubes);
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
        let world = StaticWorld::from_cubes(&cubes);
        let exposed = exposed_set(&world);
        assert_eq!(exposed.len(), 98 + 6);
        assert!(exposed.contains(&IVec3::new(2, 2, 1)));
        assert!(!exposed.contains(&IVec3::new(1, 1, 1)));
    }

    #[test]
    fn duplicates_count_once() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let world = StaticWorld::from_cubes(&[cube_at(p), cube_at(p)]);
        assert_eq!(exposed_positions(&world), [p]);
        assert_eq!(world.total(), 2);
    }

    #[test]
    fn colours_are_kept() {
        let world = StaticWorld::from_cubes(&[cube_at(Vec3::ZERO)]);
        let raw = world.exposed((0, 0))[0];
        assert_eq!(raw.color, pack_srgb8(Vec3::new(0.5, 0.25, 1.0)));
    }

    #[test]
    fn exposed_order_is_deterministic() {
        let mut cubes = block(IVec3::new(-3, 0, 28), 6);
        let world = StaticWorld::from_cubes(&cubes);
        cubes.reverse();
        let reversed = StaticWorld::from_cubes(&cubes);
        let keys: Vec<_> = world.chunk_keys().collect();
        assert_eq!(keys, reversed.chunk_keys().collect::<Vec<_>>());
        for key in keys {
            assert_eq!(world.exposed(key), reversed.exposed(key));
        }
    }

    #[test]
    fn occupancy_is_read_across_chunks() {
        let world = StaticWorld::from_cubes(&[cube_at(Vec3::new(32.0, 0.0, -1.0))]);
        assert!(world.is_occupied(IVec3::new(32, 0, -1)));
        assert!(!world.is_occupied(IVec3::new(31, 0, -1)));
    }
}
