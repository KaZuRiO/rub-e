//! The benchmark scenarios: what each one puts in the scene and moves, and the
//! camera path they all share. All pure functions, with no randomness, so every
//! run of a scenario sees the same frames.

use std::ops::Range;

use clap::ValueEnum;
use etib_scene::glam::{IVec3, Vec3};
use etib_scene::{DynamicModel, ModelCube};

/// Bumped whenever the world or the path changes, so results of different
/// scenarios are never compared.
pub const SCENARIO_VERSION: u32 = 1;
/// Frames drawn before measuring, with the camera held on the first pose.
pub const WARMUP_FRAMES: u32 = 120;
/// Frames measured, along the camera path.
pub const MEASURED_FRAMES: u32 = 600;
/// Frames drawn after the measured ones, with the camera held on the last pose.
pub const GRACE_FRAMES: u32 = 120;
/// Window size in pixels.
pub const WINDOW: (u32, u32) = (1280, 720);

/// Far plane distance the camera path needs: from the last pose, the farthest
/// corner of the terrain is about 540 units away. The bench app sets the
/// perspective camera's `far` to at least this (fovy 45 degrees, near 0.1).
pub const FAR_DISTANCE: f32 = 1000.0;

/// Cells of the scattered lattice along X and Z.
const SCATTERED_ACROSS: i32 = 85;
/// Cells of the scattered lattice along Y.
const SCATTERED_LAYERS: i32 = 28;
/// Side of a scattered cell, in world units.
const SCATTERED_CELL: f32 = 3.0;
/// How far a scattered cube strays from its cell's centre, per axis: little
/// enough to leave half a unit between any two cubes.
const SCATTERED_JITTER: f32 = 0.75;

/// Models along each side of the swarm.
const SWARM_SIDE: usize = 71;
/// Models in the swarm.
const SWARM_MODELS: usize = SWARM_SIDE * SWARM_SIDE;
/// Distance between two models at rest.
const SWARM_SPACING: f32 = 6.0;
/// Height of the swarm at rest.
const SWARM_HEIGHT: f32 = 12.0;
/// Radius of the circle each model follows.
const SWARM_RADIUS: f32 = 2.0;

/// What a run plays. Each one shows a kind of game a single world would hide.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum Scenario {
    /// Solid ground and towers: most cubes are buried.
    #[default]
    Terrain,
    /// Static cubes apart from each other and off the grid: no cube has a
    /// neighbour to hide a face against, though they still hide one another.
    Scattered,
    /// A swarm of dynamic models that all move every frame.
    Moving,
    /// The same swarm with a single model moving each frame.
    FewChanges,
    /// The terrain, with four surface blocks near the camera toggled every
    /// frame, each in its own chunk: players breaking and placing blocks.
    EditBlocks,
    /// The terrain, with a ball of it blown away in front of the camera every
    /// 120 frames and put back 60 frames later: explosions.
    EditArea,
}

impl Scenario {
    /// The name it goes by on the command line and in the results.
    pub fn name(self) -> &'static str {
        match self {
            Self::Terrain => "terrain",
            Self::Scattered => "scattered",
            Self::Moving => "moving",
            Self::FewChanges => "few-changes",
            Self::EditBlocks => "edit-blocks",
            Self::EditArea => "edit-area",
        }
    }

    /// The static cubes of the scene.
    pub fn static_cubes(self) -> Vec<ModelCube> {
        match self {
            Self::Terrain | Self::EditBlocks | Self::EditArea => world(),
            Self::Scattered => scattered(),
            Self::Moving | Self::FewChanges => Vec::new(),
        }
    }

    /// The dynamic models of the scene, in the order
    /// [`moved_models`](Self::moved_models) counts them.
    pub fn models(self) -> Vec<DynamicModel> {
        match self {
            Self::Terrain | Self::Scattered | Self::EditBlocks | Self::EditArea => Vec::new(),
            Self::Moving | Self::FewChanges => (0..SWARM_MODELS).map(swarm_model).collect(),
        }
    }

    /// The static capacity the scene is built with for `cubes` static cubes:
    /// 10 % to spare for the scenarios that edit them, so a scene full to the
    /// brim never skews their numbers.
    pub fn static_capacity(self, cubes: usize) -> usize {
        match self {
            Self::EditBlocks | Self::EditArea => cubes * 11 / 10,
            _ => cubes,
        }
    }

    /// The models that move on `frame`.
    pub fn moved_models(self, frame: u32) -> Range<usize> {
        match self {
            Self::Terrain | Self::Scattered | Self::EditBlocks | Self::EditArea => 0..0,
            Self::Moving => 0..SWARM_MODELS,
            Self::FewChanges => {
                let index = frame as usize % SWARM_MODELS;
                index..index + 1
            }
        }
    }
}

/// Where model `index` of the swarm is on `frame`: on a circle around its
/// place at rest, with neighbours out of step.
pub fn model_position(index: usize, frame: u32) -> Vec3 {
    let (row, column) = (index / SWARM_SIDE, index % SWARM_SIDE);
    let middle = (SWARM_SIDE / 2) as f32;
    let rest = Vec3::new(
        (column as f32 - middle) * SWARM_SPACING,
        SWARM_HEIGHT,
        (row as f32 - middle) * SWARM_SPACING,
    );
    let (sin, cos) = (frame as f32 * 0.03 + index as f32 * 0.7).sin_cos();
    rest + Vec3::new(cos, 0.0, sin) * SWARM_RADIUS
}

/// Model `index` of the swarm, at its place on frame 0: eight cubes in a block.
fn swarm_model(index: usize) -> DynamicModel {
    let seed = index as i32;
    let color = Vec3::new(
        0.2 + noise(seed, 0, 0, 3) * 0.8,
        0.2 + noise(seed, 0, 0, 4) * 0.8,
        0.2 + noise(seed, 0, 0, 5) * 0.8,
    );
    let corner = |bit: usize| if bit != 0 { 0.5 } else { -0.5 };
    let cubes = (0..8)
        .map(|i| ModelCube {
            position: Vec3::new(corner(i & 1), corner(i & 2), corner(i & 4)),
            color,
        })
        .collect();
    let mut model = DynamicModel::from_cubes(cubes);
    model.position = model_position(index, 0);
    model
}

/// One cube per cell of a lattice, each strayed from the centre of its cell.
fn scattered() -> Vec<ModelCube> {
    let mut cubes = Vec::new();
    let middle = SCATTERED_ACROSS / 2;
    for x in 0..SCATTERED_ACROSS {
        for y in 0..SCATTERED_LAYERS {
            for z in 0..SCATTERED_ACROSS {
                let stray = |salt| (noise(x, y, z, salt) * 2.0 - 1.0) * SCATTERED_JITTER;
                let centre = Vec3::new(
                    (x - middle) as f32 * SCATTERED_CELL,
                    (y as f32 + 0.5) * SCATTERED_CELL,
                    (z - middle) as f32 * SCATTERED_CELL,
                );
                cubes.push(ModelCube {
                    position: centre + Vec3::new(stray(0), stray(1), stray(2)),
                    color: Vec3::new(
                        0.2 + noise(x, y, z, 3) * 0.8,
                        0.2 + noise(x, y, z, 4) * 0.8,
                        0.2 + noise(x, y, z, 5) * 0.8,
                    ),
                });
            }
        }
    }
    cubes
}

/// A number in `0..1` that looks random and only depends on its arguments.
fn noise(x: i32, y: i32, z: i32, salt: u32) -> f32 {
    let mut hash = (x as u32).wrapping_mul(0x9E37_79B1)
        ^ (y as u32).wrapping_mul(0x85EB_CA77)
        ^ (z as u32).wrapping_mul(0xC2B2_AE3D)
        ^ salt.wrapping_mul(0x27D4_EB2F);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x85EB_CA77);
    hash ^= hash >> 13;
    hash = hash.wrapping_mul(0xC2B2_AE3D);
    hash ^= hash >> 16;
    // 24 bits, which an f32 holds exactly, so the result stays below 1.
    (hash >> 8) as f32 / (1u32 << 24) as f32
}

/// Terrain side, in columns.
const SIDE: i32 = 256;
/// Towers sit on a grid with this spacing, in columns.
const TOWER_SPACING: i32 = 32;
/// Footprint of a tower, in columns.
const TOWER_WIDTH: i32 = 5;

/// A `SIDE` x `SIDE` terrain filled from y = 0 up to a rolling surface, with a
/// grid of solid towers rising out of it. Cubes are 1 unit wide, centred on
/// integer coordinates; the terrain is centred on the origin.
fn world() -> Vec<ModelCube> {
    let mut cubes = Vec::with_capacity(700_000);
    for x in -SIDE / 2..SIDE / 2 {
        for z in -SIDE / 2..SIDE / 2 {
            let (fx, fz) = (x as f32, z as f32);
            let top = top(x, z);
            for y in 0..=top {
                cubes.push(ModelCube {
                    position: Vec3::new(fx, y as f32, fz),
                    color: color(x, y, z, top),
                });
            }
        }
    }
    cubes
}

/// The height of the highest cube of column `(x, z)`: the rolling surface, or
/// the tower that rises out of it.
fn top(x: i32, z: i32) -> i32 {
    let (fx, fz) = (x as f32, z as f32);
    let relief = (fx * 0.08).sin() * 0.7
        + (fz * 0.08).cos() * 0.7
        + (fx * 0.02 + fz * 0.02).sin() * 1.5
        + (fx * 0.06).sin() * (fz * 0.06).cos() * 0.5;
    let top = 8 + (relief * 2.0).round() as i32;
    tower_height(x, z).map_or(top, |height| top.max(height))
}

/// The colour of the terrain's block in `cell`, or `None` where the terrain
/// has none: outside it, or above its surface.
pub fn terrain_block(cell: IVec3) -> Option<Vec3> {
    let across = -SIDE / 2..SIDE / 2;
    if !across.contains(&cell.x) || !across.contains(&cell.z) || cell.y < 0 {
        return None;
    }
    let top = top(cell.x, cell.z);
    (cell.y <= top).then(|| color(cell.x, cell.y, cell.z, top))
}

/// How far from the camera, along X and Z, `edit-blocks` picks its blocks.
const EDIT_OFFSET: i32 = 20;

/// A column of the terrain near `position`, far enough inside it that a
/// column [`EDIT_OFFSET`] away on either side still is.
fn column_near(position: Vec3) -> (i32, i32) {
    let inside =
        |v: f32| (v.round() as i32).clamp(-SIDE / 2 + EDIT_OFFSET, SIDE / 2 - 1 - EDIT_OFFSET);
    (inside(position.x), inside(position.z))
}

/// The four surface blocks `edit-blocks` toggles on `frame`, with their
/// colour in the terrain: around the camera, 40 columns apart so each is in
/// its own chunk. The same four for two frames in a row, so each is removed
/// and then put back.
pub fn edit_blocks(frame: u32) -> [(IVec3, Vec3); 4] {
    let (x, z) = column_near(camera_at(frame & !1).0);
    [(-1, -1), (-1, 1), (1, -1), (1, 1)].map(|(dx, dz)| {
        let (x, z) = (x + dx * EDIT_OFFSET, z + dz * EDIT_OFFSET);
        let top = top(x, z);
        (IVec3::new(x, top, z), color(x, top, z, top))
    })
}

/// What `edit-area` does to its ball.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AreaEdit {
    /// Empties every cell of the ball.
    Remove,
    /// Puts the terrain back in it.
    Restore,
}

/// The radius of the `edit-area` ball, in cells.
pub const AREA_RADIUS: i32 = 32;

/// Frames between two explosions of `edit-area`; the terrain comes back half
/// way through.
const AREA_PERIOD: u32 = 120;

/// What `edit-area` does on `frame`, and the centre of its ball: on the
/// surface where the camera looks when the period starts.
pub fn area_edit(frame: u32) -> Option<(AreaEdit, IVec3)> {
    let edit = match frame % AREA_PERIOD {
        0 => AreaEdit::Remove,
        p if p == AREA_PERIOD / 2 => AreaEdit::Restore,
        _ => return None,
    };
    let target = camera_at(frame - frame % AREA_PERIOD).1;
    let inside = |v: f32| (v.round() as i32).clamp(-SIDE / 2, SIDE / 2 - 1);
    let (x, z) = (inside(target.x), inside(target.z));
    Some((edit, IVec3::new(x, top(x, z), z)))
}

/// Height of the tower covering column `(x, z)`, if any. Heights vary from
/// tower to tower so that the taller ones hide the others.
fn tower_height(x: i32, z: i32) -> Option<i32> {
    let (cx, cz) = (x.div_euclid(TOWER_SPACING), z.div_euclid(TOWER_SPACING));
    let (dx, dz) = (x.rem_euclid(TOWER_SPACING), z.rem_euclid(TOWER_SPACING));
    let start = (TOWER_SPACING - TOWER_WIDTH) / 2;
    let inside = |d: i32| (start..start + TOWER_WIDTH).contains(&d);
    (inside(dx) && inside(dz)).then(|| 22 + (cx * 7 + cz * 13).rem_euclid(5) * 6)
}

/// Green near the surface, earthy lower down, with a little variation per
/// cube so neighbours are told apart.
fn color(x: i32, y: i32, z: i32, top: i32) -> Vec3 {
    let jitter = (x * 31 + z * 17 + y * 7).rem_euclid(11) as f32 * 0.01;
    let depth = (top - y) as f32 / 30.0;
    Vec3::new(
        0.25 + jitter + depth * 0.3,
        0.55 - jitter - depth * 0.2,
        0.25 + jitter,
    )
    .clamp(Vec3::ZERO, Vec3::ONE)
}

/// Poses `(eye, target)` the camera passes through, evenly spread over the
/// measured frames: close to the towers, over them, then above the whole
/// terrain.
const KEYFRAMES: [(Vec3, Vec3); 3] = [
    // Skimming the towers.
    (
        Vec3::new(-100.0, 16.0, -110.0),
        Vec3::new(-50.0, 18.0, -60.0),
    ),
    // Flying over them.
    (Vec3::new(20.0, 70.0, 20.0), Vec3::new(60.0, 10.0, 60.0)),
    // The whole terrain in view.
    (Vec3::new(0.0, 260.0, 330.0), Vec3::ZERO),
];

/// The camera `(eye, target)` at `frame`. During warm-up it holds the first
/// pose; the measured frames interpolate linearly between the keyframes, the
/// last of them on the last pose, which is then held.
///
/// The perspective camera needs `far >= FAR_DISTANCE` to show the far view.
pub fn camera_at(frame: u32) -> (Vec3, Vec3) {
    let last = MEASURED_FRAMES - 1;
    let t = frame.saturating_sub(WARMUP_FRAMES).min(last) as f32 / last as f32
        * (KEYFRAMES.len() - 1) as f32;
    let i = (t as usize).min(KEYFRAMES.len() - 2);
    let f = t - i as f32;
    let ((e0, t0), (e1, t1)) = (KEYFRAMES[i], KEYFRAMES[i + 1]);
    (e0.lerp(e1, f), t0.lerp(t1, f))
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use etib_scene::glam::IVec3;

    use super::*;

    #[test]
    fn the_world_is_the_same_every_time() {
        assert_eq!(world(), world());
    }

    #[test]
    fn the_world_is_big_and_mostly_buried() {
        let cubes = world();
        assert!(
            (500_000..=1_000_000).contains(&cubes.len()),
            "{}",
            cubes.len()
        );
        // Most cubes have a neighbour above them.
        let set: std::collections::HashSet<_> = cubes
            .iter()
            .map(|c| c.position.to_array().map(|v| v as i32))
            .collect();
        let buried = set
            .iter()
            .filter(|[x, y, z]| set.contains(&[*x, y + 1, *z]))
            .count();
        assert!(buried * 2 > set.len());
    }

    const ALL: [Scenario; 6] = [
        Scenario::Terrain,
        Scenario::Scattered,
        Scenario::Moving,
        Scenario::FewChanges,
        Scenario::EditBlocks,
        Scenario::EditArea,
    ];

    #[test]
    fn the_edit_scenarios_start_from_the_terrain() {
        assert_eq!(Scenario::EditBlocks.static_cubes(), world());
        assert_eq!(Scenario::EditArea.static_cubes(), world());
        assert!(Scenario::EditBlocks.models().is_empty());
        assert!(Scenario::EditArea.models().is_empty());
    }

    #[test]
    fn the_edit_scenarios_leave_room() {
        assert_eq!(Scenario::EditBlocks.static_capacity(100), 110);
        assert_eq!(Scenario::EditArea.static_capacity(100), 110);
        assert_eq!(Scenario::Terrain.static_capacity(100), 100);
    }

    #[test]
    fn edit_blocks_touches_four_chunks_of_the_surface_every_frame() {
        for frame in 0..WARMUP_FRAMES + MEASURED_FRAMES + GRACE_FRAMES {
            let cells = edit_blocks(frame);
            let chunks: HashSet<_> = cells
                .iter()
                .map(|(c, _)| c.div_euclid(IVec3::splat(32)).to_array())
                .collect();
            assert_eq!(chunks.len(), 4, "frame {frame}");
            assert!(
                cells
                    .iter()
                    .all(|(c, color)| terrain_block(*c) == Some(*color)),
                "frame {frame}"
            );
        }
    }

    #[test]
    fn edit_blocks_uses_the_same_cells_two_frames_in_a_row() {
        assert_eq!(edit_blocks(10), edit_blocks(11));
        assert_ne!(edit_blocks(300), edit_blocks(400));
    }

    #[test]
    fn a_zone_is_removed_then_restored_at_the_same_place() {
        let (remove, centre) = area_edit(120).unwrap();
        assert!(matches!(remove, AreaEdit::Remove));
        let (restore, back) = area_edit(180).unwrap();
        assert!(matches!(restore, AreaEdit::Restore));
        assert_eq!(centre, back);
        assert!(area_edit(121).is_none());
    }

    #[test]
    fn terrain_block_knows_the_world() {
        for cube in world().iter().step_by(997) {
            assert_eq!(terrain_block(cube.position.as_ivec3()), Some(cube.color));
        }
        assert_eq!(terrain_block(IVec3::new(0, -1, 0)), None);
        assert_eq!(terrain_block(IVec3::new(0, 200, 0)), None);
    }

    #[test]
    fn every_scenario_builds_the_same_thing_every_time() {
        for scenario in ALL {
            assert_eq!(scenario.static_cubes(), scenario.static_cubes());
            let positions = |models: Vec<DynamicModel>| -> Vec<Vec3> {
                models.iter().map(|model| model.position).collect()
            };
            assert_eq!(positions(scenario.models()), positions(scenario.models()));
        }
    }

    #[test]
    fn terrain_is_the_world_and_nothing_else() {
        assert_eq!(Scenario::Terrain.static_cubes(), world());
        assert!(Scenario::Terrain.models().is_empty());
    }

    #[test]
    fn scattered_is_static_and_off_the_grid() {
        let cubes = Scenario::Scattered.static_cubes();
        assert!(
            (150_000..=250_000).contains(&cubes.len()),
            "{}",
            cubes.len()
        );
        assert!(Scenario::Scattered.models().is_empty());
        let on_grid = cubes
            .iter()
            .filter(|cube| cube.position == cube.position.round())
            .count();
        assert_eq!(on_grid, 0);
    }

    #[test]
    fn scattered_cubes_never_touch() {
        let cubes = Scenario::Scattered.static_cubes();
        let cell = |position: Vec3| (position / 4.0).floor().as_ivec3().to_array();
        let mut cells: HashMap<[i32; 3], Vec<Vec3>> = HashMap::new();
        for cube in &cubes {
            cells
                .entry(cell(cube.position))
                .or_default()
                .push(cube.position);
        }
        for cube in &cubes {
            let [x, y, z] = cell(cube.position);
            for near in (-1..=1).flat_map(|dx| {
                (-1..=1).flat_map(move |dy| (-1..=1).map(move |dz| [x + dx, y + dy, z + dz]))
            }) {
                for other in cells.get(&near).into_iter().flatten() {
                    let gap = (*other - cube.position).abs().max_element();
                    assert!(
                        *other == cube.position || gap > 1.0,
                        "{:?} touches {other:?}",
                        cube.position
                    );
                }
            }
        }
    }

    #[test]
    fn the_two_dynamic_scenarios_differ_only_in_what_moves() {
        let (moving, few) = (Scenario::Moving.models(), Scenario::FewChanges.models());
        assert_eq!(moving.len(), few.len());
        assert!((4_000..=6_000).contains(&moving.len()), "{}", moving.len());
        assert!(moving.iter().all(|model| model.cube_count() == 8));
        assert!(few.iter().all(|model| model.cube_count() == 8));
        assert!(Scenario::Moving.static_cubes().is_empty());
        assert!(Scenario::FewChanges.static_cubes().is_empty());
    }

    #[test]
    fn what_moves_each_frame_depends_on_the_scenario() {
        let count = Scenario::Moving.models().len();
        assert_eq!(Scenario::Terrain.moved_models(7), 0..0);
        assert_eq!(Scenario::Scattered.moved_models(7), 0..0);
        assert_eq!(Scenario::Moving.moved_models(7), 0..count);
        assert_eq!(Scenario::FewChanges.moved_models(7), 7..8);
        let wrapped = u32::try_from(count).unwrap() + 2;
        assert_eq!(Scenario::FewChanges.moved_models(wrapped), 2..3);
    }

    #[test]
    fn a_model_moves_from_one_frame_to_the_next() {
        for index in [0, 1, 2_500] {
            assert_eq!(model_position(index, 40), model_position(index, 40));
            assert_ne!(model_position(index, 40), model_position(index, 41));
        }
        let start: Vec<Vec3> = Scenario::Moving
            .models()
            .iter()
            .map(|model| model.position)
            .collect();
        assert_eq!(start[3], model_position(3, 0));
    }

    #[test]
    fn the_camera_path_is_pure_and_holds_at_both_ends() {
        assert_eq!(camera_at(300), camera_at(300));
        assert_eq!(camera_at(0), camera_at(WARMUP_FRAMES - 1));
        let end = WARMUP_FRAMES + MEASURED_FRAMES;
        assert_eq!(camera_at(end), camera_at(end + GRACE_FRAMES));
        assert_ne!(camera_at(WARMUP_FRAMES), camera_at(end));
        // Both ends of the path are measured.
        assert_eq!(camera_at(WARMUP_FRAMES), (KEYFRAMES[0].0, KEYFRAMES[0].1));
        let last = KEYFRAMES[KEYFRAMES.len() - 1];
        assert_eq!(camera_at(end - 1), (last.0, last.1));
    }
}
