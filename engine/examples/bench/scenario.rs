//! The benchmark scenario: a generated world and the camera path through it.
//! Both are pure functions, with no randomness, so every run sees the same
//! frames.

use etib_scene::ModelCube;
use etib_scene::glam::Vec3;

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

/// Terrain side, in columns.
const SIDE: i32 = 256;
/// Towers sit on a grid with this spacing, in columns.
const TOWER_SPACING: i32 = 32;
/// Footprint of a tower, in columns.
const TOWER_WIDTH: i32 = 5;

/// A `SIDE` x `SIDE` terrain filled from y = 0 up to a rolling surface, with a
/// grid of solid towers rising out of it. Cubes are 1 unit wide, centred on
/// integer coordinates; the terrain is centred on the origin.
pub fn world() -> Vec<ModelCube> {
    let mut cubes = Vec::with_capacity(700_000);
    for x in -SIDE / 2..SIDE / 2 {
        for z in -SIDE / 2..SIDE / 2 {
            let (fx, fz) = (x as f32, z as f32);
            let relief = (fx * 0.08).sin() * 0.7
                + (fz * 0.08).cos() * 0.7
                + (fx * 0.02 + fz * 0.02).sin() * 1.5
                + (fx * 0.06).sin() * (fz * 0.06).cos() * 0.5;
            let mut top = 8 + (relief * 2.0).round() as i32;
            if let Some(height) = tower_height(x, z) {
                top = top.max(height);
            }
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
