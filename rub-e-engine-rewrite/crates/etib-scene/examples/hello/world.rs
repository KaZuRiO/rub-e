//! The scene the example builds when it is given no model files: the terrain
//! of `engine/etib/examples/generate_scene.py` and the cat of
//! `generate_cat.py`, generated in place so the example runs out of the box.

use etib_scene::ModelCube;
use etib_scene::glam::Vec3;

/// A `side` x `side` floor of cubes with rolling relief, coloured green with
/// a little variation by height and position.
pub fn floor(side: i32) -> Vec<ModelCube> {
    // The frequencies are scaled so a larger floor keeps the same features.
    let scale = 100.0 / side as f32;
    let mut cubes = Vec::with_capacity((side * side) as usize);
    for x in -side / 2..side / 2 {
        for z in -side / 2..side / 2 {
            let (fx, fz) = (x as f32 * scale, z as f32 * scale);
            let relief = (fx * 0.2).sin() * 0.7
                + (fz * 0.2).cos() * 0.7
                + (fx * 0.05 + fz * 0.05).sin() * 1.5
                + (fx * 0.15).sin() * (fz * 0.15).cos() * 0.5;
            let y = (relief * 1.5) as i32 as f32;

            let height = y / 5.0;
            let jitter = noise(x, z) * 0.1 - 0.05;
            cubes.push(ModelCube {
                position: Vec3::new(x as f32, y, z as f32),
                color: Vec3::new(
                    0.25 + jitter + height * 0.1,
                    0.55 - jitter + height * 0.15,
                    0.25 + jitter,
                )
                .clamp(Vec3::ZERO, Vec3::ONE),
            });
        }
    }
    cubes
}

/// A sitting cat, about 20 cubes tall, with its origin under its body.
pub fn cat() -> Vec<ModelCube> {
    let orange = Vec3::new(1.0, 0.5, 0.1);
    let dark_orange = Vec3::new(0.8, 0.4, 0.0);
    let light_orange = Vec3::new(1.0, 0.7, 0.3);
    let green = Vec3::new(0.1, 0.8, 0.1);
    let black = Vec3::splat(0.1);
    let white = Vec3::splat(0.95);
    let pink = Vec3::new(1.0, 0.6, 0.7);

    let mut cubes = Vec::new();
    let mut ellipsoid = |centre: Vec3, radii: Vec3, color: Vec3| {
        let (min, max) = ((centre - radii).floor(), (centre + radii).ceil());
        for x in min.x as i32..=max.x as i32 {
            for y in min.y as i32..=max.y as i32 {
                for z in min.z as i32..=max.z as i32 {
                    let p = Vec3::new(x as f32, y as f32, z as f32);
                    if ((p - centre) / radii).length_squared() <= 1.0 {
                        cubes.push(ModelCube { position: p, color });
                    }
                }
            }
        }
    };

    ellipsoid(Vec3::new(0.0, 5.0, 0.0), Vec3::new(6.0, 7.0, 8.0), orange); // body
    ellipsoid(
        Vec3::new(0.0, 4.0, 3.0),
        Vec3::new(4.0, 4.0, 5.0),
        light_orange,
    ); // chest
    for x in [-3.0, 3.0] {
        ellipsoid(Vec3::new(x, 1.0, 6.0), Vec3::splat(2.0), dark_orange); // paws
    }
    ellipsoid(Vec3::new(0.0, 14.0, 2.0), Vec3::splat(5.0), orange); // head
    for i in 0..4 {
        let width = 2.0 - i as f32 * 0.5;
        for x in [-4.0, 4.0] {
            ellipsoid(
                Vec3::new(x, 18.0 + i as f32, 2.0),
                Vec3::new(width.max(0.5), 1.0, 1.0),
                dark_orange,
            ); // ears
        }
    }
    ellipsoid(Vec3::new(0.0, 13.0, 6.0), Vec3::splat(2.5), light_orange); // muzzle
    ellipsoid(Vec3::new(0.0, 14.0, 8.0), Vec3::splat(0.5), pink); // nose
    for x in [-2.5, 2.5] {
        ellipsoid(Vec3::new(x, 15.0, 6.0), Vec3::splat(1.2), green); // eyes
    }
    for i in 0..20 {
        let angle = i as f32 * std::f32::consts::PI / 16.0;
        let centre = Vec3::new(
            angle.cos() * 7.0,
            1.5 + (i as f32 / 5.0).sin(),
            angle.sin() * 7.0 - 7.0,
        );
        let color = if i % 4 < 2 { dark_orange } else { orange };
        ellipsoid(centre, Vec3::splat(1.5 - i as f32 * 0.05), color); // tail
    }

    for x in [-2.5, 2.5] {
        cubes.push(ModelCube {
            position: Vec3::new(x, 15.0, 7.0),
            color: black,
        }); // pupils
    }
    for i in 3..8 {
        for x in [-(i as f32), i as f32] {
            cubes.push(ModelCube {
                position: Vec3::new(x, 13.0, 7.0),
                color: white,
            }); // whiskers
            cubes.push(ModelCube {
                position: Vec3::new(x, 12.5, 7.5),
                color: white,
            });
        }
    }
    cubes
}

/// A repeatable value in 0..1 for a grid position, standing in for the
/// script's `random`.
fn noise(x: i32, z: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x9e37_79b9) ^ (z as u32).wrapping_mul(0x85eb_ca6b);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    (h & 0xffff) as f32 / 65535.0
}
