//! Everything pong is built of, in cubes: the Orange Vélodrome, the paddles,
//! the ball, and the digits and letters of the scores and the title.
//!
//! Z is up: the pitch lies in the XY plane.

use std::collections::{BTreeMap, HashSet};

use etib_scene::{DynamicModel, ModelCube};
use glam::Vec3;

use crate::state::{FIELD_HALF_H, FIELD_HALF_W};

pub const LEFT_COLOR: Vec3 = Vec3::new(0.2, 0.75, 1.0);
pub const RIGHT_COLOR: Vec3 = Vec3::new(1.0, 0.45, 0.1);
/// Where the scores lie, on the ground below the near touchline.
pub const SCORE_POSITION: Vec3 = Vec3::new(0.0, -FIELD_HALF_H - 5.0, 0.0);
/// Where models wait while they are not shown: far above everything.
pub const HIDDEN: Vec3 = Vec3::new(0.0, 0.0, 1000.0);

pub fn unit_cube(x: f32, y: f32, z: f32, r: f32, g: f32, b: f32) -> ModelCube {
    ModelCube {
        position: Vec3::new(x, y, z),
        color: Vec3::new(r, g, b),
    }
}

/// 9 cubes tall, 2 deep and 2 wide.
pub fn make_paddle(color: Vec3) -> DynamicModel {
    let mut cubes = Vec::new();
    for y in -4..=4 {
        for x in 0..=1 {
            for z in 0..=1 {
                cubes.push(unit_cube(
                    x as f32, y as f32, z as f32, color.x, color.y, color.z,
                ));
            }
        }
    }
    DynamicModel::from_cubes(cubes)
}

/// A 2×2×2 block centred on its origin.
pub fn make_ball() -> DynamicModel {
    let mut cubes = Vec::new();
    for x in 0..2 {
        for y in 0..2 {
            for z in 0..2 {
                cubes.push(unit_cube(
                    x as f32 - 0.5,
                    y as f32 - 0.5,
                    z as f32 - 0.5,
                    1.0,
                    0.95,
                    0.2,
                ));
            }
        }
    }
    DynamicModel::from_cubes(cubes)
}

/// The ball's shadow on the grass while it is in the air.
pub fn make_shadow() -> DynamicModel {
    let mut cubes = Vec::new();
    for x in 0..2 {
        for y in 0..2 {
            cubes.push(unit_cube(
                x as f32 - 0.5,
                y as f32 - 0.5,
                0.0,
                0.05,
                0.05,
                0.05,
            ));
        }
    }
    let mut model = DynamicModel::from_cubes(cubes);
    model.position = HIDDEN;
    model
}

/// Static geometry: Marseille Orange Velodrome Stadium. Also returns where the
/// seats are, for the spectators.
pub fn make_static_geometry() -> (Vec<ModelCube>, Vec<Vec3>) {
    let mut seats = Vec::new();
    // Ordered, so the stadium comes out the same every run; a later cube in
    // the same place replaces an earlier one.
    let mut grid: BTreeMap<(i32, i32, i32), ModelCube> = BTreeMap::new();

    let mut add_cube = |x: i32, y: i32, z: f32, (r, g, b): (f32, f32, f32)| {
        let key = (x, y, (z * 10.0).round() as i32);
        grid.insert(key, unit_cube(x as f32, y as f32, z, r, g, b));
    };

    let (half_w, half_h) = (FIELD_HALF_W as i32, FIELD_HALF_H as i32);
    let dark = (0.2, 0.2, 0.2);
    let white = (1.0, 1.0, 1.0);
    let dark_green = (0.1, 0.4, 0.1);
    let light_green = (0.15, 0.45, 0.15);

    for x in -half_w..=half_w {
        add_cube(x, half_h, 0.0, dark); // Top wall
        add_cube(x, -half_h, 0.0, dark); // Bottom wall
    }

    // --- Grass Pitch Floor & Markings ---
    let floor_z = -2.0;

    let mut circle_points = HashSet::new();
    for angle in 0..72 {
        let rad = angle as f32 * std::f32::consts::PI / 36.0;
        circle_points.insert((
            (rad.cos() * 6.0).round() as i32,
            (rad.sin() * 6.0).round() as i32,
        ));
    }

    let grass_start_x = -half_w - 40;
    let grass_end_x = half_w + 40;
    let grass_start_y = -half_h - 60; // Extend towards the camera
    let grass_end_y = half_h + 40;

    for x in grass_start_x..=grass_end_x {
        for y in grass_start_y..=grass_end_y {
            let on_pitch = (-half_w..=half_w).contains(&x) && (-half_h..=half_h).contains(&y);
            let color = if on_pitch {
                let is_border = x.abs() == half_w || y.abs() == half_h;
                let is_center_line = x == 0 || x == -1; // 2 wide
                if is_border || is_center_line || circle_points.contains(&(x, y)) {
                    white
                } else if (x + half_w) % 12 >= 6 {
                    dark_green
                } else {
                    light_green
                }
            } else if (x + grass_end_x) % 12 >= 6 {
                dark_green
            } else {
                light_green
            };
            add_cube(x, y, floor_z, color);
        }
    }

    // --- Goals Running the Entire Length of the Stadium ---
    let goal_height = 6;
    for y in -half_h..=half_h {
        // Crossbars
        add_cube(-half_w - 1, y, goal_height as f32, white);
        add_cube(half_w + 1, y, goal_height as f32, white);
        // Nets (Checkerboard pattern)
        for z in 0..goal_height {
            if (y + z) % 2 == 0 {
                add_cube(-half_w - 2, y, z as f32, (0.5, 0.5, 0.5));
                add_cube(half_w + 2, y, z as f32, (0.5, 0.5, 0.5));
            }
        }
    }
    // Side posts
    for z in 0..goal_height {
        for x in [-half_w - 1, half_w + 1] {
            add_cube(x, half_h, z as f32, white);
            add_cube(x, -half_h, z as f32, white);
        }
    }

    // --- Orange Velodrome Stadium Bleachers ---
    let sky_blue = (0.1, 0.6, 0.9);
    let off_white = (0.85, 0.85, 0.85);
    let text_color = sky_blue;
    let concrete = (0.6, 0.6, 0.6);

    let marseille = [
        "111 010 110 111 111 1 100 100 111",
        "101 101 101 100 100 1 100 100 100",
        "101 111 110 111 111 1 100 100 111",
        "101 101 101 001 100 1 100 100 100",
        "101 101 101 111 111 1 111 111 111",
    ];
    let om = ["111 111", "101 101", "101 101", "101 101", "111 101"];
    let year_1899 = [
        "010 111 111 111",
        "110 101 101 101",
        "010 111 111 111",
        "010 101 001 001",
        "111 111 111 111",
    ];
    // Whether a banner's pixel is lit at `index` along its row.
    let lit = |banner: &[&str; 5], row: i32, index: i32| {
        let line = banner[row as usize].as_bytes();
        (0..line.len() as i32).contains(&index) && line[index as usize] == b'1'
    };

    let rows_per_tier = [16, 24, 32];
    let mut current_row = 1;
    let mut current_z = 0.0;

    for (tier, &rows) in rows_per_tier.iter().enumerate() {
        for r in 1..=rows {
            let out_y_top = half_h + current_row;
            let out_x_left = -half_w - current_row;
            let out_x_right = half_w + current_row;

            current_z += 0.15 + (tier as f32 * 0.05);
            let height_z = current_z;

            let ext_x = half_w + current_row;
            let ext_y = half_h + current_row;
            let banner_row = (tier == 1 && (8..=16).contains(&r)).then(|| (16 - r) / 2);
            let has_seat = tier < 2 || r < rows - 15;

            // Top Sideline (Main Stand)
            for x in -ext_x..=ext_x {
                let mut color = off_white;
                if let Some(row) = banner_row {
                    let start_x = -32;
                    if x >= start_x && x < start_x + 66 && lit(&marseille, row, (x - start_x) / 2) {
                        color = text_color;
                    }
                }
                if (x + ext_x) % 24 == 0 {
                    color = concrete; // Double concrete stairs
                }
                add_cube(x, out_y_top, height_z, color);
                if x % 2 == 0 && has_seat {
                    seats.push(Vec3::new(x as f32, out_y_top as f32, height_z + 1.0));
                }
            }

            // Left & Right Goal lines
            let side_start_y = -half_h - 20; // Start further back
            for y in side_start_y..ext_y {
                // LEFT STAND
                let mut color_l = off_white;
                if let Some(row) = banner_row {
                    let start_y = -6;
                    if y >= start_y && y < start_y + 14 && lit(&om, row, (y - start_y) / 2) {
                        color_l = text_color;
                    }
                }
                // RIGHT STAND
                let mut color_r = off_white;
                if let Some(row) = banner_row {
                    let start_y = 14;
                    if y <= start_y && y > start_y - 30 && lit(&year_1899, row, (start_y - y) / 2) {
                        color_r = text_color;
                    }
                }
                if (y - side_start_y) % 24 == 0 {
                    color_l = concrete;
                    color_r = concrete;
                }
                add_cube(out_x_left, y, height_z, color_l);
                add_cube(out_x_right, y, height_z, color_r);

                if y % 2 == 0 && has_seat {
                    // Avoid the net area.
                    let is_net_y = (-half_h..=half_h).contains(&y);
                    for x in [out_x_left, out_x_right] {
                        let is_net_x = x.abs() <= half_w + 3;
                        if !(is_net_y && is_net_x) {
                            seats.push(Vec3::new(x as f32, y as f32, height_z + 1.0));
                        }
                    }
                }
            }

            // Iconic Velodrome undulating roof smoothly connecting to the stands
            if tier == 2 && r == rows {
                let overhang_max = 40;
                for overhang in 1..=overhang_max {
                    let roof_y = out_y_top - overhang;
                    let wave_amplitude = 6.0 * (overhang as f32 / overhang_max as f32);
                    for x in -ext_x..=ext_x {
                        let wave = (x as f32 * 0.075).sin() * wave_amplitude;
                        let roof_z = height_z + (overhang as f32 * 0.2) + wave;
                        add_cube(x, roof_y, roof_z, white);
                    }
                }

                for overhang in 1..=overhang_max {
                    let wave_amplitude = 6.0 * (overhang as f32 / overhang_max as f32);
                    let roof_limit_y = out_y_top - overhang_max;
                    for y in side_start_y..roof_limit_y {
                        let wave = (y as f32 * 0.075).cos() * wave_amplitude;
                        let roof_z = height_z + (overhang as f32 * 0.2) + wave;
                        add_cube(out_x_left + overhang, y, roof_z, white);
                        add_cube(out_x_right - overhang, y, roof_z, white);
                    }
                }
            }

            current_row += 1;
        }
    }

    (grid.into_values().collect(), seats)
}

/// A 3×5 glyph, or nothing for a character the font does not have.
fn glyph(c: char) -> [u8; 15] {
    match c.to_ascii_uppercase() {
        '0' => [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
        '1' => [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        '2' => [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        '3' => [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
        '4' => [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
        '5' => [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        '6' => [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        '7' => [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        '8' => [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        '9' => [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        'P' => [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
        'O' => [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
        'N' => [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
        'G' => [1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
        'D' => [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
        _ => [0; 15],
    }
}

/// One character's cubes, its left edge at `offset_x`, standing in X and Y.
fn make_char_cubes(c: char, offset_x: f32, color: Vec3) -> Vec<ModelCube> {
    let map = glyph(c);
    let mut cubes = Vec::new();
    for y in 0..5 {
        for x in 0..3 {
            if map[(4 - y) * 3 + x] == 1 {
                cubes.push(unit_cube(
                    offset_x + x as f32,
                    y as f32 - 2.5,
                    0.0,
                    color.x,
                    color.y,
                    color.z,
                ));
            }
        }
    }
    cubes
}

/// A score, centred on `base_x`, lying on the ground below the pitch.
pub fn make_score_model(score: u32, base_x: f32, color: Vec3) -> DynamicModel {
    const CHAR_WIDTH: f32 = 3.0;
    const SPACING: f32 = 1.0;
    let digits = score.to_string();
    let n = digits.len() as f32;
    let total_width = n * CHAR_WIDTH + (n - 1.0) * SPACING;

    let mut cubes = Vec::new();
    let mut offset_x = base_x - total_width / 2.0;
    for c in digits.chars() {
        cubes.extend(make_char_cubes(c, offset_x, color));
        offset_x += CHAR_WIDTH + SPACING;
    }
    let mut model = DynamicModel::from_cubes(cubes);
    model.position = SCORE_POSITION;
    model
}

/// A line of text starting at `base_x`, four units per character.
pub fn make_text_model(text: &str, base_x: f32, color: Vec3) -> DynamicModel {
    let mut cubes = Vec::new();
    let mut offset_x = base_x;
    for c in text.chars() {
        cubes.extend(make_char_cubes(c, offset_x, color));
        offset_x += 4.0;
    }
    DynamicModel::from_cubes(cubes)
}

/// The engine's `make_static_geometry`, copied as it was (cgmath → glam only),
/// to hold the restructured one above to the same stadium.
#[cfg(test)]
#[allow(clippy::all)]
mod reference {
    use etib_scene::ModelCube;
    use glam::Vec3;

    use super::unit_cube;
    use crate::state::{FIELD_HALF_H, FIELD_HALF_W};

    /// Static geometry: Marseille Orange Velodrome Stadium
    pub fn ref_make_static_geometry() -> (Vec<ModelCube>, Vec<Vec3>) {
        let mut seats = Vec::new();
        let mut grid: std::collections::HashMap<(i32, i32, i32), ModelCube> =
            std::collections::HashMap::new();

        let mut add_cube = |x: i32, y: i32, z: f32, r: f32, g: f32, b: f32| {
            let key = (x, y, (z * 10.0).round() as i32);
            grid.insert(key, unit_cube(x as f32, y as f32, z, r, g, b));
        };

        let xi = -(FIELD_HALF_W as i32)..=(FIELD_HALF_W as i32);
        for x in xi {
            // Top wall
            add_cube(x, FIELD_HALF_H as i32, 0.0, 0.2, 0.2, 0.2);
            // Bottom wall
            add_cube(x, -(FIELD_HALF_H as i32), 0.0, 0.2, 0.2, 0.2);
        }

        // --- Grass Pitch Floor & Markings ---
        let floor_z = -2.0; // Scaled down to match 2x

        let mut circle_points = std::collections::HashSet::new();
        for angle in 0..72 {
            // Double points
            let rad = angle as f32 * std::f32::consts::PI / 36.0;
            let cx = (rad.cos() * 6.0).round() as i32; // Double radius
            let cy = (rad.sin() * 6.0).round() as i32;
            circle_points.insert((cx, cy));
        }

        let grass_start_x = -(FIELD_HALF_W as i32) - 40;
        let grass_end_x = (FIELD_HALF_W as i32) + 40;
        let grass_start_y = -(FIELD_HALF_H as i32) - 60; // Extend towards the camera
        let grass_end_y = (FIELD_HALF_H as i32) + 40;

        for x in grass_start_x..=grass_end_x {
            for y in grass_start_y..=grass_end_y {
                let is_pitch_x = x >= -(FIELD_HALF_W as i32) && x <= (FIELD_HALF_W as i32);
                let is_pitch_y = y >= -(FIELD_HALF_H as i32) && y <= (FIELD_HALF_H as i32);

                if is_pitch_x && is_pitch_y {
                    let is_border_x = x == -(FIELD_HALF_W as i32) || x == FIELD_HALF_W as i32;
                    let is_border_y = y == -(FIELD_HALF_H as i32) || y == FIELD_HALF_H as i32;
                    let is_center_line = x == 0 || x == -1; // 2 wide
                    let is_circle = circle_points.contains(&(x, y));

                    let (r, g, b) = if is_border_x || is_border_y || is_center_line || is_circle {
                        (1.0, 1.0, 1.0) // White markings
                    } else if (x + FIELD_HALF_W as i32) % 12 >= 6 {
                        (0.1, 0.4, 0.1) // Dark green
                    } else {
                        (0.15, 0.45, 0.15) // Light green
                    };
                    add_cube(x, y, floor_z, r, g, b);
                } else {
                    // Outside pitch, just regular grass stripes
                    if (x + grass_end_x) % 12 >= 6 {
                        add_cube(x, y, floor_z, 0.1, 0.4, 0.1);
                    } else {
                        add_cube(x, y, floor_z, 0.15, 0.45, 0.15);
                    }
                }
            }
        }

        // --- Goals Running the Entire Length of the Stadium ---
        let goal_height = 6; // Double height
        for y in -(FIELD_HALF_H as i32)..=(FIELD_HALF_H as i32) {
            // Crossbars
            add_cube(
                -(FIELD_HALF_W as i32) - 1,
                y,
                goal_height as f32,
                1.0,
                1.0,
                1.0,
            );
            add_cube(
                FIELD_HALF_W as i32 + 1,
                y,
                goal_height as f32,
                1.0,
                1.0,
                1.0,
            );

            // Nets (Checkerboard pattern)
            for z in 0..goal_height {
                if (y + z) % 2 == 0 {
                    add_cube(-(FIELD_HALF_W as i32) - 2, y, z as f32, 0.5, 0.5, 0.5);
                    add_cube(FIELD_HALF_W as i32 + 2, y, z as f32, 0.5, 0.5, 0.5);
                }
            }
        }
        // Side posts
        for z in 0..goal_height {
            add_cube(
                -(FIELD_HALF_W as i32) - 1,
                FIELD_HALF_H as i32,
                z as f32,
                1.0,
                1.0,
                1.0,
            );
            add_cube(
                -(FIELD_HALF_W as i32) - 1,
                -(FIELD_HALF_H as i32),
                z as f32,
                1.0,
                1.0,
                1.0,
            );

            add_cube(
                FIELD_HALF_W as i32 + 1,
                FIELD_HALF_H as i32,
                z as f32,
                1.0,
                1.0,
                1.0,
            );
            add_cube(
                FIELD_HALF_W as i32 + 1,
                -(FIELD_HALF_H as i32),
                z as f32,
                1.0,
                1.0,
                1.0,
            );
        }

        // --- Orange Velodrome Stadium Bleachers ---
        let sky_blue = (0.1, 0.6, 0.9);
        let off_white = (0.85, 0.85, 0.85);
        let text_color = sky_blue;
        let concrete = (0.6, 0.6, 0.6);
        let pure_white = (1.0, 1.0, 1.0);

        let marseille = [
            "111 010 110 111 111 1 100 100 111",
            "101 101 101 100 100 1 100 100 100",
            "101 111 110 111 111 1 100 100 111",
            "101 101 101 001 100 1 100 100 100",
            "101 101 101 111 111 1 111 111 111",
        ];

        let om = ["111 111", "101 101", "101 101", "101 101", "111 101"];

        let year_1899 = [
            "010 111 111 111",
            "110 101 101 101",
            "010 111 111 111",
            "010 101 001 001",
            "111 111 111 111",
        ];

        let max_tiers = 3;
        let rows_per_tier = [16, 24, 32]; // Double rows
        let tier_gaps = [0, 0, 0];

        let mut current_row = 1;
        let mut current_z = 0.0;

        for tier in 0..max_tiers {
            let rows = rows_per_tier[tier];
            let h_offset = tier_gaps[tier];
            current_row += h_offset;

            for r in 1..=rows {
                let out_y_top = FIELD_HALF_H as i32 + current_row;
                let out_x_left = -(FIELD_HALF_W as i32) - current_row;
                let out_x_right = FIELD_HALF_W as i32 + current_row;

                // Keep the slope consistent with doubled rows
                let z_increment = 0.15 + (tier as f32 * 0.05);
                current_z += z_increment;
                let height_z = current_z;

                let ext_x = FIELD_HALF_W as i32 + current_row;
                let ext_y = FIELD_HALF_H as i32 + current_row;

                // Top Sideline (Main Stand)
                for x in -ext_x..=ext_x {
                    let mut color = off_white;

                    if tier == 1 && r >= 8 && r <= 16 {
                        let text_row = (16 - r) / 2;
                        let start_x = -32;
                        if x >= start_x && x < start_x + 66 {
                            let char_idx = ((x - start_x) / 2) as usize;
                            let line = marseille[text_row as usize].as_bytes();
                            if char_idx < line.len() && line[char_idx] == b'1' {
                                color = text_color;
                            }
                        }
                    }

                    if (x + ext_x) % 24 == 0 {
                        // Double concrete stairs
                        color = concrete;
                    }

                    add_cube(x, out_y_top, height_z, color.0, color.1, color.2);
                    if x % 2 == 0 && (tier < 2 || r < rows - 15) {
                        seats.push(Vec3::new(x as f32, out_y_top as f32, height_z + 1.0));
                    }
                }

                // Left & Right Goal lines
                let side_start_y = -(FIELD_HALF_H as i32) - 20; // Start further back
                for y in side_start_y..ext_y {
                    // LEFT STAND
                    let mut color_l = off_white;
                    if tier == 1 && r >= 8 && r <= 16 {
                        let text_row = (16 - r) / 2;
                        let start_y = -6;
                        if y >= start_y && y < start_y + 14 {
                            let char_idx = ((y - start_y) / 2) as usize;
                            let line = om[text_row as usize].as_bytes();
                            if char_idx < line.len() && line[char_idx] == b'1' {
                                color_l = text_color;
                            }
                        }
                    }
                    if (y - side_start_y) % 24 == 0 {
                        color_l = concrete;
                    }
                    add_cube(out_x_left, y, height_z, color_l.0, color_l.1, color_l.2);
                    if y % 2 == 0 && (tier < 2 || r < rows - 15) {
                        // Avoid net area
                        let is_net_y = y >= -(FIELD_HALF_H as i32) && y <= (FIELD_HALF_H as i32);
                        let is_net_x = out_x_left.abs() <= (FIELD_HALF_W as i32 + 3);
                        if !(is_net_y && is_net_x) {
                            seats.push(Vec3::new(out_x_left as f32, y as f32, height_z + 1.0));
                        }
                    }

                    // RIGHT STAND
                    let mut color_r = off_white;
                    if tier == 1 && r >= 8 && r <= 16 {
                        let text_row = (16 - r) / 2;
                        let start_y = 14;
                        if y <= start_y && y > start_y - 30 {
                            let char_idx = ((start_y - y) / 2) as usize;
                            let line = year_1899[text_row as usize].as_bytes();
                            if char_idx < line.len() && line[char_idx] == b'1' {
                                color_r = text_color;
                            }
                        }
                    }
                    if (y - side_start_y) % 24 == 0 {
                        color_r = concrete;
                    }
                    add_cube(out_x_right, y, height_z, color_r.0, color_r.1, color_r.2);
                    if y % 2 == 0 && (tier < 2 || r < rows - 15) {
                        // Avoid net area
                        let is_net_y = y >= -(FIELD_HALF_H as i32) && y <= (FIELD_HALF_H as i32);
                        let is_net_x = out_x_right.abs() <= (FIELD_HALF_W as i32 + 3);
                        if !(is_net_y && is_net_x) {
                            seats.push(Vec3::new(out_x_right as f32, y as f32, height_z + 1.0));
                        }
                    }
                }

                // Iconic Velodrome undulating roof smoothly connecting to the stands
                if tier == 2 && r == rows {
                    let overhang_max = 40;
                    for overhang in 1..=overhang_max {
                        let roof_y = out_y_top - overhang;
                        let wave_amplitude = 6.0 * (overhang as f32 / overhang_max as f32);
                        for x in -ext_x..=ext_x {
                            let wave = (x as f32 * 0.075).sin() * wave_amplitude; // Halve frequency, double amplitude
                            let roof_z = height_z + (overhang as f32 * 0.2) + wave;
                            add_cube(x, roof_y, roof_z, pure_white.0, pure_white.1, pure_white.2);
                        }
                    }

                    for overhang in 1..=overhang_max {
                        let wave_amplitude = 6.0 * (overhang as f32 / overhang_max as f32);
                        let roof_limit_y = out_y_top - overhang_max;
                        for y in side_start_y..roof_limit_y {
                            let roof_x_l = out_x_left + overhang;
                            let wave_l = (y as f32 * 0.075).cos() * wave_amplitude;
                            let roof_z_l = height_z + (overhang as f32 * 0.2) + wave_l;
                            add_cube(
                                roof_x_l,
                                y,
                                roof_z_l,
                                pure_white.0,
                                pure_white.1,
                                pure_white.2,
                            );

                            let roof_x_r = out_x_right - overhang;
                            let wave_r = (y as f32 * 0.075).cos() * wave_amplitude;
                            let roof_z_r = height_z + (overhang as f32 * 0.2) + wave_r;
                            add_cube(
                                roof_x_r,
                                y,
                                roof_z_r,
                                pure_white.0,
                                pure_white.1,
                                pure_white.2,
                            );
                        }
                    }
                }

                current_row += 1;
            }
        }

        (grid.into_values().collect(), seats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stadium_is_the_engines() {
        // The engine's came out of a HashMap, in no particular order.
        let key = |c: &ModelCube| {
            (
                c.position.x as i32,
                c.position.y as i32,
                (c.position.z * 10.0).round() as i32,
            )
        };
        let (mut ours, our_seats) = make_static_geometry();
        let (mut theirs, their_seats) = reference::ref_make_static_geometry();
        ours.sort_by_key(key);
        theirs.sort_by_key(key);
        assert_eq!(ours.len(), theirs.len());
        for (a, b) in ours.iter().zip(&theirs) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.color, b.color, "at {:?}", a.position);
        }
        assert_eq!(our_seats, their_seats);
    }

    #[test]
    fn the_stadium_is_the_same_every_time() {
        let (a, seats_a) = make_static_geometry();
        let (b, seats_b) = make_static_geometry();
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| x.position == y.position));
        assert_eq!(seats_a, seats_b);
        // Big enough to be the stadium, and seats for the 4000 spectators.
        assert!(a.len() > 30_000, "{}", a.len());
        assert!(seats_a.len() >= 4000, "{}", seats_a.len());
    }

    #[test]
    fn no_two_static_cubes_share_a_place() {
        let (cubes, _) = make_static_geometry();
        let mut places = HashSet::new();
        for cube in &cubes {
            let key = (
                cube.position.x as i32,
                cube.position.y as i32,
                (cube.position.z * 10.0).round() as i32,
            );
            assert!(places.insert(key), "{key:?} twice");
        }
    }

    #[test]
    fn a_score_is_centred_on_its_base() {
        for score in [0, 7, 10, 123] {
            let model = make_score_model(score, 10.0, LEFT_COLOR);
            let xs = model.cubes().iter().map(|c| c.position.x);
            let (min, max) = xs.fold((f32::MAX, f32::MIN), |(a, b), x| (a.min(x), b.max(x)));
            // Within half a cube: a glyph's outer columns may be unlit.
            assert!(
                ((min + max) / 2.0 - 10.0).abs() <= 0.5,
                "{score}: {min}..{max}"
            );
            assert_eq!(model.position, SCORE_POSITION);
        }
    }

    #[test]
    fn the_title_has_every_letter() {
        let title = make_text_model("PONG 3D", -14.0, Vec3::ONE);
        let expected: usize = "PONG 3D"
            .chars()
            .map(|c| glyph(c).iter().filter(|&&p| p == 1).count())
            .sum();
        assert_eq!(title.cube_count(), expected);
        assert!(expected > 0);
    }
}
