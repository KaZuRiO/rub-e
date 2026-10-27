//! `.model` files: one cube per line.
//!
//! ```text
//! # A comment
//! 0 0 0 1 0 0     a red cube at the origin: x y z r g b
//! 1 0 0           a white one beside it: x y z
//! ```
//!
//! A line with any other number of fields is skipped rather than refused, as
//! the engine always has; a field that is not a number is an error.

use std::fmt::Write as _;
use std::path::Path;

use glam::Vec3;

use crate::error::{Error, Result};

/// One cube of a model: where it sits and its linear RGB colour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelCube {
    /// The cube's centre, in model units: one per cube side.
    pub position: Vec3,
    /// Linear RGB, 0..1 per channel.
    pub color: Vec3,
}

/// Reads the cubes of a `.model` file.
pub fn load_model(path: impl AsRef<Path>) -> Result<Vec<ModelCube>> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    parse_model(&content)
}

/// Parses the text of a `.model` file.
pub fn parse_model(content: &str) -> Result<Vec<ModelCube>> {
    let mut cubes = Vec::new();
    for (index, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 3 && fields.len() != 6 {
            log::warn!(
                "line {} of the model has {} fields, not 3 or 6; skipped",
                index + 1,
                fields.len()
            );
            continue;
        }
        let numbers = fields
            .iter()
            .map(|field| field.parse::<f32>())
            .collect::<Result<Vec<f32>, _>>()
            .map_err(|source| Error::Parse {
                line: index + 1,
                source,
            })?;

        cubes.push(ModelCube {
            position: Vec3::new(numbers[0], numbers[1], numbers[2]),
            color: match numbers.get(3..6) {
                Some(rgb) => Vec3::new(rgb[0], rgb[1], rgb[2]),
                None => Vec3::ONE,
            },
        });
    }
    Ok(cubes)
}

/// Writes cubes in the `x y z r g b` form [`load_model`] reads back.
pub fn save_model(path: impl AsRef<Path>, cubes: &[ModelCube]) -> Result<()> {
    let mut content = String::with_capacity(cubes.len() * 32);
    for cube in cubes {
        let (p, c) = (cube.position, cube.color);
        // Writing to a String cannot fail.
        let _ = writeln!(content, "{} {} {} {} {} {}", p.x, p.y, p.z, c.x, c.y, c.z);
    }
    let path = path.as_ref();
    std::fs::write(path, content).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_both_forms_and_skips_comments() {
        let cubes = parse_model(
            "# Test model\n0.0 0.0 0.0\n1.0 2.0 3.0 1.0 0.0 0.0\n\n-1.0 -2.0 -3.0 0.0 1.0 0.0\n",
        )
        .unwrap();
        assert_eq!(cubes.len(), 3);
        assert_eq!(cubes[0].position, Vec3::ZERO);
        assert_eq!(cubes[0].color, Vec3::ONE, "white by default");
        assert_eq!(cubes[1].position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(cubes[1].color, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(cubes[2].position, Vec3::new(-1.0, -2.0, -3.0));
        assert_eq!(cubes[2].color, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn empty_and_comment_only_models_are_empty() {
        assert!(parse_model("").unwrap().is_empty());
        assert!(parse_model("# one\n# two\n\n").unwrap().is_empty());
    }

    #[test]
    fn lines_with_the_wrong_field_count_are_skipped() {
        let cubes = parse_model("1.0 2.0 3.0 4.0\n1.0 2.0\n5.0 6.0 7.0\n").unwrap();
        assert_eq!(cubes.len(), 1);
        assert_eq!(cubes[0].position, Vec3::new(5.0, 6.0, 7.0));
    }

    #[test]
    fn a_bad_number_names_its_line() {
        let err = parse_model("0 0 0\n# fine\n1 x 2\n").unwrap_err();
        assert!(matches!(err, Error::Parse { line: 3, .. }), "{err:?}");
    }

    #[test]
    fn a_missing_file_is_an_error() {
        let err = load_model("/nonexistent/path/does_not_exist.model").unwrap_err();
        assert!(matches!(err, Error::Io { .. }), "{err:?}");
    }

    #[test]
    fn saved_models_load_back() {
        let path = std::env::temp_dir().join(format!("etib-scene-{}.model", std::process::id()));
        let cubes = vec![
            ModelCube {
                position: Vec3::new(0.0, 1.0, 2.0),
                color: Vec3::new(1.0, 0.0, 0.0),
            },
            ModelCube {
                position: Vec3::new(-3.0, -4.0, -5.0),
                color: Vec3::new(0.0, 1.0, 0.5),
            },
        ];
        save_model(&path, &cubes).unwrap();
        let loaded = load_model(&path);
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.unwrap(), cubes);
    }
}
