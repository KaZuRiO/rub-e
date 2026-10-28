//! A single error type for loading what a scene is made of.

use std::error::Error as StdError;
use std::fmt;
use std::num::ParseFloatError;
use std::path::PathBuf;

/// `Result` with this crate's [`Error`] as the default error type.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Something a scene could not load, or could not make room for.
///
/// GPU work is not in here: building a scene cannot fail once the device is
/// open, and a shader that does not compile is a bug, which panics.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A file could not be read or written.
    Io {
        /// The file that was being read or written.
        path: PathBuf,
        /// Why it could not be.
        source: std::io::Error,
    },
    /// A line of a `.model` file holds something that is not a number.
    Parse {
        /// One-based, as an editor counts.
        line: usize,
        /// What was wrong with the number.
        source: ParseFloatError,
    },
    /// A skybox image could not be decoded.
    Image {
        /// The decoder's own error.
        source: Box<dyn StdError + Send + Sync>,
    },
    /// A dynamic model that would take the scene past the number of dynamic
    /// cubes it was built with room for.
    Capacity {
        /// The cubes the models would hold with this one added.
        needed: usize,
        /// The cubes the scene was built with room for.
        capacity: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, .. } => write!(f, "could not access '{}'", path.display()),
            Self::Parse { line, .. } => write!(f, "line {line} of the model holds a bad number"),
            Self::Image { .. } => f.write_str("the skybox image could not be decoded"),
            Self::Capacity { needed, capacity } => write!(
                f,
                "the dynamic models would hold {needed} cubes, but the scene has room for {capacity}"
            ),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Parse { source, .. } => Some(source),
            Self::Image { source } => Some(source.as_ref()),
            Self::Capacity { .. } => None,
        }
    }
}
