//! A single error type for loading what a scene is made of.

use std::error::Error as StdError;
use std::fmt;
use std::num::ParseFloatError;
use std::path::PathBuf;

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Something a model file or a skybox image could not be loaded from.
///
/// GPU work is not in here: building a scene cannot fail once the device is
/// open, and a shader that does not compile is a bug, which panics.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A file could not be read or written.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// A line of a `.model` file holds something that is not a number.
    Parse {
        /// One-based, as an editor counts.
        line: usize,
        source: ParseFloatError,
    },
    /// A skybox image could not be decoded.
    Image {
        source: Box<dyn StdError + Send + Sync>,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, .. } => write!(f, "could not access '{}'", path.display()),
            Self::Parse { line, .. } => write!(f, "line {line} of the model holds a bad number"),
            Self::Image { .. } => f.write_str("the skybox image could not be decoded"),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Parse { source, .. } => Some(source),
            Self::Image { source } => Some(source.as_ref()),
        }
    }
}
