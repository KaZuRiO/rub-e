//! A single error type, so callers never have to match on a backend's errors.

use std::error::Error as StdError;
use std::fmt;

/// `Result` with this crate's [`Error`] as the default error type.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Something that stopped a sound being opened, read or played.
///
/// The message says what was being attempted; where there is an underlying
/// failure — the audio backend refusing a device, a file that would not open —
/// it is kept as the [source](StdError::source), so `{:#}` under anyhow (or any
/// other chain-printing formatter) shows the detail.
#[derive(Debug)]
pub struct Error {
    context: String,
    source: Option<Box<dyn StdError + Send + Sync>>,
}

impl Error {
    /// An error that is fully described by its own message — a malformed file,
    /// a sound with no frames in it.
    pub(crate) fn new(context: impl Into<String>) -> Self {
        Self {
            context: context.into(),
            source: None,
        }
    }

    /// An error that wraps one from below: the backend, or the filesystem.
    pub(crate) fn with_source(
        context: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self {
            context: context.into(),
            source: Some(Box::new(source)),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.context)
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source.as_deref().map(|source| source as _)
    }
}
