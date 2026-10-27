//! A single error type, so callers never have to match on a backend's errors.

use std::error::Error as StdError;
use std::fmt;

/// `Result` with this crate's [`Error`] as the default error type.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Something the graphics backend refused to do.
///
/// The message says what was being attempted; the backend's own error is kept
/// as the [source](StdError::source), so `{:#}` under anyhow (or any other
/// chain-printing formatter) shows the detail.
#[derive(Debug)]
pub struct Error {
    context: &'static str,
    source: Box<dyn StdError + Send + Sync>,
}

impl Error {
    pub(crate) fn new(
        context: &'static str,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self {
            context,
            source: Box::new(source),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.context)
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(self.source.as_ref())
    }
}
