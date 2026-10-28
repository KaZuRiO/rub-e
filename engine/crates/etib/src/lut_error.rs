//! Why a lookup table was refused.

use std::fmt;

use crate::lut::Lut;

/// A table that could not be built, or a `.cube` file that could not be read
/// as one.
///
/// Every refusal says which case it is, so a caller can tell a malformed file
/// from one that is well-formed but asks for something [`Lut`] does not do.
/// Lines are counted from 1, as an editor counts them.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum LutError {
    /// A side length outside [`Lut::MIN_SIZE`]..=[`Lut::MAX_SIZE`].
    Size {
        /// The side length asked for.
        size: u32,
    },
    /// RGB data whose length is not three floats per entry of a `size`-a-side
    /// table.
    Length {
        /// The side length the data was meant for.
        size: u32,
        /// How many floats there were.
        actual: usize,
    },
    /// `LUT_3D_SIZE` followed by something that is not a whole number.
    SizeNotAWholeNumber {
        /// The line it is on.
        line: usize,
    },
    /// A second `LUT_3D_SIZE`.
    SizeTwice {
        /// The line of the second one.
        line: usize,
    },
    /// A `LUT_1D_SIZE`: a tone curve rather than a look.
    OneDimensional {
        /// The line it is on.
        line: usize,
    },
    /// A `DOMAIN_MIN` or `DOMAIN_MAX` other than the default 0..1.
    Domain {
        /// The line it is on.
        line: usize,
        /// `"DOMAIN_MIN"` or `"DOMAIN_MAX"`.
        keyword: &'static str,
        /// What the file gave instead.
        domain: [f32; 3],
    },
    /// An entry before the size that says how many to expect.
    EntryBeforeSize {
        /// The line of the entry.
        line: usize,
    },
    /// A field that should be a number and is not.
    NotANumber {
        /// The line it is on.
        line: usize,
        /// The field as written.
        field: String,
    },
    /// An entry or a domain with other than three values.
    WrongCount {
        /// The line it is on.
        line: usize,
        /// `"entry"`, `"DOMAIN_MIN"` or `"DOMAIN_MAX"`.
        what: &'static str,
        /// How many values it had.
        count: usize,
    },
    /// An entry or a domain holding an infinity or a NaN.
    NotFinite {
        /// The line it is on.
        line: usize,
        /// `"entry"`, `"DOMAIN_MIN"` or `"DOMAIN_MAX"`.
        what: &'static str,
    },
    /// A file with no `LUT_3D_SIZE` at all.
    MissingSize,
    /// A file with more or fewer entries than its declared size calls for.
    EntryCount {
        /// The side length the file declares.
        size: u32,
        /// How many entries it holds.
        actual: usize,
    },
}

impl fmt::Display for LutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let entries = |size: u32| (size as usize).pow(3);
        match self {
            Self::Size { size } => write!(
                f,
                "a lookup table is between {} and {} entries a side, not {size}",
                Lut::MIN_SIZE,
                Lut::MAX_SIZE
            ),
            Self::Length { size, actual } => write!(
                f,
                "a {size}-a-side table has {} entries, so it needs {} floats, not {actual}",
                entries(*size),
                entries(*size) * 3
            ),
            Self::SizeNotAWholeNumber { line } => {
                write!(f, "line {line}: LUT_3D_SIZE is not a whole number")
            }
            Self::SizeTwice { line } => {
                write!(f, "line {line}: the file declares LUT_3D_SIZE twice")
            }
            Self::OneDimensional { line } => write!(
                f,
                "line {line}: this is a 1D table; only 3D ones are supported"
            ),
            Self::Domain {
                line,
                keyword,
                domain,
            } => write!(
                f,
                "line {line}: {keyword} is {domain:?}; only the default 0..1 domain is supported"
            ),
            Self::EntryBeforeSize { line } => {
                write!(f, "line {line}: an entry appears before LUT_3D_SIZE")
            }
            Self::NotANumber { line, field } => {
                write!(f, "line {line}: '{field}' is not a number")
            }
            Self::WrongCount { line, what, count } => {
                write!(f, "line {line}: {what} has {count} values, not 3")
            }
            Self::NotFinite { line, what } => {
                write!(f, "line {line}: {what} holds a value that is not finite")
            }
            Self::MissingSize => f.write_str("the file declares no LUT_3D_SIZE"),
            Self::EntryCount { size, actual } => write!(
                f,
                "the file declares {size} a side, so it should hold {} entries, but it has {actual}",
                entries(*size)
            ),
        }
    }
}

impl std::error::Error for LutError {}
