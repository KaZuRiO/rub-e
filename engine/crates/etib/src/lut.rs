//! Colour lookup tables: the cube of colours a grade can be baked into.

use anyhow::{Result, bail, ensure};
use log::debug;

/// A 3D colour lookup table — a cube of output colours, indexed by an input
/// colour's three channels.
///
/// It is the general form of a grade: whatever a colourist did in their tool,
/// sampled on a regular grid and interpolated between. [`PostProcess`] applies
/// one after its own tone curve and grade, which is where the look a `.cube`
/// file describes belongs.
///
/// Both the index and the entries are gamma-encoded, the signal a display is
/// fed rather than linear light, because that is what a grading tool shows its
/// user and therefore what a table exported from one describes. The pass
/// encodes before the lookup and decodes after it, so nothing outside this type
/// has to think about it.
///
/// The table is held on the CPU. Hand it to
/// [`PostProcess::set_lut`](crate::PostProcess::set_lut), which uploads it; a
/// `Lut` that has been uploaded can be dropped.
///
/// [`PostProcess`]: crate::PostProcess
#[derive(Clone, Debug, PartialEq)]
pub struct Lut {
    size: u32,
    /// One RGBA texel per entry, red varying fastest and blue slowest, which is
    /// both what a `.cube` file lists and what a 3D texture wants.
    rgba: Vec<f32>,
}

impl Lut {
    /// The smallest useful table: one entry per corner of the cube, which
    /// trilinear interpolation turns back into the identity.
    pub const MIN_SIZE: u32 = 2;
    /// An upper bound on the side length. Tables in the wild are 17, 32 or 64 a
    /// side; this is here so a malformed header cannot ask for gigabytes.
    pub const MAX_SIZE: u32 = 128;

    /// The table that changes nothing, at `size` entries a side, clamped to the
    /// range a table can have.
    ///
    /// Two a side is enough to be exactly neutral — the corners of the cube,
    /// with interpolation filling in the rest — and is what [`PostProcess`]
    /// binds when no table is set.
    ///
    /// [`PostProcess`]: crate::PostProcess
    pub fn identity(size: u32) -> Self {
        let size = size.clamp(Self::MIN_SIZE, Self::MAX_SIZE);
        let step = (size - 1) as f32;

        let mut rgba = Vec::with_capacity((size as usize).pow(3) * 4);
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    rgba.extend_from_slice(&[
                        r as f32 / step,
                        g as f32 / step,
                        b as f32 / step,
                        1.0,
                    ]);
                }
            }
        }
        Self { size, rgba }
    }

    /// Builds a table from `size³` RGB triples, red varying fastest.
    ///
    /// The entries are output colours: the entry at `(r, g, b)` is what the
    /// colour `(r, g, b) / (size - 1)` becomes, both of them gamma-encoded as
    /// described above.
    pub fn from_rgb(size: u32, rgb: &[f32]) -> Result<Self> {
        check_size(size)?;
        let entries = (size as usize).pow(3);
        ensure!(
            rgb.len() == entries * 3,
            "a {size}-a-side table has {entries} entries, so it needs {} floats, not {}",
            entries * 3,
            rgb.len()
        );

        let mut rgba = Vec::with_capacity(entries * 4);
        for entry in rgb.as_chunks::<3>().0 {
            rgba.extend_from_slice(&[entry[0], entry[1], entry[2], 1.0]);
        }
        Ok(Self { size, rgba })
    }

    /// Parses an Adobe `.cube` file, the format every grading tool exports.
    ///
    /// Only 3D tables over the default 0..1 domain are accepted; a 1D table is
    /// a tone curve rather than a look, and a domain other than 0..1 would have
    /// to be undone somewhere, which is better refused than guessed at.
    pub fn from_cube(text: &str) -> Result<Self> {
        let mut size = None;
        let mut rgb: Vec<f32> = Vec::new();
        let mut title = None;

        for (number, line) in text.lines().enumerate() {
            // Everything from a '#' is a comment, and blank lines are skipped.
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            // Reported to a human reading the file in an editor, so 1-based.
            let line_number = number + 1;

            let mut fields = line.split_whitespace();
            let keyword = fields.next().expect("a non-empty line has a first field");
            match keyword {
                "TITLE" => {
                    title = Some(line["TITLE".len()..].trim().trim_matches('"').to_owned());
                }
                "LUT_3D_SIZE" => {
                    let parsed: u32 = fields.next().unwrap_or_default().parse().map_err(|_| {
                        anyhow::anyhow!("line {line_number}: LUT_3D_SIZE is not a whole number")
                    })?;
                    check_size(parsed)?;
                    ensure!(
                        size.is_none(),
                        "line {line_number}: the file declares LUT_3D_SIZE twice"
                    );
                    size = Some(parsed);
                    rgb.reserve((parsed as usize).pow(3) * 3);
                }
                "LUT_1D_SIZE" => {
                    bail!("line {line_number}: this is a 1D table; only 3D ones are supported")
                }
                "DOMAIN_MIN" | "DOMAIN_MAX" => {
                    let expected = if keyword == "DOMAIN_MIN" { 0.0 } else { 1.0 };
                    let domain = parse_triple(fields, line_number, keyword, "")?;
                    ensure!(
                        domain.iter().all(|value| *value == expected),
                        "line {line_number}: {keyword} is {domain:?}; only the default \
                         0..1 domain is supported"
                    );
                }
                // Anything else has to be an entry, which is three numbers.
                _ => {
                    ensure!(
                        size.is_some(),
                        "line {line_number}: an entry appears before LUT_3D_SIZE"
                    );
                    rgb.extend_from_slice(&parse_triple(fields, line_number, "entry", keyword)?);
                }
            }
        }

        let size = size.ok_or_else(|| anyhow::anyhow!("the file declares no LUT_3D_SIZE"))?;
        let entries = (size as usize).pow(3);
        ensure!(
            rgb.len() == entries * 3,
            "the file declares {size} a side, so it should hold {entries} entries, but it has {}",
            rgb.len() / 3
        );

        debug!(
            "parsed a {size}³ lookup table{}",
            match &title {
                Some(title) => format!(" titled '{title}'"),
                None => String::new(),
            }
        );
        Self::from_rgb(size, &rgb)
    }

    /// Entries along one side of the cube.
    pub fn size(&self) -> u32 {
        self.size
    }

    /// The table as RGBA texels, which is what the upload wants.
    pub(crate) fn rgba(&self) -> &[f32] {
        &self.rgba
    }
}

fn check_size(size: u32) -> Result<()> {
    ensure!(
        (Lut::MIN_SIZE..=Lut::MAX_SIZE).contains(&size),
        "a lookup table is between {} and {} entries a side, not {size}",
        Lut::MIN_SIZE,
        Lut::MAX_SIZE
    );
    Ok(())
}

/// Reads the three numbers a domain or an entry line holds. `first` is the
/// field already taken off the iterator, which for an entry is its red channel
/// and for a domain line is the keyword; the empty string means there is none.
fn parse_triple<'a>(
    fields: impl Iterator<Item = &'a str>,
    line_number: usize,
    what: &str,
    first: &'a str,
) -> Result<[f32; 3]> {
    let values: Vec<f32> = std::iter::once(first)
        .filter(|field| !field.is_empty())
        .chain(fields)
        .map(|field| {
            field
                .parse::<f32>()
                .map_err(|_| anyhow::anyhow!("line {line_number}: '{field}' is not a number"))
        })
        .collect::<Result<_>>()?;

    ensure!(
        values.len() == 3,
        "line {line_number}: {what} has {} values, not 3",
        values.len()
    );
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "line {line_number}: {what} holds a value that is not finite"
    );
    Ok([values[0], values[1], values[2]])
}

#[cfg(test)]
mod tests {
    use super::Lut;

    /// The corners of a 2-a-side identity, which is what an unset LUT binds.
    #[test]
    fn identity_spans_the_cube() {
        let lut = Lut::identity(2);
        assert_eq!(lut.size(), 2);
        // First entry is black, last is white, and red comes before green.
        assert_eq!(&lut.rgba()[..4], &[0.0, 0.0, 0.0, 1.0]);
        assert_eq!(&lut.rgba()[4..8], &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(&lut.rgba()[8..12], &[0.0, 1.0, 0.0, 1.0]);
        assert_eq!(&lut.rgba()[28..], &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn parses_a_cube_file() {
        let text = "\
# A comment
TITLE \"Two a side\"
LUT_3D_SIZE 2
DOMAIN_MIN 0.0 0.0 0.0
DOMAIN_MAX 1.0 1.0 1.0

0 0 0
1 0 0
0 1 0
1 1 0
0 0 1
1 0 1
0 1 1
1 1 1
";
        assert_eq!(Lut::from_cube(text).unwrap(), Lut::identity(2));
    }

    #[test]
    fn rejects_what_it_cannot_apply() {
        let err = |text: &str| Lut::from_cube(text).unwrap_err().to_string();

        assert!(err("LUT_1D_SIZE 16\n").contains("only 3D ones are supported"));
        assert!(err("LUT_3D_SIZE 2\n0 0 0\n").contains("should hold 8 entries, but it has 1"));
        assert!(err("0 0 0\n").contains("before LUT_3D_SIZE"));
        assert!(err("# nothing here\n").contains("no LUT_3D_SIZE"));
        assert!(err("LUT_3D_SIZE 1\n").contains("between 2 and 128"));
        assert!(err("LUT_3D_SIZE 2\nDOMAIN_MAX 0 0 4\n").contains("only the default"));
        assert!(err("LUT_3D_SIZE 2\n0 0 blue\n").contains("'blue' is not a number"));
        assert!(err("LUT_3D_SIZE 2\n0 0\n").contains("has 2 values, not 3"));
    }

    #[test]
    fn from_rgb_checks_its_length() {
        assert!(Lut::from_rgb(2, &[0.0; 24]).is_ok());
        assert!(
            Lut::from_rgb(2, &[0.0; 23])
                .unwrap_err()
                .to_string()
                .contains("needs 24 floats, not 23")
        );
    }
}
