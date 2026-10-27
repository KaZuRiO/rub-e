//! A reader for the WAV files a sound effect usually arrives in.
//!
//! WAV is a RIFF container: a header, then a sequence of chunks. Only two of
//! them matter here — `fmt ` says how the samples are stored and `data` holds
//! them — and everything else (`LIST`, `fact`, a tool's own metadata) is
//! skipped, which is what the format expects a reader to do.
//!
//! Every supported encoding is converted to `f32` in -1..1 on the way in, so
//! nothing past this module has to know what was on disk. Formats that would
//! have to be guessed at rather than read — compressed payloads, more than two
//! channels — are refused with a message that says which one it was.

use crate::error::{Error, Result};

/// What a parsed file turned out to hold.
#[derive(Debug)]
pub(crate) struct Wav {
    /// Interleaved, one entry per channel per frame.
    pub samples: Vec<f32>,
    pub channels: u16,
    pub sample_rate: u32,
}

/// The two `wFormatTag` values a plain WAV file uses, plus the wrapper that
/// defers the real one to a GUID at the end of the `fmt ` chunk.
const FORMAT_PCM: u16 = 1;
const FORMAT_FLOAT: u16 = 3;
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;

pub(crate) fn parse(bytes: &[u8]) -> Result<Wav> {
    // "RIFF" <u32 size> "WAVE", then chunks.
    let header = bytes.get(..12).ok_or_else(|| {
        Error::new("this is not a WAV file: it is too short to hold a RIFF header")
    })?;
    if &header[..4] != b"RIFF" || &header[8..12] != b"WAVE" {
        return Err(Error::new(
            "this is not a WAV file: it does not begin with a RIFF/WAVE header",
        ));
    }

    let mut format = None;
    let mut data = None;
    let mut offset = 12;

    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32(&bytes[offset + 4..offset + 8]) as usize;
        let body_at = offset + 8;
        let body = bytes.get(body_at..body_at + size).ok_or_else(|| {
            Error::new(format!(
                "the '{}' chunk claims {size} bytes but the file ends after {}",
                String::from_utf8_lossy(id),
                bytes.len() - body_at
            ))
        })?;

        match id {
            b"fmt " => format = Some(parse_format(body)?),
            b"data" => data = Some(body),
            _ => {}
        }

        // Chunks are padded to an even length, and the pad byte is not counted
        // in the size; missing that leaves every later chunk one byte out.
        offset = body_at + size + (size & 1);
    }

    let format = format.ok_or_else(|| Error::new("the file has no 'fmt ' chunk"))?;
    let data = data.ok_or_else(|| Error::new("the file has no 'data' chunk"))?;

    let samples = decode(data, &format)?;
    if samples.is_empty() {
        return Err(Error::new("the file's 'data' chunk holds no samples"));
    }
    if samples.len() % format.channels as usize != 0 {
        return Err(Error::new(format!(
            "the 'data' chunk holds {} samples, which is not a whole number of \
             {}-channel frames",
            samples.len(),
            format.channels
        )));
    }

    Ok(Wav {
        samples,
        channels: format.channels,
        sample_rate: format.sample_rate,
    })
}

struct Format {
    /// The real encoding: already resolved through the extensible GUID.
    tag: u16,
    channels: u16,
    sample_rate: u32,
    bits: u16,
}

fn parse_format(body: &[u8]) -> Result<Format> {
    if body.len() < 16 {
        return Err(Error::new(format!(
            "the 'fmt ' chunk is {} bytes; a WAV header is at least 16",
            body.len()
        )));
    }

    let mut tag = u16(&body[0..2]);
    let channels = u16(&body[2..4]);
    let sample_rate = u32(&body[4..8]);
    let bits = u16(&body[14..16]);

    // WAVE_FORMAT_EXTENSIBLE keeps the real tag in the first two bytes of a
    // 16-byte GUID, after the 2-byte cbSize, 2-byte valid-bits and 4-byte
    // channel mask that follow the common header.
    if tag == FORMAT_EXTENSIBLE {
        let guid = body.get(24..26).ok_or_else(|| {
            Error::new("the 'fmt ' chunk says WAVE_FORMAT_EXTENSIBLE but holds no sub-format GUID")
        })?;
        tag = u16(guid);
    }

    if tag != FORMAT_PCM && tag != FORMAT_FLOAT {
        return Err(Error::new(format!(
            "the file is encoded as format {tag}; only uncompressed PCM and float WAV are supported"
        )));
    }
    if channels == 0 || channels > 2 {
        return Err(Error::new(format!(
            "the file has {channels} channels; only mono and stereo are supported, because \
             anything wider has to be downmixed by a convention this cannot guess at"
        )));
    }
    if sample_rate == 0 {
        return Err(Error::new("the file declares a sample rate of 0"));
    }

    Ok(Format {
        tag,
        channels,
        sample_rate,
        bits,
    })
}

/// Converts the `data` chunk to `f32` in -1..1.
///
/// Integer samples are signed and divided by the magnitude of their most
/// negative value, which is the convention that makes a full-scale sample land
/// on exactly -1.0; 8-bit is the one exception, stored unsigned with 128 as
/// silence.
fn decode(data: &[u8], format: &Format) -> Result<Vec<f32>> {
    let bytes_per_sample = (format.bits as usize).div_ceil(8);
    let usable = data.len() - data.len() % bytes_per_sample.max(1);
    let data = &data[..usable];

    let samples = match (format.tag, format.bits) {
        (FORMAT_PCM, 8) => data
            .iter()
            .map(|byte| (*byte as f32 - 128.0) / 128.0)
            .collect(),
        (FORMAT_PCM, 16) => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32_768.0)
            .collect(),
        (FORMAT_PCM, 24) => data
            .as_chunks::<3>()
            .0
            .iter()
            // Sign-extended by landing the three bytes in the top of an i32.
            .map(|s| i32::from_le_bytes([0, s[0], s[1], s[2]]) as f32 / 2_147_483_648.0)
            .collect(),
        (FORMAT_PCM, 32) => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|s| i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / 2_147_483_648.0)
            .collect(),
        (FORMAT_FLOAT, 32) => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|s| f32::from_le_bytes([s[0], s[1], s[2], s[3]]))
            .collect(),
        (FORMAT_FLOAT, 64) => data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|s| {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "64-bit samples are mixed at 32"
                )]
                let sample =
                    f64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]) as f32;
                sample
            })
            .collect(),
        (tag, bits) => {
            let kind = if tag == FORMAT_FLOAT { "float" } else { "PCM" };
            return Err(Error::new(format!(
                "the file holds {bits}-bit {kind} samples, which is not one of the supported \
                 depths (8, 16, 24 or 32-bit PCM, or 32 or 64-bit float)"
            )));
        }
    };

    Ok(samples)
}

fn u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

fn u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a file around one `fmt ` and one `data` chunk, with a chunk of
    /// junk in between that a reader is expected to skip.
    fn file(tag: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let block_align = channels * bits.div_ceil(8);
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&tag.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&(rate * block_align as u32).to_le_bytes());
        fmt.extend_from_slice(&block_align.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());

        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF\0\0\0\0WAVE");
        for (id, body) in [
            (&b"fmt "[..], &fmt[..]),
            (b"LIST", b"junk"),
            (b"data", data),
        ] {
            out.extend_from_slice(id);
            out.extend_from_slice(&u32::try_from(body.len()).unwrap().to_le_bytes());
            out.extend_from_slice(body);
            if body.len() % 2 == 1 {
                out.push(0);
            }
        }
        let size = u32::try_from(out.len() - 8).unwrap();
        out[4..8].copy_from_slice(&size.to_le_bytes());
        out
    }

    #[test]
    fn reads_16_bit_stereo() {
        let data: Vec<u8> = [0i16, 16_384, -16_384, i16::MIN]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let wav = parse(&file(FORMAT_PCM, 2, 44_100, 16, &data)).unwrap();

        assert_eq!(wav.channels, 2);
        assert_eq!(wav.sample_rate, 44_100);
        assert_eq!(wav.samples, vec![0.0, 0.5, -0.5, -1.0]);
    }

    #[test]
    fn reads_the_other_depths() {
        // 8-bit is unsigned with 128 as silence.
        let wav = parse(&file(FORMAT_PCM, 1, 8_000, 8, &[128, 255, 0])).unwrap();
        assert_eq!(wav.samples, vec![0.0, 0.9921875, -1.0]);

        // 24-bit: 0x400000 is a quarter of full scale.
        let wav = parse(&file(FORMAT_PCM, 1, 8_000, 24, &[0x00, 0x00, 0x40])).unwrap();
        assert_eq!(wav.samples, vec![0.5]);

        let float: Vec<u8> = 0.25f32.to_le_bytes().into();
        let wav = parse(&file(FORMAT_FLOAT, 1, 8_000, 32, &float)).unwrap();
        assert_eq!(wav.samples, vec![0.25]);

        let double: Vec<u8> = 0.25f64.to_le_bytes().into();
        let wav = parse(&file(FORMAT_FLOAT, 1, 8_000, 64, &double)).unwrap();
        assert_eq!(wav.samples, vec![0.25]);
    }

    #[test]
    fn resolves_the_extensible_sub_format() {
        // The common 16 bytes, then cbSize, valid bits, channel mask, and a
        // GUID whose first two bytes are the real tag.
        let mut fmt = vec![0u8; 40];
        fmt[0..2].copy_from_slice(&FORMAT_EXTENSIBLE.to_le_bytes());
        fmt[2..4].copy_from_slice(&1u16.to_le_bytes());
        fmt[4..8].copy_from_slice(&48_000u32.to_le_bytes());
        fmt[14..16].copy_from_slice(&16u16.to_le_bytes());
        fmt[24..26].copy_from_slice(&FORMAT_PCM.to_le_bytes());

        let format = parse_format(&fmt).unwrap();
        assert_eq!(format.tag, FORMAT_PCM);
        assert_eq!(format.bits, 16);
    }

    #[test]
    fn rejects_what_it_cannot_read() {
        let err = |bytes: &[u8]| parse(bytes).unwrap_err().to_string();

        assert!(err(b"not a wav at all").contains("RIFF/WAVE header"));
        assert!(err(b"RIFF").contains("too short"));
        // ADPCM, and anything else that would have to be decompressed.
        assert!(err(&file(2, 1, 8_000, 4, &[0, 0])).contains("only uncompressed PCM"));
        assert!(err(&file(FORMAT_PCM, 6, 8_000, 16, &[0; 12])).contains("only mono and stereo"));
        assert!(err(&file(FORMAT_PCM, 1, 8_000, 12, &[0; 6])).contains("12-bit PCM"));
        assert!(err(&file(FORMAT_PCM, 1, 0, 16, &[0, 0])).contains("sample rate of 0"));
        assert!(err(&file(FORMAT_PCM, 1, 8_000, 16, &[])).contains("no samples"));
        assert!(
            err(&file(FORMAT_PCM, 2, 8_000, 16, &[0; 6])).contains("whole number of 2-channel")
        );

        // A chunk that claims more than the file holds.
        let mut truncated = file(FORMAT_PCM, 1, 8_000, 16, &[0, 0]);
        truncated.truncate(truncated.len() - 1);
        assert!(
            parse(&truncated)
                .unwrap_err()
                .to_string()
                .contains("the file ends after")
        );
    }
}
