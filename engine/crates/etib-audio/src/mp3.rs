//! MP3 decoding, behind the `mp3` feature.
//!
//! The whole file is decoded up front, like a WAV is: a [`Sound`] is samples
//! in memory, and nothing is streamed.

use std::io::{Cursor, ErrorKind};
use std::path::Path;

use log::debug;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as DecodeError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::error::{Error, Result};
use crate::sound::Sound;

impl Sound {
    /// Decodes an MP3 file from memory, mono or stereo.
    ///
    /// A damaged frame is skipped rather than failing the whole file, as a
    /// player would.
    pub fn from_mp3(bytes: &[u8]) -> Result<Self> {
        let source =
            MediaSourceStream::new(Box::new(Cursor::new(bytes.to_vec())), Default::default());
        let mut hint = Hint::new();
        hint.with_extension("mp3");
        let mut format = symphonia::default::get_probe()
            .format(
                &hint,
                source,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .map_err(|err| Error::with_source("not an MP3 file", err))?
            .format;

        let track = format
            .default_track()
            .ok_or_else(|| Error::new("the MP3 file has no audio in it"))?;
        let track_id = track.id;
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .map_err(|err| Error::with_source("cannot decode this MP3 file", err))?;

        let mut samples = Vec::new();
        let mut spec = None;
        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(DecodeError::IoError(err)) if err.kind() == ErrorKind::UnexpectedEof => break,
                Err(err) => return Err(Error::with_source("the MP3 file could not be read", err)),
            };
            if packet.track_id() != track_id {
                continue;
            }
            match decoder.decode(&packet) {
                Ok(decoded) => {
                    spec.get_or_insert(*decoded.spec());
                    let mut buffer =
                        SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                    buffer.copy_interleaved_ref(decoded);
                    samples.extend_from_slice(buffer.samples());
                }
                Err(DecodeError::DecodeError(reason)) => debug!("skipping an MP3 frame: {reason}"),
                Err(err) => {
                    return Err(Error::with_source("the MP3 file could not be decoded", err));
                }
            }
        }

        let spec = spec.ok_or_else(|| Error::new("the MP3 file has no audio in it"))?;
        let channels =
            u16::try_from(spec.channels.count()).expect("an MP3 has at most two channels");
        debug!(
            "decoded a {channels}-channel MP3 at {} Hz, {} frames",
            spec.rate,
            samples.len() / channels.max(1) as usize
        );
        Self::from_samples(samples, channels, spec.rate)
    }

    /// Decodes an MP3 file from disk. See [`from_mp3`](Self::from_mp3).
    pub fn from_mp3_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|err| Error::with_source(format!("could not read {}", path.display()), err))?;
        Self::from_mp3(&bytes).map_err(|err| {
            Error::with_source(
                format!("{} is not a sound this can play", path.display()),
                err,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first 12 KiB of the engine's `crowd_cheer.mp3`: whole frames of a
    /// real file, cut short.
    const CHEER: &[u8] = include_bytes!("../testdata/cheer.mp3");

    #[test]
    fn decodes_a_real_file() {
        let sound = Sound::from_mp3(CHEER).unwrap();
        assert!(sound.channels() == 1 || sound.channels() == 2);
        assert!(sound.sample_rate() >= 8_000);
        assert!(sound.frames() > 1_000);
        assert!(sound.samples().iter().all(|s| s.is_finite()));
    }

    #[test]
    fn refuses_what_is_not_an_mp3() {
        assert!(Sound::from_mp3(b"definitely not audio").is_err());
        assert!(Sound::from_mp3(&[]).is_err());
    }

    #[test]
    fn names_the_file_it_could_not_read() {
        let err = Sound::from_mp3_file("/nonexistent/cheer.mp3").unwrap_err();
        assert!(err.to_string().contains("cheer.mp3"));
    }
}
