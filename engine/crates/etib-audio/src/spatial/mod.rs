//! Sounds placed in the world: what a listener hears of them, frame by frame.
//!
//! Nothing here runs on the audio thread. A [`Listener`] and an emitter's
//! position become a gain and a pan on the application's thread, once a frame,
//! and reach the mixer through the same [`Voice::set_gain`] and
//! [`Voice::set_pan`] any other sound uses.
//!
//! The model is kira's spatial track, which the engine used, so a scene sounds
//! as it did: the level falls linearly in decibels from full at
//! [`Spatial::min_distance`] to silence (-60 dB) at [`Spatial::max_distance`],
//! and each ear hears what is in front of it, two ears 0.2 apart and turned
//! π/8 forward of straight out to the side.

use std::f32::consts::FRAC_PI_8;

use glam::Vec3;

use crate::engine::{Audio, Play, Voice};
use crate::sound::Sound;

mod group;
mod manager;
mod strategy;

pub use group::SoundGroup;
pub use manager::SoundManager;
pub use strategy::{EmitterState, SingleCubeStrategy, SoundStrategy};

/// Half the distance between the ears, in world units.
const EAR_OFFSET: f32 = 0.1;
/// The level at [`Spatial::max_distance`], below which nothing is heard.
const SILENCE_DB: f32 = -60.0;

/// Where the sound is heard from: a position, and the way it is facing.
///
/// Up is +Y, as it is for every camera in the workspace. Built from a camera
/// with [`looking_at`](Self::looking_at)`(camera.eye, camera.target)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Listener {
    /// Where the listener's head is, in world units.
    pub position: Vec3,
    /// Need not be normalised. A zero vector is taken as -Z.
    pub forward: Vec3,
}

impl Default for Listener {
    fn default() -> Self {
        Self::new(Vec3::ZERO, Vec3::NEG_Z)
    }
}

impl Listener {
    /// A listener at `position`, facing along `forward`, which need not be
    /// normalised.
    pub fn new(position: Vec3, forward: Vec3) -> Self {
        Self { position, forward }
    }

    /// A listener at `eye` facing `target`.
    pub fn looking_at(eye: Vec3, target: Vec3) -> Self {
        Self::new(eye, target - eye)
    }

    /// The unit forward and right vectors. Looking straight up or down leaves
    /// right undefined, so +X stands in.
    fn basis(&self) -> (Vec3, Vec3) {
        let forward = self.forward.try_normalize().unwrap_or(Vec3::NEG_Z);
        let right = forward.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
        (forward, right)
    }
}

/// How a sound fades with distance and how strongly it is placed between the
/// ears. The default is kira's, and so the engine's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spatial {
    /// Closer than this, the sound is at its full level.
    pub min_distance: f32,
    /// At and past this, it is silent.
    pub max_distance: f32,
    /// 0 puts everything in the middle; 1 lets a sound to one side vanish
    /// from the other ear entirely.
    pub strength: f32,
}

impl Default for Spatial {
    fn default() -> Self {
        Self {
            min_distance: 1.0,
            max_distance: 100.0,
            strength: 0.75,
        }
    }
}

/// What a sound at some position comes to for one listener: the values to
/// hand to [`Voice::set_gain`] (before the sound's own volume) and
/// [`Voice::set_pan`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    /// The level, where 1.0 is the sound as recorded.
    pub gain: f32,
    /// Where the sound sits between the ears: -1.0 is hard left, 1.0 hard right.
    pub pan: f32,
}

/// The level a sound `distance` away is heard at, 0 to 1.
pub fn attenuation(distance: f32, spatial: &Spatial) -> f32 {
    let span = spatial.max_distance - spatial.min_distance;
    let relative = if span > 0.0 {
        ((distance - spatial.min_distance) / span).clamp(0.0, 1.0)
    } else if distance < spatial.max_distance {
        0.0
    } else {
        1.0
    };
    let db = SILENCE_DB * relative;
    if db <= SILENCE_DB {
        0.0
    } else {
        10f32.powf(db / 20.0)
    }
}

/// How `listener` hears a sound at `position`.
pub fn spatialize(listener: &Listener, position: Vec3, spatial: &Spatial) -> Placement {
    let level = attenuation((position - listener.position).length(), spatial);
    let strength = spatial.strength.clamp(0.0, 1.0);
    if strength == 0.0 {
        return Placement {
            gain: level,
            pan: 0.0,
        };
    }

    let (forward, right) = listener.basis();
    let (cos, sin) = (FRAC_PI_8.cos(), FRAC_PI_8.sin());
    let ear = |side: f32| {
        let at = listener.position + right * (side * EAR_OFFSET);
        let facing = right * (side * cos) + forward * sin;
        let towards = (position - at).normalize_or_zero();
        let heard = (facing.dot(towards) + 1.0) / 2.0;
        level * ((1.0 - strength) + strength * heard)
    };
    let (left, right) = (ear(-1.0), ear(1.0));

    // The mixer's pan law leaves the near side at full level and takes the far
    // side down, so the louder ear sets the gain and the ratio sets the pan.
    let gain = left.max(right);
    let pan = if gain <= 0.0 {
        0.0
    } else if right >= left {
        1.0 - left / right
    } else {
        right / left - 1.0
    };
    Placement { gain, pan }
}

/// A playing sound with a place in the world.
///
/// [`update`](Self::update) it once a frame with where the listener is; that
/// is the only time anything is sent to the mixer. Like a [`Voice`], dropping
/// it does not stop the sound.
#[derive(Clone, Debug)]
pub struct SpatialVoice {
    voice: Voice,
    /// Where the sound is.
    pub position: Vec3,
    /// The sound's own level, which distance and direction then scale.
    pub volume: f32,
    /// How the level falls off with distance from the listener.
    pub spatial: Spatial,
}

impl SpatialVoice {
    /// Starts `sound` at `position`, already placed for `listener` so its
    /// first block is not heard from the middle. `play.gain` is the volume;
    /// `play.pan` is replaced by the placement.
    pub fn play(
        audio: &Audio,
        sound: &Sound,
        play: Play,
        position: Vec3,
        listener: &Listener,
    ) -> Self {
        let spatial = Spatial::default();
        let placement = spatialize(listener, position, &spatial);
        let voice = audio.play_with(
            sound,
            Play {
                gain: play.gain * placement.gain,
                pan: placement.pan,
                ..play
            },
        );
        Self {
            voice,
            position,
            volume: play.gain,
            spatial,
        }
    }

    /// Sends the gain and pan for `listener` to the mixer.
    pub fn update(&self, listener: &Listener) {
        let placement = spatialize(listener, self.position, &self.spatial);
        self.voice.set_gain(self.volume * placement.gain);
        self.voice.set_pan(placement.pan);
    }

    /// The underlying voice, for its speed, looping and so on.
    pub fn voice(&self) -> &Voice {
        &self.voice
    }

    /// Whether the sound is still playing: `false` once a one-shot has ended or
    /// the voice was stopped.
    pub fn is_playing(&self) -> bool {
        self.voice.is_playing()
    }

    /// Fades the sound out, as [`Voice::stop`] does.
    pub fn stop(&self) {
        self.voice.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mixer::pan_gains;

    fn listener() -> Listener {
        Listener::new(Vec3::ZERO, Vec3::NEG_Z)
    }

    /// What each ear gets, through the mixer's pan law.
    fn ears(placement: Placement) -> (f32, f32) {
        let (left, right) = pan_gains(placement.pan);
        (placement.gain * left, placement.gain * right)
    }

    #[test]
    fn attenuation_is_linear_in_decibels_between_the_distances() {
        let spatial = Spatial::default();
        assert_eq!(attenuation(0.0, &spatial), 1.0, "inside min_distance");
        assert_eq!(attenuation(1.0, &spatial), 1.0);
        assert_eq!(attenuation(100.0, &spatial), 0.0, "-60 dB is silence");
        assert_eq!(attenuation(500.0, &spatial), 0.0);
        // Halfway is -30 dB.
        assert!((attenuation(50.5, &spatial) - 10f32.powf(-1.5)).abs() < 1e-6);
    }

    #[test]
    fn straight_ahead_is_centred() {
        let spatial = Spatial::default();
        let placement = spatialize(&listener(), Vec3::new(0.0, 0.0, -5.0), &spatial);
        assert!(placement.pan.abs() < 1e-6);
        assert!(placement.gain > 0.0 && placement.gain < attenuation(5.0, &spatial));
    }

    #[test]
    fn right_is_right_and_left_is_left() {
        let spatial = Spatial::default();
        let right = spatialize(&listener(), Vec3::new(5.0, 0.0, 0.0), &spatial);
        let left = spatialize(&listener(), Vec3::new(-5.0, 0.0, 0.0), &spatial);
        assert!(right.pan > 0.0 && left.pan < 0.0);
        assert!((right.pan + left.pan).abs() < 1e-5, "mirror images");
        assert!((right.gain - left.gain).abs() < 1e-5);
    }

    #[test]
    fn turning_round_swaps_the_ears() {
        let behind = Listener::new(Vec3::ZERO, Vec3::Z);
        let placement = spatialize(&behind, Vec3::new(5.0, 0.0, 0.0), &Spatial::default());
        assert!(placement.pan < 0.0);
    }

    #[test]
    fn far_away_is_silent() {
        let placement = spatialize(
            &listener(),
            Vec3::new(0.0, 0.0, -200.0),
            &Spatial::default(),
        );
        assert_eq!(placement.gain, 0.0);
    }

    #[test]
    fn matches_kira_at_the_listener() {
        // The engine's ambience sat exactly on the listener, between the ears,
        // so each ear faces away from it: kira gives each one
        // 0.25 + 0.75 * (1 - cos(π/8)) / 2 there, about -11 dB.
        let placement = spatialize(&listener(), Vec3::ZERO, &Spatial::default());
        let expected = 0.25 + 0.75 * (1.0 - std::f32::consts::FRAC_PI_8.cos()) / 2.0;
        let (left, right) = ears(placement);
        assert!((left - expected).abs() < 1e-5, "{left} vs {expected}");
        assert!((right - expected).abs() < 1e-5);
    }

    #[test]
    fn the_ears_are_what_kira_would_give() {
        // Hard right at 10 units: the near ear faces it, the far one does not.
        let spatial = Spatial::default();
        let (left, right) = ears(spatialize(&listener(), Vec3::new(10.0, 0.0, 0.0), &spatial));
        let ear = |facing: f32| 0.25 + 0.75 * (facing + 1.0) / 2.0;
        let a = attenuation(10.0, &spatial);
        let cos = std::f32::consts::FRAC_PI_8.cos();
        assert!((right - a * ear(cos)).abs() < 1e-4, "{right}");
        assert!((left - a * ear(-cos)).abs() < 1e-4, "{left}");
    }

    #[test]
    fn no_spatialization_only_attenuates() {
        let spatial = Spatial {
            strength: 0.0,
            ..Spatial::default()
        };
        let placement = spatialize(&listener(), Vec3::new(10.0, 0.0, 0.0), &spatial);
        assert_eq!(placement.pan, 0.0);
        assert_eq!(placement.gain, attenuation(10.0, &spatial));
    }

    #[test]
    fn looking_straight_up_or_down_is_not_a_nan() {
        for forward in [Vec3::Y, Vec3::NEG_Y, Vec3::ZERO] {
            let listener = Listener::new(Vec3::ZERO, forward);
            let placement = spatialize(&listener, Vec3::new(3.0, 0.0, 0.0), &Spatial::default());
            assert!(placement.gain.is_finite() && placement.pan.is_finite());
        }
    }

    #[test]
    fn looking_at_points_the_listener() {
        let listener = Listener::looking_at(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO);
        assert_eq!(listener.position, Vec3::new(0.0, 0.0, 10.0));
        assert!((listener.forward.normalize() - Vec3::NEG_Z).length() < 1e-6);
    }
}
