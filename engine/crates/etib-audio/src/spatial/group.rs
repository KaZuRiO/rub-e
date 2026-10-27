//! A looping sound that follows a set of cubes.

use glam::Vec3;

use super::{Listener, SoundStrategy, SpatialVoice};
use crate::engine::{Audio, Play};
use crate::sound::Sound;

/// One looping sound, placed each frame by a [`SoundStrategy`] from the cubes
/// it belongs to.
///
/// The engine's `SoundGroup`, on a [`SpatialVoice`] rather than a kira track.
/// As with the track, dropping the group stops its sound.
pub struct SoundGroup {
    cubes: Vec<Vec3>,
    strategy: Box<dyn SoundStrategy>,
    voice: SpatialVoice,
}

impl SoundGroup {
    /// Starts `sound` looping, placed for `listener` from the first emitter
    /// `strategy` gives for `cubes`.
    pub fn new(
        audio: &Audio,
        listener: &Listener,
        cubes: Vec<Vec3>,
        sound: &Sound,
        strategy: Box<dyn SoundStrategy>,
    ) -> Self {
        let first = strategy
            .compute_emitters(&cubes, listener.position)
            .into_iter()
            .next();
        let (position, volume) = first.map_or((Vec3::ZERO, 1.0), |e| (e.position, e.volume));
        let voice = SpatialVoice::play(
            audio,
            sound,
            Play {
                gain: volume,
                looping: true,
                ..Play::default()
            },
            position,
            listener,
        );
        Self {
            cubes,
            strategy,
            voice,
        }
    }

    /// Replaces the cube positions the strategy works from.
    pub fn update_cubes(&mut self, cubes: Vec<Vec3>) {
        self.cubes = cubes;
    }

    /// Places the sound for this frame. When the strategy gives no emitter,
    /// the sound stays where it was, as it did in the engine.
    pub fn update(&mut self, listener: &Listener) {
        let emitters = self
            .strategy
            .compute_emitters(&self.cubes, listener.position);
        if let Some(state) = emitters.first() {
            self.voice.position = state.position;
            self.voice.volume = state.volume;
        }
        self.voice.update(listener);
    }

    pub fn voice(&self) -> &SpatialVoice {
        &self.voice
    }
}

impl Drop for SoundGroup {
    fn drop(&mut self) {
        self.voice.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Audio;
    use crate::sound::Waveform;
    use crate::spatial::{EmitterState, SingleCubeStrategy};

    /// Puts the emitter wherever the listener is, at half volume: the pong
    /// ambience, quieter.
    struct AtListener;

    impl SoundStrategy for AtListener {
        fn compute_emitters(&self, _cubes: &[Vec3], camera_pos: Vec3) -> Vec<EmitterState> {
            vec![EmitterState {
                position: camera_pos,
                volume: 0.5,
            }]
        }
    }

    struct Nowhere;

    impl SoundStrategy for Nowhere {
        fn compute_emitters(&self, _cubes: &[Vec3], _camera_pos: Vec3) -> Vec<EmitterState> {
            Vec::new()
        }
    }

    fn hum() -> Sound {
        Sound::tone(Waveform::Sine, 220.0, 0.1)
    }

    #[test]
    fn the_strategy_places_the_sound() {
        let audio = Audio::silent();
        let listener = Listener::default();
        let mut group = SoundGroup::new(&audio, &listener, vec![], &hum(), Box::new(AtListener));
        assert_eq!(group.voice().volume, 0.5);

        let moved = Listener::new(Vec3::new(3.0, 1.0, 2.0), Vec3::NEG_Z);
        group.update(&moved);
        assert_eq!(
            group.voice().position,
            moved.position,
            "the camera position is passed in"
        );
    }

    #[test]
    fn new_cubes_move_the_sound() {
        let audio = Audio::silent();
        let listener = Listener::default();
        let cube = Vec3::new(1.0, 2.0, 3.0);
        let mut group = SoundGroup::new(
            &audio,
            &listener,
            vec![cube],
            &hum(),
            Box::new(SingleCubeStrategy),
        );
        assert_eq!(group.voice().position, cube);

        group.update_cubes(vec![Vec3::new(9.0, 0.0, 0.0)]);
        group.update(&listener);
        assert_eq!(group.voice().position, Vec3::new(9.0, 0.0, 0.0));
    }

    #[test]
    fn no_emitter_leaves_the_sound_where_it_was() {
        let audio = Audio::silent();
        let listener = Listener::default();
        let mut group = SoundGroup::new(&audio, &listener, vec![], &hum(), Box::new(Nowhere));
        group.update(&listener);
        assert_eq!(group.voice().position, Vec3::ZERO);
        assert_eq!(group.voice().volume, 1.0);
    }
}
