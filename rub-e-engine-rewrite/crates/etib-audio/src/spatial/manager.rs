//! The listener, the looping groups and the one-shot effects, together.

use glam::Vec3;

use super::{Listener, SoundGroup, SoundStrategy, SpatialVoice};
use crate::engine::{Audio, Play};
use crate::sound::Sound;

/// Spatial sound for a scene: the engine's `SoundManager`.
///
/// It owns the [`Audio`] it plays on, so an application that wants sound
/// without a device builds it from [`Audio::new_or_silent`] or
/// [`Audio::silent`]. Call [`update`](Self::update) once a frame with the
/// camera's listener.
pub struct SoundManager {
    audio: Audio,
    listener: Listener,
    groups: Vec<SoundGroup>,
    oneshots: Vec<SpatialVoice>,
}

impl SoundManager {
    pub fn new(audio: Audio) -> Self {
        Self {
            audio,
            listener: Listener::default(),
            groups: Vec::new(),
            oneshots: Vec::new(),
        }
    }

    /// The device everything is played on, for sounds that are not placed.
    pub fn audio(&self) -> &Audio {
        &self.audio
    }

    /// The listener as of the last [`update`](Self::update).
    pub fn listener(&self) -> &Listener {
        &self.listener
    }

    /// Starts a looping group and returns its index, the order groups were
    /// added in. `cubes` are the world positions the strategy works from.
    pub fn add_group(
        &mut self,
        cubes: Vec<Vec3>,
        sound: &Sound,
        strategy: Box<dyn SoundStrategy>,
    ) -> usize {
        let group = SoundGroup::new(&self.audio, &self.listener, cubes, sound, strategy);
        self.groups.push(group);
        self.groups.len() - 1
    }

    /// Replaces the cube positions of the group at `index`; an unknown index
    /// is ignored.
    pub fn update_group_cubes(&mut self, index: usize, cubes: Vec<Vec3>) {
        if let Some(group) = self.groups.get_mut(index) {
            group.update_cubes(cubes);
        }
    }

    /// Plays `sound` once at `position`. It keeps being placed as the listener
    /// moves until it has finished.
    pub fn play_oneshot(&mut self, sound: &Sound, position: Vec3) {
        let voice = SpatialVoice::play(
            &self.audio,
            sound,
            Play::default(),
            position,
            &self.listener,
        );
        self.oneshots.push(voice);
    }

    /// Moves the listener and places every sound for it.
    pub fn update(&mut self, listener: Listener) {
        self.listener = listener;
        for group in &mut self.groups {
            group.update(&listener);
        }
        self.oneshots.retain(SpatialVoice::is_playing);
        for voice in &self.oneshots {
            voice.update(&listener);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sound::Waveform;
    use crate::spatial::SingleCubeStrategy;

    #[test]
    fn groups_are_numbered_in_order() {
        let mut sounds = SoundManager::new(Audio::silent());
        let hum = Sound::tone(Waveform::Sine, 220.0, 0.1);
        let a = sounds.add_group(vec![Vec3::ZERO], &hum, Box::new(SingleCubeStrategy));
        let b = sounds.add_group(vec![Vec3::X], &hum, Box::new(SingleCubeStrategy));
        assert_eq!((a, b), (0, 1));

        sounds.update_group_cubes(b, vec![Vec3::Y]);
        sounds.update_group_cubes(7, vec![Vec3::Y]);
        sounds.update(Listener::default());
        assert_eq!(sounds.groups[b].voice().position, Vec3::Y);
    }

    #[test]
    fn finished_oneshots_are_let_go() {
        // A silent Audio's voices are over as soon as they start.
        let mut sounds = SoundManager::new(Audio::silent());
        sounds.play_oneshot(&Sound::tone(Waveform::Square, 440.0, 0.1), Vec3::X);
        assert_eq!(sounds.oneshots.len(), 1);
        sounds.update(Listener::default());
        assert!(sounds.oneshots.is_empty());
    }

    #[test]
    fn the_listener_is_kept() {
        let mut sounds = SoundManager::new(Audio::silent());
        let listener = Listener::looking_at(Vec3::new(0.0, 5.0, 10.0), Vec3::ZERO);
        sounds.update(listener);
        assert_eq!(*sounds.listener(), listener);
    }
}
