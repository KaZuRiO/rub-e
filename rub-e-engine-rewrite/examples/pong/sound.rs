//! The crowd: an ambience that follows the listener, and a cheer at each goal.

use etib_audio::{Audio, EmitterState, Listener, Sound, SoundManager, SoundStrategy};
use glam::Vec3;
use log::warn;

const AMBIENCE: &[u8] = include_bytes!("../../assets/ambience.mp3");
const CROWD_CHEER: &[u8] = include_bytes!("../../assets/crowd_cheer.mp3");

/// Places the emitter at the listener (camera) position — effectively global/non-spatial.
struct AmbientStrategy;

impl SoundStrategy for AmbientStrategy {
    fn compute_emitters(&self, _cubes: &[Vec3], camera_pos: Vec3) -> Vec<EmitterState> {
        vec![EmitterState {
            position: camera_pos,
            volume: 1.0,
        }]
    }
}

pub struct PongSounds {
    manager: SoundManager,
    /// Taken when the game starts: there is no crowd in the menu.
    ambience: Option<Sound>,
    cheer: Option<Sound>,
}

impl PongSounds {
    /// Opens the default output, or carries on silently without one. The
    /// sounds are decoded now, so nothing is read during play.
    pub fn new() -> Self {
        Self {
            manager: SoundManager::new(Audio::new_or_silent()),
            ambience: decode("ambience", AMBIENCE),
            cheer: decode("crowd cheer", CROWD_CHEER),
        }
    }

    /// Starts the crowd's ambience, the first time only.
    pub fn start_ambience(&mut self) {
        if let Some(ambience) = self.ambience.take() {
            self.manager
                .add_group(vec![Vec3::ZERO], &ambience, Box::new(AmbientStrategy));
        }
    }

    /// A cheer from `position`.
    pub fn cheer(&mut self, position: Vec3) {
        if let Some(cheer) = &self.cheer {
            self.manager.play_oneshot(cheer, position);
        }
    }

    pub fn update(&mut self, listener: Listener) {
        self.manager.update(listener);
    }
}

/// A sound that does not decode is a sound the game does without.
fn decode(name: &str, bytes: &[u8]) -> Option<Sound> {
    Sound::from_mp3(bytes)
        .inspect_err(|err| warn!("the {name} will not be heard: {err}"))
        .ok()
}
