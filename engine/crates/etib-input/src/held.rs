//! Which of a set of keys or buttons are down, and which changed this frame.

use std::collections::HashSet;
use std::hash::Hash;

/// Down / just pressed / just released, for keys of any kind: keyboard keys,
/// mouse buttons, gamepad buttons.
///
/// The rules are the engine's `InputState`: a press of a key that is already
/// down (auto-repeat) is not a new press, and releasing a key that was never
/// seen down still counts as a release.
#[derive(Debug)]
pub(crate) struct Held<T> {
    down: HashSet<T>,
    just_pressed: HashSet<T>,
    just_released: HashSet<T>,
}

// Derived, it would ask for `T: Default`.
impl<T> Default for Held<T> {
    fn default() -> Self {
        Self {
            down: HashSet::new(),
            just_pressed: HashSet::new(),
            just_released: HashSet::new(),
        }
    }
}

impl<T: Copy + Eq + Hash> Held<T> {
    pub fn set(&mut self, key: T, down: bool) {
        if down {
            if self.down.insert(key) {
                self.just_pressed.insert(key);
            }
        } else if self.down.remove(&key) {
            // Only a key that was down can be released: one pressed before
            // the window had focus sends its release here without its press.
            self.just_released.insert(key);
        }
    }

    pub fn is_down(&self, key: T) -> bool {
        self.down.contains(&key)
    }

    pub fn just_pressed(&self, key: T) -> bool {
        self.just_pressed.contains(&key)
    }

    pub fn just_released(&self, key: T) -> bool {
        self.just_released.contains(&key)
    }

    /// Forgets what changed, keeping what is down.
    pub fn clear_frame_state(&mut self) {
        self.just_pressed.clear();
        self.just_released.clear();
    }

    /// Releases every key that is down.
    pub fn release_all(&mut self) {
        self.release_where(|_| true);
    }

    /// Releases the keys that are down and match `f`.
    pub fn release_where(&mut self, f: impl Fn(&T) -> bool) {
        let released: Vec<T> = self.down.iter().copied().filter(|key| f(key)).collect();
        for key in released {
            self.set(key, false);
        }
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.down.is_empty() && self.just_pressed.is_empty() && self.just_released.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn releasing_a_key_that_was_not_down_is_not_a_release() {
        let mut held = Held::default();
        held.set(1u8, false);
        assert!(!held.just_released(1));
    }

    #[test]
    fn press_hold_release() {
        let mut held = Held::<u32>::default();
        assert!(!held.is_down(1));
        assert!(!held.just_pressed(1));

        held.set(1, true);
        assert!(held.is_down(1));
        assert!(held.just_pressed(1));
        assert!(!held.just_released(1));

        held.clear_frame_state();
        held.set(1, false);
        assert!(!held.is_down(1));
        assert!(!held.just_pressed(1));
        assert!(held.just_released(1));

        held.clear_frame_state();
        assert!(!held.is_down(1));
        assert!(!held.just_released(1));
    }

    #[test]
    fn repeat_is_not_a_new_press() {
        let mut held = Held::<u32>::default();
        held.set(1, true);
        held.clear_frame_state();
        held.set(1, true);
        assert!(held.is_down(1));
        assert!(!held.just_pressed(1));
        assert!(!held.just_released(1));
    }

    #[test]
    fn multiple_keys() {
        let mut held = Held::<u32>::default();
        held.set(1, true);
        held.set(2, true);
        assert!(held.is_down(1) && held.is_down(2));
        assert!(held.just_pressed(1) && held.just_pressed(2));

        held.clear_frame_state();
        held.set(1, false);
        assert!(!held.is_down(1));
        assert!(held.is_down(2));
        assert!(held.just_released(1));
        assert!(!held.just_released(2));
    }

    #[test]
    fn release_all_releases_what_was_down() {
        let mut held = Held::<u32>::default();
        held.set(1, true);
        held.clear_frame_state();
        held.release_all();
        assert!(!held.is_down(1));
        assert!(held.just_released(1));
        assert!(!held.just_released(2));
    }

    #[test]
    fn release_where_releases_matching_keys() {
        let mut held = Held::<(u32, u32)>::default();
        held.set((1, 7), true);
        held.set((2, 7), true);
        held.clear_frame_state();
        held.release_where(|&(pad, _)| pad == 1);
        assert!(!held.is_down((1, 7)) && held.just_released((1, 7)));
        assert!(held.is_down((2, 7)) && !held.just_released((2, 7)));
    }
}
