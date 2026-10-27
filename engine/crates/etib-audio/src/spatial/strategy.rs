//! Where a group's sound comes from, given the cubes it belongs to.
//!
//! The engine's `SoundStrategy`, unchanged but for glam.

use glam::Vec3;

/// A single spatial emitter state for one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmitterState {
    /// World-space position of the emitter.
    pub position: Vec3,
    /// Linear volume multiplier.
    pub volume: f32,
}

/// Strategy interface for converting cube positions into one or more emitters.
pub trait SoundStrategy: Send + Sync {
    /// Compute emitters from the set of cube positions and the current camera position.
    fn compute_emitters(&self, cubes: &[Vec3], camera_pos: Vec3) -> Vec<EmitterState>;
}

/// Simplest strategy: use the first cube as the single emitter.
#[derive(Clone, Copy, Debug, Default)]
pub struct SingleCubeStrategy;

impl SoundStrategy for SingleCubeStrategy {
    fn compute_emitters(&self, cubes: &[Vec3], _camera_pos: Vec3) -> Vec<EmitterState> {
        cubes
            .first()
            .map(|&position| {
                vec![EmitterState {
                    position,
                    volume: 1.0,
                }]
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;

    fn origin() -> Vec3 {
        Vec3::new(0.0, 0.0, 0.0)
    }

    #[test]
    fn empty_cubes_returns_no_emitters() {
        let strategy = SingleCubeStrategy;
        assert!(strategy.compute_emitters(&[], origin()).is_empty());
    }

    #[test]
    fn single_cube_emitter_position_and_volume() {
        let strategy = SingleCubeStrategy;
        let cube = Vec3::new(3.0, 1.0, -2.0);
        let emitters = strategy.compute_emitters(&[cube], origin());
        assert_eq!(emitters.len(), 1);
        assert_eq!(emitters[0].position, cube);
        assert!((emitters[0].volume - 1.0).abs() < 1e-6);
    }

    #[test]
    fn multiple_cubes_uses_only_first() {
        let strategy = SingleCubeStrategy;
        let cubes = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(5.0, 5.0, 5.0),
            Vec3::new(-3.0, 2.0, 7.0),
        ];
        let emitters = strategy.compute_emitters(&cubes, origin());
        assert_eq!(emitters.len(), 1);
        assert_eq!(emitters[0].position, cubes[0]);
    }

    #[test]
    fn camera_position_does_not_affect_output() {
        let strategy = SingleCubeStrategy;
        let cube = Vec3::new(1.0, 2.0, 3.0);
        let e1 = strategy.compute_emitters(&[cube], Vec3::new(0.0, 0.0, 0.0));
        let e2 = strategy.compute_emitters(&[cube], Vec3::new(100.0, 200.0, 300.0));
        assert_eq!(e1[0].position, e2[0].position);
        assert!((e1[0].volume - e2[0].volume).abs() < 1e-6);
    }
}
