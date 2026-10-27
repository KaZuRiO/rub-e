//! Models that move at runtime: groups of cubes transformed as one rigid body.

use std::path::Path;

use glam::{Mat4, Quat, Vec3};

use crate::cube::CubeRaw;
use crate::error::{Error, Result};
use crate::model::{ModelCube, load_model};

/// A group of cubes that moves, turns and scales as a unit.
///
/// The cubes are in model space; `position`, `rotation` and `scale` place the
/// group in the world, and apply to the cubes themselves as well as to where
/// they sit, so a model scaled by 2 is made of cubes twice the size.
#[derive(Debug, Clone)]
pub struct DynamicModel {
    cubes: Vec<ModelCube>,
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: f32,
}

impl DynamicModel {
    /// Loads a `.model` file, placed at the origin with no rotation.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::from_cubes(load_model(path)?))
    }

    pub fn from_cubes(cubes: Vec<ModelCube>) -> Self {
        Self {
            cubes,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: 1.0,
        }
    }

    pub fn cubes(&self) -> &[ModelCube] {
        &self.cubes
    }

    /// The cubes, for changing their colour or their place within the model.
    pub fn cubes_mut(&mut self) -> &mut [ModelCube] {
        &mut self.cubes
    }

    /// How many instances the model occupies.
    pub fn cube_count(&self) -> usize {
        self.cubes.len()
    }

    /// The model's world transform.
    pub fn transform(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(Vec3::splat(self.scale), self.rotation, self.position)
    }

    pub(crate) fn instances(&self) -> impl Iterator<Item = CubeRaw> + '_ {
        let group = self.transform();
        self.cubes.iter().map(move |cube| {
            CubeRaw::new(
                group * Mat4::from_translation(cube.position),
                cube.color.extend(1.0),
            )
        })
    }
}

/// Names a dynamic model in its [`Scene`](crate::Scene).
///
/// Ids are not reused: removing a model retires its id for good, so an id kept
/// after the model it named was removed reaches nothing, even once another
/// model has taken the same slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DynamicId {
    index: u32,
    generation: u32,
}

/// One place a model can be in, and how many models it has held.
#[derive(Debug)]
struct Slot {
    /// Bumped on every removal, so that ids handed out before it go stale.
    generation: u32,
    model: Option<DynamicModel>,
}

/// The scene's dynamic models: reusable slots behind generational ids, a cap
/// on the total number of cubes, and whether anything changed since the last
/// upload.
///
/// GPU-free; the [`Scene`](crate::Scene) owns the buffer it fills.
#[derive(Debug)]
pub(crate) struct DynamicModels {
    slots: Vec<Slot>,
    capacity: usize,
    dirty: bool,
}

impl DynamicModels {
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: Vec::new(),
            capacity,
            dirty: false,
        }
    }

    /// Adds a model, reusing the first free slot, and returns its id.
    ///
    /// # Errors
    ///
    /// [`Error::Capacity`] if the cubes of all the models would no longer fit,
    /// in which case nothing is added.
    pub fn add(&mut self, model: DynamicModel) -> Result<DynamicId> {
        let needed = self.cube_count() + model.cube_count();
        if needed > self.capacity {
            return Err(Error::Capacity {
                needed,
                capacity: self.capacity,
            });
        }
        self.dirty = true;
        let index = match self.slots.iter().position(|slot| slot.model.is_none()) {
            Some(index) => index,
            None => {
                self.slots.push(Slot {
                    generation: 0,
                    model: None,
                });
                self.slots.len() - 1
            }
        };
        let slot = &mut self.slots[index];
        slot.model = Some(model);
        Ok(DynamicId {
            index: u32::try_from(index).expect("more than u32::MAX dynamic models"),
            generation: slot.generation,
        })
    }

    /// Removes a model; a stale or already removed id is ignored.
    pub fn remove(&mut self, id: DynamicId) {
        if let Some(slot) = self.slot_mut(id)
            && slot.model.take().is_some()
        {
            slot.generation = slot.generation.wrapping_add(1);
            self.dirty = true;
        }
    }

    pub fn get(&self, id: DynamicId) -> Option<&DynamicModel> {
        let slot = self.slots.get(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.model.as_ref()
    }

    /// The model, marked as changed so it is uploaded again.
    pub fn get_mut(&mut self, id: DynamicId) -> Option<&mut DynamicModel> {
        let slot = self
            .slots
            .get_mut(id.index as usize)
            .filter(|slot| slot.generation == id.generation)?;
        let model = slot.model.as_mut()?;
        self.dirty = true;
        Some(model)
    }

    /// The slot `id` names, if it still names it.
    fn slot_mut(&mut self, id: DynamicId) -> Option<&mut Slot> {
        self.slots
            .get_mut(id.index as usize)
            .filter(|slot| slot.generation == id.generation)
    }

    /// Cubes across all the live models.
    pub fn cube_count(&self) -> usize {
        self.models().map(DynamicModel::cube_count).sum()
    }

    fn models(&self) -> impl Iterator<Item = &DynamicModel> {
        self.slots.iter().filter_map(|slot| slot.model.as_ref())
    }

    /// Every live cube, packed, if anything changed since the last call.
    pub fn take_changes(&mut self) -> Option<Vec<CubeRaw>> {
        if !std::mem::take(&mut self.dirty) {
            return None;
        }
        Some(self.models().flat_map(DynamicModel::instances).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    fn cube(x: f32, y: f32, z: f32) -> ModelCube {
        ModelCube {
            position: Vec3::new(x, y, z),
            color: Vec3::ONE,
        }
    }

    fn translation(raw: &CubeRaw) -> Vec3 {
        Vec3::new(raw.model[3][0], raw.model[3][1], raw.model[3][2])
    }

    #[test]
    fn moving_the_model_moves_its_cubes() {
        let mut model = DynamicModel::from_cubes(vec![cube(1.0, 2.0, 3.0)]);
        model.position = Vec3::new(10.0, 0.0, 0.0);
        let raw = model.instances().next().unwrap();
        assert_eq!(translation(&raw), Vec3::new(11.0, 2.0, 3.0));
        assert_eq!(raw.color, [1.0; 4]);

        model.cubes_mut()[0].position.x = 5.0;
        assert_eq!(translation(&model.instances().next().unwrap()).x, 15.0);
    }

    #[test]
    fn scale_applies_to_where_cubes_sit() {
        let mut model = DynamicModel::from_cubes(vec![cube(1.0, 0.0, 0.0)]);
        model.scale = 2.0;
        let raw = model.instances().next().unwrap();
        assert_eq!(translation(&raw), Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(raw.model[0][0], 2.0, "and to the cube itself");
    }

    #[test]
    fn rotation_turns_the_group() {
        let mut model = DynamicModel::from_cubes(vec![cube(1.0, 0.0, 0.0)]);
        // 90° around Y takes +X to -Z.
        model.rotation = Quat::from_rotation_y(90f32.to_radians());
        let raw = model.instances().next().unwrap();
        assert!(translation(&raw).abs_diff_eq(Vec3::new(0.0, 0.0, -1.0), 1e-5));
    }

    #[test]
    fn empty_models_have_no_instances() {
        let model = DynamicModel::from_cubes(vec![]);
        assert_eq!(model.cube_count(), 0);
        assert_eq!(model.instances().count(), 0);
    }

    fn model_of(cubes: usize) -> DynamicModel {
        DynamicModel::from_cubes(vec![cube(0.0, 0.0, 0.0); cubes])
    }

    #[test]
    fn ids_are_stable_and_freed_slots_reused() {
        let mut models = DynamicModels::new(10);
        let a = models.add(model_of(1)).unwrap();
        let b = models.add(model_of(2)).unwrap();
        assert_ne!(a, b);
        assert_eq!(models.cube_count(), 3);

        models.remove(a);
        assert!(models.get(a).is_none());
        assert!(models.get(b).is_some());
        models.remove(a);

        let c = models.add(model_of(1)).unwrap();
        assert_eq!(models.slots.len(), 2, "the freed slot is reused");
        assert!(models.get(c).is_some());
    }

    #[test]
    fn a_stale_id_does_not_reach_the_model_that_reused_its_slot() {
        let mut models = DynamicModels::new(10);
        let a = models.add(model_of(1)).unwrap();
        models.remove(a);
        let b = models.add(model_of(2)).unwrap();
        assert!(models.get(a).is_none());
        assert!(models.get_mut(a).is_none());
        assert_eq!(models.get(b).unwrap().cube_count(), 2);
        models.remove(a);
        assert!(models.get(b).is_some(), "a stale id removes nothing");
    }

    #[test]
    fn ids_stay_stale_across_many_reuses() {
        let mut models = DynamicModels::new(10);
        let mut old = Vec::new();
        for _ in 0..100 {
            let id = models.add(model_of(1)).unwrap();
            assert!(old.iter().all(|stale| models.get(*stale).is_none()));
            models.remove(id);
            old.push(id);
        }
        assert_eq!(models.slots.len(), 1);
    }

    #[test]
    fn adding_past_capacity_is_an_error() {
        let mut models = DynamicModels::new(2);
        let err = models.add(model_of(3)).unwrap_err();
        assert!(
            matches!(
                err,
                Error::Capacity {
                    needed: 3,
                    capacity: 2
                }
            ),
            "{err}"
        );
        assert_eq!(models.cube_count(), 0);
        assert_eq!(
            err.to_string(),
            "the dynamic models would hold 3 cubes, but the scene has room for 2"
        );
    }

    #[test]
    fn changes_are_packed_once() {
        let mut models = DynamicModels::new(10);
        assert!(models.take_changes().is_none(), "nothing yet");

        let id = models.add(model_of(3)).unwrap();
        models.add(model_of(1)).unwrap();
        assert_eq!(models.take_changes().unwrap().len(), 4);
        assert!(
            models.take_changes().is_none(),
            "until something changes again"
        );

        models.get_mut(id).unwrap().position.x = 1.0;
        assert_eq!(models.take_changes().unwrap().len(), 4);

        models.remove(id);
        assert_eq!(models.take_changes().unwrap().len(), 1);
    }
}
