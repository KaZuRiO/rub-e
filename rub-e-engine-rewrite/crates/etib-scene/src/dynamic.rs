//! Models that move at runtime: groups of cubes transformed as one rigid body.

use std::path::Path;

use glam::{Mat4, Quat, Vec3};

use crate::cube::CubeRaw;
use crate::error::Result;
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

/// The scene's dynamic models: slots that keep their ids, a cap on the total
/// number of cubes, and whether anything changed since the last upload.
///
/// GPU-free; the [`Scene`](crate::Scene) owns the buffer it fills.
#[derive(Debug)]
pub(crate) struct DynamicModels {
    models: Vec<Option<DynamicModel>>,
    capacity: usize,
    dirty: bool,
}

impl DynamicModels {
    pub fn new(capacity: usize) -> Self {
        Self {
            models: Vec::new(),
            capacity,
            dirty: false,
        }
    }

    /// Adds a model, reusing the first free slot, and returns its id.
    ///
    /// # Panics
    ///
    /// If the cubes of all the models would no longer fit in the capacity.
    pub fn add(&mut self, model: DynamicModel) -> usize {
        let total = self.cube_count() + model.cube_count();
        assert!(
            total <= self.capacity,
            "dynamic models would hold {total} cubes, but the scene has room for {}",
            self.capacity
        );
        self.dirty = true;
        match self.models.iter().position(Option::is_none) {
            Some(id) => {
                self.models[id] = Some(model);
                id
            }
            None => {
                self.models.push(Some(model));
                self.models.len() - 1
            }
        }
    }

    /// Removes a model; an unknown or already removed id is ignored.
    pub fn remove(&mut self, id: usize) {
        if let Some(slot) = self.models.get_mut(id)
            && slot.take().is_some()
        {
            self.dirty = true;
        }
    }

    pub fn get(&self, id: usize) -> Option<&DynamicModel> {
        self.models.get(id)?.as_ref()
    }

    /// The model, marked as changed so it is uploaded again.
    pub fn get_mut(&mut self, id: usize) -> Option<&mut DynamicModel> {
        let model = self.models.get_mut(id)?.as_mut()?;
        self.dirty = true;
        Some(model)
    }

    /// Cubes across all the live models.
    pub fn cube_count(&self) -> usize {
        self.models
            .iter()
            .flatten()
            .map(DynamicModel::cube_count)
            .sum()
    }

    /// Every live cube, packed, if anything changed since the last call.
    pub fn take_changes(&mut self) -> Option<Vec<CubeRaw>> {
        if !std::mem::take(&mut self.dirty) {
            return None;
        }
        Some(
            self.models
                .iter()
                .flatten()
                .flat_map(DynamicModel::instances)
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn ids_are_stable_and_freed_slots_reused() {
        let mut models = DynamicModels::new(10);
        let a = models.add(DynamicModel::from_cubes(vec![cube(0.0, 0.0, 0.0)]));
        let b = models.add(DynamicModel::from_cubes(vec![cube(1.0, 0.0, 0.0); 2]));
        assert_eq!((a, b), (0, 1));
        assert_eq!(models.cube_count(), 3);

        models.remove(a);
        assert!(models.get(a).is_none());
        assert!(models.get(b).is_some());
        models.remove(a);
        models.remove(999);

        let c = models.add(DynamicModel::from_cubes(vec![cube(2.0, 0.0, 0.0)]));
        assert_eq!(c, a, "the freed slot is reused");
        assert!(models.get_mut(999).is_none());
    }

    #[test]
    #[should_panic(expected = "room for 2")]
    fn exceeding_the_capacity_panics() {
        let mut models = DynamicModels::new(2);
        models.add(DynamicModel::from_cubes(vec![cube(0.0, 0.0, 0.0); 2]));
        models.add(DynamicModel::from_cubes(vec![cube(0.0, 0.0, 0.0)]));
    }

    #[test]
    fn changes_are_packed_once() {
        let mut models = DynamicModels::new(10);
        assert!(models.take_changes().is_none(), "nothing yet");

        let id = models.add(DynamicModel::from_cubes(vec![cube(0.0, 0.0, 0.0); 3]));
        models.add(DynamicModel::from_cubes(vec![cube(0.0, 0.0, 0.0)]));
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
