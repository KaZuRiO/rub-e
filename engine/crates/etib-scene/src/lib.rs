//! Cube scenes for etib: a camera, static and dynamic cube models, frustum
//! culling on the GPU, and an HDR skybox.
//!
//! Everything is drawn through `etib-core`, and the result is a
//! scene-referred `Rgba16Float` texture rather than a finished image: tone
//! mapping is the presentation pass's job (`etib::PostProcess`). Nor does this
//! crate know about windows or keyboards — the [`CameraController`] is told
//! which way to [`Move`], and the application decides which keys mean that.
//!
//! ```no_run
//! # use etib_scene::*;
//! # fn go(device: &etib_core::Device, encoder: &mut etib_core::Encoder) -> Result<()> {
//! let cubes = load_model("world.model")?;
//! let mut scene = Scene::new(device, &cubes, 1024, 1280, 800);
//! let ship = scene.add_dynamic(DynamicModel::load("ship.model")?)?;
//!
//! let mut camera = Camera::new(
//!     glam::Vec3::new(0.0, 30.0, 80.0),
//!     glam::Vec3::ZERO,
//!     glam::Vec3::Y,
//!     1280.0 / 800.0,
//!     Projection::Perspective { fovy: 45.0 },
//!     0.1,
//!     500.0,
//! );
//!
//! // Each frame:
//! scene.dynamic_mut(ship).unwrap().position.x += 0.1;
//! scene.render(device, encoder, &mut camera);
//! // ... then tone map `scene.target_view()` onto the swapchain.
//! # Ok(())
//! # }
//! ```

mod camera;
mod cube;
mod culling;
mod dynamic;
mod error;
mod model;
mod scene;
mod skybox;

pub use camera::{Camera, CameraController, CameraMode, CameraRaw, Move, Projection};
pub use cube::{CubeRaw, INDICES, VERTICES, Vertex};
pub use dynamic::{DynamicId, DynamicModel};
pub use error::{Error, Result};
pub use model::{ModelCube, load_model, parse_model, save_model};
pub use scene::Scene;

pub use glam;
