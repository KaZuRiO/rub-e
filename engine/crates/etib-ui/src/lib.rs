//! egui on top of an [`etib_core`] frame.
//!
//! [`EguiLayer`] owns the egui context, the winit input bridge and the paint
//! backend, and paints into the same encoder as the rest of the frame:
//!
//! ```no_run
//! # use etib_core::{Device, Encoder, Output, TextureView};
//! # use etib_ui::{EguiLayer, PaintTarget, egui};
//! # fn frame(ui: &mut EguiLayer, device: &Device, encoder: &mut Encoder,
//! #          window: &etib_ui::winit::window::Window, view: &TextureView) {
//! let target = PaintTarget {
//!     window,
//!     view,
//!     size_in_pixels: (1280, 800),
//!     output: Output::SDR,
//! };
//! ui.render(device, encoder, target, |ui| {
//!     egui::Panel::left("controls").show(ui, |ui| ui.label("hello"));
//! });
//! # }
//! ```
//!
//! This crate is separate from the `etib` framework, and optional there: an
//! application that draws no interface never builds egui at all. It depends
//! only on [`etib_core`], so it can also be used with a frame loop of your own.
//!
//! # Extended-range surfaces
//!
//! egui assumes it is painting standard-range output, and picks its encoding
//! from the target's format — neither branch of which is right for a float HDR
//! surface. On one of those, [`EguiLayer`] paints into a standard-range texture
//! of its own and composites it at the display's SDR white, which is where an
//! interface belongs anyway. It costs one texture and one pass, and only on
//! such a surface; see `src/composite.rs` for the details.
//!
//! # Interop
//!
//! This is the one place in the workspace that reaches past [`etib_core`] to
//! the graphics API, because `egui-wgpu` speaks wgpu directly. It does so
//! through the core wrappers' `raw` accessors rather than holding wgpu objects
//! of its own, so there is still only one device, one queue and one encoder.

mod composite;
mod layer;

pub use layer::{EguiLayer, PaintTarget};

// Re-exported so applications build their panels against the same egui this
// crate paints with, and can reach winit's `Window` type without a second
// dependency.
pub use {egui, winit};
