//! The GPU layer underneath [`etib`](https://docs.rs/etib): textures, buffers,
//! bindings, pipelines and passes, with the graphics API kept out of sight.
//!
//! wgpu is an implementation detail here. Nothing in the API takes or returns a
//! wgpu type, the errors are this crate's own, and the async device handshake
//! is done for you, so an application can be written against [`Device`],
//! [`Encoder`] and friends without learning wgpu's descriptor vocabulary:
//!
//! ```no_run
//! # use etib_core::*;
//! # fn go(device: &Device) {
//! let shader = Shader::wgsl(device, "paint", "/* WGSL source */");
//! let target = StorageTexture::new(device, "canvas", 1280, 800, StorageFormat::Rgba8Unorm);
//!
//! let layout = BindLayout::build(device, "paint")
//!     .storage_texture(Stage::Compute, target.format())
//!     .finish();
//! let bindings = layout.group(device, "paint", &[target.view()]);
//! let pipeline = ComputePipeline::new(device, "paint", &shader, "main", &layout);
//!
//! let mut encoder = device.encoder("frame");
//! encoder
//!     .compute_pass("paint")
//!     .dispatch_grid(&pipeline, &bindings, target.size(), (8, 8));
//! device.submit(encoder);
//! # }
//! ```
//!
//! Start at [`Instance::open`], which opens a [`Device`] and a [`Surface`] for
//! a window in one call.
//!
//! # Interop
//!
//! Some libraries — an egui paint backend, a profiler — speak wgpu directly and
//! have to be handed the real objects. Every wrapper therefore has a `raw()` or
//! `raw_mut()` accessor, and the wgpu version this crate was built against is
//! re-exported as [`etib_core::wgpu`](wgpu) so callers cannot link a second,
//! mismatched copy of it.
//!
//! # Logging
//!
//! Resource creation logs at debug, adapter and surface changes at info,
//! per-dispatch detail at trace, through the [`log`] facade. No logger is
//! installed.

mod binding;
mod buffer;
mod device;
mod display;
mod encoder;
mod error;
mod instance;
mod pipeline;
mod shader;
mod surface;
mod texture;
mod timing;

#[cfg(any(test, feature = "test-util"))]
pub mod test_util;

pub use binding::{Access, BindGroup, BindLayout, BindLayoutBuilder, Bindable, Stage};
pub use buffer::{
    DrawIndexedArgs, IndexBuffer, IndexFormat, IndirectBuffer, StorageBuffer, UniformBuffer,
    VertexBuffer, VertexSource,
};
pub use device::Device;
pub use display::{ColorSpace, Display, Hdr, Output, SCRGB_REFERENCE_NITS, Vsync};
pub use encoder::{Color, ComputePass, DepthLoad, Encoder, Load, RenderPass};
pub use error::{Error, Result};
pub use instance::Instance;
pub use pipeline::{
    Blend, Compare, ComputePipeline, Cull, Depth, RasterDesc, RenderPipeline, StepMode,
    VertexAttribute, VertexFormat, VertexLayout,
};
pub use shader::Shader;
pub use surface::{Acquired, Frame, Surface};
pub use texture::{
    DepthTexture, Filter, RenderTexture, Sampler, StorageFormat, StorageTexture, Texture2d,
    Texture3d, TextureCube, TextureFormat, TextureView,
};
pub use timing::{GpuTimings, PassTime};

// For `#[derive(Pod)]` on shader parameter structs, and for interop.
pub use {bytemuck, wgpu};
