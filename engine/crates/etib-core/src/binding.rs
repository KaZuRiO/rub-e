//! What a pipeline's shaders can see: bind group layouts and bind groups.
//!
//! Bindings are numbered by position. The first thing declared on a
//! [`BindLayout`] is `@binding(0)`, the second `@binding(1)`, and a bind group
//! built from that layout lists its resources in the same order.

use bytemuck::Pod;
use log::debug;

use crate::buffer::{DrawIndexedArgs, IndirectBuffer, StorageBuffer, UniformBuffer};
use crate::device::Device;
use crate::texture::{Sampler, StorageFormat, TextureView};

/// Which shader stage a binding is visible to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Compute,
    Vertex,
    Fragment,
    /// Every stage: for a resource such as a camera that a compute pass, a
    /// vertex shader and a fragment shader all read, so that one layout and
    /// one bind group serve every pipeline that uses it.
    All,
}

impl Stage {
    fn raw(self) -> wgpu::ShaderStages {
        match self {
            Self::Compute => wgpu::ShaderStages::COMPUTE,
            Self::Vertex => wgpu::ShaderStages::VERTEX,
            Self::Fragment => wgpu::ShaderStages::FRAGMENT,
            Self::All => {
                wgpu::ShaderStages::COMPUTE
                    | wgpu::ShaderStages::VERTEX
                    | wgpu::ShaderStages::FRAGMENT
            }
        }
    }
}

/// Whether a shader may write a storage buffer, or only read it. It has to
/// agree with the WGSL: `var<storage, read>` or `var<storage, read_write>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    ReadWrite,
}

/// The shape of a bind group: what it holds, in what order, and who can see it.
pub struct BindLayout {
    inner: wgpu::BindGroupLayout,
    /// What each binding holds, in binding order.
    kinds: Vec<Kind>,
}

impl BindLayout {
    /// Starts describing a layout. Declare one binding per call, in binding
    /// order, then [`BindLayoutBuilder::finish`].
    pub fn build<'a>(device: &'a Device, label: &str) -> BindLayoutBuilder<'a> {
        BindLayoutBuilder {
            device,
            label: label.to_owned(),
            entries: Vec::new(),
            kinds: Vec::new(),
        }
    }

    /// Binds concrete resources, in the order the layout declared them.
    ///
    /// # Panics
    ///
    /// If the number of resources does not match the layout, or one of them is
    /// a different kind of resource from the one its binding declares.
    pub fn group(&self, device: &Device, label: &str, bindings: &[&dyn Bindable]) -> BindGroup {
        let given: Vec<Kind> = bindings.iter().map(|binding| binding.kind()).collect();
        check_kinds(label, &self.kinds, &given);

        let entries: Vec<_> = bindings
            .iter()
            .enumerate()
            .map(|(index, binding)| wgpu::BindGroupEntry {
                binding: index as u32,
                resource: binding.resource(),
            })
            .collect();

        BindGroup(device.checked("bind group", label, |raw| {
            raw.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.inner,
                entries: &entries,
            })
        }))
    }

    pub(crate) fn raw(&self) -> &wgpu::BindGroupLayout {
        &self.inner
    }
}

/// Builds a [`BindLayout`], one binding at a time.
pub struct BindLayoutBuilder<'a> {
    device: &'a Device,
    label: String,
    entries: Vec<wgpu::BindGroupLayoutEntry>,
    kinds: Vec<Kind>,
}

impl BindLayoutBuilder<'_> {
    /// A texture the stage writes with `textureStore`. Its format has to match
    /// the texture that will be bound.
    pub fn storage_texture(mut self, stage: Stage, format: StorageFormat) -> Self {
        self.push(
            stage,
            wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: format.format().raw(),
                view_dimension: wgpu::TextureViewDimension::D2,
            },
        );
        self
    }

    /// A six-layer array texture the stage writes with `textureStore`, which
    /// is how a compute shader fills a [`TextureCube`](crate::TextureCube)
    /// through its [`storage_view`](crate::TextureCube::storage_view).
    pub fn storage_texture_array(mut self, stage: Stage, format: StorageFormat) -> Self {
        self.push(
            stage,
            wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: format.format().raw(),
                view_dimension: wgpu::TextureViewDimension::D2Array,
            },
        );
        self
    }

    /// A texture the stage samples.
    pub fn texture(mut self, stage: Stage) -> Self {
        self.push(
            stage,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
        );
        self
    }

    /// A three-dimensional texture the stage samples, such as a
    /// [`Texture3d`](crate::Texture3d) holding a lookup table.
    pub fn texture_3d(mut self, stage: Stage) -> Self {
        self.push(
            stage,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
        );
        self
    }

    /// A cube texture the stage samples by direction, such as a
    /// [`TextureCube`](crate::TextureCube) holding a skybox.
    pub fn texture_cube(mut self, stage: Stage) -> Self {
        self.push(
            stage,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::Cube,
                multisampled: false,
            },
        );
        self
    }

    /// The sampler used to read a texture binding.
    pub fn sampler(mut self, stage: Stage) -> Self {
        self.push(
            stage,
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        );
        self
    }

    /// A [`UniformBuffer<T>`]. `T` fixes the size the shader's struct must have.
    pub fn uniform<T: Pod>(mut self, stage: Stage) -> Self {
        self.push(
            stage,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(UniformBuffer::<T>::size()),
            },
        );
        self
    }

    /// A [`StorageBuffer<T>`]: a `var<storage>` array of `T`. `T` fixes the
    /// smallest size the buffer may have, one element.
    pub fn storage_buffer<T: Pod>(mut self, stage: Stage, access: Access) -> Self {
        self.push_storage(stage, access, size_of::<T>() as u64);
        self
    }

    /// An [`IndirectBuffer`], for the shader that writes the draw arguments:
    /// an array of `DrawIndexedIndirectArgs`, or a single one.
    pub fn indirect_buffer(mut self, stage: Stage, access: Access) -> Self {
        self.push_storage(stage, access, size_of::<DrawIndexedArgs>() as u64);
        self
    }

    pub fn finish(self) -> BindLayout {
        debug!(
            "creating bind group layout '{}' with {} binding(s)",
            self.label,
            self.entries.len()
        );
        let inner = self
            .device
            .checked("bind group layout", &self.label, |raw| {
                raw.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&self.label),
                    entries: &self.entries,
                })
            });
        BindLayout {
            inner,
            kinds: self.kinds,
        }
    }

    fn push_storage(&mut self, stage: Stage, access: Access, min_size: u64) {
        self.push(
            stage,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: access == Access::Read,
                },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(min_size),
            },
        );
    }

    fn push(&mut self, stage: Stage, ty: wgpu::BindingType) {
        self.kinds.push(match ty {
            wgpu::BindingType::Sampler(_) => Kind::Sampler,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                ..
            } => Kind::Uniform,
            wgpu::BindingType::Buffer { .. } => Kind::Storage,
            _ => Kind::View,
        });
        self.entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.entries.len() as u32,
            visibility: stage.raw(),
            ty,
            count: None,
        });
    }
}

/// Resources bound together for one pass.
pub struct BindGroup(wgpu::BindGroup);

impl BindGroup {
    pub(crate) fn raw(&self) -> &wgpu::BindGroup {
        &self.0
    }
}

/// A resource that can fill one slot of a [`BindGroup`].
///
/// Implemented for [`TextureView`], [`Sampler`], [`UniformBuffer`],
/// [`StorageBuffer`] and [`IndirectBuffer`]; it is not meant to be implemented
/// outside this crate.
pub trait Bindable: sealed::Sealed {
    #[doc(hidden)]
    fn resource(&self) -> wgpu::BindingResource<'_>;

    #[doc(hidden)]
    fn kind(&self) -> Kind;
}

impl Bindable for TextureView {
    fn resource(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::TextureView(self.raw())
    }

    fn kind(&self) -> Kind {
        Kind::View
    }
}

impl Bindable for Sampler {
    fn resource(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Sampler(self.raw())
    }

    fn kind(&self) -> Kind {
        Kind::Sampler
    }
}

impl<T: Pod> Bindable for UniformBuffer<T> {
    fn resource(&self) -> wgpu::BindingResource<'_> {
        self.raw().as_entire_binding()
    }

    fn kind(&self) -> Kind {
        Kind::Uniform
    }
}

impl<T: Pod> Bindable for StorageBuffer<T> {
    fn resource(&self) -> wgpu::BindingResource<'_> {
        self.raw().as_entire_binding()
    }

    fn kind(&self) -> Kind {
        Kind::Storage
    }
}

impl Bindable for IndirectBuffer {
    fn resource(&self) -> wgpu::BindingResource<'_> {
        self.raw().as_entire_binding()
    }

    fn kind(&self) -> Kind {
        Kind::Storage
    }
}

/// Refuses a bind group whose resources do not match its layout: first in
/// number, then kind by kind, naming the first slot that differs.
fn check_kinds(label: &str, declared: &[Kind], given: &[Kind]) {
    assert_eq!(
        given.len(),
        declared.len(),
        "bind group '{label}' has {} resources but its layout declares {}",
        given.len(),
        declared.len()
    );
    if let Some((index, (declared, given))) = declared
        .iter()
        .zip(given)
        .enumerate()
        .find(|(_, (declared, given))| declared != given)
    {
        panic!(
            "bind group '{label}': binding {index} is declared as a {declared} but was given a {given}"
        );
    }
}

use sealed::Kind;

mod sealed {
    use std::fmt;

    use bytemuck::Pod;

    use crate::buffer::{IndirectBuffer, StorageBuffer, UniformBuffer};
    use crate::texture::{Sampler, TextureView};

    pub trait Sealed {}

    /// What a binding holds, as far as matching a resource to it goes. The
    /// finer detail — dimension, format, access — is left to the backend.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Kind {
        View,
        Sampler,
        Uniform,
        Storage,
    }

    impl fmt::Display for Kind {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(match self {
                Self::View => "texture view",
                Self::Sampler => "sampler",
                Self::Uniform => "uniform buffer",
                Self::Storage => "storage buffer",
            })
        }
    }

    impl Sealed for TextureView {}
    impl Sealed for Sampler {}
    impl<T: Pod> Sealed for UniformBuffer<T> {}
    impl<T: Pod> Sealed for StorageBuffer<T> {}
    impl Sealed for IndirectBuffer {}
}

#[cfg(test)]
mod tests {
    use super::sealed::Kind;
    use super::*;

    #[test]
    #[should_panic(
        expected = "bind group 'post': binding 1 is declared as a sampler but was given a texture view"
    )]
    fn a_sampler_in_a_texture_slot_is_refused() {
        check_kinds(
            "post",
            &[Kind::View, Kind::Sampler],
            &[Kind::View, Kind::View],
        );
    }

    #[test]
    #[should_panic(expected = "has 1 resources but its layout declares 2")]
    fn the_count_is_still_checked_first() {
        check_kinds("post", &[Kind::View, Kind::Sampler], &[Kind::View]);
    }

    #[test]
    fn matching_kinds_pass() {
        check_kinds(
            "x",
            &[Kind::Uniform, Kind::Storage],
            &[Kind::Uniform, Kind::Storage],
        );
    }
}
