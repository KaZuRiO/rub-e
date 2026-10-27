//! What a pipeline's shaders can see: bind group layouts and bind groups.
//!
//! Bindings are numbered by position. The first thing declared on a
//! [`BindLayout`] is `@binding(0)`, the second `@binding(1)`, and a bind group
//! built from that layout lists its resources in the same order.

use bytemuck::Pod;
use log::debug;

use crate::buffer::UniformBuffer;
use crate::device::Device;
use crate::texture::{Sampler, StorageFormat, TextureView};

/// Which shader stage a binding is visible to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Compute,
    Vertex,
    Fragment,
}

impl Stage {
    fn raw(self) -> wgpu::ShaderStages {
        match self {
            Self::Compute => wgpu::ShaderStages::COMPUTE,
            Self::Vertex => wgpu::ShaderStages::VERTEX,
            Self::Fragment => wgpu::ShaderStages::FRAGMENT,
        }
    }
}

/// The shape of a bind group: what it holds, in what order, and who can see it.
pub struct BindLayout {
    inner: wgpu::BindGroupLayout,
    entries: usize,
}

impl BindLayout {
    /// Starts describing a layout. Declare one binding per call, in binding
    /// order, then [`BindLayoutBuilder::finish`].
    pub fn build<'a>(device: &'a Device, label: &str) -> BindLayoutBuilder<'a> {
        BindLayoutBuilder {
            device,
            label: label.to_owned(),
            entries: Vec::new(),
        }
    }

    /// Binds concrete resources, in the order the layout declared them.
    ///
    /// # Panics
    ///
    /// If the number of resources does not match the layout.
    pub fn group(&self, device: &Device, label: &str, bindings: &[&dyn Bindable]) -> BindGroup {
        assert_eq!(
            bindings.len(),
            self.entries,
            "bind group '{label}' has {} resources but its layout declares {}",
            bindings.len(),
            self.entries
        );

        let entries: Vec<_> = bindings
            .iter()
            .enumerate()
            .map(|(index, binding)| wgpu::BindGroupEntry {
                binding: index as u32,
                resource: binding.resource(),
            })
            .collect();

        BindGroup(
            device
                .raw()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &self.inner,
                    entries: &entries,
                }),
        )
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

    pub fn finish(self) -> BindLayout {
        debug!(
            "creating bind group layout '{}' with {} binding(s)",
            self.label,
            self.entries.len()
        );
        let inner = self
            .device
            .raw()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&self.label),
                entries: &self.entries,
            });
        BindLayout {
            inner,
            entries: self.entries.len(),
        }
    }

    fn push(&mut self, stage: Stage, ty: wgpu::BindingType) {
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
/// Implemented for [`TextureView`], [`Sampler`] and [`UniformBuffer`]; it is
/// not meant to be implemented outside this crate.
pub trait Bindable: sealed::Sealed {
    #[doc(hidden)]
    fn resource(&self) -> wgpu::BindingResource<'_>;
}

impl Bindable for TextureView {
    fn resource(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::TextureView(self.raw())
    }
}

impl Bindable for Sampler {
    fn resource(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Sampler(self.raw())
    }
}

impl<T: Pod> Bindable for UniformBuffer<T> {
    fn resource(&self) -> wgpu::BindingResource<'_> {
        self.raw().as_entire_binding()
    }
}

mod sealed {
    use bytemuck::Pod;

    use crate::buffer::UniformBuffer;
    use crate::texture::{Sampler, TextureView};

    pub trait Sealed {}

    impl Sealed for TextureView {}
    impl Sealed for Sampler {}
    impl<T: Pod> Sealed for UniformBuffer<T> {}
}
