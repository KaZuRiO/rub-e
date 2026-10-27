//! Uniform buffers: the small, per-frame parameter blocks shaders read.

use bytemuck::Pod;

use crate::device::Device;

/// A buffer holding one `T`, rewritten as often as you like.
///
/// `T` must be [`Pod`] and laid out to match the shader's struct: `#[repr(C)]`,
/// with WGSL's alignment rules in mind.
pub struct UniformBuffer<T> {
    buffer: wgpu::Buffer,
    _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: Pod> UniformBuffer<T> {
    pub fn new(device: &Device, label: &str) -> Self {
        let buffer = device.raw().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size_of::<T>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            _marker: std::marker::PhantomData,
        }
    }

    /// Queues a write of the whole buffer. It lands before any work submitted
    /// after this call, so writing just before recording the pass is correct.
    pub fn write(&self, device: &Device, value: &T) {
        device
            .raw_queue()
            .write_buffer(&self.buffer, 0, bytemuck::bytes_of(value));
    }

    pub(crate) fn raw(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub(crate) fn size() -> u64 {
        size_of::<T>() as u64
    }
}
