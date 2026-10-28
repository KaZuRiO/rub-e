//! GPU buffers: uniform parameter blocks, vertex and index data, storage
//! arrays, and the indirect draw arguments a shader can write.

use bytemuck::{Pod, Zeroable};
use log::debug;

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
    /// A buffer for one `T`, uninitialised until the first
    /// [`write`](Self::write).
    ///
    /// WGSL rounds a uniform struct up to a multiple of 16 bytes, so a `T`
    /// whose size is not one is refused at compile time: its fields would not
    /// line up with the shader's.
    pub fn new(device: &Device, label: &str) -> Self {
        const {
            assert!(
                size_of::<T>().is_multiple_of(16),
                "a uniform struct's size must be a multiple of 16 bytes (WGSL alignment)"
            );
        }
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

/// Per-vertex or per-instance data a render pipeline reads through its vertex
/// layout, uploaded from a slice.
///
/// `T` is one element — one vertex, or one instance — and the pipeline's
/// [`VertexLayout`](crate::VertexLayout) says how its bytes map to shader
/// inputs.
pub struct VertexBuffer<T> {
    buffer: wgpu::Buffer,
    len: u32,
    _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: Pod> VertexBuffer<T> {
    /// # Panics
    ///
    /// If `data` is empty: a zero-sized buffer cannot be bound.
    pub fn new(device: &Device, label: &str, data: &[T]) -> Self {
        assert!(!data.is_empty(), "vertex buffer '{label}' cannot be empty");
        let buffer = create_init(
            device,
            label,
            bytemuck::cast_slice(data),
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );
        Self {
            buffer,
            len: element_count(label, data.len()),
            _marker: std::marker::PhantomData,
        }
    }

    /// Queues a write of `data` starting at element `offset`.
    ///
    /// # Panics
    ///
    /// If the write would run past the end of the buffer.
    pub fn write(&self, device: &Device, offset: u32, data: &[T]) {
        write_elements(device, &self.buffer, self.len, offset, data);
    }

    /// The number of elements.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Always false: an empty buffer is refused at creation.
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// Which integer type an [`IndexBuffer`] holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexFormat {
    /// Sixteen-bit indices.
    U16,
    /// Thirty-two-bit indices.
    U32,
}

impl IndexFormat {
    pub(crate) fn raw(self) -> wgpu::IndexFormat {
        match self {
            Self::U16 => wgpu::IndexFormat::Uint16,
            Self::U32 => wgpu::IndexFormat::Uint32,
        }
    }
}

/// The indices of an indexed draw, uploaded from a slice.
pub struct IndexBuffer {
    buffer: wgpu::Buffer,
    format: IndexFormat,
    len: u32,
}

impl IndexBuffer {
    /// Sixteen-bit indices, enough for any mesh under 65536 vertices.
    ///
    /// # Panics
    ///
    /// If `indices` is empty.
    pub fn u16(device: &Device, label: &str, indices: &[u16]) -> Self {
        // A buffer's size has to be a multiple of four, which an odd number of
        // u16 indices is not.
        let mut bytes = bytemuck::cast_slice::<u16, u8>(indices).to_vec();
        bytes.resize(bytes.len().next_multiple_of(4), 0);
        Self::new(device, label, &bytes, IndexFormat::U16, indices.len())
    }

    /// Thirty-two-bit indices.
    ///
    /// # Panics
    ///
    /// If `indices` is empty.
    pub fn u32(device: &Device, label: &str, indices: &[u32]) -> Self {
        Self::new(
            device,
            label,
            bytemuck::cast_slice(indices),
            IndexFormat::U32,
            indices.len(),
        )
    }

    fn new(device: &Device, label: &str, bytes: &[u8], format: IndexFormat, len: usize) -> Self {
        assert!(len > 0, "index buffer '{label}' cannot be empty");
        let buffer = create_init(device, label, bytes, wgpu::BufferUsages::INDEX);
        Self {
            buffer,
            format,
            len: element_count(label, len),
        }
    }

    /// Which integer type the indices are.
    pub fn format(&self) -> IndexFormat {
        self.format
    }

    /// The number of indices, which is what a draw of the whole buffer covers.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Always false: an empty buffer is refused at creation.
    pub fn is_empty(&self) -> bool {
        false
    }

    pub(crate) fn raw(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

/// An array of `T` a shader reads or writes as `var<storage>`.
///
/// It can also be bound as a vertex buffer, which is what lets a compute pass
/// write the instances a later draw consumes — culling being the case this
/// exists for. Whether the shader may write it is declared on the layout
/// ([`BindLayoutBuilder::storage_buffer`](crate::BindLayoutBuilder::storage_buffer)),
/// not here.
pub struct StorageBuffer<T> {
    buffer: wgpu::Buffer,
    len: u32,
    _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: Pod> StorageBuffer<T> {
    /// An array of `len` zeroed elements.
    ///
    /// # Panics
    ///
    /// If `len` is zero.
    pub fn new(device: &Device, label: &str, len: u32) -> Self {
        assert!(len > 0, "storage buffer '{label}' cannot be empty");
        let size = u64::from(len) * size_of::<T>() as u64;
        debug!("allocating {size}-byte storage buffer '{label}' ({len} elements)");
        let buffer = device.raw().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.next_multiple_of(4),
            usage: storage_usages(),
            mapped_at_creation: false,
        });
        Self {
            buffer,
            len,
            _marker: std::marker::PhantomData,
        }
    }

    /// An array holding a copy of `data`.
    ///
    /// # Panics
    ///
    /// If `data` is empty.
    pub fn from_slice(device: &Device, label: &str, data: &[T]) -> Self {
        assert!(!data.is_empty(), "storage buffer '{label}' cannot be empty");
        let buffer = create_init(device, label, bytemuck::cast_slice(data), storage_usages());
        Self {
            buffer,
            len: element_count(label, data.len()),
            _marker: std::marker::PhantomData,
        }
    }

    /// Queues a write of `data` starting at element `offset`.
    ///
    /// # Panics
    ///
    /// If the write would run past the end of the buffer.
    pub fn write(&self, device: &Device, offset: u32, data: &[T]) {
        write_elements(device, &self.buffer, self.len, offset, data);
    }

    /// The number of elements.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Always false: an empty buffer is refused at creation.
    pub fn is_empty(&self) -> bool {
        false
    }

    pub(crate) fn raw(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

fn storage_usages() -> wgpu::BufferUsages {
    // COPY_SRC so what a shader wrote can be read back, by a test or a tool.
    wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::VERTEX
        | wgpu::BufferUsages::COPY_DST
        | wgpu::BufferUsages::COPY_SRC
}

/// The arguments of one indexed draw, as the GPU reads them from an
/// [`IndirectBuffer`]. Mirrors WGSL's
///
/// ```wgsl
/// struct DrawIndexedIndirectArgs {
///     index_count: u32,
///     instance_count: atomic<u32>,   // or u32
///     first_index: u32,
///     base_vertex: i32,
///     first_instance: u32,
/// }
/// ```
///
/// `first_instance` other than zero needs an optional feature this crate does
/// not ask for, so leave it at zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct DrawIndexedArgs {
    /// How many indices each instance draws.
    pub index_count: u32,
    /// How many instances are drawn; what a culling pass counts up.
    pub instance_count: u32,
    /// The first index read from the index buffer.
    pub first_index: u32,
    /// Added to every index before the vertex is fetched.
    pub base_vertex: i32,
    /// The first instance's index; must stay zero, see above.
    pub first_instance: u32,
}

/// Indexed draws whose arguments live on the GPU, so a compute pass can decide
/// how many instances get drawn without the CPU waiting to find out.
///
/// It binds as a storage array of [`DrawIndexedArgs`] for the shader that fills
/// it, and is then handed to
/// [`RenderPass::draw_indexed_indirect`](crate::RenderPass::draw_indexed_indirect).
pub struct IndirectBuffer {
    buffer: wgpu::Buffer,
    len: u32,
}

impl IndirectBuffer {
    /// Room for `len` draws, zeroed.
    ///
    /// # Panics
    ///
    /// If `len` is zero.
    pub fn new(device: &Device, label: &str, len: u32) -> Self {
        assert!(len > 0, "indirect buffer '{label}' cannot be empty");
        let size = u64::from(len) * size_of::<DrawIndexedArgs>() as u64;
        debug!("allocating {size}-byte indirect buffer '{label}' ({len} draws)");
        let buffer = device.raw().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self { buffer, len }
    }

    /// Queues a write of draw `index`. Like [`UniformBuffer::write`] it lands
    /// before any work submitted afterwards, which is how a culling pass's
    /// counter is put back to zero each frame.
    ///
    /// # Panics
    ///
    /// If `index` is out of range.
    pub fn write(&self, device: &Device, index: u32, args: &DrawIndexedArgs) {
        write_elements(
            device,
            &self.buffer,
            self.len,
            index,
            std::slice::from_ref(args),
        );
    }

    /// The number of draws it holds.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Always false: an empty buffer is refused at creation.
    pub fn is_empty(&self) -> bool {
        false
    }

    pub(crate) fn raw(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub(crate) fn offset_of(&self, index: u32) -> u64 {
        assert!(
            index < self.len,
            "draw {index} is out of range of an indirect buffer of {}",
            self.len
        );
        u64::from(index) * size_of::<DrawIndexedArgs>() as u64
    }
}

/// Anything [`RenderPass::set_vertex_buffer`](crate::RenderPass::set_vertex_buffer)
/// can read from: a [`VertexBuffer`] or a [`StorageBuffer`].
///
/// Not meant to be implemented outside this crate.
pub trait VertexSource: sealed::Sealed {
    #[doc(hidden)]
    fn vertex_buffer(&self) -> &wgpu::Buffer;
}

impl<T: Pod> VertexSource for VertexBuffer<T> {
    fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl<T: Pod> VertexSource for StorageBuffer<T> {
    fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

mod sealed {
    use bytemuck::Pod;

    pub trait Sealed {}

    impl<T: Pod> Sealed for super::VertexBuffer<T> {}
    impl<T: Pod> Sealed for super::StorageBuffer<T> {}
}

/// A slice's length as the `u32` the GPU counts elements in.
///
/// # Panics
///
/// Past `u32::MAX` elements, which no draw or dispatch could address anyway.
fn element_count(label: &str, len: usize) -> u32 {
    u32::try_from(len)
        .unwrap_or_else(|_| panic!("buffer '{label}' holds more than u32::MAX elements"))
}

fn create_init(
    device: &Device,
    label: &str,
    bytes: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;

    debug!("allocating {}-byte buffer '{label}'", bytes.len());
    device
        .raw()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage,
        })
}

fn write_elements<T: Pod>(
    device: &Device,
    buffer: &wgpu::Buffer,
    len: u32,
    offset: u32,
    data: &[T],
) {
    check_range(len, offset, data.len());
    device.raw_queue().write_buffer(
        buffer,
        u64::from(offset) * size_of::<T>() as u64,
        bytemuck::cast_slice(data),
    );
}

fn check_range(len: u32, offset: u32, count: usize) {
    let end = u64::from(offset) + count as u64;
    assert!(
        end <= u64::from(len),
        "writing {count} element(s) at {offset} overruns a buffer of {len}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util;

    #[test]
    fn draw_args_match_the_gpu_layout() {
        assert_eq!(size_of::<DrawIndexedArgs>(), 20);
        assert_eq!(
            size_of::<DrawIndexedArgs>(),
            size_of::<wgpu::util::DrawIndexedIndirectArgs>()
        );
    }

    #[test]
    fn buffers_remember_their_length() {
        let Some(device) = test_util::device() else {
            return;
        };
        let vertices = VertexBuffer::new(&device, "test vertices", &[[0.0f32; 3]; 24]);
        assert_eq!(vertices.len(), 24);

        // An odd number of u16 indices is padded underneath, not in the count.
        let indices = IndexBuffer::u16(&device, "test indices", &[0, 1, 2]);
        assert_eq!(indices.len(), 3);
        assert_eq!(indices.format(), IndexFormat::U16);
        assert_eq!(indices.raw().size(), 8);

        let storage = StorageBuffer::<[f32; 4]>::new(&device, "test storage", 100);
        assert_eq!(storage.len(), 100);
        assert_eq!(storage.raw().size(), 1600);
        storage.write(&device, 98, &[[1.0; 4]; 2]);

        let indirect = IndirectBuffer::new(&device, "test indirect", 2);
        assert_eq!(indirect.raw().size(), 40);
        assert_eq!(indirect.offset_of(1), 20);
        indirect.write(&device, 1, &DrawIndexedArgs::default());
    }

    #[test]
    fn writes_up_to_the_end_are_allowed() {
        check_range(4, 0, 4);
        check_range(4, 3, 1);
        check_range(4, 4, 0);
    }

    #[test]
    #[should_panic(expected = "overruns")]
    fn writes_past_the_end_are_refused() {
        check_range(4, 3, 2);
    }
}
