//! Compiled pipelines: the compute kind, a minimal screen-filling render kind,
//! and the rasterizing kind that draws meshes from vertex buffers.

use log::info;

use crate::binding::BindLayout;
use crate::device::Device;
use crate::shader::Shader;
use crate::texture::{DepthTexture, TextureFormat};

/// A compute pipeline: one entry point, and the bind group layouts it reads.
pub struct ComputePipeline(wgpu::ComputePipeline);

impl ComputePipeline {
    /// A pipeline with a single bind group, `@group(0)`.
    pub fn new(
        device: &Device,
        label: &str,
        shader: &Shader,
        entry_point: &str,
        layout: &BindLayout,
    ) -> Self {
        Self::with_layouts(device, label, shader, entry_point, &[layout])
    }

    /// A pipeline with several bind groups: the nth layout is `@group(n)`.
    /// Bind them with [`ComputePass::set_bind_group`](crate::ComputePass::set_bind_group).
    pub fn with_layouts(
        device: &Device,
        label: &str,
        shader: &Shader,
        entry_point: &str,
        layouts: &[&BindLayout],
    ) -> Self {
        let pipeline_layout = create_layout(device, label, layouts);

        let pipeline = device
            .raw()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: shader.raw(),
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });
        info!("compute pipeline '{label}' created");

        Self(pipeline)
    }

    pub(crate) fn raw(&self) -> &wgpu::ComputePipeline {
        &self.0
    }
}

/// How a pass's output combines with what is already in the target.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Blend {
    /// Overwrite the target.
    #[default]
    Replace,
    /// Alpha-blend over it, with colours already multiplied by their alpha —
    /// which is what egui and most interface rendering produces.
    Premultiplied,
}

impl Blend {
    fn raw(self) -> Option<wgpu::BlendState> {
        match self {
            Self::Replace => None,
            Self::Premultiplied => Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
        }
    }
}

/// A render pipeline for passes that fill their target.
///
/// There are no vertex buffers: the vertex shader generates its positions from
/// the vertex index, which is all a fullscreen triangle or quad needs. Nothing
/// is blended, culled or depth-tested.
pub struct RenderPipeline(wgpu::RenderPipeline);

impl RenderPipeline {
    #[allow(clippy::too_many_arguments)]
    pub fn screen(
        device: &Device,
        label: &str,
        shader: &Shader,
        entry_points: (&str, &str),
        target: TextureFormat,
        blend: Blend,
        layout: &BindLayout,
    ) -> Self {
        let (vertex_entry, fragment_entry) = entry_points;

        let pipeline_layout = create_layout(device, label, &[layout]);

        let pipeline = device
            .raw()
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader.raw(),
                    entry_point: Some(vertex_entry),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader.raw(),
                    entry_point: Some(fragment_entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target.raw(),
                        blend: blend.raw(),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    // An oversized triangle is clipped, not culled, so winding
                    // is irrelevant here.
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        info!("render pipeline '{label}' created for target {target:?}");

        Self(pipeline)
    }

    /// A pipeline that rasterizes geometry from vertex buffers, optionally
    /// depth-tested.
    pub fn raster(device: &Device, desc: &RasterDesc<'_>) -> Self {
        let (vertex_entry, fragment_entry) = desc.entry_points;
        let label = desc.label;
        let pipeline_layout = create_layout(device, label, desc.layouts);

        // wgpu's vertex layouts borrow their attribute lists, so the converted
        // attributes have to outlive the descriptor built over them.
        let attributes: Vec<Vec<wgpu::VertexAttribute>> = desc
            .vertex
            .iter()
            .map(|layout| layout.attributes.iter().map(|a| a.raw()).collect())
            .collect();
        let buffers: Vec<Option<wgpu::VertexBufferLayout<'_>>> = desc
            .vertex
            .iter()
            .zip(&attributes)
            .map(|(layout, attributes)| {
                Some(wgpu::VertexBufferLayout {
                    array_stride: layout.stride,
                    step_mode: layout.step.raw(),
                    attributes,
                })
            })
            .collect();

        let pipeline = device
            .raw()
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: desc.shader.raw(),
                    entry_point: Some(vertex_entry),
                    compilation_options: Default::default(),
                    buffers: &buffers,
                },
                fragment: Some(wgpu::FragmentState {
                    module: desc.shader.raw(),
                    entry_point: Some(fragment_entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: desc.target.raw(),
                        blend: desc.blend.raw(),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: desc.cull.raw(),
                    ..Default::default()
                },
                depth_stencil: desc.depth.map(|depth| wgpu::DepthStencilState {
                    format: DepthTexture::FORMAT,
                    depth_write_enabled: Some(depth.write),
                    depth_compare: Some(depth.compare.raw()),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        info!(
            "raster pipeline '{label}' created for target {:?}{}",
            desc.target,
            if desc.depth.is_some() {
                " with depth"
            } else {
                ""
            }
        );

        Self(pipeline)
    }

    pub(crate) fn raw(&self) -> &wgpu::RenderPipeline {
        &self.0
    }
}

/// Everything [`RenderPipeline::raster`] needs to know.
pub struct RasterDesc<'a> {
    pub label: &'a str,
    pub shader: &'a Shader,
    /// The vertex and fragment entry points.
    pub entry_points: (&'a str, &'a str),
    /// The bind group layouts; the nth is `@group(n)`.
    pub layouts: &'a [&'a BindLayout],
    /// The vertex buffers; the nth is bound to slot n.
    pub vertex: &'a [VertexLayout<'a>],
    pub target: TextureFormat,
    /// Depth testing against a [`DepthTexture`], or none. A pipeline with
    /// depth must be drawn in a pass opened with
    /// [`Encoder::render_pass_with_depth`](crate::Encoder::render_pass_with_depth).
    pub depth: Option<Depth>,
    pub blend: Blend,
    pub cull: Cull,
}

/// How a pipeline uses the depth buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Depth {
    /// Whether passing fragments record their depth. Off for transparent
    /// surfaces and for a skybox drawn behind everything.
    pub write: bool,
    pub compare: Compare,
}

impl Default for Depth {
    /// The usual opaque geometry: nearer wins, and records itself.
    fn default() -> Self {
        Self {
            write: true,
            compare: Compare::Less,
        }
    }
}

/// When a fragment passes the depth test, against what is already there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compare {
    Less,
    LessEqual,
    Always,
}

impl Compare {
    fn raw(self) -> wgpu::CompareFunction {
        match self {
            Self::Less => wgpu::CompareFunction::Less,
            Self::LessEqual => wgpu::CompareFunction::LessEqual,
            Self::Always => wgpu::CompareFunction::Always,
        }
    }
}

/// Which triangles are discarded by winding. Counter-clockwise is the front.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Cull {
    #[default]
    None,
    Back,
    Front,
}

impl Cull {
    fn raw(self) -> Option<wgpu::Face> {
        match self {
            Self::None => None,
            Self::Back => Some(wgpu::Face::Back),
            Self::Front => Some(wgpu::Face::Front),
        }
    }
}

/// How one vertex buffer's elements map to shader inputs.
#[derive(Clone, Copy, Debug)]
pub struct VertexLayout<'a> {
    /// The size of one element, in bytes: usually `size_of::<T>()`.
    pub stride: u64,
    pub step: StepMode,
    pub attributes: &'a [VertexAttribute],
}

/// Whether a vertex buffer advances once per vertex or once per instance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StepMode {
    #[default]
    Vertex,
    Instance,
}

impl StepMode {
    fn raw(self) -> wgpu::VertexStepMode {
        match self {
            Self::Vertex => wgpu::VertexStepMode::Vertex,
            Self::Instance => wgpu::VertexStepMode::Instance,
        }
    }
}

/// One shader input, `@location(n)`, read from an element of a vertex buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexAttribute {
    pub location: u32,
    /// Where it starts within the element, in bytes.
    pub offset: u64,
    pub format: VertexFormat,
}

impl VertexAttribute {
    fn raw(self) -> wgpu::VertexAttribute {
        wgpu::VertexAttribute {
            format: self.format.raw(),
            offset: self.offset,
            shader_location: self.location,
        }
    }
}

/// The type of a [`VertexAttribute`], named after the WGSL type it arrives as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexFormat {
    /// `f32`
    Float32,
    /// `vec2<f32>`
    Float32x2,
    /// `vec3<f32>`
    Float32x3,
    /// `vec4<f32>`
    Float32x4,
    /// `u32`
    Uint32,
    /// `vec2<u32>`
    Uint32x2,
    /// `vec4<u32>`
    Uint32x4,
    /// `i32`
    Sint32,
    /// `vec4<f32>` from four bytes in 0..255, normalised to 0..1.
    Unorm8x4,
}

impl VertexFormat {
    /// The number of bytes it occupies in the buffer.
    pub fn size(self) -> u64 {
        self.raw().size()
    }

    fn raw(self) -> wgpu::VertexFormat {
        match self {
            Self::Float32 => wgpu::VertexFormat::Float32,
            Self::Float32x2 => wgpu::VertexFormat::Float32x2,
            Self::Float32x3 => wgpu::VertexFormat::Float32x3,
            Self::Float32x4 => wgpu::VertexFormat::Float32x4,
            Self::Uint32 => wgpu::VertexFormat::Uint32,
            Self::Uint32x2 => wgpu::VertexFormat::Uint32x2,
            Self::Uint32x4 => wgpu::VertexFormat::Uint32x4,
            Self::Sint32 => wgpu::VertexFormat::Sint32,
            Self::Unorm8x4 => wgpu::VertexFormat::Unorm8x4,
        }
    }
}

fn create_layout(device: &Device, label: &str, layouts: &[&BindLayout]) -> wgpu::PipelineLayout {
    let layouts: Vec<_> = layouts.iter().map(|layout| Some(layout.raw())).collect();
    device
        .raw()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &layouts,
            ..Default::default()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binding::{Access, Stage};
    use crate::buffer::{
        DrawIndexedArgs, IndexBuffer, IndirectBuffer, StorageBuffer, UniformBuffer, VertexBuffer,
    };
    use crate::encoder::{Color, DepthLoad, Load};
    use crate::test_util;
    use crate::texture::{RenderTexture, StorageFormat};

    #[test]
    fn vertex_formats_know_their_size() {
        assert_eq!(VertexFormat::Float32.size(), 4);
        assert_eq!(VertexFormat::Float32x3.size(), 12);
        assert_eq!(VertexFormat::Uint32x4.size(), 16);
        assert_eq!(VertexFormat::Unorm8x4.size(), 4);
    }

    const MESH: &str = "
        struct Camera { view_proj: mat4x4<f32> }
        @group(0) @binding(0) var<uniform> camera: Camera;

        struct Out { @builtin(position) position: vec4<f32>, @location(0) color: vec3<f32> }

        @vertex
        fn vs(@location(0) position: vec3<f32>, @location(1) offset: vec3<f32>, @location(2) color: vec3<f32>) -> Out {
            return Out(camera.view_proj * vec4(position + offset, 1.0), color);
        }

        @fragment
        fn fs(in: Out) -> @location(0) vec4<f32> {
            return vec4(in.color, 1.0);
        }
    ";

    #[test]
    fn instanced_mesh_draws_with_depth() {
        let Some(device) = test_util::device() else {
            return;
        };
        let shader = Shader::wgsl(&device, "test mesh", MESH);
        let layout = BindLayout::build(&device, "test camera")
            .uniform::<[[f32; 4]; 4]>(Stage::Vertex)
            .finish();
        let target = RenderTexture::new(
            &device,
            "test target",
            8,
            8,
            StorageFormat::Rgba16Float.format(),
        );
        let depth = DepthTexture::new(&device, "test depth", 8, 8);

        let pipeline = RenderPipeline::raster(
            &device,
            &RasterDesc {
                label: "test mesh",
                shader: &shader,
                entry_points: ("vs", "fs"),
                layouts: &[&layout],
                vertex: &[
                    VertexLayout {
                        stride: 12,
                        step: StepMode::Vertex,
                        attributes: &[VertexAttribute {
                            location: 0,
                            offset: 0,
                            format: VertexFormat::Float32x3,
                        }],
                    },
                    VertexLayout {
                        stride: 24,
                        step: StepMode::Instance,
                        attributes: &[
                            VertexAttribute {
                                location: 1,
                                offset: 0,
                                format: VertexFormat::Float32x3,
                            },
                            VertexAttribute {
                                location: 2,
                                offset: 12,
                                format: VertexFormat::Float32x3,
                            },
                        ],
                    },
                ],
                target: target.format(),
                depth: Some(Depth::default()),
                blend: Blend::Replace,
                cull: Cull::Back,
            },
        );

        let camera = UniformBuffer::<[[f32; 4]; 4]>::new(&device, "test camera");
        camera.write(
            &device,
            &[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        );
        let bindings = layout.group(&device, "test camera", &[&camera]);
        let vertices = VertexBuffer::new(
            &device,
            "test vertices",
            &[[0.0f32, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5]],
        );
        let instances = StorageBuffer::from_slice(
            &device,
            "test instances",
            &[[0.0f32; 6], [0.1, 0.0, 0.0, 1.0, 1.0, 1.0]],
        );
        let indices = IndexBuffer::u16(&device, "test indices", &[0, 1, 2]);
        let indirect = IndirectBuffer::new(&device, "test indirect", 1);
        indirect.write(
            &device,
            0,
            &DrawIndexedArgs {
                index_count: 3,
                instance_count: 2,
                ..Default::default()
            },
        );

        let mut encoder = device.encoder("test");
        {
            let mut pass = encoder.render_pass_with_depth(
                "test",
                target.view(),
                Load::Clear(Color::BLACK),
                &depth,
                DepthLoad::Clear(1.0),
            );
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings);
            pass.set_vertex_buffer(0, &vertices);
            pass.set_vertex_buffer(1, &instances);
            pass.set_index_buffer(&indices);
            pass.draw_indexed(0..indices.len(), 0, 0..2);
            pass.draw_indexed_indirect(&indirect, 0);
        }
        device.submit(encoder);
        device
            .raw()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device lost");
    }

    /// A miniature of the culling pass: five storage buffers in one compute
    /// stage, which is more than the downlevel limit allows, and an indirect
    /// buffer the shader counts the survivors into.
    const CULL: &str = "
        struct DrawIndexedIndirectArgs {
            index_count: u32, instance_count: atomic<u32>, first_index: u32, base_vertex: i32, first_instance: u32,
        }
        @group(0) @binding(0) var<storage, read> values: array<f32>;
        @group(0) @binding(1) var<storage, read_write> kept: array<f32>;
        @group(0) @binding(2) var<storage, read_write> draw: DrawIndexedIndirectArgs;
        @group(0) @binding(3) var<storage, read> thresholds: array<f32>;
        @group(0) @binding(4) var<storage, read_write> scratch: array<u32>;
        @group(1) @binding(0) var<uniform> count: u32;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            if id.x >= count { return; }
            scratch[id.x] = id.x;
            if values[id.x] > thresholds[0] {
                let slot = atomicAdd(&draw.instance_count, 1u);
                kept[slot] = values[id.x];
            }
        }
    ";

    #[test]
    fn compute_writes_an_indirect_draw() {
        let Some(device) = test_util::device() else {
            return;
        };
        let shader = Shader::wgsl(&device, "test cull", CULL);
        let buffers = BindLayout::build(&device, "test cull buffers")
            .storage_buffer::<f32>(Stage::Compute, Access::Read)
            .storage_buffer::<f32>(Stage::Compute, Access::ReadWrite)
            .indirect_buffer(Stage::Compute, Access::ReadWrite)
            .storage_buffer::<f32>(Stage::Compute, Access::Read)
            .storage_buffer::<u32>(Stage::Compute, Access::ReadWrite)
            .finish();
        let params = BindLayout::build(&device, "test cull params")
            .uniform::<[u32; 4]>(Stage::Compute)
            .finish();
        let pipeline = ComputePipeline::with_layouts(
            &device,
            "test cull",
            &shader,
            "main",
            &[&buffers, &params],
        );

        let values: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let values = StorageBuffer::from_slice(&device, "test values", &values);
        let kept = StorageBuffer::<f32>::new(&device, "test kept", 100);
        let indirect = IndirectBuffer::new(&device, "test indirect", 1);
        let thresholds = StorageBuffer::from_slice(&device, "test thresholds", &[89.5f32]);
        let scratch = StorageBuffer::<u32>::new(&device, "test scratch", 100);
        let count = UniformBuffer::<[u32; 4]>::new(&device, "test count");
        count.write(&device, &[100, 0, 0, 0]);
        indirect.write(
            &device,
            0,
            &DrawIndexedArgs {
                index_count: 36,
                ..Default::default()
            },
        );

        let group0 = buffers.group(
            &device,
            "test cull buffers",
            &[&values, &kept, &indirect, &thresholds, &scratch],
        );
        let group1 = params.group(&device, "test cull params", &[&count]);

        let readback = device.raw().create_buffer(&wgpu::BufferDescriptor {
            label: Some("test readback"),
            size: 20,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.encoder("test");
        {
            let mut pass = encoder.compute_pass("test cull");
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group0);
            pass.set_bind_group(1, &group1);
            pass.dispatch_workgroups(2, 1, 1);
        }
        encoder
            .raw_mut()
            .copy_buffer_to_buffer(indirect.raw(), 0, &readback, 0, 20);
        device.submit(encoder);

        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| result.expect("map failed"));
        device
            .raw()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device lost");
        let args: DrawIndexedArgs =
            *bytemuck::from_bytes(&readback.slice(..).get_mapped_range().expect("mapped"));
        assert_eq!(
            args,
            DrawIndexedArgs {
                index_count: 36,
                instance_count: 10,
                ..Default::default()
            }
        );
    }
}
