//! Compiled pipelines: the compute kind, and a minimal screen-filling render kind.

use log::info;

use crate::binding::BindLayout;
use crate::device::Device;
use crate::shader::Shader;
use crate::texture::TextureFormat;

/// A compute pipeline: one entry point, one bind group layout.
pub struct ComputePipeline(wgpu::ComputePipeline);

impl ComputePipeline {
    pub fn new(
        device: &Device,
        label: &str,
        shader: &Shader,
        entry_point: &str,
        layout: &BindLayout,
    ) -> Self {
        let pipeline_layout =
            device
                .raw()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[Some(layout.raw())],
                    ..Default::default()
                });

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

        let pipeline_layout =
            device
                .raw()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[Some(layout.raw())],
                    ..Default::default()
                });

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

    pub(crate) fn raw(&self) -> &wgpu::RenderPipeline {
        &self.0
    }
}
