//! A grid of instanced cubes, rasterized with a depth buffer.
//!
//! Run it with:
//!
//! ```text
//! cargo run --example cube
//! ```
//!
//! The frame pipeline is two passes:
//!   1. a render pass draws 4096 instances of one cube, depth-tested, into an
//!      Rgba16Float render texture — some of them emissive, well above 1.0,
//!   2. the post-processing pass tone maps that texture onto the swapchain.
//!
//! It is the smallest complete use of the raster side of `etib-core`: vertex
//! and index buffers, a depth texture, a pipeline with per-vertex and
//! per-instance layouts, and a draw that is indexed and indirect. The
//! instances live in a `StorageBuffer` bound as a vertex buffer and the draw
//! arguments in an `IndirectBuffer` — the two halves GPU culling writes, so a
//! culling pass can be put in front of this draw without changing it.
//!
//! The window title shows the frame rate. `--sdr` and `--hdr` force the
//! surface either way, as in the other examples.

use std::f32::consts::TAU;

use anyhow::Result;
use etib::core::{
    Compare, Cull, Depth, DepthLoad, DepthTexture, DrawIndexedArgs, IndexBuffer, IndirectBuffer,
    RasterDesc, StepMode, StorageBuffer, VertexAttribute, VertexBuffer, VertexFormat, VertexLayout,
    bytemuck,
};
use etib::{
    App, BindGroup, BindLayout, Blend, Color, Device, Frame, Gpu, Grade, Hdr, Load, Output,
    PostProcess, RenderPipeline, RenderTexture, Shader, Stage, StorageFormat, UniformBuffer,
};
use glam::{Mat4, Vec3};
use log::info;

/// Cubes along each side of the grid.
const GRID: u32 = 16;
/// Distance between neighbouring cube centres.
const SPACING: f32 = 2.5;

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,etib=info,etib_core=info,cube=info"),
    )
    .format_timestamp_millis()
    .init();

    let hdr = if std::env::args().any(|arg| arg == "--sdr") {
        Hdr::Off
    } else if std::env::args().any(|arg| arg == "--hdr") {
        Hdr::Force
    } else {
        Hdr::Allow
    };

    etib::run(
        etib::Config {
            title: "etib - cubes".to_owned(),
            hdr,
            ..Default::default()
        },
        |gpu| Ok(CubeApp::new(gpu)),
    )
}

/// One corner of the cube mesh. Mirrors the per-vertex inputs of `vs_main`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

/// One cube in the grid. Mirrors the per-instance inputs of `vs_main`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    offset: [f32; 3],
    scale: f32,
    /// Linear colour, with the emission multiplier in alpha.
    color: [f32; 4],
}

/// Mirror of the Camera struct in cube.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Camera {
    view_proj: [[f32; 4]; 4],
    sun: [f32; 4],
}

struct CubeApp {
    pipeline: RenderPipeline,
    camera: UniformBuffer<Camera>,
    bindings: BindGroup,
    vertices: VertexBuffer<Vertex>,
    indices: IndexBuffer,
    instances: StorageBuffer<Instance>,
    draw: IndirectBuffer,
    /// The scene-referred image the cubes are drawn into, and its depth.
    scene: RenderTexture,
    depth: DepthTexture,
    post: PostProcess,
    grade: Grade,
    output: Output,
    /// Orbit angle of the camera, in radians.
    angle: f32,
    /// Seconds since the display and the title were last refreshed.
    since_poll: f32,
}

impl CubeApp {
    fn new(gpu: &Gpu) -> Self {
        let device = &gpu.device;
        let (width, height) = gpu.size();

        let shader = Shader::wgsl(device, "cube", include_str!("cube.wgsl"));
        let layout = BindLayout::build(device, "cube camera")
            .uniform::<Camera>(Stage::Vertex)
            .finish();
        let camera = UniformBuffer::new(device, "cube camera");
        let bindings = layout.group(device, "cube camera", &[&camera]);

        let scene_format = StorageFormat::Rgba16Float.format();
        let pipeline = RenderPipeline::raster(
            device,
            &RasterDesc {
                label: "cube",
                shader: &shader,
                entry_points: ("vs_main", "fs_main"),
                layouts: &[&layout],
                vertex: &[
                    VertexLayout {
                        stride: size_of::<Vertex>() as u64,
                        step: StepMode::Vertex,
                        attributes: &[
                            VertexAttribute {
                                location: 0,
                                offset: 0,
                                format: VertexFormat::Float32x3,
                            },
                            VertexAttribute {
                                location: 1,
                                offset: 12,
                                format: VertexFormat::Float32x3,
                            },
                        ],
                    },
                    VertexLayout {
                        stride: size_of::<Instance>() as u64,
                        step: StepMode::Instance,
                        attributes: &[
                            VertexAttribute {
                                location: 2,
                                offset: 0,
                                format: VertexFormat::Float32x3,
                            },
                            VertexAttribute {
                                location: 3,
                                offset: 12,
                                format: VertexFormat::Float32,
                            },
                            VertexAttribute {
                                location: 4,
                                offset: 16,
                                format: VertexFormat::Float32x4,
                            },
                        ],
                    },
                ],
                target: scene_format,
                depth: Some(Depth {
                    write: true,
                    compare: Compare::Less,
                }),
                blend: Blend::Replace,
                cull: Cull::Back,
            },
        );

        let (mesh, mesh_indices) = cube_mesh();
        let vertices = VertexBuffer::new(device, "cube vertices", &mesh);
        let indices = IndexBuffer::u16(device, "cube indices", &mesh_indices);
        let instances = StorageBuffer::from_slice(device, "cube instances", &grid());

        // Written once here; a culling pass would rewrite `instance_count`
        // every frame instead.
        let draw = IndirectBuffer::new(device, "cube draw", 1);
        draw.write(
            device,
            0,
            &DrawIndexedArgs {
                index_count: indices.len(),
                instance_count: instances.len(),
                ..Default::default()
            },
        );
        info!("drawing {} cubes", instances.len());

        let scene = RenderTexture::new(device, "cube scene", width, height, scene_format);
        let depth = DepthTexture::new(device, "cube depth", width, height);
        let post = PostProcess::new(device, gpu.format(), scene.view());

        Self {
            pipeline,
            camera,
            bindings,
            vertices,
            indices,
            instances,
            draw,
            scene,
            depth,
            post,
            grade: Grade::default(),
            output: Output::detect(gpu.color_space(), &gpu.surface.display()),
            angle: 0.0,
            // Due straight away, so the title shows a rate from the start.
            since_poll: 1.0,
        }
    }

    fn write_camera(&self, device: &Device) {
        let (width, height) = self.scene.size();
        let extent = GRID as f32 * SPACING;
        let eye = Vec3::new(self.angle.cos(), 0.6, self.angle.sin()) * extent * 1.4;
        let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        let projection = Mat4::perspective_rh(
            50f32.to_radians(),
            width as f32 / height as f32,
            0.1,
            extent * 4.0,
        );
        self.camera.write(
            device,
            &Camera {
                view_proj: (projection * view).to_cols_array_2d(),
                sun: Vec3::new(-0.4, -1.0, -0.3)
                    .normalize()
                    .extend(0.0)
                    .to_array(),
            },
        );
    }
}

impl App for CubeApp {
    fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        self.depth.resize(&gpu.device, width, height);
        if self.scene.resize(&gpu.device, width, height) {
            self.post.rebind(&gpu.device, self.scene.view());
        }
    }

    fn update(&mut self, dt: f32) {
        self.angle = (self.angle + dt * 0.25).rem_euclid(TAU);
        self.since_poll += dt;
    }

    fn render(&mut self, frame: Frame<'_>) {
        let Frame {
            gpu,
            encoder,
            view,
            stats,
        } = frame;

        if self.since_poll >= 1.0 {
            self.since_poll = 0.0;
            self.output = Output::detect(gpu.color_space(), &gpu.surface.display());
            gpu.window
                .set_title(&format!("etib - cubes - {:.0} fps", stats.fps));
        }

        self.write_camera(&gpu.device);

        // 1. The cubes, into the scene-referred texture.
        {
            let mut pass = encoder.render_pass_with_depth(
                "cube",
                self.scene.view(),
                Load::Clear(Color {
                    r: 0.02,
                    g: 0.025,
                    b: 0.04,
                    a: 1.0,
                }),
                &self.depth,
                DepthLoad::Clear(1.0),
            );
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bindings);
            pass.set_vertex_buffer(0, &self.vertices);
            pass.set_vertex_buffer(1, &self.instances);
            pass.set_index_buffer(&self.indices);
            pass.draw_indexed_indirect(&self.draw, 0);
        }

        // 2. Tone mapped onto the swapchain image.
        self.post
            .draw(&gpu.device, encoder, view, &self.grade, self.output);
    }
}

/// A unit cube centred on the origin: four corners per face so each face has
/// its own normal, wound counter-clockwise seen from outside.
fn cube_mesh() -> (Vec<Vertex>, Vec<u16>) {
    // Each face as its normal and two edges whose cross product is that
    // normal, which is what makes the winding come out counter-clockwise.
    let faces = [
        (Vec3::X, Vec3::Y, Vec3::Z),
        (Vec3::NEG_X, Vec3::Z, Vec3::Y),
        (Vec3::Y, Vec3::Z, Vec3::X),
        (Vec3::NEG_Y, Vec3::X, Vec3::Z),
        (Vec3::Z, Vec3::X, Vec3::Y),
        (Vec3::NEG_Z, Vec3::Y, Vec3::X),
    ];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (normal, u, v) in faces {
        let base = vertices.len() as u16;
        for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let position = (normal + u * su + v * sv) * 0.5;
            vertices.push(Vertex {
                position: position.to_array(),
                normal: normal.to_array(),
            });
        }
        indices.extend([0, 1, 2, 0, 2, 3].map(|i| base + i));
    }
    (vertices, indices)
}

/// `GRID`³ cubes centred on the origin, coloured by position, with one in
/// seven emissive so the image has highlights for the tone curve to compress.
fn grid() -> Vec<Instance> {
    let half = (GRID - 1) as f32 * SPACING * 0.5;
    let mut instances = Vec::with_capacity((GRID * GRID * GRID) as usize);
    for x in 0..GRID {
        for y in 0..GRID {
            for z in 0..GRID {
                let t = Vec3::new(x as f32, y as f32, z as f32) / (GRID - 1) as f32;
                let index = instances.len();
                instances.push(Instance {
                    offset: [
                        x as f32 * SPACING - half,
                        y as f32 * SPACING - half,
                        z as f32 * SPACING - half,
                    ],
                    scale: 1.0,
                    color: [
                        0.2 + 0.8 * t.x,
                        0.2 + 0.8 * t.y,
                        0.2 + 0.8 * t.z,
                        if index % 7 == 0 { 6.0 } else { 1.0 },
                    ],
                });
            }
        }
    }
    instances
}
