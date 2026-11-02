//! The cube every instance draws, and what each instance says about it.

use etib_core::{StepMode, VertexAttribute, VertexFormat, VertexLayout};
use glam::{Mat4, Vec3, Vec4};

/// One corner of the cube mesh. Mirrors `VertexInput` in `cube.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// Where the corner is, on a unit cube centred on the origin.
    pub position: [f32; 3],
    /// Linear RGB. The cube shader colours by instance and ignores it.
    pub color: [f32; 3],
    /// The face's outward normal, for lighting.
    pub normal: [f32; 3],
}

impl Vertex {
    /// Vertex buffer slot 0: locations 0 to 2.
    pub const LAYOUT: VertexLayout<'static> = VertexLayout {
        stride: size_of::<Self>() as u64,
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
            VertexAttribute {
                location: 2,
                offset: 24,
                format: VertexFormat::Float32x3,
            },
        ],
    };
}

const fn vertex(position: [f32; 3], color: [f32; 3], normal: [f32; 3]) -> Vertex {
    Vertex {
        position,
        color,
        normal,
    }
}

/// A unit cube centred on the origin, four corners per face so that each face
/// has its own normal.
pub const VERTICES: &[Vertex] = &[
    // Front (+z)
    vertex([-0.5, 0.5, 0.5], [0.5, 0.0, 0.5], [0.0, 0.0, 1.0]),
    vertex([-0.5, -0.5, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]),
    vertex([0.5, -0.5, 0.5], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]),
    vertex([0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    // Back (-z)
    vertex([-0.5, 0.5, -0.5], [0.5, 0.0, 0.5], [0.0, 0.0, -1.0]),
    vertex([-0.5, -0.5, -0.5], [0.0, 0.5, 0.5], [0.0, 0.0, -1.0]),
    vertex([0.5, -0.5, -0.5], [0.5, 0.5, 0.0], [0.0, 0.0, -1.0]),
    vertex([0.5, 0.5, -0.5], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
    // Top (+y)
    vertex([-0.5, 0.5, -0.5], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]),
    vertex([-0.5, 0.5, 0.5], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]),
    vertex([0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    vertex([0.5, 0.5, -0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    // Bottom (-y)
    vertex([-0.5, -0.5, 0.5], [0.0, 0.5, 0.5], [0.0, -1.0, 0.0]),
    vertex([-0.5, -0.5, -0.5], [0.0, 0.5, 0.5], [0.0, -1.0, 0.0]),
    vertex([0.5, -0.5, -0.5], [0.5, 0.5, 0.0], [0.0, -1.0, 0.0]),
    vertex([0.5, -0.5, 0.5], [0.5, 0.5, 0.0], [0.0, -1.0, 0.0]),
    // Right (+x)
    vertex([0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    vertex([0.5, -0.5, 0.5], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]),
    vertex([0.5, -0.5, -0.5], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]),
    vertex([0.5, 0.5, -0.5], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    // Left (-x)
    vertex([-0.5, 0.5, -0.5], [0.5, 0.0, 0.5], [-1.0, 0.0, 0.0]),
    vertex([-0.5, -0.5, -0.5], [0.0, 0.5, 0.5], [-1.0, 0.0, 0.0]),
    vertex([-0.5, -0.5, 0.5], [0.0, 0.5, 0.5], [-1.0, 0.0, 0.0]),
    vertex([-0.5, 0.5, 0.5], [0.5, 0.0, 0.5], [-1.0, 0.0, 0.0]),
];

/// Two triangles per face, counter-clockwise seen from outside.
pub const INDICES: &[u16] = &[
    0, 1, 2, 0, 2, 3, // front (+Z)
    4, 7, 6, 4, 6, 5, // back (-Z)
    8, 9, 10, 8, 10, 11, // top (+Y)
    12, 13, 14, 12, 14, 15, // bottom (-Y)
    16, 17, 18, 16, 18, 19, // right (+X)
    20, 21, 22, 20, 22, 23, // left (-X)
];

/// One cube as the GPU sees it: where it is and what colour. Mirrors
/// `InstanceInput` in `cube.wgsl` and `CubeRaw` in `cull.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CubeRaw {
    /// Model to world space, column-major as WGSL reads it.
    pub model: [[f32; 4]; 4],
    /// Linear RGBA; alpha is unused.
    pub color: [f32; 4],
}

impl CubeRaw {
    /// An instance placed by `model`, coloured `color`.
    pub fn new(model: Mat4, color: Vec4) -> Self {
        Self {
            model: model.to_cols_array_2d(),
            color: color.to_array(),
        }
    }

    /// Vertex buffer slot 1, per instance: the model matrix as four columns at
    /// locations 3 to 6, and the colour at 7.
    pub const LAYOUT: VertexLayout<'static> = VertexLayout {
        stride: size_of::<Self>() as u64,
        step: StepMode::Instance,
        attributes: &[
            VertexAttribute {
                location: 3,
                offset: 0,
                format: VertexFormat::Float32x4,
            },
            VertexAttribute {
                location: 4,
                offset: 16,
                format: VertexFormat::Float32x4,
            },
            VertexAttribute {
                location: 5,
                offset: 32,
                format: VertexFormat::Float32x4,
            },
            VertexAttribute {
                location: 6,
                offset: 48,
                format: VertexFormat::Float32x4,
            },
            VertexAttribute {
                location: 7,
                offset: 64,
                format: VertexFormat::Float32x4,
            },
        ],
    };
}

/// One static cube as the GPU sees it: 16 bytes rather than [`CubeRaw`]'s 80.
/// Static cubes are unit cubes on the axes, so a position is all the
/// transform they need. Mirrors `StaticCube` in `cull.wgsl` and
/// `static_cube.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StaticCubeRaw {
    /// The cube's centre, in world units.
    pub position: [f32; 3],
    /// sRGB-encoded RGBA, 8 bits each, red in the low byte; alpha is 255.
    /// See [`StaticCubeRaw::new`].
    pub color: u32,
}

impl StaticCubeRaw {
    /// A cube at `position`, coloured `color` in linear RGB. Each channel is
    /// clamped to 0..1 and stored sRGB-encoded on 8 bits, which keeps dark
    /// colours from banding as linear 8-bit values would.
    pub fn new(position: Vec3, color: Vec3) -> Self {
        Self {
            position: position.to_array(),
            color: pack_srgb8(color),
        }
    }
}

/// `linear` clamped to 0..1, sRGB-encoded and rounded to 8 bits per channel,
/// red in the low byte and an opaque alpha in the high one: what WGSL's
/// `unpack4x8unorm` reads back.
pub(crate) fn pack_srgb8(linear: Vec3) -> u32 {
    let [r, g, b] = linear.to_array().map(|channel| u32::from(srgb8(channel)));
    r | g << 8 | b << 16 | 0xFF << 24
}

/// One linear channel as an sRGB-encoded byte.
fn srgb8(linear: f32) -> u8 {
    let linear = if linear.is_nan() {
        0.0
    } else {
        linear.clamp(0.0, 1.0)
    };
    let encoded = if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    #[expect(
        clippy::cast_possible_truncation,
        reason = "clamped to 0..1 above, so the rounded value is 0..=255"
    )]
    let byte = (encoded * 255.0).round() as u8;
    byte
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_match_the_structs() {
        assert_eq!(size_of::<Vertex>(), 36);
        assert_eq!(size_of::<CubeRaw>(), 80);
        for layout in [Vertex::LAYOUT, CubeRaw::LAYOUT] {
            let last = layout.attributes.last().unwrap();
            assert_eq!(last.offset + last.format.size(), layout.stride);
        }
        let locations: Vec<u32> = CubeRaw::LAYOUT
            .attributes
            .iter()
            .map(|a| a.location)
            .collect();
        assert_eq!(locations, [3, 4, 5, 6, 7]);
    }

    #[test]
    fn mesh_has_six_faces() {
        assert_eq!(VERTICES.len(), 24);
        assert_eq!(INDICES.len(), 36);
        assert!(INDICES.iter().all(|&i| (i as usize) < VERTICES.len()));
    }

    /// Back-face culling depends on it: every triangle has to face the way its
    /// normal points.
    #[test]
    fn triangles_wind_counter_clockwise_from_outside() {
        for triangle in INDICES.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|i| Vec3::from(VERTICES[triangle[i] as usize].position));
            let facing = (b - a).cross(c - a);
            let normal = Vec3::from(VERTICES[triangle[0] as usize].normal);
            assert!(
                facing.dot(normal) > 0.0,
                "triangle {triangle:?} faces inwards"
            );
        }
    }

    /// The sRGB decoding the static cube shader does.
    fn srgb_to_linear(encoded: f32) -> f32 {
        if encoded <= 0.04045 {
            encoded / 12.92
        } else {
            ((encoded + 0.055) / 1.055).powf(2.4)
        }
    }

    #[test]
    fn static_instances_are_sixteen_bytes() {
        assert_eq!(size_of::<StaticCubeRaw>(), 16);
        assert_eq!(std::mem::offset_of!(StaticCubeRaw, color), 12);
        let raw = StaticCubeRaw::new(Vec3::new(1.0, 2.0, 3.0), Vec3::ONE);
        assert_eq!(raw.position, [1.0, 2.0, 3.0]);
        assert_eq!(raw.color, 0xFFFF_FFFF);
    }

    #[test]
    fn colours_pack_as_srgb_bytes() {
        assert_eq!(pack_srgb8(Vec3::ONE), 0xFFFF_FFFF);
        assert_eq!(pack_srgb8(Vec3::ZERO), 0xFF00_0000);
        assert_eq!(pack_srgb8(Vec3::new(1.0, 0.0, 0.0)), 0xFF00_00FF);
        assert_eq!(pack_srgb8(Vec3::new(0.0, 0.0, 1.0)), 0xFFFF_0000);
        // Out of range is clamped.
        assert_eq!(pack_srgb8(Vec3::new(2.0, -1.0, 0.0)), 0xFF00_00FF);
    }

    #[test]
    fn dark_colours_survive_the_round_trip() {
        for v in [0.002, 0.01, 0.05, 0.2, 0.5, 0.9] {
            let byte = pack_srgb8(Vec3::splat(v)) & 0xFF;
            let back = srgb_to_linear(byte as f32 / 255.0);
            assert!((back - v).abs() <= v * 0.05 + 0.0005, "{v} -> {back}");
        }
    }

    #[test]
    fn instances_carry_their_transform() {
        let raw = CubeRaw::new(Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)), Vec4::ONE);
        assert_eq!(raw.model[3], [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(raw.color, [1.0; 4]);
    }
}
