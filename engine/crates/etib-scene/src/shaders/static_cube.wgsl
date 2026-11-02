// Static cubes, drawn from storage rather than from an instance buffer: the
// culling pass lists the index of each visible cube in `visible`, and instance
// `i` of the indirect draw is the cube `statics[visible[i]]`. The fragment
// shader comes from shading.wgsl, appended to this file.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    frustum: array<vec4<f32>, 6>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
};

// Mirrors StaticCubeRaw: 16 bytes, the colour packed in the last four.
struct StaticCube {
    position: vec3<f32>,
    color: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<storage, read> statics: array<StaticCube>;

@group(1) @binding(1)
var<storage, read> visible: array<u32>;

// The sRGB transfer function undone, per channel.
fn srgb_to_linear(encoded: vec3<f32>) -> vec3<f32> {
    let low = encoded / 12.92;
    let high = pow((encoded + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, encoded <= vec3<f32>(0.04045));
}

@vertex
fn vs_main(vertex: VertexInput, @builtin(instance_index) instance: u32) -> VertexOutput {
    let cube = statics[visible[instance]];

    var out: VertexOutput;
    out.color = srgb_to_linear(unpack4x8unorm(cube.color).rgb);
    let world_pos = vec4<f32>(vertex.position + cube.position, 1.0);
    out.world_position = world_pos.xyz;
    // A translation leaves normals as they are.
    out.world_normal = vertex.normal;
    out.clip_position = camera.view_proj * world_pos;
    return out;
}
