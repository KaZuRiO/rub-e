// Skybox shader for rendering environment cubemap
// Renders a fullscreen triangle and samples the cubemap based on view direction

struct Camera {
    view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(1) @binding(0)
var env_map: texture_cube<f32>;

@group(1) @binding(1)
var env_sampler: sampler;

@group(1) @binding(2)
var<uniform> rotation: mat4x4<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc_pos: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Generate fullscreen triangle using vertex index
    // Vertex 0: (-1, -1), Vertex 1: (3, -1), Vertex 2: (-1, 3)
    // This creates a triangle that covers the entire screen
    let x = f32((vertex_index & 1u) << 2u) - 1.0;
    let y = f32((vertex_index & 2u) << 1u) - 1.0;

    // Position at far plane (z = 1.0 in WGPU's [0, 1] depth range)
    out.position = vec4<f32>(x, y, 1.0, 1.0);

    // Pass NDC coordinates to fragment shader
    out.ndc_pos = vec2<f32>(x, y);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Use interpolated NDC position from vertex shader
    let ndc_pos = in.ndc_pos;

    // Reconstruct ray direction from NDC position
    let clip_pos = vec4<f32>(ndc_pos, 1.0, 1.0);
    let view_pos = camera.inv_proj * clip_pos;
    let view_ray = view_pos.xyz / view_pos.w;

    // Transform to world space (w=0 for direction)
    var direction = (camera.inv_view * vec4<f32>(view_ray, 0.0)).xyz;
    
    // Apply skybox rotation
    direction = (rotation * vec4<f32>(direction, 0.0)).xyz;
    direction = normalize(direction);

    // Sample the environment cubemap
    let color = textureSample(env_map, env_sampler, direction);

    return color;
}
