struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    frustum:        array<vec4<f32>, 6>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
};

struct InstanceInput {
    @location(3) model_matrix_0: vec4<f32>,
    @location(4) model_matrix_1: vec4<f32>,
    @location(5) model_matrix_2: vec4<f32>,
    @location(6) model_matrix_3: vec4<f32>,
    @location(7) instance_color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Reconstruct model matrix from instance attributes
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    
    out.color = instance.instance_color.rgb;

    // Transform position to world space
    let world_pos = model_matrix * vec4<f32>(vertex.position, 1.0);
    out.world_position = world_pos.xyz;

    // Transform normal to world space (using model matrix, assuming uniform scale)
    out.world_normal = normalize((model_matrix * vec4<f32>(vertex.normal, 0.0)).xyz);

    out.clip_position = camera.view_proj * world_pos;
    return out;
}
