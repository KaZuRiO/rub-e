struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
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

// Phong lighting model constants
const ka: f32 = 0.2; // ambient coefficient
const kd: f32 = 0.5; // diffuse coefficient
const ks: f32 = 0.4; // specular coefficient
const shininess: f32 = 32.0; // specular shininess

const light_position: vec3<f32> = vec3<f32>(0.0, -150.0, 100.0);
const light_color: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
const camera_position: vec3<f32> = vec3<f32>(0.0, 10.0, 40.0);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Normalize interpolated normal
    let normal = normalize(in.world_normal);

    // Ambient component
    let ambient = ka * light_color;

    // Diffuse component
    let light_dir = normalize(light_position - in.world_position);
    let diff = max(dot(normal, light_dir), 0.0);
    let diffuse = kd * diff * light_color;

    // Specular component (Phong)
    let view_dir = normalize(camera_position - in.world_position);
    let reflect_dir = reflect(-light_dir, normal);
    let spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
    let specular = ks * spec * light_color;

    // Combine all components
    let lighting = ambient + diffuse + specular;

    return vec4<f32>(in.color * lighting, 1.0);
}
