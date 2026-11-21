// The cube fragment shader, shared by the dynamic (cube.wgsl) and static
// (static_cube.wgsl) cube pipelines. It is appended to either, which declare
// `VertexOutput` and `camera`.

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
