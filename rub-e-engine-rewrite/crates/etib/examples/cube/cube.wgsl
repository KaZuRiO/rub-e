// A grid of lit cubes, one instance each, depth-tested into a scene-referred
// Rgba16Float target. Tone mapping happens later, in the post-processing pass.

struct Camera {
    view_proj: mat4x4<f32>,
    // xyz: the direction light travels towards the scene; w unused.
    sun: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Per instance: where the cube sits, how big it is, and its colour.
    @location(2) offset: vec3<f32>,
    @location(3) scale: f32,
    @location(4) color: vec4<f32>,
) -> VertexOut {
    var out: VertexOut;
    out.position = camera.view_proj * vec4(position * scale + offset, 1.0);
    out.normal = normal;
    // Alpha carries the emission multiplier, so some cubes land well above 1.0
    // and the tone curve has something to do.
    out.color = color.rgb * color.a;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let light = max(dot(normalize(in.normal), -camera.sun.xyz), 0.0);
    let ambient = 0.15;
    return vec4(in.color * (ambient + light), 1.0);
}
