// Copies the compute shader's offscreen texture onto the swapchain image using
// a single oversized triangle, which avoids the diagonal seam a two-triangle
// quad can produce and needs no vertex buffer.

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Emits (0,0), (2,0), (0,2) in UV space, which covers the unit square.
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);

    var out: VertexOutput;
    out.uv = vec2<f32>(x, y);
    // UV origin is top-left but NDC +y is up, so y is negated rather than
    // simply remapped.
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // The source is a linear (non-sRGB) texture and the surface is sRGB, so the
    // hardware applies the transfer function on write. Nothing to do here.
    return textureSample(source, source_sampler, in.uv);
}
