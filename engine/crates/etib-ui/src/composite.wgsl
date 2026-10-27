// Composites the interface, drawn into a standard-range texture, onto an
// extended-range surface.
//
// The source is an sRGB texture, so sampling it returns linear values, and it
// holds colours already multiplied by their alpha. Scaling the colour by
// `sdr_white` keeps that premultiplication valid (alpha is untouched) and puts
// the interface exactly at the display's SDR white, where an interface belongs:
// menus and text should sit at paper white, not be driven into the highlights.

struct Params {
    // The value that means SDR white on the target surface.
    sdr_white: f32,
}

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);

    var out: VertexOutput;
    out.uv = vec2<f32>(x, y);
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ui = textureSample(source, source_sampler, in.uv);
    return vec4<f32>(ui.rgb * params.sdr_white, ui.a);
}
