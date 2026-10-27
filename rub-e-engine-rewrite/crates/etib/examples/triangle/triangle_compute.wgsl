// Compute-shader rasterizer: draws a single rotating triangle into a storage
// texture. There is no vertex/fragment stage here -- every invocation owns one
// texel and decides for itself whether that texel is covered by the triangle.
//
// The target is a float texture and what is written to it is scene-referred:
// the fill runs past 1.0 and is brought back into display range later, by the
// post-processing pass.

struct Params {
    // Render target size in texels.
    resolution: vec2<f32>,
    // Rotation angle in radians.
    angle: f32,
    // Circumradius of the triangle, in the aspect-corrected space below.
    radius: f32,
    // Triangle fill colour, linear RGB (the CPU converts from sRGB).
    color: vec4<f32>,
    // How brightly the fill burns, as a multiplier on the colour. Above 1.0 the
    // triangle is brighter than the display can show, which is the point: the
    // post-processing pass is what brings it back into range.
    intensity: f32,
}

@group(0) @binding(0) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> params: Params;

const TAU: f32 = 6.283185307179586;
const BACKGROUND: vec3<f32> = vec3<f32>(0.011, 0.013, 0.020);

// Twice the signed area of the triangle (a, b, p). Positive when p lies to the
// left of the directed edge a -> b.
fn edge_function(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> f32 {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// Perpendicular distance from p to the line a -> b, signed like edge_function.
fn edge_distance(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> f32 {
    return edge_function(a, b, p) / max(length(b - a), 1e-6);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output_tex);

    // The dispatch is rounded up to whole workgroups, so the tail invocations
    // fall outside the texture and must not write.
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let resolution = vec2<f32>(f32(dims.x), f32(dims.y));
    let aspect = resolution.x / resolution.y;

    // Texel centre -> [-1, 1] clip-ish space, corrected for aspect ratio and
    // flipped so +y points up (texture rows run top to bottom).
    let ndc = ((vec2<f32>(gid.xy) + 0.5) / resolution) * 2.0 - 1.0;
    let p = vec2<f32>(ndc.x * aspect, -ndc.y);

    // World-space size of one texel, used to antialias the edges by exactly one
    // pixel regardless of window size.
    let texel = 2.0 * aspect / resolution.x;

    // Three vertices evenly spaced around a circle, rotated by `angle`.
    var verts: array<vec2<f32>, 3>;
    for (var i = 0u; i < 3u; i++) {
        let theta = params.angle + f32(i) * (TAU / 3.0);
        verts[i] = vec2<f32>(cos(theta), sin(theta)) * params.radius;
    }

    // Normalise winding so "inside" is always the positive half-space.
    var winding = 1.0;
    if edge_function(verts[0], verts[1], verts[2]) < 0.0 {
        winding = -1.0;
    }

    let d0 = edge_distance(verts[0], verts[1], p) * winding;
    let d1 = edge_distance(verts[1], verts[2], p) * winding;
    let d2 = edge_distance(verts[2], verts[0], p) * winding;

    // Signed distance to the triangle boundary; positive inside.
    let signed_distance = min(d0, min(d1, d2));
    let coverage = smoothstep(-texel, texel, signed_distance);

    // Gentle radial vignette so the otherwise flat background has some depth.
    let vignette = 1.0 - 0.35 * smoothstep(0.2, 1.6, length(p));
    let background = BACKGROUND * vignette;

    // Shade the fill from the centroid outwards. This keeps the hue the user
    // picked intact while still reading as a lit surface.
    let centre_bias = 1.0 - smoothstep(0.0, params.radius, length(p));
    let fill = params.color.rgb * mix(0.72, 1.0, centre_bias) * params.intensity;

    let rgb = mix(background, fill, coverage);
    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(rgb, 1.0));
}
