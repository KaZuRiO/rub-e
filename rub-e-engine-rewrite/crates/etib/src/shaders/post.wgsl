// Post-processing: maps a scene-referred (HDR) image to a display-referred one
// and applies a grade, and optionally a lookup table, on the way. Drawn as a
// single oversized triangle, like the blit pass -- WGSL has no include, so the
// vertex stage is repeated here.
//
// Two of the uniforms describe the surface being written to rather than the
// look. `headroom` is how far above SDR white the display can be driven, and
// `sdr_white` is the value that means SDR white on the surface. On an ordinary
// sRGB surface both are 1.0 and every path below collapses to "compress into
// 0..1"; on an extended-range (scRGB) surface they open the top end up.

struct Grade {
    // Colour filter, multiplied in before the tone curve. w is unused.
    tint: vec4<f32>,
    // Linear exposure multiplier, 2^stops.
    exposure: f32,
    contrast: f32,
    saturation: f32,
    // One of the TONEMAP_* constants below.
    tonemap: u32,
    // Multiple of SDR white the display can show. 1.0 on a standard-range one.
    headroom: f32,
    // The value that means SDR white on this surface. 1.0 on a standard-range
    // one; on scRGB, the OS's SDR white level over scRGB's 80-nit reference.
    sdr_white: f32,
    // Entries per side of the lookup table, or 0.0 when there is none, which is
    // how this shader knows to skip that stage.
    lut_size: f32,
    // How much of the table's result to mix in, 0..1.
    lut_strength: f32,
}

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> grade: Grade;
// A cube of output colours indexed by an input colour. Neutral and unused when
// `lut_size` is 0.0; it is still bound, because a binding cannot be left empty.
@group(0) @binding(3) var lut: texture_3d<f32>;

const TONEMAP_NONE: u32 = 0u;
const TONEMAP_REINHARD: u32 = 1u;
const TONEMAP_ACES: u32 = 2u;

// Linear mid-grey. Contrast pivots here so that scaling it brightens highlights
// and deepens shadows without shifting the exposure of the image as a whole.
const MID_GREY: f32 = 0.18;

const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

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

// Reinhard, applied to luminance rather than per channel: the ratio between the
// channels survives, so bright areas desaturate far less than they would if
// each channel were compressed on its own.
//
// `headroom` is the value the curve saturates at, so on a display with room
// above SDR white the highlights roll off towards that instead of towards
// white. At a headroom of 1.0 this is plain Reinhard.
fn tonemap_reinhard(color: vec3<f32>, headroom: f32) -> vec3<f32> {
    return color / (1.0 + dot(color, LUMA) / headroom);
}

// Narkowicz's curve fit to the ACES filmic tone mapper: an S-curve with a toe
// that lifts shadows gently and a long shoulder that rolls highlights off
// instead of clipping them.
fn tonemap_aces(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp(
        (color * (a * color + b)) / (color * (c * color + d) + e),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

// Brings a colour whose brightest channel overshoots the display into range by
// scaling all three by the same factor.
//
// Clamping each channel on its own would be wrong here: a saturated colour has
// one channel far above the others, so clamping cuts that one alone, changes
// the ratios between them, and drags the result towards white. Ember at three
// times its base brightness maps to (1.68, 0.25, 0.06) on a display with 1.21
// of headroom; clamping gives (1.21, 0.25, 0.06), which is a duller orange,
// while scaling gives (1.21, 0.18, 0.04) -- the same orange, slightly dimmer.
// Trading a little brightness for the hue is the better bargain, and the one
// the tone curve already assumed it was making.
fn fit_to_headroom(color: vec3<f32>, headroom: f32) -> vec3<f32> {
    let peak = max(color.r, max(color.g, color.b));
    // A colour that already fits gets a scale of exactly 1.0 and comes back
    // untouched, so this only ever acts on overshooting highlights. The epsilon
    // is there for black, where the division would otherwise be 0/0.
    let scale = min(1.0, headroom / max(peak, 1e-6));
    return color * scale;
}

// The sRGB transfer function and its inverse, which the lookup below needs and
// nothing else here does: every other stage works in linear light, and the
// surface applies its own encoding on write.
fn encode(color: vec3<f32>) -> vec3<f32> {
    let low = color * 12.92;
    let high = 1.055 * pow(color, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, color <= vec3<f32>(0.0031308));
}

fn decode(color: vec3<f32>) -> vec3<f32> {
    let low = color / 12.92;
    let high = pow((color + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, color <= vec3<f32>(0.04045));
}

// Looks a colour up in the table, which holds a grade sampled on a regular
// grid; the hardware's trilinear filtering interpolates between the entries.
//
// Three details decide whether the result is the look its author saw.
//
// The first is what space the table is indexed in. A grading tool shows its
// user the signal a display is fed -- gamma-encoded -- and a table exported
// from one is a function of that signal, not of linear light. So the colour is
// encoded on the way in and the table's answer decoded on the way out. Skipping
// that pair would not be a subtle error: mid-grey sits at 0.18 in linear and
// 0.46 encoded, so every entry of the table would be read at the wrong place,
// and the shadows a look lifts would be crushed instead.
//
// The second is where the entries sit: a table of `size` entries spans 0..1
// with one at each end, so the coordinate of the nth is (n + 0.5) / size -- the
// centre of its texel -- and a colour of 0.0 or 1.0 has to land exactly on the
// first or last of them rather than half a texel outside, where clamping would
// flatten the ends of the grade.
//
// The third is what to do with a colour the cube does not cover. The table is
// defined over 0..1, but on a display with headroom a highlight can sit above
// that, and squashing it into the cube would throw away the range the tone
// curve just worked to keep. So the lookup is done on the part that fits, and
// whatever sat above SDR white is carried over the result -- the graded colour,
// still with its highlight on top.
fn apply_lut(color: vec3<f32>, size: f32, strength: f32) -> vec3<f32> {
    let inside = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    let coord = (encode(inside) * (size - 1.0) + 0.5) / size;
    let graded = decode(textureSample(lut, source_sampler, coord).rgb) + (color - inside);
    return mix(color, graded, strength);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Scene-referred: the compute pass writes into a float target, so values
    // here run well past 1.0.
    var color = textureSample(source, source_sampler, in.uv).rgb;

    // 1. Exposure and colour filter, both of which belong before the curve:
    //    they decide what the tone mapper is given to compress.
    color *= grade.exposure * grade.tint.rgb;

    // 2. Tone curve. Everything after this point is display-referred, where 1.0
    //    is SDR white and the display can take up to `headroom`.
    switch grade.tonemap {
        case TONEMAP_REINHARD: {
            color = tonemap_reinhard(color, grade.headroom);
        }
        case TONEMAP_ACES: {
            // The fit targets standard range and saturates at SDR white; it
            // leaves any headroom unused rather than inventing an extension.
            color = tonemap_aces(color);
        }
        default: {
            // No curve: whatever does not fit clips, at the display's peak
            // rather than at SDR white.
            color = clamp(color, vec3<f32>(0.0), vec3<f32>(grade.headroom));
        }
    }

    // 3. Grade the mapped image.
    color = max((color - MID_GREY) * grade.contrast + MID_GREY, vec3<f32>(0.0));
    color = max(mix(vec3<f32>(dot(color, LUMA)), color, grade.saturation), vec3<f32>(0.0));

    // 4. Look the result up in the table, when there is one. It comes last
    //    because a table is a finished look, baked over an image that has
    //    already been tone mapped and graded -- which is the image a colourist
    //    had in front of them when they made it.
    if grade.lut_size > 0.0 {
        color = apply_lut(max(color, vec3<f32>(0.0)), grade.lut_size, grade.lut_strength);
    }

    // 5. Fit whatever is left above the display's ceiling, which the curve does
    //    not guarantee on its own: Reinhard compresses luminance rather than
    //    each channel, and contrast, saturation and the table all run after it.
    color = fit_to_headroom(max(color, vec3<f32>(0.0)), grade.headroom);

    // 6. Place the finished image on the surface. Values stay linear either
    //    way: an sRGB surface has the hardware apply the transfer function on
    //    write, and an extended-range one carries a linear signal by
    //    definition. All that changes is where white sits, and how far above it
    //    anything is allowed to go.
    return vec4<f32>(color * grade.sdr_white, 1.0);
}
