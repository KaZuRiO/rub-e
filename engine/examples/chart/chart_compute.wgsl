// A tone-mapping test chart, drawn scene-referred into a float storage texture.
//
// Nothing here is tone mapped or graded: every value is what the scene is,
// which for most of the chart means well above 1.0. What reaches the display is
// entirely the post-processing pass's doing, which is the point -- switching the
// tone curve changes the picture and nothing else does.
//
// Four zones, top to bottom:
//
//   1. A grey wedge in stops, -6 to +5 around mid-grey, with a white notch
//      marking mid-grey itself. Shows where each curve puts the midtones and
//      where it stops telling patches apart.
//   2. Four patches at 1, 2, 4 and 8 times SDR white. On a standard-range
//      surface they collapse into one white block; on an extended-range one
//      they become a staircase -- unless the curve saturates at white.
//   3. Near-saturated hues at rising intensity. Per-channel clipping marches
//      them towards white, a luminance curve holds their hue, and the filmic
//      fit desaturates them smoothly.
//   4. A continuous version of zone 1, where the shape of the roll-off shows:
//      hard edges, gradients, and any kink where a curve changes behaviour.

struct Params {
    // Fraction of the width taken by the control panel. The chart lays itself
    // out to the right of it, so the two never overlap.
    inset: f32,
}

@group(0) @binding(0) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> params: Params;

const BACKGROUND: vec3<f32> = vec3<f32>(0.006, 0.007, 0.010);
const MID_GREY: f32 = 0.18;

// Zone bounds in normalised coordinates: (y0, y1) here, with x spanning the
// area left over beside the panel. Origin is top left.
const WEDGE: vec2<f32> = vec2<f32>(0.06, 0.22);
const LEVELS: vec2<f32> = vec2<f32>(0.28, 0.44);
const HUES: vec2<f32> = vec2<f32>(0.50, 0.76);
const RAMP: vec2<f32> = vec2<f32>(0.82, 0.96);

// Breathing room between the panel and the chart, and at the right edge.
const MARGIN: f32 = 0.03;

/// Turns a zone's vertical extent into a rectangle in the area beside the panel.
fn zone(rows: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(params.inset + MARGIN, rows.x, 1.0 - MARGIN, rows.y);
}

// Stops either side of mid-grey covered by the wedge and the ramp.
const LOW_STOP: f32 = -6.0;
const HIGH_STOP: f32 = 5.0;
const WEDGE_STEPS: f32 = 12.0;

// Fraction of each cell left as a gutter, so patches read as separate.
const GUTTER: f32 = 0.08;

fn inside(rect: vec4<f32>, uv: vec2<f32>) -> bool {
    return uv.x >= rect.x && uv.x <= rect.z && uv.y >= rect.y && uv.y <= rect.w;
}

/// Position within a zone, 0..1 on each axis.
fn local(rect: vec4<f32>, uv: vec2<f32>) -> vec2<f32> {
    return (uv - rect.xy) / (rect.zw - rect.xy);
}

/// Whether a cell coordinate falls in the gutter rather than on the patch.
fn in_gutter(cell: f32) -> bool {
    let edge = GUTTER * 0.5;
    return cell < edge || cell > 1.0 - edge;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output_tex);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(f32(dims.x), f32(dims.y));
    var color = BACKGROUND;

    // 1. Grey wedge, one patch per stop.
    let wedge = zone(WEDGE);
    if inside(wedge, uv) {
        let p = local(wedge, uv);
        let step = floor(p.x * WEDGE_STEPS);
        let cell = fract(p.x * WEDGE_STEPS);
        if !in_gutter(cell) {
            let stop = LOW_STOP + step;
            color = vec3<f32>(MID_GREY * exp2(stop));

            // Mark mid-grey, so the eye has somewhere to anchor.
            if stop == 0.0 && p.y > 0.82 && abs(cell - 0.5) < 0.22 {
                color = vec3<f32>(1.0);
            }
        }
    }

    // 2. Multiples of SDR white.
    let levels = zone(LEVELS);
    if inside(levels, uv) {
        let p = local(levels, uv);
        let step = floor(p.x * 4.0);
        let cell = fract(p.x * 4.0);
        if !in_gutter(cell) {
            color = vec3<f32>(exp2(step));
        }
    }

    // 3. Hues across, intensity down.
    let hue_grid = zone(HUES);
    if inside(hue_grid, uv) {
        let p = local(hue_grid, uv);
        let col = floor(p.x * 6.0);
        let row = floor(p.y * 4.0);
        let cell_x = fract(p.x * 6.0);
        let cell_y = fract(p.y * 4.0);
        if !in_gutter(cell_x) && !in_gutter(cell_y) {
            var hues = array<vec3<f32>, 6>(
                vec3<f32>(1.0, 0.0, 0.0),
                vec3<f32>(1.0, 1.0, 0.0),
                vec3<f32>(0.0, 1.0, 0.0),
                vec3<f32>(0.0, 1.0, 1.0),
                vec3<f32>(0.0, 0.0, 1.0),
                vec3<f32>(1.0, 0.0, 1.0),
            );
            // Pulled a little off the gamut corners: a fully pure primary has
            // nothing in its other two channels, so clipping could not shift
            // its hue and zone 3 would have nothing to show.
            let hue = mix(hues[u32(col)], vec3<f32>(1.0), 0.12);
            color = hue * exp2(row);
        }
    }

    // 4. The same range as zone 1, continuous.
    let ramp = zone(RAMP);
    if inside(ramp, uv) {
        let p = local(ramp, uv);
        color = vec3<f32>(MID_GREY * exp2(mix(LOW_STOP, HIGH_STOP, p.x)));
    }

    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}
