// Chunk-level frustum culling compute pass.
//
// One thread per chunk. Tests the chunk's pre-computed AABB against the camera
// frustum and writes 1 (visible) or 0 (culled) into chunk_visible[].
// The per-cube cull pass reads chunk_visible[] to skip entire culled chunks.
//
// A row with a count of 0 is never visible. Row 0 is kept that way: every
// free slot of the instance buffer points at it, so a slot no chunk holds is
// never drawn, and so is the row of a chunk that was emptied.

struct CameraUniform {
    view_proj:      mat4x4<f32>,
    inv_view:       mat4x4<f32>,
    inv_proj:       mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    frustum:        array<vec4<f32>, 6>,
};

struct AABB {
    min: vec3<f32>,
    max: vec3<f32>,
}

// Flat layout matching the Rust ChunkRaw repr(C) — avoids vec3 alignment issues.
// Byte offsets:  0  4  8  12 | 16 20 24 28 | 32 | 36 40 44
struct ChunkRaw {
    aabb_min_x: f32,
    aabb_min_y: f32,
    aabb_min_z: f32,
    pad0:       f32,
    aabb_max_x: f32,
    aabb_max_y: f32,
    aabb_max_z: f32,
    start_idx:  u32,
    count:      u32,
    pad1_x:     u32,
    pad1_y:     u32,
    pad1_z:     u32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<storage, read> chunks: array<ChunkRaw>;

@group(1) @binding(1)
var<storage, read_write> chunk_visible: array<u32>;

// Whether the box is at least partly inside the frustum planes the CPU
// computed for this frame (left, right, bottom, top, near, far).
fn intersects_aabb(aabb: AABB) -> bool {
    for (var i = 0; i < 6; i++) {
        let plane = camera.frustum[i];
        let p = select(aabb.min, aabb.max, plane.xyz >= vec3<f32>(0.0));
        if (dot(plane.xyz, p) + plane.w < 0.0) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&chunks) {
        return;
    }
    let c = chunks[idx];
    let aabb = AABB(
        vec3<f32>(c.aabb_min_x, c.aabb_min_y, c.aabb_min_z),
        vec3<f32>(c.aabb_max_x, c.aabb_max_y, c.aabb_max_z),
    );
    chunk_visible[idx] = u32(c.count > 0u && intersects_aabb(aabb));
}
