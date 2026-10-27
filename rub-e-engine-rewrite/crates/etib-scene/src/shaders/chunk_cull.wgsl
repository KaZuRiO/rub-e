// Chunk-level frustum culling compute pass.
//
// One thread per chunk. Tests the chunk's pre-computed AABB against the camera
// frustum and writes 1 (visible) or 0 (culled) into chunk_visible[].
// The per-cube cull pass reads chunk_visible[] to skip entire culled chunks.

struct CameraUniform {
    view_proj:      mat4x4<f32>,
    inv_view:       mat4x4<f32>,
    inv_proj:       mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
};

struct Plane {
    normal:   vec3<f32>,
    distance: f32,
}

struct Frustum {
    planes: array<Plane, 6>,
}

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

fn normalize_plane(v: vec4<f32>) -> Plane {
    let len = length(v.xyz);
    return Plane(v.xyz / len, v.w / len);
}

fn frustum_from_view_proj(vp: mat4x4<f32>) -> Frustum {
    let m = transpose(vp);
    var planes: array<Plane, 6>;
    planes[0] = normalize_plane(m[3] + m[0]); // Left
    planes[1] = normalize_plane(m[3] - m[0]); // Right
    planes[2] = normalize_plane(m[3] + m[1]); // Bottom
    planes[3] = normalize_plane(m[3] - m[1]); // Top
    planes[4] = normalize_plane(m[3] + m[2]); // Near
    planes[5] = normalize_plane(m[3] - m[2]); // Far
    return Frustum(planes);
}

fn intersects_aabb(frustum: Frustum, aabb: AABB) -> bool {
    for (var i = 0; i < 6; i++) {
        let plane = frustum.planes[i];
        let px = select(aabb.min.x, aabb.max.x, plane.normal.x >= 0.0);
        let py = select(aabb.min.y, aabb.max.y, plane.normal.y >= 0.0);
        let pz = select(aabb.min.z, aabb.max.z, plane.normal.z >= 0.0);
        if (dot(plane.normal, vec3<f32>(px, py, pz)) + plane.distance < 0.0) {
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
    chunk_visible[idx] = u32(intersects_aabb(frustum_from_view_proj(camera.view_proj), aabb));
}
