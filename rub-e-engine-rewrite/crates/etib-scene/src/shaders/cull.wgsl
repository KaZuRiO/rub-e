struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
};

struct Plane {
    normal: vec3<f32>,
    distance: f32,
}

struct Frustum {
    planes: array<Plane, 6>,
}

struct AABB {
    min: vec3<f32>,
    max: vec3<f32>,
}

struct CubeRaw {
    model: mat4x4<f32>,
    color: vec4<f32>,
}

struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<storage, read> all_instances: array<CubeRaw>;

@group(1) @binding(1)
var<storage, read_write> visible_instances: array<CubeRaw>;

@group(1) @binding(2)
var<storage, read_write> draw_cmd: DrawIndexedIndirectArgs;

// Chunk-visibility written by the chunk_cull pass that runs before this one.
@group(1) @binding(3)
var<storage, read> cube_chunk_ids: array<u32>;

@group(1) @binding(4)
var<storage, read> chunk_visible: array<u32>;

fn normalize_plane(v: vec4<f32>) -> Plane {
    let len = length(v.xyz);
    return Plane(v.xyz / len, v.w / len);
}

fn frustum_from_view_proj(view_proj: mat4x4<f32>) -> Frustum {
    // Transpose to get rows as vectors
    let m = transpose(view_proj);
    let row0 = m[0];
    let row1 = m[1];
    let row2 = m[2];
    let row3 = m[3];

    var planes: array<Plane, 6>;
    planes[0] = normalize_plane(row3 + row0); // Left
    planes[1] = normalize_plane(row3 - row0); // Right
    planes[2] = normalize_plane(row3 + row1); // Bottom
    planes[3] = normalize_plane(row3 - row1); // Top
    planes[4] = normalize_plane(row3 + row2); // Near
    planes[5] = normalize_plane(row3 - row2); // Far
    return Frustum(planes);
}

fn plane_distance(plane: Plane, point: vec3<f32>) -> f32 {
    return dot(plane.normal, point) + plane.distance;
}

fn intersects_aabb(frustum: Frustum, aabb: AABB) -> bool {
    for (var i = 0; i < 6; i++) {
        let plane = frustum.planes[i];
        let px = select(aabb.min.x, aabb.max.x, plane.normal.x >= 0.0);
        let py = select(aabb.min.y, aabb.max.y, plane.normal.y >= 0.0);
        let pz = select(aabb.min.z, aabb.max.z, plane.normal.z >= 0.0);
        if (plane_distance(plane, vec3<f32>(px, py, pz)) < 0.0) {
            return false;
        }
    }
    return true;
}

fn get_aabb_from_instance(model: mat4x4<f32>) -> AABB {
    let world_center = model[3].xyz;
    let right = model[0].xyz;
    let up = model[1].xyz;
    let forward = model[2].xyz;
    let new_extent_x = (abs(right.x) + abs(up.x) + abs(forward.x)) * 0.5;
    let new_extent_y = (abs(right.y) + abs(up.y) + abs(forward.y)) * 0.5;
    let new_extent_z = (abs(right.z) + abs(up.z) + abs(forward.z)) * 0.5;
    let new_extent = vec3<f32>(new_extent_x, new_extent_y, new_extent_z);
    return AABB(world_center - new_extent, world_center + new_extent);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&all_instances)) {
        return;
    }

    // 1. Chunk-level early exit — skip cubes whose entire chunk was culled.
    if chunk_visible[cube_chunk_ids[index]] == 0u {
        return;
    }

    let instance = all_instances[index];
    let frustum = frustum_from_view_proj(camera.view_proj);
    let aabb = get_aabb_from_instance(instance.model);

    // 2. Per-cube frustum culling
    if (intersects_aabb(frustum, aabb)) {
        let out_index = atomicAdd(&draw_cmd.instance_count, 1u);
        visible_instances[out_index] = instance;
    }
}
