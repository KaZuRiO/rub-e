struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    frustum:        array<vec4<f32>, 6>,
};

struct AABB {
    min: vec3<f32>,
    max: vec3<f32>,
}

// Mirrors StaticCubeRaw: 16 bytes, the colour packed in the last four.
struct StaticCube {
    position: vec3<f32>,
    color: u32,
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
var<storage, read> all_instances: array<StaticCube>;

@group(1) @binding(1)
var<storage, read_write> visible_indices: array<u32>;

@group(1) @binding(2)
var<storage, read_write> draw_cmd: DrawIndexedIndirectArgs;

// Chunk-visibility written by the chunk_cull pass that runs before this one.
@group(1) @binding(3)
var<storage, read> cube_chunk_ids: array<u32>;

@group(1) @binding(4)
var<storage, read> chunk_visible: array<u32>;

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
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&all_instances)) {
        return;
    }

    // 1. Chunk-level early exit — skip cubes whose entire chunk was culled.
    if chunk_visible[cube_chunk_ids[index]] == 0u {
        return;
    }

    // Static cubes are unit cubes on the axes.
    let center = all_instances[index].position;
    let aabb = AABB(center - vec3<f32>(0.5), center + vec3<f32>(0.5));

    // 2. Per-cube frustum culling
    if (intersects_aabb(aabb)) {
        let out_index = atomicAdd(&draw_cmd.instance_count, 1u);
        visible_indices[out_index] = index;
    }
}
