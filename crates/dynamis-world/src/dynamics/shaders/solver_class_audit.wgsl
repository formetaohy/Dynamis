@group(0) @binding(0) var<storage, read> class_tokens: array<u32>;
@group(0) @binding(1) var<storage, read> block_first_body: array<u32>;
@group(0) @binding(2) var<storage, read> block_second_body: array<u32>;
@group(0) @binding(3) var<storage, read> first_order_bodies: array<u32>;
@group(0) @binding(4) var<storage, read> first_order_blocks: array<u32>;
@group(0) @binding(5) var<storage, read> second_order_bodies: array<u32>;
@group(0) @binding(6) var<storage, read> second_order_blocks: array<u32>;
@group(0) @binding(7) var<storage, read> first_of_body: array<u32>;
@group(0) @binding(8) var<storage, read> second_of_body: array<u32>;
@group(0) @binding(9) var<storage, read> segments: array<u32>;
@group(0) @binding(10) var<storage, read_write> class_conflicts: array<atomic<u32>>;

fn first_lane_conflicts(index: u32, first: u32, bucket: u32, live: u32) -> u32 {
    let start = i32(first) - 1;
    if (start < 0) {
        return 0u;
    }
    let owner = first_order_bodies[u32(start)];
    var end = u32(start);
    while (end < live && first_order_bodies[end] == owner) {
        end = end + 1u;
    }
    var conflicts = 0u;
    for (var slot = u32(start); slot < end; slot = slot + 1u) {
        let other = first_order_blocks[slot];
        if (other > index && class_tokens[other] != 0u && token_class(class_tokens[other]) == bucket) {
            conflicts = conflicts + 1u;
        }
    }
    return conflicts;
}

fn second_lane_conflicts(index: u32, first: u32, bucket: u32, live: u32) -> u32 {
    let start = i32(first) - 1;
    if (start < 0) {
        return 0u;
    }
    let owner = second_order_bodies[u32(start)];
    var end = u32(start);
    while (end < live && second_order_bodies[end] == owner) {
        end = end + 1u;
    }
    var conflicts = 0u;
    for (var slot = u32(start); slot < end; slot = slot + 1u) {
        let other = second_order_blocks[slot];
        if (other > index && class_tokens[other] != 0u && token_class(class_tokens[other]) == bucket) {
            conflicts = conflicts + 1u;
        }
    }
    return conflicts;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let live = block_live();
    let stride = grid_stride(groups);
    for (var index = global_index(gid); index < live; index = index + stride) {
        let token = class_tokens[index];
        if (token == 0u) {
            continue;
        }
        let bucket = token_class(token);
        if (bucket >= SOLVER_CLASS_OVERFLOW) {
            continue;
        }
        let first_body = block_first_body[index];
        let second_body = block_second_body[index];
        let conflicts = first_lane_conflicts(index, first_of_body[first_body], bucket, live)
            + first_lane_conflicts(index, first_of_body[second_body], bucket, live)
            + second_lane_conflicts(index, second_of_body[first_body], bucket, live)
            + second_lane_conflicts(index, second_of_body[second_body], bucket, live);
        if (conflicts > 0u) {
            atomicAdd(&class_conflicts[0], conflicts);
        }
    }
}
