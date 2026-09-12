@group(0) @binding(0) var<storage, read_write> class_tokens: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> block_first_body: array<u32>;
@group(0) @binding(2) var<storage, read> block_second_body: array<u32>;
@group(0) @binding(3) var<storage, read> first_order_bodies: array<u32>;
@group(0) @binding(4) var<storage, read> first_order_blocks: array<u32>;
@group(0) @binding(5) var<storage, read> second_order_bodies: array<u32>;
@group(0) @binding(6) var<storage, read> second_order_blocks: array<u32>;
@group(0) @binding(7) var<storage, read> first_of_body: array<u32>;
@group(0) @binding(8) var<storage, read> second_of_body: array<u32>;
@group(0) @binding(9) var<storage, read> segments: array<u32>;
@group(0) @binding(10) var<storage, read> current_round: array<u32>;
@group(0) @binding(11) var<storage, read_write> class_counts: array<atomic<u32>>;

fn claims_current_round(other: u32, index: u32, round: u32) -> bool {
    if (other >= index) {
        return false;
    }
    let token = atomicLoad(&class_tokens[other]);
    return token == 0u || token_round(token) == round;
}

fn class_conflict(index: u32, round: u32, live: u32) -> bool {
    let first_start = i32(first_of_body[block_first_body[index]]) - 1;
    if (first_start >= 0) {
        var end = u32(first_start);
        let body = first_order_bodies[end];
        while (end < live && first_order_bodies[end] == body) {
            end = end + 1u;
        }
        for (var slot = u32(first_start); slot < end; slot = slot + 1u) {
            if (claims_current_round(first_order_blocks[slot], index, round)) {
                return true;
            }
        }
    }
    let second_start = i32(second_of_body[block_second_body[index]]) - 1;
    if (second_start >= 0) {
        var end = u32(second_start);
        let body = second_order_bodies[end];
        while (end < live && second_order_bodies[end] == body) {
            end = end + 1u;
        }
        for (var slot = u32(second_start); slot < end; slot = slot + 1u) {
            if (claims_current_round(second_order_blocks[slot], index, round)) {
                return true;
            }
        }
    }
    return false;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let live = block_live();
    let round = current_round[0];
    let stride = grid_stride(groups);
    for (var index = global_index(gid); index < live; index = index + stride) {
        if (atomicLoad(&class_tokens[index]) != 0u) {
            continue;
        }
        if (class_conflict(index, round, live)) {
            continue;
        }
        atomicStore(&class_tokens[index], class_token(round, round));
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn settle(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let live = block_live();
    let round = current_round[0];
    let stride = grid_stride(groups);
    for (var index = global_index(gid); index < live; index = index + stride) {
        var token = atomicLoad(&class_tokens[index]);
        if (token == 0u) {
            let bucket = select(round, SOLVER_CLASS_OVERFLOW, class_conflict(index, round, live));
            token = class_token(round, bucket);
            atomicStore(&class_tokens[index], token);
        }
        atomicAdd(&class_counts[token_class(token)], 1u);
    }
}
