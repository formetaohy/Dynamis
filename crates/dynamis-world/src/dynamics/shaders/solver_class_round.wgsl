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

fn neighbour_blocks_round(other: u32, index: u32, round: u32) -> bool {
    if (other >= index) {
        return false;
    }
    let token = atomicLoad(&class_tokens[other]);
    return token == 0u || token_round(token) >= round;
}

fn first_lane_claims_lower(index: u32, first: u32, round: u32, live: u32) -> bool {
    let start = i32(first) - 1;
    if (start < 0) {
        return false;
    }
    let owner = first_order_bodies[u32(start)];
    var end = u32(start);
    while (end < live && first_order_bodies[end] == owner) {
        end = end + 1u;
    }
    for (var slot = u32(start); slot < end; slot = slot + 1u) {
        if (neighbour_blocks_round(first_order_blocks[slot], index, round)) {
            return true;
        }
    }
    return false;
}

fn second_lane_claims_lower(index: u32, first: u32, round: u32, live: u32) -> bool {
    let start = i32(first) - 1;
    if (start < 0) {
        return false;
    }
    let owner = second_order_bodies[u32(start)];
    var end = u32(start);
    while (end < live && second_order_bodies[end] == owner) {
        end = end + 1u;
    }
    for (var slot = u32(start); slot < end; slot = slot + 1u) {
        if (neighbour_blocks_round(second_order_blocks[slot], index, round)) {
            return true;
        }
    }
    return false;
}

fn round_claimed_by_lower(index: u32, round: u32, live: u32) -> bool {
    let first_body = block_first_body[index];
    let second_body = block_second_body[index];
    return first_lane_claims_lower(index, first_of_body[first_body], round, live)
        || first_lane_claims_lower(index, first_of_body[second_body], round, live)
        || second_lane_claims_lower(index, second_of_body[first_body], round, live)
        || second_lane_claims_lower(index, second_of_body[second_body], round, live);
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
        if (round_claimed_by_lower(index, round, live)) {
            continue;
        }
        atomicStore(&class_tokens[index], class_token(round, round));
    }
}
