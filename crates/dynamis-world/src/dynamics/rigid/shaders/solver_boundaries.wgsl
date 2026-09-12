@group(0) @binding(0) var<storage, read> segments: array<u32>;
@group(0) @binding(1) var<storage, read> a_bodies: array<u32>;
@group(0) @binding(2) var<storage, read> b_bodies: array<u32>;
@group(0) @binding(3) var<storage, read> a_payload: array<u32>;
@group(0) @binding(4) var<storage, read_write> first_a: array<u32>;
@group(0) @binding(5) var<storage, read_write> first_b: array<u32>;
@group(0) @binding(6) var<storage, read> contacts: array<Contact>;
@group(0) @binding(7) var<storage, read> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(8) var<storage, read_write> block_counts: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> contact_counts: array<atomic<u32>>;

fn extent() -> u32 {
    return segments[SOLVER_BLOCK_CONTACT] + segments[SOLVER_BLOCK_CONSTRAINT];
}

fn work(index: u32) {
    let contact_blocks = segments[SOLVER_BLOCK_CONTACT];
    let first_body = a_bodies[index];
    if (index == 0u || a_bodies[index - 1u] != first_body) {
        first_a[first_body] = index + 1u;
    }
    let second_body = b_bodies[index];
    if (index == 0u || b_bodies[index - 1u] != second_body) {
        first_b[second_body] = index + 1u;
    }
    let block = a_payload[index];
    var resolves = false;
    if (block < contact_blocks) {
        resolves = contact_block_resolves(contacts[block]);
    } else {
        resolves = constraint_runtime[block - contact_blocks].broken == 0u;
    }
    if (!resolves) {
        return;
    }
    atomicAdd(&block_counts[first_body], 1u);
    atomicAdd(&block_counts[second_body], 1u);
    if (block < contact_blocks) {
        atomicAdd(&contact_counts[first_body], 1u);
        atomicAdd(&contact_counts[second_body], 1u);
    }
}

