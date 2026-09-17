@group(0) @binding(0) var<storage, read> contacts_raw: array<Contact>;
@group(0) @binding(1) var<storage, read> valid: array<u32>;
@group(0) @binding(2) var<storage, read> ranks: array<u32>;
@group(0) @binding(3) var<storage, read> block_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(5) var<storage, read_write> contact_matched: array<u32>;
@group(0) @binding(6) var<storage, read_write> count_holder: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> refused: array<atomic<u32>>;

const BLOCK_SIZE: u32 = 256u;

fn work(index: u32) {
    if (valid[index] == 0u) {
        return;
    }
    let dest = block_offsets[index / BLOCK_SIZE] + ranks[index];
    if (dest >= arrayLength(&contacts)) {
        atomicAdd(&refused[0], 1u);
        return;
    }
    var compacted = contacts_raw[index];
    compacted.carried_normal = 0.0;
    compacted.carried_tangent = 0.0;
    contacts[dest] = compacted;
    contact_matched[dest] = 0u;
}

