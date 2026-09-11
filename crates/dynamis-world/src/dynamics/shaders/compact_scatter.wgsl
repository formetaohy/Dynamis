@group(0) @binding(0) var<storage, read> contacts_raw: array<Contact>;
@group(0) @binding(1) var<storage, read> valid: array<u32>;
@group(0) @binding(2) var<storage, read> ranks: array<u32>;
@group(0) @binding(3) var<storage, read> block_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(5) var<storage, read_write> contact_matched: array<u32>;
@group(0) @binding(6) var<storage, read_write> count_holder: array<atomic<u32>>;

const BLOCK_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let live = min(atomicLoad(&count_holder[0]), arrayLength(&contacts_raw));
    let stride = grid_stride(groups);
    for (var index = global_index(gid); index < live; index = index + stride) {
        if (valid[index] == 0u) {
            continue;
        }
        let dest = block_offsets[index / BLOCK_SIZE] + ranks[index];
        contacts[dest] = contacts_raw[index];
        contact_matched[dest] = 0u;
    }
}
