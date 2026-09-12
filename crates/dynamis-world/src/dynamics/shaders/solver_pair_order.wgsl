@group(0) @binding(0) var<storage, read> a_payload: array<u32>;
@group(0) @binding(1) var<storage, read> block_second_body: array<u32>;
@group(0) @binding(2) var<storage, read_write> b_bodies: array<u32>;
@group(0) @binding(3) var<storage, read_write> b_blocks: array<u32>;
@group(0) @binding(4) var<storage, read> block_count: array<u32>;

fn extent() -> u32 {
    return block_count[0];
}

fn work(index: u32) {
    b_bodies[index] = block_second_body[a_payload[index]];
    b_blocks[index] = index;
}

