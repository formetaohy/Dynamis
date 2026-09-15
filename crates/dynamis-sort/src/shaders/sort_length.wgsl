@group(0) @binding(0) var<storage, read> keys_len: array<u32>;
@group(0) @binding(1) var<storage, read> count_holder: array<u32>;
@group(0) @binding(2) var<storage, read_write> length_holder: array<u32>;

@compute @workgroup_size(1)
fn main() {
    length_holder[0] = min(count_holder[0], arrayLength(&keys_len));
}
