const BIN_COUNT: u32 = 256u;
const SHIFT: u32 = __SHIFT__u;

@group(0) @binding(0) var<storage, read> keys_lo: array<u32>;
@group(0) @binding(1) var<storage, read> keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>, BIN_COUNT>;
@group(0) @binding(3) var<storage, read> count_holder: array<u32>;
@group(0) @binding(4) var<storage, read_write> positions: array<u32>;

fn digit_at(index: u32) -> u32 {
    let key = select(keys_hi[index], keys_lo[index], SHIFT < 32u);
    return (key >> (SHIFT & 31u)) & 0xFFu;
}

fn prefix_of(digit: u32) -> u32 {
    var prefix = 0u;
    for (var i = 0u; i < digit; i = i + 1u) {
        prefix = prefix + atomicLoad(&histogram[i]);
    }
    return prefix;
}

@compute @workgroup_size(BIN_COUNT)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    let count = max(count_holder[0], 1u);
    let digit = lid.x;
    let prefix = prefix_of(digit);
    var rank = 0u;
    for (var index = 0u; index < count; index = index + 1u) {
        if (digit_at(index) == digit) {
            positions[index] = prefix + rank;
            rank = rank + 1u;
        }
    }
}
