@group(0) @binding(0) var<storage, read> count: array<u32>;
@group(0) @binding(1) var<storage, read_write> args: array<u32>;

@compute @workgroup_size(1u)
fn main() {
    let elements = count[0];
    args[0] = (elements + 63u) / 64u;
    args[1] = 1u;
    args[2] = 1u;
    args[3] = 0u;
    args[4] = (elements + 255u) / 256u;
    args[5] = 1u;
    args[6] = 1u;
    args[7] = 0u;
}
