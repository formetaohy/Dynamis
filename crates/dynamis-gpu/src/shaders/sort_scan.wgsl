const BIN_COUNT: u32 = 256u;

@group(0) @binding(0) var<storage, read_write> histogram: array<atomic<u32>, BIN_COUNT>;
@group(0) @binding(1) var<storage, read_write> buckets: array<atomic<u32>, BIN_COUNT>;

@compute @workgroup_size(BIN_COUNT)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    let index = lid.x;
    var cumulative = 0u;
    for (var i = 0u; i <= index; i = i + 1u) {
        cumulative = cumulative + atomicLoad(&histogram[i]);
    }
    atomicStore(&buckets[index], cumulative);
}
