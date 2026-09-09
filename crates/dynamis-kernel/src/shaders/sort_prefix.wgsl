@group(0) @binding(0) var<storage, read_write> histogram: array<atomic<u32>, 256>;
@group(0) @binding(1) var<storage, read_write> cursor: array<atomic<u32>, 256>;

var<workgroup> scratch: array<u32, 256>;

@compute @workgroup_size(256u)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    let index = lid.x;
    scratch[index] = atomicLoad(&histogram[index]);
    workgroupBarrier();
    var value = scratch[index];
    var stride = 1u;
    for (var step = 0u; step < 8u; step = step + 1u) {
        var source = 0u;
        if (index >= stride) {
            source = scratch[index - stride];
        }
        workgroupBarrier();
        value = value + source;
        scratch[index] = value;
        workgroupBarrier();
        stride = stride * 2u;
    }
    atomicStore(&cursor[index], value - atomicLoad(&histogram[index]));
    atomicStore(&histogram[index], 0u);
}
