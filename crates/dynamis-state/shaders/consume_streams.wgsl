@group(0) @binding(0) var<storage, read_write> row_streams: RowStreams;
@group(0) @binding(1) var<storage, read_write> counters: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x != 0u) {
        return;
    }
    row_streams = RowStreams(0u, 0u, 0u, 0u, 0u);
    let step = COUNTER_STEP * COUNTER_STRIDE_WORDS;
    atomicStore(&counters[step], atomicLoad(&counters[step]) + 1u);
}
