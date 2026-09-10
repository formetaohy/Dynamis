@group(0) @binding(0) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(1) var<storage, read_write> scratch: array<BodyState>;
@group(0) @binding(2) var<storage, read> move_src: array<u32>;
@group(0) @binding(3) var<storage, read> move_fresh: array<u32>;
@group(0) @binding(4) var<storage, read> fresh_states: array<BodyState>;
@group(0) @binding(5) var<uniform> params: SimParams;

const CLEAR_ROW: u32 = 0xFFFFFFFFu;
const NO_FRESH: u32 = 0xFFFFFFFFu;

/// The final row layout lands in the scratch lanes first, so no gather ever reads
/// a row another lane is writing.
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= params.body_count) {
        return;
    }
    let source = move_src[index];
    let fresh_index = move_fresh[index];
    if (source == CLEAR_ROW) {
        scratch[index] = BodyState();
        return;
    }
    if (fresh_index != NO_FRESH) {
        scratch[index] = fresh_states[fresh_index];
        return;
    }
    scratch[index] = body_states[source];
}
