struct ConstraintCommandRecord {
    kind: u32,
    slot: u32,
    _pad0: u32,
    _pad1: u32,
    constraint: Constraint,
}

@group(0) @binding(0) var<storage, read> commands: array<ConstraintCommandRecord>;
@group(0) @binding(1) var<storage, read_write> constraints: array<Constraint>;
@group(0) @binding(2) var<storage, read> command_count: atomic<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x != 0u) {
        return;
    }
    let count = atomicLoad(&command_count);
    for (var i = 0u; i < count; i = i + 1u) {
        let command = commands[i];
        if (command.kind == COMMAND_CONSTRAINT_ADD) {
            constraints[command.slot] = command.constraint;
        } else if (command.kind == COMMAND_CONSTRAINT_REMOVE) {
            constraints[command.slot].kind = CONSTRAINT_INVALID;
        }
    }
}
