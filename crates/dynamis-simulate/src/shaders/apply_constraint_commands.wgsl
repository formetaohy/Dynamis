struct ConstraintCommand {
    kind: u32,
    slot: u32,
    tail: u32,
    constraint_id: u32,
    generation: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read> commands: array<ConstraintCommand>;
@group(0) @binding(1) var<storage, read_write> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(2) var<storage, read_write> command_count: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x != 0u) {
        return;
    }
    let count = atomicLoad(&command_count[0]);
    for (var i = 0u; i < count; i = i + 1u) {
        let command = commands[i];
        if (command.kind == COMMAND_CONSTRAINT_ADD) {
            constraint_runtime[command.slot] = ConstraintRuntime(
                array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                0u,
                command.constraint_id,
                command.generation,
                0u,
            );
        } else {
            let first = constraint_runtime[command.slot];
            let second = constraint_runtime[command.tail];
            constraint_runtime[command.slot] = second;
            constraint_runtime[command.tail] = first;
        }
    }
}
