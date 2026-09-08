struct ConstraintCommandRecord {
    kind: u32,
    slot: u32,
    _pad0: u32,
    _pad1: u32,
    constraint: Constraint,
}

@group(0) @binding(0) var<storage, read> commands: array<ConstraintCommandRecord>;
@group(0) @binding(1) var<storage, read_write> constraints: array<Constraint>;
@group(0) @binding(2) var<storage, read> command_count: u32;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x != 0u) {
        return;
    }
    let count = command_count;
    for (var i = 0u; i < count; i = i + 1u) {
        let command = commands[i];
        if (command.kind == COMMAND_CONSTRAINT_ADD) {
            constraints[command.slot] = command.constraint;
        } else if (command.kind == COMMAND_CONSTRAINT_REMOVE) {
            constraints[command.slot].kind = CONSTRAINT_INVALID;
        } else if (command.kind == COMMAND_CONSTRAINT_PATCH) {
            var current = constraints[command.slot];
            let replacement = command.constraint;
            current.kind = replacement.kind;
            current.flags = replacement.flags;
            current.anchor_a = replacement.anchor_a;
            current.anchor_b = replacement.anchor_b;
            current.axis_a = replacement.axis_a;
            current.axis_b = replacement.axis_b;
            current.distance = replacement.distance;
            current.limit_min = replacement.limit_min;
            current.limit_max = replacement.limit_max;
            current.swing_a = replacement.swing_a;
            current.swing_b = replacement.swing_b;
            current.motor_speed = replacement.motor_speed;
            current.motor_max_force = replacement.motor_max_force;
            current.spring_frequency = replacement.spring_frequency;
            current.spring_damping_ratio = replacement.spring_damping_ratio;
            current.break_force = replacement.break_force;
            current.break_torque = replacement.break_torque;
            current.gear_ratio = replacement.gear_ratio;
            current.pulley_fixed_a = replacement.pulley_fixed_a;
            current.pulley_fixed_b = replacement.pulley_fixed_b;
            constraints[command.slot] = current;
        }
    }
}
