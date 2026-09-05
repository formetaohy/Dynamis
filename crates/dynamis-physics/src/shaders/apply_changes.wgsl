struct BodyCommand {
    kind: u32,
    slot: u32,
    extra: u32,
    _pad: u32,
    body: RigidBody,
}

@group(0) @binding(0) var<storage, read> commands: array<BodyCommand>;
@group(0) @binding(1) var<storage, read_write> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read> command_count: atomic<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x != 0u) {
        return;
    }
    let count = atomicLoad(&command_count);
    for (var i = 0u; i < count; i = i + 1u) {
        let command = commands[i];
        if (command.kind == 0u) {
            bodies[command.slot] = command.body;
        } else if (command.kind == 1u) {
            if (command.slot != command.extra) {
                bodies[command.slot] = bodies[command.extra];
            }
        } else if (command.kind == 2u) {
            var body = bodies[command.slot];
            if ((command.extra & 1u) != 0u) {
                body.position = command.body.position;
            }
            if ((command.extra & 2u) != 0u) {
                body.velocity = command.body.velocity;
            }
            if ((command.extra & 4u) != 0u) {
                body.inverse_mass = command.body.inverse_mass;
            }
            if ((command.extra & 8u) != 0u) {
                body.radius = command.body.radius;
            }
            if ((command.extra & 16u) != 0u) {
                body.restitution = command.body.restitution;
            }
            if ((command.extra & 32u) != 0u) {
                body.orientation = command.body.orientation;
            }
            if ((command.extra & 64u) != 0u) {
                body.angular_velocity = command.body.angular_velocity;
            }
            if ((command.extra & 128u) != 0u) {
                body.friction = command.body.friction;
            }
            bodies[command.slot] = body;
        } else if (command.kind == 3u) {
            var body = bodies[command.slot];
            body.force = body.force + command.body.force;
            bodies[command.slot] = body;
        } else if (command.kind == 4u) {
            var body = bodies[command.slot];
            body.torque = body.torque + command.body.torque;
            bodies[command.slot] = body;
        } else {
            var body = bodies[command.slot];
            let impulse = command.body.velocity;
            body.velocity = body.velocity + impulse * body.inverse_mass;
            if ((command.extra & 1u) != 0u) {
                let arm = command.body.position - body.position;
                body.angular_velocity =
                    body.angular_velocity + inverse_inertia(body) * cross(arm, impulse);
            }
            bodies[command.slot] = body;
        }
    }
}
