struct BodyCommand {
    kind: u32,
    slot: u32,
    extra: u32,
    _pad: u32,
    body: RigidBody,
    collider: Collider,
}

@group(0) @binding(0) var<storage, read> commands: array<BodyCommand>;
@group(0) @binding(1) var<storage, read_write> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read_write> colliders: array<Collider>;
@group(0) @binding(3) var<storage, read> command_count: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> wake_flags: array<atomic<u32>>;

fn body_woken(body: RigidBody) -> RigidBody {
    var woken = body;
    woken.flags = woken.flags & ~BODY_SLEEPING;
    woken.sleep_timer = 0.0;
    return woken;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x != 0u) {
        return;
    }
    let count = atomicLoad(&command_count);
    for (var i = 0u; i < count; i = i + 1u) {
        let command = commands[i];
        if (command.kind == COMMAND_ADD) {
            bodies[command.slot] = command.body;
            colliders[command.slot] = command.collider;
        } else if (command.kind == COMMAND_REMOVE) {
            if (command.slot != command.extra) {
                bodies[command.slot] = bodies[command.extra];
                colliders[command.slot] = colliders[command.extra];
            }
        } else if (command.kind == COMMAND_PATCH) {
            var body = bodies[command.slot];
            if ((command.extra & PATCH_POSITION) != 0u) {
                body.position = command.body.position;
            }
            if ((command.extra & PATCH_VELOCITY) != 0u) {
                body.velocity = command.body.velocity;
            }
            if ((command.extra & PATCH_INVERSE_MASS) != 0u) {
                body.inverse_mass = command.body.inverse_mass;
            }
            if ((command.extra & PATCH_RESTITUTION) != 0u) {
                body.restitution = command.body.restitution;
            }
            if ((command.extra & PATCH_ORIENTATION) != 0u) {
                body.orientation = command.body.orientation;
            }
            if ((command.extra & PATCH_ANGULAR_VELOCITY) != 0u) {
                body.angular_velocity = command.body.angular_velocity;
            }
            if ((command.extra & PATCH_FRICTION) != 0u) {
                body.friction = command.body.friction;
            }
            if ((command.extra & PATCH_GROUP) != 0u) {
                body.collision_group = command.body.collision_group;
            }
            if ((command.extra & PATCH_MASK) != 0u) {
                body.collision_mask = command.body.collision_mask;
            }
            if ((command.extra & PATCH_KINEMATIC) != 0u) {
                body.flags = command.body.flags;
                body.inverse_mass = command.body.inverse_mass;
                body.inverse_inertia_body = command.body.inverse_inertia_body;
            }
            if ((command.extra & PATCH_COLLIDER) != 0u) {
                body.inverse_inertia_body = command.body.inverse_inertia_body;
                colliders[command.slot] = command.collider;
            }
            if ((body.flags & BODY_SLEEPING) != 0u) {
                atomicOr(&wake_flags[command.slot], 1u);
            }
            body = body_woken(body);
            bodies[command.slot] = body;
        } else if (command.kind == COMMAND_FORCE) {
            var body = bodies[command.slot];
            body.force = body.force + command.body.force;
            if ((body.flags & BODY_SLEEPING) != 0u) {
                atomicOr(&wake_flags[command.slot], 1u);
            }
            body = body_woken(body);
            bodies[command.slot] = body;
        } else if (command.kind == COMMAND_TORQUE) {
            var body = bodies[command.slot];
            body.torque = body.torque + command.body.torque;
            if ((body.flags & BODY_SLEEPING) != 0u) {
                atomicOr(&wake_flags[command.slot], 1u);
            }
            body = body_woken(body);
            bodies[command.slot] = body;
        } else if (command.kind == COMMAND_IMPULSE) {
            var body = bodies[command.slot];
            let impulse = command.body.velocity;
            body.velocity = body.velocity + impulse * body.inverse_mass;
            body.angular_velocity =
                body.angular_velocity + apply_inverse_inertia(body, cross(command.body.position - body.position, impulse));
            if ((body.flags & BODY_SLEEPING) != 0u) {
                atomicOr(&wake_flags[command.slot], 1u);
            }
            body = body_woken(body);
            bodies[command.slot] = body;
        } else if (command.kind == COMMAND_SLEEP) {
            var body = bodies[command.slot];
            body.flags = body.flags | BODY_SLEEPING;
            body.velocity = vec3f(0.0);
            body.angular_velocity = vec3f(0.0);
            body.sleep_timer = 0.0;
            atomicStore(&wake_flags[command.slot], 0u);
            bodies[command.slot] = body;
        } else if (command.kind == COMMAND_WAKE) {
            var body = bodies[command.slot];
            body.flags = body.flags & ~BODY_SLEEPING;
            body.sleep_timer = 0.0;
            atomicOr(&wake_flags[command.slot], 1u);
            bodies[command.slot] = body;
        }
    }
}
