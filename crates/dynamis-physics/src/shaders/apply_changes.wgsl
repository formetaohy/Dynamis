struct RigidBody {
    position: vec3f,
    _pad0: f32,
    velocity: vec3f,
    _pad1: f32,
    inverse_mass: f32,
    radius: f32,
    restitution: f32,
    _pad2: f32,
}

struct BodyCommand {
    kind: u32,
    slot: u32,
    extra: u32,
    _pad: u32,
    record: RigidBody,
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
            bodies[command.slot] = command.record;
        } else if (command.kind == 1u) {
            if (command.slot != command.extra) {
                bodies[command.slot] = bodies[command.extra];
            }
        } else {
            var body = bodies[command.slot];
            if ((command.extra & 1u) != 0u) {
                body.position = command.record.position;
            }
            if ((command.extra & 2u) != 0u) {
                body.velocity = command.record.velocity;
            }
            if ((command.extra & 4u) != 0u) {
                body.inverse_mass = command.record.inverse_mass;
            }
            if ((command.extra & 8u) != 0u) {
                body.radius = command.record.radius;
            }
            if ((command.extra & 16u) != 0u) {
                body.restitution = command.record.restitution;
            }
            bodies[command.slot] = body;
        }
    }
}
