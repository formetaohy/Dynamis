@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read> contacts: array<Contact>;
@group(0) @binding(3) var<storage, read> contact_count: atomic<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= atomicLoad(&contact_count)) {
        return;
    }
    let contact = contacts[index];
    var first = bodies[contact.a];
    var second = bodies[contact.b];
    let first_weight = first.inverse_mass;
    let second_weight = second.inverse_mass;
    let weight_sum = first_weight + second_weight;
    if (weight_sum == 0.0) {
        return;
    }
    let delta = second.position - first.position;
    let distance = length(delta);
    let radius_sum = first.radius + second.radius;
    let depth = radius_sum - distance;
    if (depth <= 0.0) {
        return;
    }
    var normal = vec3f(0.0, 1.0, 0.0);
    if (distance > 1e-6) {
        normal = delta / distance;
    }
    let correction = max(depth - params.slop, 0.0) * params.relaxation / weight_sum;
    first.position = first.position - normal * (correction * first_weight);
    second.position = second.position + normal * (correction * second_weight);

    let arm_first = normal * first.radius;
    let arm_second = -normal * second.radius;
    let inv_inertia_first = inverse_inertia(first);
    let inv_inertia_second = inverse_inertia(second);

    var velocity_first = first.velocity + cross(first.angular_velocity, arm_first);
    var velocity_second = second.velocity + cross(second.angular_velocity, arm_second);
    let relative_velocity = velocity_second - velocity_first;
    let normal_speed = dot(relative_velocity, normal);

    if (normal_speed < 0.0) {
        let cross_normal_first = cross(arm_first, normal);
        let cross_normal_second = cross(arm_second, normal);
        let normal_mass = weight_sum
            + dot(cross(inv_inertia_first * cross_normal_first, arm_first), normal)
            + dot(cross(inv_inertia_second * cross_normal_second, arm_second), normal);
        var restitution = max(first.restitution, second.restitution);
        if (normal_speed > -params.restitution_threshold) {
            restitution = 0.0;
        }
        let normal_impulse = -(1.0 + restitution) * normal_speed / normal_mass;
        let impulse = normal * normal_impulse;
        first.velocity = first.velocity - impulse * first_weight;
        second.velocity = second.velocity + impulse * second_weight;
        first.angular_velocity =
            first.angular_velocity - inv_inertia_first * cross(arm_first, impulse);
        second.angular_velocity =
            second.angular_velocity + inv_inertia_second * cross(arm_second, impulse);

        velocity_first = first.velocity + cross(first.angular_velocity, arm_first);
        velocity_second = second.velocity + cross(second.angular_velocity, arm_second);
        let slip = velocity_second - velocity_first
            - normal * dot(velocity_second - velocity_first, normal);
        let slip_speed = length(slip);
        if (slip_speed > 1e-6) {
            let tangent = slip / slip_speed;
            let cross_tangent_first = cross(arm_first, tangent);
            let cross_tangent_second = cross(arm_second, tangent);
            let tangent_mass = weight_sum
                + dot(cross(inv_inertia_first * cross_tangent_first, arm_first), tangent)
                + dot(cross(inv_inertia_second * cross_tangent_second, arm_second), tangent);
            let friction = sqrt(first.friction * second.friction);
            let friction_limit = friction * normal_impulse;
            let friction_impulse =
                clamp(-dot(velocity_second - velocity_first, tangent) / tangent_mass,
                    -friction_limit, friction_limit);
            let impulse_tangent = tangent * friction_impulse;
            first.velocity = first.velocity - impulse_tangent * first_weight;
            second.velocity = second.velocity + impulse_tangent * second_weight;
            first.angular_velocity =
                first.angular_velocity - inv_inertia_first * cross(arm_first, impulse_tangent);
            second.angular_velocity =
                second.angular_velocity + inv_inertia_second * cross(arm_second, impulse_tangent);
        }
    }
    bodies[contact.a] = first;
    bodies[contact.b] = second;
}
