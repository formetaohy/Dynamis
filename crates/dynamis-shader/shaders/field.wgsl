fn field_reaches(field: Field, center: vec3f, reach: vec2u) -> bool {
    if (!filters_intersect(vec2u(field.collision_group, field.collision_mask), reach)) {
        return false;
    }
    if (field.region == FIELD_REGION_GLOBAL) {
        return true;
    }
    let offset = center - field.position;
    if (field.region == FIELD_REGION_SPHERE) {
        return dot(offset, offset) <= field.radius * field.radius;
    }
    let local = quat_rotate(quat_conjugate(field.orientation), offset);
    return all(abs(local) <= field.half_extents);
}

fn field_acceleration(field: Field, center: vec3f, gravity: vec3f) -> vec3f {
    var acceleration = field.push - gravity * field.buoyancy;
    let offset = center - field.position;
    if (field.pull != 0.0) {
        let distance = length(offset);
        if (distance > 0.0) {
            acceleration = acceleration - offset * (field.pull / distance);
        }
    }
    if (field.swirl != 0.0) {
        acceleration = acceleration + cross(field.axis, offset) * field.swirl;
    }
    return acceleration;
}

fn field_drag(field: Field, velocity: vec3f, dt: f32) -> vec3f {
    let relative = velocity - field.medium;
    let linear = relative * (field.linear_drag * dt / (1.0 + field.linear_drag * dt));
    let quadratic = relative * min(field.quadratic_drag * length(relative) * dt, 1.0);
    return -linear - quadratic;
}

struct FieldForce {
    acceleration: vec3f,
    velocity: vec3f,
    spin: f32,
}

fn field_force(
    field: Field,
    center: vec3f,
    velocity: vec3f,
    gravity: vec3f,
    reach: vec2u,
    dt: f32,
) -> FieldForce {
    var force: FieldForce;
    force.acceleration = vec3f(0.0);
    force.velocity = vec3f(0.0);
    force.spin = 1.0;
    if (!field_reaches(field, center, reach)) {
        return force;
    }
    force.acceleration = field_acceleration(field, center, gravity);
    force.velocity = field_drag(field, velocity, dt);
    force.spin = 1.0 / (1.0 + field.angular_drag * dt);
    return force;
}
