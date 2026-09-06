@group(0) @binding(0) var<storage, read> bodies: array<RigidBody>;
@group(0) @binding(1) var<storage, read> colliders: array<Collider>;
@group(0) @binding(2) var<storage, read> pair_keys_hi: array<u32>;
@group(0) @binding(3) var<storage, read> pair_keys_lo: array<u32>;
@group(0) @binding(4) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(5) var<storage, read_write> contact_count: atomic<u32>;
@group(0) @binding(6) var<storage, read> pair_count: atomic<u32>;

fn contact_emit(contact: ptr<function, Contact>, normal: vec3f) {
    (*contact).point_count = 0u;
    (*contact).normal = normal;
}

fn manifold_push(contact: ptr<function, Contact>, point: vec3f, depth: f32) {
    let count = (*contact).point_count;
    if (count >= CONTACT_MAX_POINTS) {
        return;
    }
    (*contact).points[count] = ManifoldPoint(point, depth, 0.0, 0.0, 0.0, 0.0);
    (*contact).point_count = count + 1u;
}

fn sphere_sphere(
    first: RigidBody, first_collider: Collider,
    second: RigidBody, second_collider: Collider,
) -> Contact {
    var contact: Contact;
    let delta = second.position - first.position;
    let distance = length(delta);
    let radius_sum = first_collider.radius + second_collider.radius;
    contact_emit(&contact, sign_normalize(delta));
    if (distance > radius_sum) {
        return contact;
    }
    let normal = sign_normalize(delta);
    let depth = radius_sum - distance;
    let point = first.position + normal * (first_collider.radius - depth * 0.5);
    manifold_push(&contact, point, depth);
    return contact;
}

fn box_deep_normal(point: vec3f, box_body: RigidBody, box_collider: Collider) -> vec3f {
    let q = quat_mul(box_body.orientation, box_collider.local_rotation);
    let local = quat_rotate(quat_conjugate(q), point - box_center(box_body, box_collider));
    let penetration = box_collider.half_extents - abs(local);
    let axis = largest_axis(penetration);
    var facing = vec3f(0.0);
    if (axis == 0u) {
        facing = vec3f(select(1.0, -1.0, local.x > 0.0), 0.0, 0.0);
    } else if (axis == 1u) {
        facing = vec3f(0.0, select(1.0, -1.0, local.y > 0.0), 0.0);
    } else {
        facing = vec3f(0.0, 0.0, select(1.0, -1.0, local.z > 0.0));
    }
    return -quat_rotate(q, facing);
}

fn sphere_box(
    sphere: RigidBody, sphere_collider: Collider,
    box_body: RigidBody, box_collider: Collider,
) -> Contact {
    var contact: Contact;
    let center = sphere.position;
    let closest = closest_point_box(center, box_body, box_collider);
    let delta = closest - center;
    let distance = length(delta);
    let radius = sphere_collider.radius;
    contact_emit(&contact, sign_normalize(delta));
    if (distance >= radius) {
        return contact;
    }
    var normal = sign_normalize(delta);
    if (distance <= 1e-6) {
        normal = box_deep_normal(center, box_body, box_collider);
    }
    let depth = radius - distance;
    let point = closest - normal * (depth * 0.5);
    manifold_push(&contact, point, depth);
    return contact;
}

fn capsule_segment(body: RigidBody, collider: Collider) -> Segment {
    let axis = quat_rotate(body.orientation, vec3f(0.0, 1.0, 0.0));
    let center = box_center(body, collider);
    return Segment(center - axis * collider.half_height, center + axis * collider.half_height);
}

fn sphere_capsule(
    sphere: RigidBody, sphere_collider: Collider,
    capsule: RigidBody, capsule_collider: Collider,
) -> Contact {
    var contact: Contact;
    let seg = capsule_segment(capsule, capsule_collider);
    let closest = closest_point_segment(sphere.position, seg.start, seg.end);
    let delta = closest - sphere.position;
    let distance = length(delta);
    let radius_sum = sphere_collider.radius + capsule_collider.radius;
    contact_emit(&contact, sign_normalize(delta));
    if (distance > radius_sum) {
        return contact;
    }
    let normal = sign_normalize(delta);
    let depth = radius_sum - distance;
    let point = closest - normal * (depth * 0.5);
    manifold_push(&contact, point, depth);
    return contact;
}

fn closest_points_segments(a0: vec3f, a1: vec3f, b0: vec3f, b1: vec3f) -> Segment {
    let d1 = a1 - a0;
    let d2 = b1 - b0;
    let r = a0 - b0;
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);
    var s = 0.0;
    var t = 0.0;
    if (a <= 1e-10 && e <= 1e-10) {
        s = 0.0;
        t = 0.0;
    } else if (a <= 1e-10) {
        s = 0.0;
        t = clamp(f / e, 0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if (e <= 1e-10) {
            t = 0.0;
            s = clamp(-c / a, 0.0, 1.0);
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            if (denom != 0.0) {
                s = clamp((b * f - c * e) / denom, 0.0, 1.0);
            } else {
                s = 0.0;
            }
            t = (b * s + f) / e;
            if (t < 0.0) {
                t = 0.0;
                s = clamp(-c / a, 0.0, 1.0);
            } else if (t > 1.0) {
                t = 1.0;
                s = clamp((b - c) / a, 0.0, 1.0);
            }
        }
    }
    return Segment(a0 + d1 * s, b0 + d2 * t);
}

fn capsule_capsule(
    first: RigidBody, first_collider: Collider,
    second: RigidBody, second_collider: Collider,
) -> Contact {
    var contact: Contact;
    let seg_a = capsule_segment(first, first_collider);
    let seg_b = capsule_segment(second, second_collider);
    let closest = closest_points_segments(seg_a.start, seg_a.end, seg_b.start, seg_b.end);
    let delta = closest.end - closest.start;
    let distance = length(delta);
    let radius_sum = first_collider.radius + second_collider.radius;
    contact_emit(&contact, sign_normalize(delta));
    if (distance > radius_sum) {
        return contact;
    }
    let normal = sign_normalize(delta);
    let depth = radius_sum - distance;
    let point = (closest.start + closest.end) * 0.5;
    manifold_push(&contact, point, depth);
    return contact;
}

fn box_box_sat(
    first: RigidBody, first_collider: Collider,
    second: RigidBody, second_collider: Collider,
) -> Contact {
    var contact: Contact;
    let axes_a = box_rotated_axes(first, first_collider);
    let axes_b = box_rotated_axes(second, second_collider);
    let he_a = first_collider.half_extents;
    let he_b = second_collider.half_extents;
    let center_a = box_center(first, first_collider);
    let center_b = box_center(second, second_collider);
    let delta = center_b - center_a;
    var best = 1e30;
    var best_axis = vec3f(0.0, 1.0, 0.0);

    var i = 0;
    loop {
        if (i >= 3) {
            break;
        }
        let axis = axes_a[i];
        let ra = abs(dot(delta, axis));
        let rb = abs(dot(axes_b[0], axis)) * he_b.x + abs(dot(axes_b[1], axis)) * he_b.y + abs(dot(axes_b[2], axis)) * he_b.z;
        let overlap = he_a[i] + rb - ra;
        if (overlap < best) {
            best = overlap;
            best_axis = axis;
        }
        i = i + 1;
    }
    i = 0;
    loop {
        if (i >= 3) {
            break;
        }
        let axis = axes_b[i];
        let projected_delta = abs(dot(delta, axis));
        let projected_a = abs(dot(axes_a[0], axis)) * he_a.x + abs(dot(axes_a[1], axis)) * he_a.y + abs(dot(axes_a[2], axis)) * he_a.z;
        let overlap = he_b[i] + projected_a - projected_delta;
        if (overlap < best) {
            best = overlap;
            best_axis = axis;
        }
        i = i + 1;
    }

    for (var ia = 0u; ia < 3u; ia = ia + 1u) {
        for (var ib = 0u; ib < 3u; ib = ib + 1u) {
            let axis = cross(axes_a[ia], axes_b[ib]);
            let len = length(axis);
            if (len < 1e-8) {
                continue;
            }
            let n = axis / len;
            let ra = abs(dot(delta, n));
            let rb = abs(dot(axes_a[0], n)) * he_a.x + abs(dot(axes_a[1], n)) * he_a.y + abs(dot(axes_a[2], n)) * he_a.z
                + abs(dot(axes_b[0], n)) * he_b.x + abs(dot(axes_b[1], n)) * he_b.y + abs(dot(axes_b[2], n)) * he_b.z;
            let overlap = rb - ra;
            if (overlap < best) {
                best = overlap;
                best_axis = n;
            }
        }
    }

    if (best <= 0.0) {
        contact_emit(&contact, vec3f(0.0, 1.0, 0.0));
        return contact;
    }
    let signed = select(best_axis, -best_axis, dot(best_axis, delta) < 0.0);
    let depth = best;
    let point = (center_a + center_b) * 0.5;
    contact_emit(&contact, signed);
    manifold_push(&contact, point, depth);
    return contact;
}

fn box_capsule(
    box_body: RigidBody, box_collider: Collider,
    capsule: RigidBody, capsule_collider: Collider,
) -> Contact {
    var contact: Contact;
    let seg = capsule_segment(capsule, capsule_collider);
    let closest_a = closest_point_box(seg.start, box_body, box_collider);
    let closest_b = closest_point_box(seg.end, box_body, box_collider);
    let delta_a = closest_a - seg.start;
    let delta_b = closest_b - seg.end;
    let depth_a = capsule_collider.radius - length(delta_a);
    let depth_b = capsule_collider.radius - length(delta_b);
    var depth = depth_a;
    var normal = sign_normalize(delta_a);
    var point = closest_a;
    if (depth_b > depth) {
        depth = depth_b;
        normal = sign_normalize(delta_b);
        point = closest_b;
    }
    contact_emit(&contact, normal);
    if (depth <= 0.0) {
        return contact;
    }
    manifold_push(&contact, point, depth);
    return contact;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= atomicLoad(&pair_count)) {
        return;
    }
    if (index > 0u && pair_keys_hi[index] == pair_keys_hi[index - 1u] && pair_keys_lo[index] == pair_keys_lo[index - 1u]) {
        return;
    }
    let first_slot = pair_keys_hi[index];
    let second_slot = pair_keys_lo[index];
    let first = bodies[first_slot];
    let second = bodies[second_slot];
    if (first.inverse_mass == 0.0 && (first.flags & BODY_KINEMATIC) == 0u &&
        second.inverse_mass == 0.0 && (second.flags & BODY_KINEMATIC) == 0u) {
        return;
    }
    if ((first.collision_group & second.collision_mask) == 0u ||
        (second.collision_group & first.collision_mask) == 0u) {
        return;
    }
    let first_collider = colliders[first_slot];
    let second_collider = colliders[second_slot];
    var contact: Contact;
    if (first_collider.shape == SHAPE_SPHERE && second_collider.shape == SHAPE_SPHERE) {
        contact = sphere_sphere(first, first_collider, second, second_collider);
    } else if (first_collider.shape == SHAPE_SPHERE && second_collider.shape == SHAPE_BOX) {
        contact = sphere_box(first, first_collider, second, second_collider);
    } else if (first_collider.shape == SHAPE_SPHERE && second_collider.shape == SHAPE_CAPSULE) {
        contact = sphere_capsule(first, first_collider, second, second_collider);
    } else if (first_collider.shape == SHAPE_BOX && second_collider.shape == SHAPE_SPHERE) {
        let swapped = sphere_box(second, second_collider, first, first_collider);
        contact = swapped;
        contact.normal = -contact.normal;
    } else if (first_collider.shape == SHAPE_BOX && second_collider.shape == SHAPE_BOX) {
        contact = box_box_sat(first, first_collider, second, second_collider);
    } else if (first_collider.shape == SHAPE_BOX && second_collider.shape == SHAPE_CAPSULE) {
        contact = box_capsule(first, first_collider, second, second_collider);
    } else if (first_collider.shape == SHAPE_CAPSULE && second_collider.shape == SHAPE_SPHERE) {
        let swapped = sphere_capsule(second, second_collider, first, first_collider);
        contact = swapped;
        contact.normal = -contact.normal;
    } else if (first_collider.shape == SHAPE_CAPSULE && second_collider.shape == SHAPE_BOX) {
        let swapped = box_capsule(second, second_collider, first, first_collider);
        contact = swapped;
        contact.normal = -contact.normal;
    } else {
        contact = capsule_capsule(first, first_collider, second, second_collider);
    }
    contact.a = first_slot;
    contact.b = second_slot;
    if (contact.point_count > 0u) {
        let slot = atomicAdd(&contact_count, 1u);
        if (slot < arrayLength(&contacts)) {
            contacts[slot] = contact;
        }
    }
}
