@group(0) @binding(0) var<storage, read> bodies: array<RigidBody>;
@group(0) @binding(1) var<storage, read> colliders: array<Collider>;
@group(0) @binding(2) var<storage, read> pair_keys_hi: array<u32>;
@group(0) @binding(3) var<storage, read> pair_keys_lo: array<u32>;
@group(0) @binding(4) var<storage, read_write> contacts_raw: array<Contact>;
@group(0) @binding(5) var<storage, read_write> contact_valid: array<u32>;
@group(0) @binding(6) var<storage, read_write> pair_count: atomic<u32>;
@group(0) @binding(7) var<storage, read> joint_hi: array<u32>;
@group(0) @binding(8) var<storage, read> joint_lo: array<u32>;
@group(0) @binding(9) var<storage, read> joint_count: array<u32>;
@group(0) @binding(10) var<uniform> params: SimParams;

fn contact_emit(contact: ptr<function, Contact>, normal: vec3f) {
    (*contact).point_count = 0u;
    (*contact).normal = normal;
}

fn pair_joined(first_body: u32, second_body: u32) -> bool {
    let count = joint_count[0];
    if (count == 0u) {
        return false;
    }
    let a = min(first_body, second_body);
    let b = max(first_body, second_body);
    var lo = 0u;
    var hi = count;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (joint_hi[mid] < a || (joint_hi[mid] == a && joint_lo[mid] < b)) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo < count && joint_hi[lo] == a && joint_lo[lo] == b;
}

fn sphere_sphere(
    first: RigidBody, first_collider: Collider,
    second: RigidBody, second_collider: Collider,
) -> Contact {
    var contact: Contact;
    let first_center = box_center(first, first_collider);
    let second_center = box_center(second, second_collider);
    let delta = second_center - first_center;
    let distance = length(delta);
    let radius_sum = first_collider.radius + second_collider.radius;
    contact_emit(&contact, sign_normalize(delta));
    if (distance > radius_sum) {
        return contact;
    }
    var normal = sign_normalize(delta);
    if (distance <= 1e-6) {
        let relative = relative_velocity(first, second, second_center, first_center);
        normal = select(normal, -normalize(relative), length(relative) > 1e-6);
    }
    let depth = radius_sum - distance;
    let point = first_center + normal * (first_collider.radius - depth * 0.5);
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
    let center = sphere.position + quat_rotate(sphere.orientation, sphere_collider.local_offset);
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
    let q = quat_mul(body.orientation, collider.local_rotation);
    let axis = quat_rotate(q, vec3f(0.0, 1.0, 0.0));
    let center = box_center(body, collider);
    return Segment(center - axis * collider.half_height, center + axis * collider.half_height);
}

fn sphere_capsule(
    sphere: RigidBody, sphere_collider: Collider,
    capsule: RigidBody, capsule_collider: Collider,
) -> Contact {
    var contact: Contact;
    let seg = capsule_segment(capsule, capsule_collider);
    let center = sphere.position + quat_rotate(sphere.orientation, sphere_collider.local_offset);
    let closest = closest_point_segment(center, seg.start, seg.end);
    let delta = closest - center;
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

fn box_face_corners(body: RigidBody, collider: Collider, face_normal: vec3f) -> array<vec3f, 4> {
    let axes = box_rotated_axes(body, collider);
    let center = box_center(body, collider);
    var axis = 0u;
    for (var i = 1u; i < 3u; i = i + 1u) {
        if (abs(dot(face_normal, axes[i])) > abs(dot(face_normal, axes[axis]))) {
            axis = i;
        }
    }
    let sign = select(1.0, -1.0, dot(face_normal, axes[axis]) < 0.0);
    let n = axes[axis] * sign;
    let u = axes[(axis + 1u) % 3u];
    let v = axes[(axis + 2u) % 3u];
    let e = collider.half_extents[axis];
    var corners: array<vec3f, 4>;
    corners[0] = center + n * e + u * collider.half_extents[(axis + 1u) % 3u] + v * collider.half_extents[(axis + 2u) % 3u];
    corners[1] = center + n * e - u * collider.half_extents[(axis + 1u) % 3u] + v * collider.half_extents[(axis + 2u) % 3u];
    corners[2] = center + n * e - u * collider.half_extents[(axis + 1u) % 3u] - v * collider.half_extents[(axis + 2u) % 3u];
    corners[3] = center + n * e + u * collider.half_extents[(axis + 1u) % 3u] - v * collider.half_extents[(axis + 2u) % 3u];
    return corners;
}

fn clip_polygon(points: array<vec3f, 8>, count: u32, plane_point: vec3f, plane_normal: vec3f, out_points: ptr<function, array<vec3f, 8>>) -> u32 {
    var out_count = 0u;
    for (var i = 0u; i < count; i = i + 1u) {
        let current = points[i];
        let next = points[(i + 1u) % count];
        let current_dist = dot(current - plane_point, plane_normal);
        let next_dist = dot(next - plane_point, plane_normal);
        if (current_dist <= 0.0) {
            if (out_count < 8u) {
                (*out_points)[out_count] = current;
                out_count = out_count + 1u;
            }
        }
        if (current_dist * next_dist < 0.0) {
            let t = current_dist / (current_dist - next_dist);
            if (out_count < 8u) {
                (*out_points)[out_count] = current + (next - current) * t;
                out_count = out_count + 1u;
            }
        }
    }
    return out_count;
}

fn box_box_sat(
    first: RigidBody, first_collider: Collider,
    second: RigidBody, second_collider: Collider,
) -> Contact {
    var contact: Contact;
    contact_emit(&contact, vec3f(0.0, 1.0, 0.0));
    let box_radius = max(max(first_collider.half_extents.x, first_collider.half_extents.y), first_collider.half_extents.z);
    let other_radius = max(max(second_collider.half_extents.x, second_collider.half_extents.y), second_collider.half_extents.z);
    let swapped = box_radius < other_radius;
    var ref_body = first;
    var ref_collider = first_collider;
    var hit_body = second;
    var hit_collider = second_collider;
    if (swapped) {
        ref_body = second;
        ref_collider = second_collider;
        hit_body = first;
        hit_collider = first_collider;
    }
    let axes_a = box_rotated_axes(ref_body, ref_collider);
    let axes_b = box_rotated_axes(hit_body, hit_collider);
    let he_a = ref_collider.half_extents;
    let he_b = hit_collider.half_extents;
    let center_a = box_center(ref_body, ref_collider);
    let center_b = box_center(hit_body, hit_collider);
    let delta = center_b - center_a;
    var best = 1e30;
    var best_axis = vec3f(0.0, 1.0, 0.0);
    var best_side = 0u;
    var index = 0u;
    loop {
        if (index >= 3u) {
            break;
        }
        let axis = axes_a[index];
        let ra = abs(dot(delta, axis));
        let rb = abs(dot(axes_b[0], axis)) * he_b.x + abs(dot(axes_b[1], axis)) * he_b.y + abs(dot(axes_b[2], axis)) * he_b.z;
        let overlap = he_a[index] + rb - ra;
        if (overlap < best) {
            best = overlap;
            best_axis = axis;
            best_side = index;
        }
        index = index + 1u;
    }
    index = 0u;
    loop {
        if (index >= 3u) {
            break;
        }
        let axis = axes_b[index];
        let projected_delta = abs(dot(delta, axis));
        let projected_a = abs(dot(axes_a[0], axis)) * he_a.x + abs(dot(axes_a[1], axis)) * he_a.y + abs(dot(axes_a[2], axis)) * he_a.z;
        let overlap = he_b[index] + projected_a - projected_delta;
        if (overlap < best) {
            best = overlap;
            best_axis = axis;
            best_side = 3u + index;
        }
        index = index + 1u;
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
                best_side = 6u + ia * 3u + ib;
            }
        }
    }
    if (best <= 0.0) {
        return contact;
    }
    let signed = select(best_axis, -best_axis, dot(best_axis, delta) < 0.0);
    if (best_side >= 6u) {
        let depth = best;
        let point = box_face_point(hit_body, hit_collider, -signed);
        contact.normal = select(signed, -signed, swapped);
        manifold_push(&contact, point, depth);
        return contact;
    }
    var reference: RigidBody;
    var reference_collider: Collider;
    var incident: RigidBody;
    var incident_collider: Collider;
    var face_normal = signed;
    if (best_side < 3u) {
        reference = ref_body;
        reference_collider = ref_collider;
        incident = hit_body;
        incident_collider = hit_collider;
    } else {
        reference = hit_body;
        reference_collider = hit_collider;
        incident = ref_body;
        incident_collider = ref_collider;
        face_normal = -signed;
    }
    let ref_normal = face_normal;
    let ref_center = box_face_point(reference, reference_collider, ref_normal);
    let ref_corners = box_face_corners(reference, reference_collider, ref_normal);
    let incident_ax = box_rotated_axes(incident, incident_collider);
    var face_axis = 0u;
    for (var i = 1u; i < 3u; i = i + 1u) {
        if (abs(dot(incident_ax[i], ref_normal)) > abs(dot(incident_ax[face_axis], ref_normal))) {
            face_axis = i;
        }
    }
    let incident_normal = incident_ax[face_axis] * select(1.0, -1.0, dot(incident_ax[face_axis], ref_normal) > 0.0);
    let incident_corners = box_face_corners(incident, incident_collider, incident_normal);
    var polygon_a: array<vec3f, 8>;
    var polygon_b: array<vec3f, 8>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        polygon_a[i] = incident_corners[i];
    }
    var polygon_count = 4u;
    for (var side = 0u; side < 4u; side = side + 1u) {
        let current = ref_corners[side];
        let next = ref_corners[(side + 1u) % 4u];
        let plane_normal = normalize(cross(next - current, ref_normal));
        if (side % 2u == 0u) {
            polygon_count = clip_polygon(polygon_a, polygon_count, current, plane_normal, &polygon_b);
        } else {
            polygon_count = clip_polygon(polygon_b, polygon_count, current, plane_normal, &polygon_a);
        }
        if (polygon_count == 0u) {
            return contact;
        }
    }
    let clipped = polygon_a;
    contact.normal = select(signed, -signed, swapped);
    var candidates: array<ManifoldPoint, 8>;
    var candidate_count = 0u;
    for (var i = 0u; i < polygon_count; i = i + 1u) {
        let depth = dot(ref_center - clipped[i], ref_normal);
        if (depth >= 0.0 && candidate_count < 8u) {
            candidates[candidate_count] = ManifoldPoint(clipped[i] - ref_normal * (depth * 0.5), depth, 0.0, 0.0, 0.0, 0.0);
            candidate_count = candidate_count + 1u;
        }
    }
    if (candidate_count == 0u) {
        return contact;
    }
    for (var i = 0u; i < candidate_count; i = i + 1u) {
        for (var j = i + 1u; j < candidate_count; j = j + 1u) {
            if (candidates[j].depth > candidates[i].depth) {
                let tmp = candidates[i];
                candidates[i] = candidates[j];
                candidates[j] = tmp;
            }
        }
    }
    var keep = min(candidate_count, CONTACT_MAX_POINTS);
    for (var i = 0u; i < keep; i = i + 1u) {
        manifold_push(&contact, candidates[i].position, candidates[i].depth);
    }
    return contact;
}

fn box_capsule(
    box_body: RigidBody, box_collider: Collider,
    capsule: RigidBody, capsule_collider: Collider,
) -> Contact {
    var contact: Contact;
    contact_emit(&contact, vec3f(0.0, 1.0, 0.0));
    let seg = capsule_segment(capsule, capsule_collider);
    var candidates: array<ManifoldPoint, 8>;
    var candidate_normals: array<vec3f, 8>;
    var candidate_count = 0u;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let t = f32(i) * (1.0 / 2.0);
        let point = seg.start + (seg.end - seg.start) * t;
        let closest = closest_point_box(point, box_body, box_collider);
        let delta = point - closest;
        let distance = length(delta);
        let depth = capsule_collider.radius - distance;
        if (depth > 0.0) {
            var normal = sign_normalize(delta);
            if (distance <= 1e-6) {
                normal = box_deep_normal(point, box_body, box_collider);
            }
            let contact_point = closest + normal * (depth * 0.5);
            var found = false;
            for (var existing = 0u; existing < candidate_count; existing = existing + 1u) {
                if (length(candidates[existing].position - contact_point) < 0.05) {
                    found = true;
                }
            }
            if (found) {
                continue;
            }
            candidates[candidate_count] = ManifoldPoint(contact_point, depth, 0.0, 0.0, 0.0, 0.0);
            candidate_normals[candidate_count] = normal;
            candidate_count = candidate_count + 1u;
        }
    }
    if (candidate_count == 0u) {
        return contact;
    }
    var best_depth = -1e30;
    var best_normal = vec3f(0.0, 1.0, 0.0);
    for (var i = 0u; i < candidate_count; i = i + 1u) {
        if (candidates[i].depth > best_depth) {
            best_depth = candidates[i].depth;
            best_normal = candidate_normals[i];
        }
    }
    contact.normal = best_normal;
    var keep = min(candidate_count, 2u);
    for (var i = 0u; i < keep; i = i + 1u) {
        manifold_push(&contact, candidates[i].position, candidates[i].depth);
    }
    return contact;
}

fn manifold_from_hit(contact: ptr<function, Contact>, hit: ShapeHit) {
    if (hit.distance <= 0.0) {
        manifold_push(contact, hit.point, -hit.distance);
    }
}

fn plane_convex(plane: WorldShape, convex: WorldShape) -> Contact {
    var contact: Contact;
    let n = plane_normal(plane);
    let center_side = dot(convex.center - plane.center, n);
    let facing = select(n, -n, center_side < 0.0);
    contact_emit(&contact, facing);
    var points: array<vec3f, 4>;
    let count = convex_sample_points(convex, -facing, &points);
    for (var i = 0u; i < count; i = i + 1u) {
        let depth = dot(plane.center - points[i], facing);
        if (depth > 0.0) {
            manifold_push(&contact, points[i] + facing * (depth * 0.5), depth);
        }
    }
    return contact;
}

fn scaled_shape(collider: Collider) -> bool {
    return collider.scale.x != 1.0 || collider.scale.y != 1.0 || collider.scale.z != 1.0;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= atomicLoad(&pair_count)) {
        return;
    }
    contact_valid[index] = 0u;
    if (index > 0u && pair_keys_hi[index] == pair_keys_hi[index - 1u] && pair_keys_lo[index] == pair_keys_lo[index - 1u]) {
        return;
    }
    let first_slot = pair_keys_hi[index];
    let second_slot = pair_keys_lo[index];
    let first_body_slot = first_slot / 4u;
    let second_body_slot = second_slot / 4u;
    let first = bodies[first_body_slot];
    let second = bodies[second_body_slot];
    if (first_body_slot == second_body_slot) {
        return;
    }
    if (body_is_static(first) && body_is_static(second)) {
        return;
    }
    let first_collider = colliders[first_slot];
    let second_collider = colliders[second_slot];
    if (first_collider.kind == SHAPE_NONE || second_collider.kind == SHAPE_NONE) {
        return;
    }
    if (!collider_filter_intersects(first, first_collider, second, second_collider)) {
        return;
    }
    if (pair_joined(first_body_slot, second_body_slot)) {
        return;
    }
    var contact: Contact;
    var generated = false;
    let sensor = collider_is_sensor(first_collider) || collider_is_sensor(second_collider);
    let first_world_geom = first_collider.kind == SHAPE_MESH || first_collider.kind == SHAPE_HEIGHTFIELD || first_collider.kind == SHAPE_PLANE;
    let second_world_geom = second_collider.kind == SHAPE_MESH || second_collider.kind == SHAPE_HEIGHTFIELD || second_collider.kind == SHAPE_PLANE;
    if (first_world_geom && second_world_geom) {
        return;
    }
    if (first_world_geom) {
        let world_second = world_collider(second, second_collider);
        if (first_collider.kind == SHAPE_PLANE) {
            let world_plane = world_collider(first, first_collider);
            contact = plane_convex(world_plane, world_second);
            generated = contact.point_count > 0u;
        } else {
            let hit = scene_convex_hit(first_collider.source, first_collider.scale, world_second);
            if (hit.distance <= 0.0) {
                contact_emit(&contact, hit.normal);
                generated = true;
                if (!scene_convex_manifold(first_collider.source, first_collider.scale, world_second, &contact)) {
                    manifold_from_hit(&contact, hit);
                }
            }
        }
    } else if (second_world_geom) {
        let world_first = world_collider(first, first_collider);
        if (second_collider.kind == SHAPE_PLANE) {
            let world_plane = world_collider(second, second_collider);
            let swapped = plane_convex(world_plane, world_first);
            contact = swapped;
            contact.normal = -contact.normal;
            generated = contact.point_count > 0u;
        } else {
            let hit = scene_convex_hit(second_collider.source, second_collider.scale, world_first);
            if (hit.distance <= 0.0) {
                contact_emit(&contact, -hit.normal);
                generated = true;
                if (!scene_convex_manifold(second_collider.source, second_collider.scale, world_first, &contact)) {
                    let reversed_hit = ShapeHit(hit.distance, hit.point, -hit.normal);
                    manifold_from_hit(&contact, reversed_hit);
                }
            }
        }
    } else if (scaled_shape(first_collider) || scaled_shape(second_collider)) {
        let world_first = world_collider(first, first_collider);
        let world_second = world_collider(second, second_collider);
        let hit = convex_hit(world_first, world_second);
        if (hit.distance <= 0.0) {
            contact_emit(&contact, hit.normal);
            generated = true;
            if (!convex_pair_manifold(world_first, world_second, hit.normal, &contact)) {
                manifold_from_hit(&contact, hit);
            }
        }
    } else if (first_collider.kind == SHAPE_CYLINDER || first_collider.kind == SHAPE_HULL || second_collider.kind == SHAPE_CYLINDER || second_collider.kind == SHAPE_HULL) {
        let world_first = world_collider(first, first_collider);
        let world_second = world_collider(second, second_collider);
        let hit = convex_hit(world_first, world_second);
        if (hit.distance <= 0.0) {
            contact_emit(&contact, hit.normal);
            generated = true;
            if (!convex_pair_manifold(world_first, world_second, hit.normal, &contact)) {
                manifold_from_hit(&contact, hit);
            }
        }
    } else {
        let shape_a = first_collider.kind;
        let shape_b = second_collider.kind;
        if (shape_a == SHAPE_SPHERE && shape_b == SHAPE_SPHERE) {
            contact = sphere_sphere(first, first_collider, second, second_collider);
            generated = true;
        } else if (shape_a == SHAPE_SPHERE && shape_b == SHAPE_CUBOID) {
            contact = sphere_box(first, first_collider, second, second_collider);
            generated = true;
        } else if (shape_a == SHAPE_SPHERE && shape_b == SHAPE_CAPSULE) {
            contact = sphere_capsule(first, first_collider, second, second_collider);
            generated = true;
        } else if (shape_a == SHAPE_CUBOID && shape_b == SHAPE_SPHERE) {
            let swapped = sphere_box(second, second_collider, first, first_collider);
            contact = swapped;
            contact.normal = -contact.normal;
            generated = true;
        } else if (shape_a == SHAPE_CUBOID && shape_b == SHAPE_CUBOID) {
            contact = box_box_sat(first, first_collider, second, second_collider);
            generated = true;
        } else if (shape_a == SHAPE_CUBOID && shape_b == SHAPE_CAPSULE) {
            contact = box_capsule(first, first_collider, second, second_collider);
            generated = true;
        } else if (shape_a == SHAPE_CAPSULE && shape_b == SHAPE_SPHERE) {
            let swapped = sphere_capsule(second, second_collider, first, first_collider);
            contact = swapped;
            contact.normal = -contact.normal;
            generated = true;
        } else if (shape_a == SHAPE_CAPSULE && shape_b == SHAPE_CUBOID) {
            let swapped = box_capsule(second, second_collider, first, first_collider);
            contact = swapped;
            contact.normal = -contact.normal;
            generated = true;
        } else if (shape_a == SHAPE_CAPSULE && shape_b == SHAPE_CAPSULE) {
            contact = capsule_capsule(first, first_collider, second, second_collider);
            generated = true;
        } else {
            let world_first = world_collider(first, first_collider);
            let world_second = world_collider(second, second_collider);
            let hit = convex_hit(world_first, world_second);
            if (hit.distance <= 0.0) {
                contact_emit(&contact, hit.normal);
                generated = true;
                if (!convex_pair_manifold(world_first, world_second, hit.normal, &contact)) {
                    manifold_from_hit(&contact, hit);
                }
            }
        }
    }
    if (!generated) {
        return;
    }
    contact.a = first_slot;
    contact.b = second_slot;
    contact.sensor = select(0u, 1u, sensor);
    contact.first_body_id = first.body_id;
    contact.second_body_id = second.body_id;
    contact.first_generation = first.generation;
    contact.second_generation = second.generation;
    contact.friction = material_combine(first_collider.friction, second_collider.friction, params.friction_combine);
    contact.restitution = material_combine(first_collider.restitution, second_collider.restitution, params.restitution_combine);
    contact.rolling_friction = max(first_collider.rolling_friction, second_collider.rolling_friction);
    contact.spin_friction = max(first_collider.spin_friction, second_collider.spin_friction);
    if (contact.point_count > 0u) {
        contacts_raw[index] = contact;
        contact_valid[index] = 1u;
    }
}
