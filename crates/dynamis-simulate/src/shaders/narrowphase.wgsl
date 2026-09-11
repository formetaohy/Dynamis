@group(0) @binding(0) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(1) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(2) var<storage, read> colliders: array<Collider>;
@group(0) @binding(3) var<storage, read> pair_major: array<u32>;
@group(0) @binding(4) var<storage, read> pair_minor: array<u32>;
@group(0) @binding(5) var<storage, read_write> contacts_raw: array<Contact>;
@group(0) @binding(6) var<storage, read_write> contact_valid: array<u32>;
@group(0) @binding(7) var<storage, read_write> pair_count: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> joint_major: array<u32>;
@group(0) @binding(9) var<storage, read> joint_minor: array<u32>;
@group(0) @binding(10) var<storage, read_write> joint_count: array<atomic<u32>>;
@group(0) @binding(11) var<uniform> params: SimParams;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn contact_emit(contact: ptr<function, Contact>, normal: vec3f) {
    (*contact).point_count = 0u;
    (*contact).normal = normal;
}

fn pair_joined(first_body: u32, second_body: u32) -> bool {
    let count = min(atomicLoad(&joint_count[0]), arrayLength(&joint_major));
    if (count == 0u) {
        return false;
    }
    let a = min(first_body, second_body);
    let b = max(first_body, second_body);
    var lo = 0u;
    var hi = count;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (joint_major[mid] < a || (joint_major[mid] == a && joint_minor[mid] < b)) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo < count && joint_major[lo] == a && joint_minor[lo] == b;
}

fn sphere_sphere(
    first: Body, first_collider: Collider,
    second: Body, second_collider: Collider,
) -> Contact {
    var contact: Contact;
    let first_center = box_center(first.state, first_collider);
    let second_center = box_center(second.state, second_collider);
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

fn box_deep_normal(point: vec3f, box_body: Body, box_collider: Collider) -> vec3f {
    let q = quat_mul(box_body.state.orientation, box_collider.local_rotation);
    let local = quat_rotate(quat_conjugate(q), point - box_center(box_body.state, box_collider));
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
    sphere: Body, sphere_collider: Collider,
    box_body: Body, box_collider: Collider,
) -> Contact {
    var contact: Contact;
    let center = sphere.state.position + quat_rotate(sphere.state.orientation, sphere_collider.local_offset);
    let closest = closest_point_box(center, box_body.state, box_collider);
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

fn capsule_segment(body: Body, collider: Collider) -> Segment {
    let q = quat_mul(body.state.orientation, collider.local_rotation);
    let axis = quat_rotate(q, vec3f(0.0, 1.0, 0.0));
    let center = box_center(body.state, collider);
    return Segment(center - axis * collider.half_height, center + axis * collider.half_height);
}

fn sphere_capsule(
    sphere: Body, sphere_collider: Collider,
    capsule: Body, capsule_collider: Collider,
) -> Contact {
    var contact: Contact;
    let seg = capsule_segment(capsule, capsule_collider);
    let center = sphere.state.position + quat_rotate(sphere.state.orientation, sphere_collider.local_offset);
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
    first: Body, first_collider: Collider,
    second: Body, second_collider: Collider,
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

fn box_support(body: Body, collider: Collider, direction: vec3f) -> vec3f {
    let axes = box_rotated_axes(body.state, collider);
    var point = box_center(body.state, collider);
    for (var index = 0u; index < 3u; index = index + 1u) {
        point = point + axes[index] * select(-collider.half_extents[index], collider.half_extents[index], dot(axes[index], direction) > 0.0);
    }
    return point;
}

fn box_face(body: Body, collider: Collider, face_normal: vec3f, corners: ptr<function, array<vec3f, 4>>) -> vec3f {
    let axes = box_rotated_axes(body.state, collider);
    let center = box_center(body.state, collider);
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
    let half_u = collider.half_extents[(axis + 1u) % 3u];
    let half_v = collider.half_extents[(axis + 2u) % 3u];
    let face_center = center + n * collider.half_extents[axis];
    var edges: array<vec3f, 4>;
    edges[0] = u * half_u + v * half_v;
    edges[1] = -u * half_u + v * half_v;
    edges[2] = -u * half_u - v * half_v;
    edges[3] = u * half_u - v * half_v;
    for (var index = 0u; index < 4u; index = index + 1u) {
        (*corners)[index] = face_center + edges[index];
    }
    return face_center;
}

fn clip_polygon(points: array<vec3f, 8>, count: u32, plane_point: vec3f, plane_normal: vec3f, out_points: ptr<function, array<vec3f, 8>>) -> u32 {
    var out_count = 0u;
    for (var i = 0u; i < count; i = i + 1u) {
        let current = points[i];
        let next = points[(i + 1u) % count];
        let current_dist = dot(current - plane_point, plane_normal) - CLIP_MARGIN;
        let next_dist = dot(next - plane_point, plane_normal) - CLIP_MARGIN;
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

const FACE_AXIS_BIAS: f32 = 1e-3;

struct BoxGeometry {
    body: Body,
    collider: Collider,
    axes: array<vec3f, 3>,
    half_extents: vec3f,
    center: vec3f,
}

fn box_geometry(body: Body, collider: Collider) -> BoxGeometry {
    var geometry: BoxGeometry;
    geometry.body = body;
    geometry.collider = collider;
    geometry.axes = box_rotated_axes(body.state, collider);
    geometry.half_extents = collider.half_extents;
    geometry.center = box_center(body.state, collider);
    return geometry;
}

fn support_radius(geometry: BoxGeometry, axis: vec3f) -> f32 {
    return abs(dot(geometry.axes[0], axis)) * geometry.half_extents.x
        + abs(dot(geometry.axes[1], axis)) * geometry.half_extents.y
        + abs(dot(geometry.axes[2], axis)) * geometry.half_extents.z;
}

fn face_overlap(face: BoxGeometry, other: BoxGeometry, axis_index: u32, delta: vec3f) -> f32 {
    let axis = face.axes[axis_index];
    return face.half_extents[axis_index] + support_radius(other, axis) - abs(dot(delta, axis));
}

fn edge_overlap(left: BoxGeometry, right: BoxGeometry, axis: vec3f, delta: vec3f) -> f32 {
    return support_radius(left, axis) + support_radius(right, axis) - abs(dot(delta, axis));
}

fn box_box_sat(
    first: Body, first_collider: Collider,
    second: Body, second_collider: Collider,
) -> Contact {
    var contact: Contact;
    contact_emit(&contact, vec3f(0.0, 1.0, 0.0));
    let left = box_geometry(first, first_collider);
    let right = box_geometry(second, second_collider);
    let delta = right.center - left.center;
    var face_depth = 1e30;
    var face_axis = left.axes[0];
    var face_on_right = false;
    for (var index = 0u; index < 3u; index = index + 1u) {
        let overlap = face_overlap(left, right, index, delta);
        if (overlap < face_depth) {
            face_depth = overlap;
            face_axis = left.axes[index];
            face_on_right = false;
        }
        let other = face_overlap(right, left, index, delta);
        if (other < face_depth) {
            face_depth = other;
            face_axis = right.axes[index];
            face_on_right = true;
        }
    }
    var edge_depth = 1e30;
    var edge_axis = face_axis;
    for (var i = 0u; i < 3u; i = i + 1u) {
        for (var j = 0u; j < 3u; j = j + 1u) {
            let crossed = cross(left.axes[i], right.axes[j]);
            let len = length(crossed);
            if (len < 1e-8) {
                continue;
            }
            let axis = crossed / len;
            let overlap = edge_overlap(left, right, axis, delta);
            if (overlap < edge_depth) {
                edge_depth = overlap;
                edge_axis = axis;
            }
        }
    }
    let extent = max(max(left.half_extents.x, left.half_extents.y), left.half_extents.z)
        + max(max(right.half_extents.x, right.half_extents.y), right.half_extents.z);
    let edge_axis_separates = edge_depth + FACE_AXIS_BIAS * extent < face_depth;
    let depth = select(face_depth, edge_depth, edge_axis_separates);
    if (depth <= 0.0) {
        return contact;
    }
    let axis = select(face_axis, edge_axis, edge_axis_separates);
    let signed = select(axis, -axis, dot(axis, delta) < 0.0);
    if (edge_axis_separates) {
        let point = (box_support(first, first_collider, signed) + box_support(second, second_collider, -signed)) * 0.5;
        contact.normal = signed;
        manifold_push(&contact, point, depth);
        return contact;
    }
    var reference = left;
    var incident = right;
    var ref_normal = signed;
    if (face_on_right) {
        reference = right;
        incident = left;
        ref_normal = -signed;
    }
    var ref_corners: array<vec3f, 4>;
    let ref_center = box_face(reference.body, reference.collider, ref_normal, &ref_corners);
    var incident_axis = 0u;
    for (var i = 1u; i < 3u; i = i + 1u) {
        if (abs(dot(incident.axes[i], ref_normal)) > abs(dot(incident.axes[incident_axis], ref_normal))) {
            incident_axis = i;
        }
    }
    let incident_normal = incident.axes[incident_axis] * select(1.0, -1.0, dot(incident.axes[incident_axis], ref_normal) > 0.0);
    var incident_corners: array<vec3f, 4>;
    box_face(incident.body, incident.collider, incident_normal, &incident_corners);
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
            break;
        }
    }
    var candidates: array<ManifoldPoint, CONTACT_MAX_POINTS>;
    var candidate_count = 0u;
    for (var i = 0u; i < polygon_count && candidate_count < CONTACT_MAX_POINTS; i = i + 1u) {
        let point_depth = dot(ref_center - polygon_a[i], ref_normal);
        if (point_depth >= 0.0) {
            candidates[candidate_count] = ManifoldPoint(polygon_a[i] - ref_normal * (point_depth * 0.5), point_depth, 0.0, 0.0, 0.0, 0.0);
            candidate_count = candidate_count + 1u;
        }
    }
    if (candidate_count == 0u) {
        for (var i = 0u; i < 4u && candidate_count < CONTACT_MAX_POINTS; i = i + 1u) {
            let point_depth = dot(ref_center - incident_corners[i], ref_normal);
            if (point_depth >= 0.0) {
                candidates[candidate_count] = ManifoldPoint(incident_corners[i] - ref_normal * (point_depth * 0.5), point_depth, 0.0, 0.0, 0.0, 0.0);
                candidate_count = candidate_count + 1u;
            }
        }
    }
    contact.normal = signed;
    if (candidate_count == 0u) {
        manifold_push(&contact, ref_center - ref_normal * (face_depth * 0.5), face_depth);
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
    for (var i = 0u; i < candidate_count; i = i + 1u) {
        manifold_push(&contact, candidates[i].position, candidates[i].depth);
    }
    return contact;
}

fn box_capsule(
    box_body: Body, box_collider: Collider,
    capsule: Body, capsule_collider: Collider,
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
        let closest = closest_point_box(point, box_body.state, box_collider);
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
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= min(atomicLoad(&pair_count[0]), arrayLength(&pair_major))) {
        return;
    }
    contact_valid[index] = 0u;
    if (index > 0u && pair_major[index] == pair_major[index - 1u] && pair_minor[index] == pair_minor[index - 1u]) {
        return;
    }
    let first_slot = pair_major[index];
    let second_slot = pair_minor[index];
    let first_body_slot = first_slot / MAX_COLLIDERS_PER_BODY;
    let second_body_slot = second_slot / MAX_COLLIDERS_PER_BODY;
    let first = load_body(first_body_slot);
    let second = load_body(second_body_slot);
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
        let world_second = world_collider(second.state, second_collider);
        if (first_collider.kind == SHAPE_PLANE) {
            let world_plane = world_collider(first.state, first_collider);
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
        let world_first = world_collider(first.state, first_collider);
        if (second_collider.kind == SHAPE_PLANE) {
            let world_plane = world_collider(second.state, second_collider);
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
        let world_first = world_collider(first.state, first_collider);
        let world_second = world_collider(second.state, second_collider);
        let hit = convex_hit(world_first, world_second);
        if (hit.distance <= 0.0) {
            contact_emit(&contact, hit.normal);
            generated = true;
            if (!convex_pair_manifold(world_first, world_second, hit.normal, &contact)) {
                manifold_from_hit(&contact, hit);
            }
        }
    } else if (first_collider.kind == SHAPE_CYLINDER || first_collider.kind == SHAPE_HULL || second_collider.kind == SHAPE_CYLINDER || second_collider.kind == SHAPE_HULL) {
        let world_first = world_collider(first.state, first_collider);
        let world_second = world_collider(second.state, second_collider);
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
            let world_first = world_collider(first.state, first_collider);
            let world_second = world_collider(second.state, second_collider);
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
    contact.first_body_id = first.state.body_id;
    contact.second_body_id = second.state.body_id;
    contact.first_generation = first.state.generation;
    contact.second_generation = second.state.generation;
    contact.friction = material_combine(first_collider.friction, second_collider.friction, params.friction_combine);
    contact.restitution = material_combine(first_collider.restitution, second_collider.restitution, params.restitution_combine);
    contact.rolling_friction = max(first_collider.rolling_friction, second_collider.rolling_friction);
    contact.spin_friction = max(first_collider.spin_friction, second_collider.spin_friction);
    contact.events = (first_collider.flags & second_collider.flags) & (COLLIDER_EVENT_BEGIN_END | COLLIDER_EVENT_PERSIST);
    if (contact.point_count > 0u) {
        contacts_raw[index] = contact;
        contact_valid[index] = 1u;
    }
}
