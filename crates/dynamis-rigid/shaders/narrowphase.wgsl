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
@group(0) @binding(11) var<uniform> params: StepParams;
@group(0) @binding(12) var<storage, read> collider_owners: array<u32>;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn manifold_emit(contact: ptr<function, Contact>, normal: vec3f) {
    (*contact).point_count = 0u;
    (*contact).normal = normal;
    (*contact).triangle = NO_TRIANGLE;
}

fn pair_margin(first: Body, second: Body) -> f32 {
    if (!body_has_ccd(first.desc) && !body_has_ccd(second.desc)) {
        return params.contact_margin;
    }
    let closing = length(second.state.velocity - first.state.velocity) * params.dt;
    return params.contact_margin + closing;
}

fn sphere_sphere(
first: Body, first_collider: Collider,
second: Body, second_collider: Collider,
margin: f32,
) -> Contact {
    var contact: Contact;
    let first_center = box_center(first.state, first_collider);
    let second_center = box_center(second.state, second_collider);
    let delta = second_center - first_center;
    let distance = length(delta);
    var normal = sign_normalize(delta);
    if (distance <= 1e-6) {
        let relative = relative_velocity(first, second, second_center, first_center);
        normal = select(normal, -normalize(relative), length(relative) > 1e-6);
    }
    manifold_emit(&contact, normal);
    let radius_sum = first_collider.radius + second_collider.radius;
    if (distance > radius_sum + margin) {
        return contact;
    }
    let depth = radius_sum - distance;
    let point = first_center + normal * (first_collider.radius - depth * 0.5);
    manifold_push(&contact, point, depth, feature_point());
    return contact;
}

struct BoxFace {
    normal: vec3f,
    clearance: f32,
    index: u32,
}

fn box_nearest_face(point: vec3f, box_body: Body, box_collider: Collider) -> BoxFace {
    let q = quat_mul(box_body.state.orientation, box_collider.local_rotation);
    let local = quat_rotate(quat_conjugate(q), point - box_center(box_body.state, box_collider));
    let clearance = box_collider.half_extents - abs(local);
    let axis = smallest_axis(clearance);
    var facing = vec3f(0.0);
    if (axis == 0u) {
        facing = vec3f(select(1.0, -1.0, local.x > 0.0), 0.0, 0.0);
    } else if (axis == 1u) {
        facing = vec3f(0.0, select(1.0, -1.0, local.y > 0.0), 0.0);
    } else {
        facing = vec3f(0.0, 0.0, select(1.0, -1.0, local.z > 0.0));
    }
    var face: BoxFace;
    face.normal = -quat_rotate(q, facing);
    face.clearance = clearance[axis];
    face.index = axis * 2u + select(0u, 1u, local[axis] > 0.0);
    return face;
}

fn sphere_box(
sphere: Body, sphere_collider: Collider,
box_body: Body, box_collider: Collider,
margin: f32,
) -> Contact {
    var contact: Contact;
    let center = sphere.state.position + quat_rotate(sphere.state.orientation, sphere_collider.local_offset);
    let closest = closest_point_box(center, box_body.state, box_collider);
    let delta = closest - center;
    let distance = length(delta);
    let radius = sphere_collider.radius;
    var normal = sign_normalize(delta);
    var depth = radius - distance;
    if (distance <= 1e-6) {
        let face = box_nearest_face(center, box_body, box_collider);
        normal = -face.normal;
        depth = radius + face.clearance;
    }
    manifold_emit(&contact, normal);
    if (depth <= -margin) {
        return contact;
    }
    let point = closest - normal * (depth * 0.5);
    manifold_push(&contact, point, depth, feature_point());
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
margin: f32,
) -> Contact {
    var contact: Contact;
    let seg = capsule_segment(capsule, capsule_collider);
    let center = sphere.state.position + quat_rotate(sphere.state.orientation, sphere_collider.local_offset);
    let closest = closest_point_segment(center, seg.start, seg.end);
    let delta = closest - center;
    let distance = length(delta);
    let radius_sum = sphere_collider.radius + capsule_collider.radius;
    manifold_emit(&contact, sign_normalize(delta));
    if (distance > radius_sum + margin) {
        return contact;
    }
    let normal = sign_normalize(delta);
    let depth = radius_sum - distance;
    let point = closest - normal * (depth * 0.5);
    manifold_push(&contact, point, depth, feature_point());
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
margin: f32,
) -> Contact {
    var contact: Contact;
    let seg_a = capsule_segment(first, first_collider);
    let seg_b = capsule_segment(second, second_collider);
    let closest = closest_points_segments(seg_a.start, seg_a.end, seg_b.start, seg_b.end);
    let delta = closest.end - closest.start;
    let distance = length(delta);
    let radius_sum = first_collider.radius + second_collider.radius;
    manifold_emit(&contact, sign_normalize(delta));
    if (distance > radius_sum + margin) {
        return contact;
    }
    let normal = sign_normalize(delta);
    let depth = radius_sum - distance;
    let point = (closest.start + closest.end) * 0.5;
    manifold_push(&contact, point, depth, feature_point());
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

fn box_face(
    body: Body,
    collider: Collider,
    face_normal: vec3f,
    corners: ptr<function, array<vec3f, 4>>,
    ids: ptr<function, array<u32, 4>>,
    out_face: ptr<function, u32>,
) -> vec3f {
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
    (*ids)[0] = box_corner_id(axis, sign, 1.0, 1.0);
    (*ids)[1] = box_corner_id(axis, sign, -1.0, 1.0);
    (*ids)[2] = box_corner_id(axis, sign, -1.0, -1.0);
    (*ids)[3] = box_corner_id(axis, sign, 1.0, -1.0);
    *out_face = axis * 2u + select(0u, 1u, sign > 0.0);
    return face_center;
}

fn clip_polygon(
    points: array<vec3f, 8>,
    ids: array<u32, 8>,
    count: u32,
    plane_point: vec3f,
    plane_normal: vec3f,
    group: u32,
    side: u32,
    out_points: ptr<function, array<vec3f, 8>>,
    out_ids: ptr<function, array<u32, 8>>,
) -> u32 {
    var out_count = 0u;
    for (var i = 0u; i < count; i = i + 1u) {
        let current = points[i];
        let next = points[(i + 1u) % count];
        let current_dist = dot(current - plane_point, plane_normal) - CLIP_MARGIN;
        let next_dist = dot(next - plane_point, plane_normal) - CLIP_MARGIN;
        if (current_dist <= 0.0) {
            if (out_count < 8u) {
                (*out_points)[out_count] = current;
                (*out_ids)[out_count] = ids[i];
                out_count = out_count + 1u;
            }
        }
        if (current_dist * next_dist < 0.0) {
            let t = current_dist / (current_dist - next_dist);
            if (out_count < 8u) {
                (*out_points)[out_count] = current + (next - current) * t;
                (*out_ids)[out_count] = feature_field_clip(group * 32u + side * 8u + i);
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
margin: f32,
) -> Contact {
    var contact: Contact;
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
    var edge_first_axis = 0u;
    var edge_second_axis = 0u;
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
                edge_first_axis = i;
                edge_second_axis = j;
            }
        }
    }
    let extent = max(max(left.half_extents.x, left.half_extents.y), left.half_extents.z)
    + max(max(right.half_extents.x, right.half_extents.y), right.half_extents.z);
    let edge_axis_separates = edge_depth + FACE_AXIS_BIAS * extent < face_depth;
    let depth = select(face_depth, edge_depth, edge_axis_separates);
    if (depth <= -margin) {
        return contact;
    }
    let axis = select(face_axis, edge_axis, edge_axis_separates);
    let signed = select(axis, -axis, dot(axis, delta) < 0.0);
    manifold_emit(&contact, signed);
    if (edge_axis_separates) {
        let point = (box_support(first, first_collider, signed) + box_support(second, second_collider, -signed)) * 0.5;
        manifold_push(&contact, point, depth, feature_edge(edge_first_axis, edge_second_axis));
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
    var ref_ids: array<u32, 4>;
    var ref_face = 0u;
    let ref_center = box_face(reference.body, reference.collider, ref_normal, &ref_corners, &ref_ids, &ref_face);
    var incident_axis = 0u;
    for (var i = 1u; i < 3u; i = i + 1u) {
        if (abs(dot(incident.axes[i], ref_normal)) > abs(dot(incident.axes[incident_axis], ref_normal))) {
            incident_axis = i;
        }
    }
    let incident_normal = incident.axes[incident_axis] * select(1.0, -1.0, dot(incident.axes[incident_axis], ref_normal) > 0.0);
    var incident_corners: array<vec3f, 4>;
    var incident_ids: array<u32, 4>;
    var incident_face = 0u;
    box_face(incident.body, incident.collider, incident_normal, &incident_corners, &incident_ids, &incident_face);
    let ref_field = feature_field_face(ref_face);
    var polygon_a: array<vec3f, 8>;
    var polygon_b: array<vec3f, 8>;
    var polygon_ids_a: array<u32, 8>;
    var polygon_ids_b: array<u32, 8>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        polygon_a[i] = incident_corners[i];
        polygon_ids_a[i] = incident_ids[i];
    }
    var polygon_count = 4u;
    for (var side = 0u; side < 4u; side = side + 1u) {
        let current = ref_corners[side];
        let next = ref_corners[(side + 1u) % 4u];
        let plane_normal = normalize(cross(next - current, ref_normal));
        if (side % 2u == 0u) {
            polygon_count = clip_polygon(polygon_a, polygon_ids_a, polygon_count, current, plane_normal, incident_face, side, &polygon_b, &polygon_ids_b);
        } else {
            polygon_count = clip_polygon(polygon_b, polygon_ids_b, polygon_count, current, plane_normal, incident_face, side, &polygon_a, &polygon_ids_a);
        }
        if (polygon_count == 0u) {
            break;
        }
    }
    var candidates: array<ManifoldPoint, MANIFOLD_CANDIDATES>;
    var candidate_count = 0u;
    for (var i = 0u; i < polygon_count && candidate_count < MANIFOLD_CANDIDATES; i = i + 1u) {
        let point_depth = dot(ref_center - polygon_a[i], ref_normal);
        if (point_depth >= -margin) {
            candidates[candidate_count] = manifold_candidate(
                polygon_a[i] - ref_normal * (point_depth * 0.5),
                point_depth,
                select(
                    feature_face(ref_field, polygon_ids_a[i]),
                    feature_face(polygon_ids_a[i], ref_field),
                    face_on_right,
                ),
            );
            candidate_count = candidate_count + 1u;
        }
    }
    if (candidate_count == 0u) {
        for (var i = 0u; i < 4u && candidate_count < MANIFOLD_CANDIDATES; i = i + 1u) {
            let point_depth = dot(ref_center - incident_corners[i], ref_normal);
            if (point_depth >= -margin) {
                candidates[candidate_count] = manifold_candidate(
                    incident_corners[i] - ref_normal * (point_depth * 0.5),
                    point_depth,
                    select(
                        feature_face(ref_field, incident_ids[i]),
                        feature_face(incident_ids[i], ref_field),
                        face_on_right,
                    ),
                );
                candidate_count = candidate_count + 1u;
            }
        }
    }
    if (candidate_count == 0u) {
        manifold_push(&contact, ref_center - ref_normal * (face_depth * 0.5), face_depth, feature_point());
        return contact;
    }
    manifold_keep(&contact, &candidates, candidate_count);
    return contact;
}

fn contact_triangle_of(contact: Contact, world_geom: bool) -> u32 {
    return select(NO_TRIANGLE, contact.triangle, world_geom);
}

fn manifold_from_hit(contact: ptr<function, Contact>, hit: ShapeHit, margin: f32) {
    if (hit.distance <= margin) {
        manifold_push(contact, hit.point, -hit.distance, feature_point());
    }
}

fn plane_convex(plane: WorldShape, convex: WorldShape, margin: f32) -> Contact {
    var contact: Contact;
    let n = plane_normal(plane);
    let center_side = dot(convex.center - plane.center, n);
    let facing = select(n, -n, center_side < 0.0);
    manifold_emit(&contact, facing);
    var points: array<vec3f, FEATURE_MAX>;
    var ids: array<u32, FEATURE_MAX>;
    var feature = 0u;
    var ring = false;
    let count = shape_feature(convex, -facing, &points, &ids, &feature, &ring);
    var candidates: array<ManifoldPoint, MANIFOLD_CANDIDATES>;
    var candidate_count = 0u;
    for (var i = 0u; i < count; i = i + 1u) {
        let depth = dot(plane.center - points[i], facing);
        if (depth <= -margin) {
            continue;
        }
        candidates[candidate_count] = manifold_candidate(
            points[i] + facing * (depth * 0.5),
            depth,
            feature_vertex(feature_field_face(0u), ids[i]),
        );
        candidate_count = candidate_count + 1u;
    }
    manifold_keep(&contact, &candidates, candidate_count);
    return contact;
}

fn scaled_shape(collider: Collider) -> bool {
    return collider.scale.x != 1.0 || collider.scale.y != 1.0 || collider.scale.z != 1.0;
}

fn work(index: u32) {
    contact_valid[index] = 0u;
    let first_slot = pair_major[index];
    let second_slot = pair_minor[index];
    let first_body_slot = collider_owners[first_slot];
    let second_body_slot = collider_owners[second_slot];
    let first = load_body(first_body_slot);
    let second = load_body(second_body_slot);
    if (first_body_slot == NO_BODY || second_body_slot == NO_BODY || first_body_slot == second_body_slot) {
        return;
    }
    if (!body_moves(first.desc) && !body_moves(second.desc)) {
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
    if (joined_by_joint(first_body_slot, second_body_slot)) {
        return;
    }
    var contact: Contact;
    var generated = false;
    let sensor = collider_is_sensor(first_collider) || collider_is_sensor(second_collider);
    let first_world_geom = shape_world_geometry(first_collider.kind);
    let second_world_geom = shape_world_geometry(second_collider.kind);
    if (first_world_geom && second_world_geom) {
        return;
    }    let margin = pair_margin(first, second);
    if (first_world_geom) {
        let scene = world_collider(first.state, first_collider);
        let world_second = world_collider(second.state, second_collider);
        if (first_collider.kind == SHAPE_PLANE) {
            contact = plane_convex(scene, world_second, margin);
            generated = contact.point_count > 0u;
        } else {
            let hit = scene_convex_hit(scene, world_second);
            if (hit.distance <= margin) {
                manifold_emit(&contact, hit.normal);
                contact.triangle = hit.triangle;
                generated = true;
                if (!scene_convex_manifold(scene, world_second, hit.triangle, margin, &contact)) {
                    manifold_from_hit(&contact, hit, margin);
                }
            }
        }
    } else if (second_world_geom) {
        let scene = world_collider(second.state, second_collider);
        let world_first = world_collider(first.state, first_collider);
        if (second_collider.kind == SHAPE_PLANE) {
            let swapped = plane_convex(scene, world_first, margin);
            contact = swapped;
            contact.normal = -contact.normal;
            contact_mirror_features(&contact);
            generated = contact.point_count > 0u;
        } else {
            let hit = scene_convex_hit(scene, world_first);
            if (hit.distance <= margin) {
                manifold_emit(&contact, -hit.normal);
                contact.triangle = hit.triangle;
                generated = true;
                if (!scene_convex_manifold(scene, world_first, hit.triangle, margin, &contact)) {
                    let reversed_hit = ShapeHit(hit.distance, hit.point, -hit.normal, hit.triangle);
                    manifold_from_hit(&contact, reversed_hit, margin);
                }
                contact_mirror_features(&contact);
            }
        }
    } else if (scaled_shape(first_collider) || scaled_shape(second_collider)) {
        let world_first = world_collider(first.state, first_collider);
        let world_second = world_collider(second.state, second_collider);
        let hit = convex_hit(world_first, world_second);
        if (hit.distance <= margin) {
            manifold_emit(&contact, hit.normal);
            generated = true;
            if (!convex_pair_manifold(world_first, world_second, hit.normal, &contact)) {
                manifold_from_hit(&contact, hit, margin);
            }
        }
    } else if (!shape_analytic(first_collider.kind) || !shape_analytic(second_collider.kind)) {
        let world_first = world_collider(first.state, first_collider);
        let world_second = world_collider(second.state, second_collider);
        let hit = convex_hit(world_first, world_second);
        if (hit.distance <= margin) {
            manifold_emit(&contact, hit.normal);
            generated = true;
            if (!convex_pair_manifold(world_first, world_second, hit.normal, &contact)) {
                manifold_from_hit(&contact, hit, margin);
            }
        }
    } else {
        let shape_a = first_collider.kind;
        let shape_b = second_collider.kind;
        if (shape_a == SHAPE_SPHERE && shape_b == SHAPE_SPHERE) {
            contact = sphere_sphere(first, first_collider, second, second_collider, margin);
            generated = true;
        } else if (shape_a == SHAPE_SPHERE && shape_b == SHAPE_CUBOID) {
            contact = sphere_box(first, first_collider, second, second_collider, margin);
            generated = true;
        } else if (shape_a == SHAPE_SPHERE && shape_b == SHAPE_CAPSULE) {
            contact = sphere_capsule(first, first_collider, second, second_collider, margin);
            generated = true;
        } else if (shape_a == SHAPE_CUBOID && shape_b == SHAPE_SPHERE) {
            let swapped = sphere_box(second, second_collider, first, first_collider, margin);
            contact = swapped;
            contact.normal = -contact.normal;
            generated = true;
        } else if (shape_a == SHAPE_CUBOID && shape_b == SHAPE_CUBOID) {
            contact = box_box_sat(first, first_collider, second, second_collider, margin);
            generated = true;
        } else if (shape_a == SHAPE_CAPSULE && shape_b == SHAPE_SPHERE) {
            let swapped = sphere_capsule(second, second_collider, first, first_collider, margin);
            contact = swapped;
            contact.normal = -contact.normal;
            generated = true;
        } else if (shape_a == SHAPE_CAPSULE && shape_b == SHAPE_CAPSULE) {
            contact = capsule_capsule(first, first_collider, second, second_collider, margin);
            generated = true;
        } else {
            let world_first = world_collider(first.state, first_collider);
            let world_second = world_collider(second.state, second_collider);
            let hit = convex_hit(world_first, world_second);
            if (hit.distance <= margin) {
                manifold_emit(&contact, hit.normal);
                generated = true;
                if (!convex_pair_manifold(world_first, world_second, hit.normal, &contact)) {
                    manifold_from_hit(&contact, hit, margin);
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
    let first_triangle = contact_triangle_of(contact, first_world_geom);
    let second_triangle = contact_triangle_of(contact, second_world_geom);
    let first_surface = surface_material(first_collider, first_triangle);
    let second_surface = surface_material(second_collider, second_triangle);
    contact.surface = select(
        triangle_surface_index(first_collider, first_triangle),
        triangle_surface_index(second_collider, second_triangle),
        second_world_geom,
    );
    contact.friction = material_combine(first_surface.friction, second_surface.friction, params.friction_combine);
    contact.restitution = material_combine(first_surface.restitution, second_surface.restitution, params.restitution_combine);
    contact.rolling_friction = max(first_surface.rolling_friction, second_surface.rolling_friction);
    contact.spin_friction = max(first_surface.spin_friction, second_surface.spin_friction);
    let relaxation = contact_relaxation(first_collider, second_collider);
    contact.relaxation = relaxation.x;
    contact.damping_ratio = relaxation.y;
    contact.events = (first_collider.flags & second_collider.flags) & (EVENT_MODE_BEGIN_END | EVENT_MODE_PERSIST);
    if (contact.point_count > 0u) {
        manifold_anchor(&contact, first, second);
        contacts_raw[index] = contact;
        contact_valid[index] = 1u;
    }
}

