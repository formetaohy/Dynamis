fn ray_sphere(origin: vec3f, direction: vec3f, extent: f32, center: vec3f, radius: f32) -> ShapeHit {
    let offset = origin - center;
    let projection = dot(offset, direction);
    let discriminant = projection * projection - (dot(offset, offset) - radius * radius);
    if (discriminant < 0.0) {
        return no_hit();
    }
    let root = sqrt(discriminant);
    var t = -projection - root;
    if (t < 0.0) {
        t = -projection + root;
    }
    if (t < 0.0 || t > extent) {
        return no_hit();
    }
    let point = origin + direction * t;
    let normal = normalize(point - center);
    return ShapeHit(t, point, normal);
}

fn ray_box(origin: vec3f, direction: vec3f, extent: f32, center: vec3f, q: vec4f, half_extents: vec3f) -> ShapeHit {
    let local_origin = quat_rotate(quat_conjugate(q), origin - center);
    let local_direction = quat_rotate(quat_conjugate(q), direction);
    let inv = 1.0 / local_direction;
    var tmin = 0.0;
    var tmax = extent;
    var normal_axis = vec3f(0.0);
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        var o = local_origin[axis];
        var d = local_direction[axis];
        var n = vec3f(0.0);
        if (axis == 0u) {
            n = vec3f(1.0, 0.0, 0.0);
        } else if (axis == 1u) {
            n = vec3f(0.0, 1.0, 0.0);
        } else {
            n = vec3f(0.0, 0.0, 1.0);
        }
        let half = half_extents[axis];
        var t1 = (-half - o) * inv[axis];
        var t2 = (half - o) * inv[axis];
        var normal_candidate = n;
        if (t1 > t2) {
            let tmp = t1;
            t1 = t2;
            t2 = tmp;
            normal_candidate = -n;
        }
        if (t1 > tmin) {
            tmin = t1;
            normal_axis = normal_candidate;
        }
        if (t2 < tmax) {
            tmax = t2;
        }
        if (tmin > tmax) {
            return no_hit();
        }
    }
    if (tmin < 0.0 || tmin > extent) {
        return no_hit();
    }
    let local_point = local_origin + local_direction * tmin;
    let point = center + quat_rotate(q, local_point);
    let normal = quat_rotate(q, normal_axis);
    return ShapeHit(tmin, point, normal);
}

fn ray_segment(origin: vec3f, direction: vec3f, extent: f32, seg: Segment, radius: f32) -> ShapeHit {
    var best = no_hit();
    for (var sample = 0u; sample <= 8u; sample = sample + 1u) {
        let t = f32(sample) * (1.0 / 8.0);
        let sphere_center = seg.start + (seg.end - seg.start) * t;
        let hit = ray_sphere(origin, direction, extent, sphere_center, radius);
        if (hit.distance < best.distance) {
            best = hit;
        }
    }
    return best;
}

fn ray_capsule(origin: vec3f, direction: vec3f, extent: f32, center: vec3f, axis: vec3f, half_height: f32, radius: f32) -> ShapeHit {
    let seg = Segment(center - axis * half_height, center + axis * half_height);
    return ray_segment(origin, direction, extent, seg, radius);
}

fn ray_cylinder(origin: vec3f, direction: vec3f, extent: f32, center: vec3f, axis: vec3f, half_height: f32, radius: f32) -> ShapeHit {
    var best = no_hit();
    let seg = Segment(center - axis * half_height, center + axis * half_height);
    for (var sample = 0u; sample <= 4u; sample = sample + 1u) {
        let t = f32(sample) * (1.0 / 4.0);
        let circle_center = seg.start + (seg.end - seg.start) * t;
        let hit = ray_sphere(origin, direction, extent, circle_center, radius);
        if (hit.distance < best.distance) {
            best = hit;
        }
    }
    return best;
}

fn ray_triangle(origin: vec3f, direction: vec3f, extent: f32, a: vec3f, b: vec3f, c: vec3f) -> ShapeHit {
    let edge1 = b - a;
    let edge2 = c - a;
    let p = cross(direction, edge2);
    let det = dot(edge1, p);
    if (abs(det) < 1e-8) {
        return no_hit();
    }
    let inv_det = 1.0 / det;
    let t_vec = origin - a;
    let u = dot(t_vec, p) * inv_det;
    if (u < 0.0 || u > 1.0) {
        return no_hit();
    }
    let q = cross(t_vec, edge1);
    let v = dot(direction, q) * inv_det;
    if (v < 0.0 || u + v > 1.0) {
        return no_hit();
    }
    let t = dot(edge2, q) * inv_det;
    if (t < 0.0 || t > extent) {
        return no_hit();
    }
    let point = origin + direction * t;
    let normal = sign_normalize(cross(edge1, edge2));
    return ShapeHit(t, point, select(normal, -normal, det < 0.0));
}

fn scene_local_point(scene: WorldShape, point: vec3f) -> vec3f {
    return quat_rotate(quat_conjugate(scene.rotation), point - scene.center);
}

fn scene_local_shape(scene: WorldShape, world: WorldShape) -> WorldShape {
    var local = world;
    local.center = scene_local_point(scene, world.center);
    local.rotation = quat_mul(quat_conjugate(scene.rotation), world.rotation);
    return local;
}

fn scene_place_point(scene: WorldShape, point: vec3f) -> vec3f {
    return scene.center + quat_rotate(scene.rotation, point);
}

fn scene_place_normal(scene: WorldShape, normal: vec3f) -> vec3f {
    return sign_normalize(quat_rotate(scene.rotation, normal));
}

fn ray_scene(scene: WorldShape, origin: vec3f, direction: vec3f, extent: f32) -> ShapeHit {
    let local = scene_raycast(
        scene,
        scene_local_point(scene, origin),
        quat_rotate(quat_conjugate(scene.rotation), direction),
        extent,
    );
    if (local.distance == NO_HIT) {
        return local;
    }
    var hit = local;
    hit.point = scene_place_point(scene, local.point);
    hit.normal = scene_place_normal(scene, local.normal);
    return hit;
}

fn scene_raycast(scene: WorldShape, origin: vec3f, direction: vec3f, extent: f32) -> ShapeHit {
    let source = shape_sources[scene.source];
    var stack: array<u32, 64>;
    var stack_count = 1u;
    stack[0] = source.node_offset;
    var best = no_hit();
    while (stack_count > 0u) {
        stack_count = stack_count - 1u;
        let node_index = stack[stack_count];
        let node = shape_nodes[node_index];
        let t = aabb_ray_hit(node.min * scene.scale, node.max * scene.scale, origin, direction, extent);
        if (t == NO_HIT) {
            continue;
        }
        if (node.leaf == 1u) {
            for (var i = 0u; i < node.right; i = i + 1u) {
                let tri = shape_triangles[source.triangle_offset + node.left + i];
                let a = shape_vertices[source.vertex_offset + tri.a].xyz * scene.scale;
                let b = shape_vertices[source.vertex_offset + tri.b].xyz * scene.scale;
                let c = shape_vertices[source.vertex_offset + tri.c].xyz * scene.scale;
                let hit = ray_triangle(origin, direction, extent, a, b, c);
                if (hit.distance < best.distance) {
                    best = hit;
                }
            }
        } else {
            if (stack_count + 2u > 64u) {
                return best;
            }
            stack[stack_count] = node.left;
            stack_count = stack_count + 1u;
            stack[stack_count] = node.right;
            stack_count = stack_count + 1u;
        }
    }
    return best;
}

fn triangle_points(source_index: u32, triangle_index: u32, scale: vec3f) -> array<vec3f, 3> {
    let source = shape_sources[source_index];
    let tri = shape_triangles[source.triangle_offset + triangle_index];
    let points: array<vec3f, 3> = array(
        shape_vertices[source.vertex_offset + tri.a].xyz * scale,
        shape_vertices[source.vertex_offset + tri.b].xyz * scale,
        shape_vertices[source.vertex_offset + tri.c].xyz * scale,
    );
    return points;
}

fn plane_distance(triangle: u32, source_index: u32, scale: vec3f, world: WorldShape) -> ConvexClosest {
    let points = triangle_points(source_index, triangle, scale);
    var n = sign_normalize(cross(points[1] - points[0], points[2] - points[0]));
    var offset = dot(world.center - points[0], n);
    if (offset < 0.0) {
        n = -n;
        offset = -offset;
    }
    let deep = support(world, -n);
    let extent = dot(deep - world.center, -n);
    let separation = offset - extent;
    var result: ConvexClosest;
    result.distance = separation;
    result.point_a = world.center - n * extent;
    result.point_b = result.point_a;
    result.normal = n;
    result.penetrating = separation < 0.0;
    return result;
}

fn scene_convex_closest(scene: WorldShape, world: WorldShape, out_triangle: ptr<function, u32>) -> ConvexClosest {
    var result = scene_local_closest(scene, world, out_triangle);
    if (result.distance == 3.402823466e38) {
        return result;
    }
    result.point_a = scene_place_point(scene, result.point_a);
    result.point_b = scene_place_point(scene, result.point_b);
    result.normal = scene_place_normal(scene, result.normal);
    return result;
}

fn scene_local_closest(scene: WorldShape, world: WorldShape, out_triangle: ptr<function, u32>) -> ConvexClosest {
    let source = shape_sources[scene.source];
    let local = scene_local_shape(scene, world);
    var result: ConvexClosest;
    result.distance = 3.402823466e38;
    result.point_a = vec3f(0.0);
    result.point_b = vec3f(0.0);
    result.normal = vec3f(0.0, 1.0, 0.0);
    result.penetrating = false;
    if (source.node_count == 0u) {
        for (var i = 0u; i < source.triangle_count; i = i + 1u) {
            let candidate = plane_distance(i, scene.source, scene.scale, local);
            if (candidate.penetrating) { result = candidate; *out_triangle = i; return result; }
            if (candidate.distance < result.distance) { result = candidate; *out_triangle = i; }
        }
        return result;
    }
    let world_aabb = world_aabb_of(local);
    let pad = (world_aabb.max - world_aabb.min) * 0.5;
    var stack: array<u32, 64>;
    var stack_count = 1u;
    stack[0] = source.node_offset;
    while (stack_count > 0u) {
        stack_count = stack_count - 1u;
        let node_index = stack[stack_count];
        let node = shape_nodes[node_index];
        if (node.min.x * scene.scale.x - pad.x > world_aabb.max.x || node.max.x * scene.scale.x + pad.x < world_aabb.min.x ||
            node.min.y * scene.scale.y - pad.y > world_aabb.max.y || node.max.y * scene.scale.y + pad.y < world_aabb.min.y ||
            node.min.z * scene.scale.z - pad.z > world_aabb.max.z || node.max.z * scene.scale.z + pad.z < world_aabb.min.z) {
            continue;
        }
        if (node.leaf == 1u) {
            for (var i = 0u; i < node.right; i = i + 1u) {
                let candidate = plane_distance(node.left + i, scene.source, scene.scale, local);
                if (candidate.penetrating) {
                    result = candidate;
                    *out_triangle = node.left + i;
                    return result;
                }
                if (candidate.distance < result.distance) {
                    result = candidate;
                    *out_triangle = node.left + i;
                }
            }
        } else {
            if (stack_count + 2u > 64u) {
                continue;
            }
            stack[stack_count] = node.left;
            stack_count = stack_count + 1u;
            stack[stack_count] = node.right;
            stack_count = stack_count + 1u;
        }
    }
    return result;
}

fn scene_convex_hit(scene: WorldShape, world: WorldShape) -> ShapeHit {
    var triangle: u32;
    triangle = 0u;
    let closest = scene_convex_closest(scene, world, &triangle);
    if (closest.distance == 3.402823466e38) {
        return no_hit();
    }
    return ShapeHit(closest.distance, (closest.point_a + closest.point_b) * 0.5, closest.normal);
}

fn triangle_plane_margin(triangle: u32, source_index: u32, scale: vec3f, world: WorldShape) -> f32 {
    let points = triangle_points(source_index, triangle, scale);
    let n = sign_normalize(cross(points[1] - points[0], points[2] - points[0]));
    let offset = dot(world.center - points[0], n);
    return abs(offset);
}

fn scene_plane_margin(scene: WorldShape, world: WorldShape, out_triangle: ptr<function, u32>) -> f32 {
    let source = shape_sources[scene.source];
    let local = scene_local_shape(scene, world);
    let world_aabb = world_aabb_of(local);
    let pad = (world_aabb.max - world_aabb.min) * 0.5;
    var best = 3.402823466e38;
    var stack: array<u32, 64>;
    var stack_count = 1u;
    stack[0] = source.node_offset;
    if (source.node_count == 0u) {
        for (var i = 0u; i < source.triangle_count; i = i + 1u) {
            let margin = triangle_plane_margin(i, scene.source, scene.scale, local);
            if (margin < best) {
                best = margin;
                *out_triangle = i;
            }
        }
        return best;
    }
    while (stack_count > 0u) {
        stack_count = stack_count - 1u;
        let node_index = stack[stack_count];
        let node = shape_nodes[node_index];
        if (node.min.x * scene.scale.x - pad.x > world_aabb.max.x || node.max.x * scene.scale.x + pad.x < world_aabb.min.x ||
            node.min.y * scene.scale.y - pad.y > world_aabb.max.y || node.max.y * scene.scale.y + pad.y < world_aabb.min.y ||
            node.min.z * scene.scale.z - pad.z > world_aabb.max.z || node.max.z * scene.scale.z + pad.z < world_aabb.min.z) {
            continue;
        }
        if (node.leaf == 1u) {
            for (var i = 0u; i < node.right; i = i + 1u) {
                let margin = triangle_plane_margin(node.left + i, scene.source, scene.scale, local);
                if (margin < best) {
                    best = margin;
                    *out_triangle = node.left + i;
                }
            }
        } else {
            if (stack_count + 2u > 64u) {
                continue;
            }
            stack[stack_count] = node.left;
            stack_count = stack_count + 1u;
            stack[stack_count] = node.right;
            stack_count = stack_count + 1u;
        }
    }
    return best;
}

fn scene_sweep_hit(
    scene: WorldShape,
    moving: WorldShape,
    start: vec3f,
    direction: vec3f,
    max_dist: f32,
) -> ShapeHit {
    let root = shape_nodes[shape_sources[scene.source].node_offset];
    let moving_bounds = world_aabb_of(moving);
    let probe_radius = length((moving_bounds.max - moving_bounds.min) * 0.5);
    if (probe_radius <= 0.0) {
        return no_hit();
    }
    var triangle = 0u;
    var moved = moving;
    moved.center = start;
    let origin = scene_plane_margin(scene, moved, &triangle);
    if (origin <= probe_radius) {
        return no_hit();
    }
    let box_center = scene_place_point(scene, (root.min + root.max) * 0.5 * scene.scale);
    let box_half = (root.max - root.min) * 0.5 * scene.scale + vec3f(probe_radius);
    let entry = ray_box(start, direction, max_dist, box_center, scene.rotation, box_half);
    if (entry.distance >= max_dist) {
        return no_hit();
    }
    let scan_start = max(entry.distance - probe_radius, 0.0);
    var scan = scan_start;
    var lo = scan_start;
    var hi = -1.0;
    for (var iter = 0u; iter < 4096u; iter = iter + 1u) {
        scan = scan + probe_radius;
        if (scan >= max_dist) {
            return no_hit();
        }
        moved.center = start + direction * scan;
        let margin = scene_plane_margin(scene, moved, &triangle);
        if (margin <= probe_radius) {
            hi = scan;
            break;
        }
        lo = scan;
    }
    if (hi < 0.0) {
        return no_hit();
    }
    for (var iter = 0u; iter < 24u; iter = iter + 1u) {
        let mid = (lo + hi) * 0.5;
        moved.center = start + direction * mid;
        let margin = scene_plane_margin(scene, moved, &triangle);
        if (margin <= probe_radius) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    moved.center = start + direction * (hi + 1e-4);
    let closest = scene_convex_closest(scene, moved, &triangle);
    if (closest.penetrating) {
        return ShapeHit(hi, closest.point_a, closest.normal);
    }
    return ShapeHit(hi, (closest.point_a + closest.point_b) * 0.5, closest.normal);
}

fn ray_scaled_shape(world: WorldShape, origin: vec3f, direction: vec3f, extent: f32, expand: f32) -> ShapeHit {
    let unscaled = world.scale.x == 1.0 && world.scale.y == 1.0 && world.scale.z == 1.0;
    if (unscaled) {
        if (world.kind == SHAPE_SPHERE) {
            return ray_sphere(origin, direction, extent, world.center, world.radius + expand);
        }
        if (world.kind == SHAPE_CUBOID) {
            return ray_box(origin, direction, extent, world.center, world.rotation, world.half_extents + vec3f(expand));
        }
        if (world.kind == SHAPE_CAPSULE) {
            let axis = shape_axis(world);
            return ray_capsule(origin, direction, extent, world.center, axis, world.half_height, world.radius + expand);
        }
        if (world.kind == SHAPE_CYLINDER) {
            let axis = shape_axis(world);
            return ray_cylinder(origin, direction, extent, world.center, axis, world.half_height, world.radius + expand);
        }
        return no_hit();
    }
    let inv_rotation = quat_conjugate(world.rotation);
    let unscale = 1.0 / world.scale;
    let local_origin = quat_rotate(inv_rotation, origin - world.center) * unscale;
    let local_direction = quat_rotate(inv_rotation, direction) * unscale;
    var local: ShapeHit;
    if (world.kind == SHAPE_SPHERE) {
        local = ray_sphere(local_origin, local_direction, extent, vec3f(0.0), world.radius + expand);
    } else if (world.kind == SHAPE_CAPSULE) {
        local = ray_capsule(local_origin, local_direction, extent, vec3f(0.0), vec3f(0.0, 1.0, 0.0), world.half_height, world.radius + expand);
    } else if (world.kind == SHAPE_CYLINDER) {
        local = ray_cylinder(local_origin, local_direction, extent, vec3f(0.0), vec3f(0.0, 1.0, 0.0), world.half_height, world.radius + expand);
    } else {
        return no_hit();
    }
    if (local.distance == NO_HIT) {
        return no_hit();
    }
    var hit: ShapeHit;
    hit.distance = local.distance;
    hit.point = world.center + quat_rotate(world.rotation, (local_origin + local_direction * local.distance) * world.scale);
    hit.normal = quat_rotate(world.rotation, sign_normalize(local.normal * unscale));
    return hit;
}

fn scene_convex_manifold(
    scene: WorldShape,
    world: WorldShape,
    margin: f32,
    contact: ptr<function, Contact>,
) -> bool {
    let source = shape_sources[scene.source];
    let local = scene_local_shape(scene, world);
    let world_aabb = world_aabb_of(local);
    let pad = (world_aabb.max - world_aabb.min) * 0.5;
    var candidates: array<ManifoldPoint, 8>;
    var candidate_count = 0u;
    if (source.node_count == 0u) {
        for (var i = 0u; i < source.triangle_count; i = i + 1u) {
            let probe = plane_distance(i, scene.source, scene.scale, local);
            if (probe.distance <= margin) {
                candidates[candidate_count] = ManifoldPoint(scene_place_point(scene, probe.point_a), -probe.distance, 0.0, 0.0, 0.0, 0.0);
                candidate_count = candidate_count + 1u;
                if (candidate_count >= 8u) {
                    break;
                }
            }
        }
    } else {
        var stack: array<u32, 64>;
        var stack_count = 1u;
        stack[0] = source.node_offset;
        while (stack_count > 0u && candidate_count < 8u) {
            stack_count = stack_count - 1u;
            let node_index = stack[stack_count];
            let node = shape_nodes[node_index];
            if (node.min.x * scene.scale.x - pad.x > world_aabb.max.x || node.max.x * scene.scale.x + pad.x < world_aabb.min.x ||
                node.min.y * scene.scale.y - pad.y > world_aabb.max.y || node.max.y * scene.scale.y + pad.y < world_aabb.min.y ||
                node.min.z * scene.scale.z - pad.z > world_aabb.max.z || node.max.z * scene.scale.z + pad.z < world_aabb.min.z) {
                continue;
            }
            if (node.leaf == 1u) {
                for (var i = 0u; i < node.right && candidate_count < 8u; i = i + 1u) {
                    let probe = plane_distance(node.left + i, scene.source, scene.scale, local);
                    if (probe.distance <= margin) {
                        candidates[candidate_count] = ManifoldPoint(scene_place_point(scene, probe.point_a), -probe.distance, 0.0, 0.0, 0.0, 0.0);
                        candidate_count = candidate_count + 1u;
                    }
                }
            } else {
                if (stack_count + 2u > 64u) {
                    continue;
                }
                stack[stack_count] = node.left;
                stack_count = stack_count + 1u;
                stack[stack_count] = node.right;
                stack_count = stack_count + 1u;
            }
        }
    }
    if (candidate_count == 0u) {
        return false;
    }
    var keep = min(candidate_count, CONTACT_MAX_POINTS);
    for (var i = 0u; i < candidate_count; i = i + 1u) {
        for (var j = i + 1u; j < candidate_count; j = j + 1u) {
            if (candidates[j].depth > candidates[i].depth) {
                let tmp = candidates[i];
                candidates[i] = candidates[j];
                candidates[j] = tmp;
            }
        }
    }
    for (var i = 0u; i < keep; i = i + 1u) {
        let point = candidates[i].position;
        var merged = false;
        for (var j = 0u; j < i; j = j + 1u) {
            if (length(point - candidates[j].position) < 0.05 * shape_scale(world)) {
                merged = true;
            }
        }
        if (!merged) {
            manifold_push(contact, point, candidates[i].depth);
        }
    }
    return contact.point_count > 0u;
}
