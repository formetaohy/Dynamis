const SWEEP_TOLERANCE: f32 = 1e-4;

fn nearest_hit(held: ShapeHit, candidate: ShapeHit) -> ShapeHit {
    if (candidate.distance < held.distance) {
        return candidate;
    }
    return held;
}

fn ray_sphere(origin: vec3f, direction: vec3f, extent: f32, center: vec3f, radius: f32) -> ShapeHit {
    let offset = origin - center;
    let quadratic = dot(direction, direction);
    let linear = dot(offset, direction);
    let constant = dot(offset, offset) - radius * radius;
    let discriminant = linear * linear - quadratic * constant;
    if (quadratic <= 0.0 || discriminant < 0.0) {
        return no_hit();
    }
    let root = sqrt(discriminant);
    var t = (-linear - root) / quadratic;
    if (t < 0.0) {
        t = (-linear + root) / quadratic;
    }
    if (t < 0.0 || t > extent) {
        return no_hit();
    }
    let point = origin + direction * t;
    let normal = normalize(point - center);
    return ShapeHit(t, point, normal, NO_TRIANGLE);
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
    return ShapeHit(tmin, point, normal, NO_TRIANGLE);
}

fn ray_quadratic_root(quadratic: f32, linear: f32, constant: f32) -> f32 {
    let discriminant = linear * linear - quadratic * constant;
    if (quadratic <= 0.0 || discriminant < 0.0) {
        return NO_HIT;
    }
    let root = sqrt(discriminant);
    let near = (-linear - root) / quadratic;
    if (near >= 0.0) {
        return near;
    }
    let far = (-linear + root) / quadratic;
    if (far >= 0.0) {
        return far;
    }
    return NO_HIT;
}

fn ray_sphere_candidate(
    origin: vec3f,
    direction: vec3f,
    extent: f32,
    center: vec3f,
    radius: f32,
    hemisphere: vec3f,
) -> ShapeHit {
    let offset = origin - center;
    let t = ray_quadratic_root(dot(direction, direction), dot(offset, direction), dot(offset, offset) - radius * radius);
    if (t == NO_HIT || t > extent) {
        return no_hit();
    }
    let local = offset + direction * t;
    if (length(hemisphere) > 0.0 && dot(local, hemisphere) < 0.0) {
        return no_hit();
    }
    return ShapeHit(t, origin + direction * t, sign_normalize(local), NO_TRIANGLE);
}

fn ray_swept_segment(
    origin: vec3f,
    direction: vec3f,
    extent: f32,
    start: vec3f,
    end: vec3f,
    radius: f32,
) -> ShapeHit {
    let basis = end - start;
    let basis_squared = dot(basis, basis);
    if (basis_squared <= 0.0) {
        return ray_sphere_candidate(origin, direction, extent, start, radius, vec3f(0.0));
    }
    let offset = origin - start;
    let axial_offset = dot(basis, offset);
    let axial_direction = dot(basis, direction);
    let inverse_basis = 1.0 / basis_squared;
    let perpendicular_offset = offset - basis * (axial_offset * inverse_basis);
    let perpendicular_direction = direction - basis * (axial_direction * inverse_basis);
    var best = no_hit();
    let side = ray_quadratic_root(
        dot(perpendicular_direction, perpendicular_direction),
        dot(perpendicular_offset, perpendicular_direction),
        dot(perpendicular_offset, perpendicular_offset) - radius * radius,
    );
    if (side != NO_HIT && side <= extent) {
        let axial = axial_offset + side * axial_direction;
        if (axial >= 0.0 && axial <= basis_squared) {
            let normal = sign_normalize(perpendicular_offset + perpendicular_direction * side);
            best = nearest_hit(best, ShapeHit(side, origin + direction * side, normal, NO_TRIANGLE));
        }
    }
    best = nearest_hit(best, ray_sphere_candidate(origin, direction, extent, start, radius, -basis));
    best = nearest_hit(best, ray_sphere_candidate(origin, direction, extent, end, radius, basis));
    return best;
}

fn ray_capsule(origin: vec3f, direction: vec3f, extent: f32, center: vec3f, axis: vec3f, half_height: f32, radius: f32) -> ShapeHit {
    return ray_swept_segment(origin, direction, extent, center - axis * half_height, center + axis * half_height, radius);
}

fn ray_cylinder(origin: vec3f, direction: vec3f, extent: f32, center: vec3f, axis: vec3f, half_height: f32, radius: f32) -> ShapeHit {
    let start = center - axis * half_height;
    let end = center + axis * half_height;
    let basis = end - start;
    let basis_squared = dot(basis, basis);
    let radius_squared = radius * radius;
    if (basis_squared <= 0.0) {
        return ray_disc(origin, direction, extent, center, axis, radius_squared);
    }
    let offset = origin - start;
    let axial_offset = dot(basis, offset);
    let axial_direction = dot(basis, direction);
    var best = no_hit();
    let inverse_basis = 1.0 / basis_squared;
    let perpendicular_offset = offset - basis * (axial_offset * inverse_basis);
    let perpendicular_direction = direction - basis * (axial_direction * inverse_basis);
    let side = ray_quadratic_root(
        dot(perpendicular_direction, perpendicular_direction),
        dot(perpendicular_offset, perpendicular_direction),
        dot(perpendicular_offset, perpendicular_offset) - radius_squared,
    );
    if (side != NO_HIT && side <= extent) {
        let axial = axial_offset + side * axial_direction;
        if (axial >= 0.0 && axial <= basis_squared) {
            let normal = sign_normalize(perpendicular_offset + perpendicular_direction * side);
            best = nearest_hit(best, ShapeHit(side, origin + direction * side, normal, NO_TRIANGLE));
        }
    }
    if (abs(axial_direction) > 0.0) {
        for (var cap = 0u; cap < 2u; cap = cap + 1u) {
            let cap_at = select(axial_offset, basis_squared - axial_offset, cap == 1u);
            let t = cap_at / select(-axial_direction, axial_direction, cap == 1u);
            if (t < 0.0 || t > extent) {
                continue;
            }
            let local = offset + direction * t - basis * select(0.0, 1.0, cap == 1u);
            if (dot(local, local) <= radius_squared) {
                let normal = sign_normalize(select(-basis, basis, cap == 1u));
                best = nearest_hit(best, ShapeHit(t, origin + direction * t, normal, NO_TRIANGLE));
            }
        }
    }
    return best;
}

fn ray_disc(origin: vec3f, direction: vec3f, extent: f32, center: vec3f, axis: vec3f, radius_squared: f32) -> ShapeHit {
    let travel = dot(direction, axis);
    if (abs(travel) < 1e-8) {
        return no_hit();
    }
    let t = dot(center - origin, axis) / travel;
    if (t < 0.0 || t > extent) {
        return no_hit();
    }
    let local = origin + direction * t - center;
    if (dot(local, local) > radius_squared) {
        return no_hit();
    }
    return ShapeHit(t, origin + direction * t, select(-axis, axis, travel < 0.0), NO_TRIANGLE);
}

fn shape_sweep(moving: WorldShape, static_target: WorldShape, start: vec3f, direction: vec3f, max_dist: f32) -> ShapeHit {
    let heading = normalize(direction);
    if (static_target.kind == SHAPE_PLANE) {
        let normal = plane_normal(static_target);
        let travel = dot(heading, normal);
        if (travel >= 0.0) {
            return no_hit();
        }
        let extent = dot(support(moving, -normal) - moving.center, -normal);
        let offset = dot(start - static_target.center, normal);
        let time = max((offset - extent) / (-travel), 0.0);
        if (time > max_dist) {
            return no_hit();
        }
        return ShapeHit(time, start + heading * time, normal, NO_TRIANGLE);
    }
    if (static_target.kind == SHAPE_MESH || static_target.kind == SHAPE_HEIGHTFIELD) {
        return scene_convex_sweep(static_target, moving, start, direction, max_dist);
    }
    let tolerance = SWEEP_TOLERANCE;
    var moved = moving;
    var time = 0.0;
    var normal = sign_normalize(moving.center - static_target.center);
    var simplex: array<SimplexPoint, 4>;
    var count = 0u;
    for (var iteration = 0u; iteration < 8u; iteration = iteration + 1u) {
        moved.center = start + heading * time;
        let closest = convex_closest(moved, static_target, &simplex, &count);
        if (closest.penetrating || closest.distance < tolerance) {
            return ShapeHit(time, start + heading * time, normal, NO_TRIANGLE);
        }
        normal = -closest.normal;
        time = time + closest.distance;
        if (time > max_dist) {
            return no_hit();
        }
    }
    return no_hit();
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
    return ShapeHit(t, point, select(normal, -normal, det < 0.0), NO_TRIANGLE);
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
                    best.triangle = node.left + i;
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

fn triangle_surface_index(collider: Collider, triangle: u32) -> u32 {
    if (triangle == NO_TRIANGLE) {
        return NO_SURFACE;
    }
    return triangle_surface(collider, triangle).surface;
}

fn surface_material(collider: Collider, triangle: u32) -> Surface {
    if (triangle == NO_TRIANGLE) {
        return collider_surface(collider);
    }
    let surface = triangle_surface(collider, triangle);
    if (surface.surface == NO_SURFACE) {
        return collider_surface(collider);
    }
    return surface.material;
}

fn triangle_surface(collider: Collider, triangle: u32) -> Triangle {
    return shape_triangles[shape_sources[collider.source].triangle_offset + triangle];
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
    return ShapeHit(closest.distance, (closest.point_a + closest.point_b) * 0.5, closest.normal, triangle);
}

fn scene_sweep_triangle(
    scene: WorldShape,
    moving: WorldShape,
    start: vec3f,
    direction: vec3f,
    triangle: u32,
    max_dist: f32,
) -> ShapeHit {
    let local = triangle_points(scene.source, triangle, scene.scale);
    let first = scene_place_point(scene, local[0]);
    let second = scene_place_point(scene, local[1]);
    let third = scene_place_point(scene, local[2]);
    let normal = sign_normalize(cross(second - first, third - first));
    var offset = dot(start - first, normal);
    var facing = normal;
    if (offset < 0.0) {
        facing = -normal;
        offset = -offset;
    }
    let travel = dot(direction, facing);
    if (travel >= 0.0) {
        return no_hit();
    }
    let extent = dot(support(moving, -facing) - moving.center, -facing);
    let time = max((offset - extent) / (-travel), 0.0);
    if (time > max_dist) {
        return no_hit();
    }
    return ShapeHit(time, start + direction * time, facing, triangle);
}

fn scene_convex_sweep(
    scene: WorldShape,
    moving: WorldShape,
    start: vec3f,
    direction: vec3f,
    max_dist: f32,
) -> ShapeHit {
    let source = shape_sources[scene.source];
    var local = scene_local_shape(scene, moving);
    local.center = scene_local_point(scene, start);
    let bounds = world_aabb_of(local);
    let reach = quat_rotate(quat_conjugate(scene.rotation), direction * max_dist) * scene.scale;
    var swept: Aabb;
    swept.min = min(bounds.min * scene.scale, bounds.min * scene.scale + reach);
    swept.max = max(bounds.max * scene.scale, bounds.max * scene.scale + reach);
    var best = no_hit();
    if (source.node_count == 0u) {
        for (var i = 0u; i < source.triangle_count; i = i + 1u) {
            let hit = scene_sweep_triangle(scene, moving, start, direction, i, max_dist);
            if (hit.distance < best.distance) {
                best = hit;
            }
        }
        return best;
    }
    var stack: array<u32, 64>;
    var stack_count = 1u;
    stack[0] = source.node_offset;
    while (stack_count > 0u) {
        stack_count = stack_count - 1u;
        let node = shape_nodes[stack[stack_count]];
        var node_box: Aabb;
        node_box.min = node.min * scene.scale;
        node_box.max = node.max * scene.scale;
        if (!aabb_overlaps(node_box, swept)) {
            continue;
        }
        if (node.leaf == 1u) {
            for (var i = 0u; i < node.right; i = i + 1u) {
                let hit = scene_sweep_triangle(scene, moving, start, direction, node.left + i, max_dist);
                if (hit.distance < best.distance) {
                    best = hit;
                }
            }
        } else if (stack_count + 2u <= 64u) {
            stack[stack_count] = node.left;
            stack_count = stack_count + 1u;
            stack[stack_count] = node.right;
            stack_count = stack_count + 1u;
        }
    }
    return best;
}

fn shape_ray(world: WorldShape, origin: vec3f, direction: vec3f, extent: f32) -> ShapeHit {
    let heading = normalize(direction);
    if (world.kind == SHAPE_PLANE) {
        let normal = plane_normal(world);
        let travel = dot(heading, normal);
        if (abs(travel) < 1e-8) {
            return no_hit();
        }
        let t = dot(world.center - origin, normal) / travel;
        if (t < 0.0 || t > extent) {
            return no_hit();
        }
        return ShapeHit(t, origin + heading * t, normal, NO_TRIANGLE);
    }
    if (world.kind == SHAPE_MESH || world.kind == SHAPE_HEIGHTFIELD || world.kind == SHAPE_HULL) {
        return ray_scene(world, origin, heading, extent);
    }
    let unscaled = world.scale.x == 1.0 && world.scale.y == 1.0 && world.scale.z == 1.0;
    if (unscaled) {
        if (world.kind == SHAPE_SPHERE) {
            return ray_sphere(origin, heading, extent, world.center, world.radius);
        }
        if (world.kind == SHAPE_CUBOID) {
            return ray_box(origin, heading, extent, world.center, world.rotation, world.half_extents);
        }
        if (world.kind == SHAPE_CAPSULE) {
            return ray_capsule(origin, heading, extent, world.center, shape_axis(world), world.half_height, world.radius);
        }
        if (world.kind == SHAPE_CYLINDER) {
            return ray_cylinder(origin, heading, extent, world.center, shape_axis(world), world.half_height, world.radius);
        }
        return no_hit();
    }
    let inv_rotation = quat_conjugate(world.rotation);
    let unscale = 1.0 / world.scale;
    let local_origin = quat_rotate(inv_rotation, origin - world.center) * unscale;
    let local_direction = quat_rotate(inv_rotation, heading) * unscale;
    let local_extent = extent * length(local_direction);
    var local: ShapeHit;
    if (world.kind == SHAPE_SPHERE) {
        local = ray_sphere(local_origin, local_direction, local_extent, vec3f(0.0), world.radius);
    } else if (world.kind == SHAPE_CAPSULE) {
        local = ray_capsule(local_origin, local_direction, local_extent, vec3f(0.0), vec3f(0.0, 1.0, 0.0), world.half_height, world.radius);
    } else if (world.kind == SHAPE_CYLINDER) {
        local = ray_cylinder(local_origin, local_direction, local_extent, vec3f(0.0), vec3f(0.0, 1.0, 0.0), world.half_height, world.radius);
    } else {
        return no_hit();
    }
    if (local.distance == NO_HIT) {
        return no_hit();
    }
    var hit: ShapeHit;
    hit.distance = local.distance;
    hit.triangle = NO_TRIANGLE;
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
                candidates[candidate_count] = manifold_candidate(scene_place_point(scene, probe.point_a), -probe.distance, feature_triangle(i));
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
                        candidates[candidate_count] = manifold_candidate(scene_place_point(scene, probe.point_a), -probe.distance, feature_triangle(node.left + i));
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
            manifold_push(contact, point, candidates[i].depth, candidates[i].feature);
        }
    }
    return contact.point_count > 0u;
}
