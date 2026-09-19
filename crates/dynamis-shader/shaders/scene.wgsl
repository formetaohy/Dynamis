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
    if (shape_triangle_scene(static_target.kind)) {
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

struct GridSpan {
    first: vec2u,
    last: vec2u,
    empty: bool,
}

fn height_grid_cell_size(source: ShapeSource, scale: vec3f) -> vec2f {
    let cells = vec2f(f32(source.grid_cols - 1u), f32(source.grid_rows - 1u));
    return (source.local_max.xz - source.local_min.xz) * scale.xz / cells;
}

fn grid_span(source: ShapeSource, scale: vec3f, box_min: vec3f, box_max: vec3f) -> GridSpan {
    let low = source.local_min.xz * scale.xz;
    let high = source.local_max.xz * scale.xz;
    let size = height_grid_cell_size(source, scale);
    let last_cell = vec2u(source.grid_cols - 2u, source.grid_rows - 2u);
    var span: GridSpan;
    span.empty = box_max.x < low.x
        || box_max.z < low.y
        || box_min.x > high.x
        || box_min.z > high.y
        || box_min.y > source.local_max.y * scale.y
        || box_max.y < source.local_min.y * scale.y;
    let first = (box_min.xz - low) / size;
    let last = (box_max.xz - low) / size;
    span.first = min(vec2u(u32(max(floor(first.x), 0.0)), u32(max(floor(first.y), 0.0))), last_cell);
    span.last = min(vec2u(u32(max(floor(last.x), 0.0)), u32(max(floor(last.y), 0.0))), last_cell);
    return span;
}

fn grid_cell_first_triangle(source: ShapeSource, cell: vec2u) -> u32 {
    return (cell.y * (source.grid_cols - 1u) + cell.x) * 2u;
}

fn grid_cell_bounds(source: ShapeSource, scale: vec3f) -> Aabb {
    var bounds: Aabb;
    bounds.min = source.local_min * scale;
    bounds.max = source.local_max * scale;
    return bounds;
}

fn scene_raycast(scene: WorldShape, origin: vec3f, direction: vec3f, extent: f32) -> ShapeHit {
    if (!shape_source(scene.kind)) {
        return no_hit();
    }
    let source = shape_sources[scene.source];
    if (shape_height_grid(source.kind)) {
        return ray_grid(scene, source, origin, direction, extent);
    }
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
                let hit = ray_scene_triangle(scene, node.left + i, origin, direction, extent);
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

fn ray_scene_triangle(scene: WorldShape, triangle: u32, origin: vec3f, direction: vec3f, extent: f32) -> ShapeHit {
    let points = triangle_points(scene.source, triangle, scene.scale);
    var hit = ray_triangle(origin, direction, extent, points[0], points[1], points[2]);
    if (hit.distance != NO_HIT) {
        hit.triangle = triangle;
    }
    return hit;
}

fn ray_grid(scene: WorldShape, source: ShapeSource, origin: vec3f, direction: vec3f, extent: f32) -> ShapeHit {
    let bounds = grid_cell_bounds(source, scene.scale);
    let entry = aabb_ray_hit(bounds.min, bounds.max, origin, direction, extent);
    if (entry == NO_HIT) {
        return no_hit();
    }
    let size = height_grid_cell_size(source, scene.scale);
    let cells = i32(source.grid_cols - 1u);
    let rows = i32(source.grid_rows - 1u);
    let start = origin + direction * entry;
    let last_cell = vec2i(cells - 1, rows - 1);
    var cell = clamp(vec2i(floor((start.xz - bounds.min.xz) / size)), vec2i(0), last_cell);
    let vertical = direction.x == 0.0 && direction.z == 0.0;
    var travel = vec2f(3.402823466e38, 3.402823466e38);
    var advance = travel;
    if (!vertical) {
        for (var axis = 0u; axis < 2u; axis = axis + 1u) {
            let heading = select(direction.z, direction.x, axis == 0u);
            if (heading == 0.0) {
                continue;
            }
            let cell_size = select(size.y, size.x, axis == 0u);
            let cell_min = bounds.min[axis] + f32(cell[axis]) * cell_size;
            let boundary = select(cell_min, cell_min + cell_size, heading > 0.0);
            travel[axis] = entry + abs((boundary - start[axis]) / heading);
            advance[axis] = abs(cell_size / heading);
        }
    }
    var best = no_hit();
    loop {
        if (cell.x < 0 || cell.y < 0 || cell.x >= cells || cell.y >= rows) {
            break;
        }
        let first = grid_cell_first_triangle(source, vec2u(cell));
        for (var which = 0u; which < 2u; which = which + 1u) {
            let hit = ray_scene_triangle(scene, first + which, origin, direction, extent);
            if (hit.distance < best.distance) {
                best = hit;
            }
        }
        if (vertical) {
            break;
        }
        if (travel.x < travel.y) {
            cell.x = cell.x + select(-1, 1, direction.x > 0.0);
            travel.x = travel.x + advance.x;
        } else {
            cell.y = cell.y + select(-1, 1, direction.z > 0.0);
            travel.y = travel.y + advance.y;
        }
    }
    return best;
}

fn triangle_surface_index(collider: Collider, triangle: u32) -> u32 {
    if (triangle == NO_TRIANGLE) {
        return NO_SURFACE;
    }
    let source = shape_sources[collider.source];
    if (shape_height_grid(source.kind)) {
        if (source.cell_count == 0u) {
            return NO_SURFACE;
        }
        return shape_cells[source.cell_offset + triangle / 2u].surface;
    }
    return shape_triangles[source.triangle_offset + triangle].surface;
}

fn surface_material(collider: Collider, triangle: u32) -> Surface {
    if (triangle == NO_TRIANGLE) {
        return collider_surface(collider);
    }
    let source = shape_sources[collider.source];
    if (shape_height_grid(source.kind)) {
        if (source.cell_count == 0u) {
            return collider_surface(collider);
        }
        let cell = shape_cells[source.cell_offset + triangle / 2u];
        if (cell.surface == NO_SURFACE) {
            return collider_surface(collider);
        }
        return cell.material;
    }
    let surface = shape_triangles[source.triangle_offset + triangle];
    if (surface.surface == NO_SURFACE) {
        return collider_surface(collider);
    }
    return surface.material;
}

fn triangle_corner(source: ShapeSource, corner: u32, scale: vec3f) -> vec3f {
    return shape_vertices[source.vertex_offset + corner].xyz * scale;
}

fn triangle_points(source_index: u32, triangle_index: u32, scale: vec3f) -> array<vec3f, 3> {
    let source = shape_sources[source_index];
    if (shape_height_grid(source.kind)) {
        let cells_per_row = source.grid_cols - 1u;
        let cell = triangle_index / 2u;
        let row = cell / cells_per_row;
        let col = cell - row * cells_per_row;
        let near = row * source.grid_cols + col;
        let far = near + source.grid_cols;
        if (triangle_index % 2u == 0u) {
            return array(
                triangle_corner(source, near, scale),
                triangle_corner(source, far, scale),
                triangle_corner(source, near + 1u, scale),
            );
        }
        return array(
            triangle_corner(source, far, scale),
            triangle_corner(source, far + 1u, scale),
            triangle_corner(source, near + 1u, scale),
        );
    }
    let tri = shape_triangles[source.triangle_offset + triangle_index];
    return array(
        triangle_corner(source, tri.a, scale),
        triangle_corner(source, tri.b, scale),
        triangle_corner(source, tri.c, scale),
    );
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
    var weights: vec3f;
    var result: ConvexClosest;
    result.distance = separation;
    result.point_a = deep;
    result.point_b = closest_on_triangle(deep, points[0], points[1], points[2], &weights);
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
    var result: ConvexClosest;
    result.distance = 3.402823466e38;
    result.point_a = vec3f(0.0);
    result.point_b = vec3f(0.0);
    result.normal = vec3f(0.0, 1.0, 0.0);
    result.penetrating = false;
    if (!shape_source(scene.kind)) {
        return result;
    }
    let source = shape_sources[scene.source];
    let local = scene_local_shape(scene, world);
    let world_aabb = world_aabb_of(local);
    let pad = (world_aabb.max - world_aabb.min) * 0.5;
    if (shape_height_grid(source.kind)) {
        let span = grid_span(source, scene.scale, world_aabb.min - pad, world_aabb.max + pad);
        if (span.empty) {
            return result;
        }
        for (var row = span.first.y; row <= span.last.y; row = row + 1u) {
            for (var col = span.first.x; col <= span.last.x; col = col + 1u) {
                let first = grid_cell_first_triangle(source, vec2u(col, row));
                for (var which = 0u; which < 2u; which = which + 1u) {
                    if (closest_candidate(first + which, scene, local, &result, out_triangle)) {
                        return result;
                    }
                }
            }
        }
        return result;
    }
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
                if (closest_candidate(node.left + i, scene, local, &result, out_triangle)) {
                    return result;
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

fn closest_candidate(
    triangle: u32,
    scene: WorldShape,
    local: WorldShape,
    result: ptr<function, ConvexClosest>,
    out_triangle: ptr<function, u32>,
) -> bool {
    let candidate = plane_distance(triangle, scene.source, scene.scale, local);
    if (candidate.penetrating) {
        *result = candidate;
        *out_triangle = triangle;
        return true;
    }
    if (candidate.distance < (*result).distance) {
        *result = candidate;
        *out_triangle = triangle;
    }
    return false;
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

fn bounds_gap(first: Aabb, second: Aabb) -> f32 {
    let delta = max(max(first.min - second.max, second.min - first.max), vec3f(0.0));
    return length(delta);
}

fn scene_mesh_gap(scene: WorldShape, world: WorldShape) -> ShapeHit {
    let source = shape_sources[scene.source];
    let local = scene_local_shape(scene, world);
    let box = world_aabb_of(local);
    var best = 3.402823466e38;
    var point = vec3f(0.0);
    var normal = vec3f(0.0, 1.0, 0.0);
    var triangle = NO_TRIANGLE;
    var stack: array<u32, 64>;
    var stack_count = 1u;
    stack[0] = source.node_offset;
    while (stack_count > 0u) {
        stack_count = stack_count - 1u;
        let node = shape_nodes[stack[stack_count]];
        var node_box: Aabb;
        node_box.min = node.min * scene.scale;
        node_box.max = node.max * scene.scale;
        if (bounds_gap(node_box, box) >= best) {
            continue;
        }
        if (node.leaf == 1u) {
            for (var i = 0u; i < node.right; i = i + 1u) {
                let index = node.left + i;
                let candidate = plane_distance(index, scene.source, scene.scale, local);
                if (candidate.distance < best) {
                    best = candidate.distance;
                    point = (candidate.point_a + candidate.point_b) * 0.5;
                    normal = candidate.normal;
                    triangle = index;
                }
            }
        } else if (stack_count + 2u <= 64u) {
            stack[stack_count] = node.left;
            stack_count = stack_count + 1u;
            stack[stack_count] = node.right;
            stack_count = stack_count + 1u;
        }
    }
    if (best == 3.402823466e38) {
        return no_hit();
    }
    return ShapeHit(best, scene_place_point(scene, point), scene_place_normal(scene, normal), triangle);
}

fn scene_gap(scene: WorldShape, world: WorldShape) -> ShapeHit {
    if (shape_height_grid(scene.kind)) {
        return scene_convex_hit(scene, world);
    }
    return scene_mesh_gap(scene, world);
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
    if (!shape_source(scene.kind)) {
        return no_hit();
    }
    let source = shape_sources[scene.source];
    var local = scene_local_shape(scene, moving);
    local.center = scene_local_point(scene, start);
    let bounds = world_aabb_of(local);
    let reach = quat_rotate(quat_conjugate(scene.rotation), direction * max_dist) * scene.scale;
    var swept: Aabb;
    swept.min = min(bounds.min * scene.scale, bounds.min * scene.scale + reach);
    swept.max = max(bounds.max * scene.scale, bounds.max * scene.scale + reach);
    var best = no_hit();
    if (shape_height_grid(source.kind)) {
        let span = grid_span(source, scene.scale, swept.min, swept.max);
        if (span.empty) {
            return best;
        }
        for (var row = span.first.y; row <= span.last.y; row = row + 1u) {
            for (var col = span.first.x; col <= span.last.x; col = col + 1u) {
                let first = grid_cell_first_triangle(source, vec2u(col, row));
                for (var which = 0u; which < 2u; which = which + 1u) {
                    let hit = scene_sweep_triangle(scene, moving, start, direction, first + which, max_dist);
                    if (hit.distance < best.distance) {
                        best = hit;
                    }
                }
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
    if (shape_source(world.kind)) {
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

/// The reference plane a scene contact is resolved along: its two in-plane axes, so the whole
/// contact area is compared as one flat picture however many triangles the surface is made of.
struct ScenePlane {
    origin: vec3f,
    normal: vec3f,
    first: vec3f,
    second: vec3f,
}

fn scene_plane_of(origin: vec3f, normal: vec3f) -> ScenePlane {
    var helper = vec3f(1.0, 0.0, 0.0);
    if (abs(normal.x) > 0.9) {
        helper = vec3f(0.0, 1.0, 0.0);
    }
    var plane: ScenePlane;
    plane.origin = origin;
    plane.normal = normal;
    plane.first = normalize(cross(normal, helper));
    plane.second = cross(normal, plane.first);
    return plane;
}

fn scene_plane_flat(plane: ScenePlane, point: vec3f) -> vec2f {
    return vec2f(dot(point - plane.origin, plane.first), dot(point - plane.origin, plane.second));
}

fn scene_plane_cross(first: vec2f, second: vec2f) -> f32 {
    return first.x * second.y - first.y * second.x;
}

fn scene_plane_holds(corner: array<vec2f, 3>, point: vec2f) -> bool {
    let first = scene_plane_cross(corner[1] - corner[0], point - corner[0]) >= -CLIP_MARGIN;
    let second = scene_plane_cross(corner[2] - corner[1], point - corner[1]) >= -CLIP_MARGIN;
    let third = scene_plane_cross(corner[0] - corner[2], point - corner[2]) >= -CLIP_MARGIN;
    return first && second && third;
}

fn scene_plane_crossing(start: vec2f, finish: vec2f, edge_start: vec2f, edge_finish: vec2f, out_t: ptr<function, f32>) -> bool {
    let heading = finish - start;
    let edge = edge_finish - edge_start;
    let denominator = scene_plane_cross(heading, edge);
    if (abs(denominator) < 1e-12) {
        return false;
    }
    let offset = edge_start - start;
    let along = scene_plane_cross(offset, edge) / denominator;
    let across = scene_plane_cross(offset, heading) / denominator;
    if (along < 0.0 || along > 1.0 || across < 0.0 || across > 1.0) {
        return false;
    }
    *out_t = along;
    return true;
}

/// One scene triangle projected into the reference plane and wound so that its inside is one side.
fn scene_plane_triangle(triangle: u32, scene: WorldShape, plane: ScenePlane) -> array<vec2f, 3> {
    let points = triangle_points(scene.source, triangle, scene.scale);
    let first = scene_plane_flat(plane, points[0]);
    let second = scene_plane_flat(plane, points[1]);
    let third = scene_plane_flat(plane, points[2]);
    if (scene_plane_cross(second - first, third - first) < 0.0) {
        return array<vec2f, 3>(first, third, second);
    }
    return array<vec2f, 3>(first, second, third);
}

/// The part of a body's support feature one scene triangle covers: which feature points it holds, and
/// how far along each feature edge it starts and stops. Merging triangles instead of collecting their
/// clips keeps the contact area whole: a surface made of many triangles answers the extremes of the
/// area the body actually covers, so a body spanning a tessellated floor is supported across its whole
/// footprint rather than by the triangles a walk happened to visit first.
struct SceneCover {
    held: u32,
    crossed: u32,
    entered: array<f32, FEATURE_MAX>,
    left: array<f32, FEATURE_MAX>,
}

fn scene_cover_new() -> SceneCover {
    var cover: SceneCover;
    cover.held = 0u;
    cover.crossed = 0u;
    return cover;
}

fn scene_cover_triangle(
    cover: ptr<function, SceneCover>,
    feature: array<vec2f, FEATURE_MAX>,
    count: u32,
    edges: u32,
    closed: bool,
    corner: array<vec2f, 3>,
) {
    for (var index = 0u; index < count; index = index + 1u) {
        if (scene_plane_holds(corner, feature[index])) {
            (*cover).held = (*cover).held | (1u << index);
        }
    }
    for (var index = 0u; index < edges; index = index + 1u) {
        let next = select(index + 1u, (index + 1u) % count, closed);
        for (var side = 0u; side < 3u; side = side + 1u) {
            var along = 0.0;
            if (!scene_plane_crossing(feature[index], feature[next], corner[side], corner[(side + 1u) % 3u], &along)) {
                continue;
            }
            let first = ((*cover).crossed & (1u << index)) == 0u;
            if (first || along < (*cover).entered[index]) {
                (*cover).entered[index] = along;
            }
            if (first || along > (*cover).left[index]) {
                (*cover).left[index] = along;
            }
            (*cover).crossed = (*cover).crossed | (1u << index);
        }
    }
}

/// The points one scene contact asserts: every feature point the surface holds and, on every feature
/// edge that leaves the surface, where it enters and where it leaves. Each point lies on the body's
/// own surface, and its depth is how far it reaches below the plane the contact is resolved along.
fn scene_candidate_push(
    candidates: ptr<function, array<ManifoldPoint, MANIFOLD_CANDIDATES>>,
    candidate_count: ptr<function, u32>,
    scene: WorldShape,
    plane: ScenePlane,
    margin: f32,
    point: vec3f,
    feature: u32,
) {
    if (*candidate_count >= MANIFOLD_CANDIDATES) {
        return;
    }
    let depth = dot(plane.origin - point, plane.normal);
    if (depth < -margin) {
        return;
    }
    let position = scene_place_point(scene, point - plane.normal * (depth * 0.5));
    (*candidates)[*candidate_count] = manifold_candidate(position, depth, feature);
    *candidate_count = *candidate_count + 1u;
}

fn scene_cover_candidates(
    cover: SceneCover,
    scene: WorldShape,
    plane: ScenePlane,
    margin: f32,
    points: array<vec3f, FEATURE_MAX>,
    ids: array<u32, FEATURE_MAX>,
    count: u32,
    edges: u32,
    closed: bool,
    candidates: ptr<function, array<ManifoldPoint, MANIFOLD_CANDIDATES>>,
    candidate_count: ptr<function, u32>,
) {
    let face = feature_field_face(0u);
    for (var index = 0u; index < count; index = index + 1u) {
        if ((cover.held & (1u << index)) != 0u) {
            scene_candidate_push(candidates, candidate_count, scene, plane, margin, points[index], feature_vertex(face, ids[index]));
        }
    }
    for (var index = 0u; index < edges; index = index + 1u) {
        if ((cover.crossed & (1u << index)) == 0u) {
            continue;
        }
        let next = select(index + 1u, (index + 1u) % count, closed);
        if ((cover.held & (1u << index)) == 0u) {
            scene_candidate_push(
                candidates,
                candidate_count,
                scene,
                plane,
                margin,
                mix(points[index], points[next], cover.entered[index]),
                feature_vertex(face, feature_field_clip(index)),
            );
        }
        if ((cover.held & (1u << next)) == 0u) {
            scene_candidate_push(
                candidates,
                candidate_count,
                scene,
                plane,
                margin,
                mix(points[index], points[next], cover.left[index]),
                feature_vertex(face, feature_field_clip(FEATURE_MAX + index)),
            );
        }
    }
}

fn scene_convex_manifold(
    scene: WorldShape,
    world: WorldShape,
    reference: u32,
    margin: f32,
    contact: ptr<function, Contact>,
) -> bool {
    if (!shape_source(scene.kind) || reference == NO_TRIANGLE) {
        return false;
    }
    let source = shape_sources[scene.source];
    let local = scene_local_shape(scene, world);
    let reference_points = triangle_points(scene.source, reference, scene.scale);
    var normal = sign_normalize(cross(reference_points[1] - reference_points[0], reference_points[2] - reference_points[0]));
    if (dot(local.center - reference_points[0], normal) < 0.0) {
        normal = -normal;
    }
    let plane = scene_plane_of(reference_points[0], normal);
    var points: array<vec3f, FEATURE_MAX>;
    var ids: array<u32, FEATURE_MAX>;
    var face = 0u;
    var closed = false;
    let count = shape_feature(local, -normal, &points, &ids, &face, &closed);
    if (count == 0u) {
        return false;
    }
    var feature: array<vec2f, FEATURE_MAX>;
    for (var index = 0u; index < count; index = index + 1u) {
        feature[index] = scene_plane_flat(plane, points[index]);
    }
    let edges = select(count - 1u, count, closed);
    var cover = scene_cover_new();
    let world_aabb = world_aabb_of(local);
    let pad = (world_aabb.max - world_aabb.min) * 0.5;
    if (shape_height_grid(source.kind)) {
        let span = grid_span(source, scene.scale, world_aabb.min - pad, world_aabb.max + pad);
        if (span.empty) {
            return false;
        }
        for (var row = span.first.y; row <= span.last.y; row = row + 1u) {
            for (var col = span.first.x; col <= span.last.x; col = col + 1u) {
                let first = grid_cell_first_triangle(source, vec2u(col, row));
                for (var which = 0u; which < 2u; which = which + 1u) {
                    scene_cover_triangle(&cover, feature, count, edges, closed, scene_plane_triangle(first + which, scene, plane));
                }
            }
        }
    } else {
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
                    scene_cover_triangle(&cover, feature, count, edges, closed, scene_plane_triangle(node.left + i, scene, plane));
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
    var candidates: array<ManifoldPoint, MANIFOLD_CANDIDATES>;
    var candidate_count = 0u;
    scene_cover_candidates(cover, scene, plane, margin, points, ids, count, edges, closed, &candidates, &candidate_count);
    if (candidate_count == 0u) {
        return false;
    }
    manifold_keep(contact, &candidates, candidate_count);
    return contact.point_count > 0u;
}

