fn support_local(world: WorldShape, direction: vec3f) -> vec3f {
    let d = direction;
    let n = normalize(direction);
    if (world.kind == SHAPE_SPHERE) {
        return n * world.radius;
    }
    if (world.kind == SHAPE_CUBOID) {
        let signs = select(vec3f(-1.0), vec3f(1.0), d > vec3f(0.0));
        return signs * world.half_extents;
    }
    if (world.kind == SHAPE_CAPSULE) {
        let tip = vec3f(0.0, 1.0, 0.0) * world.half_height * select(-1.0, 1.0, d.y > 0.0);
        return tip + n * world.radius;
    }
    if (world.kind == SHAPE_CYLINDER) {
        let base = vec3f(0.0, 1.0, 0.0) * world.half_height * select(-1.0, 1.0, d.y > 0.0);
        let radial = d - vec3f(0.0, d.y, 0.0);
        if (length(radial) < 1e-6) {
            return base;
        }
        return base + normalize(radial) * world.radius;
    }
    if (world.kind == SHAPE_TRIANGLE) {
        let source = shape_sources[world.source];
        let tri = shape_triangles[source.triangle_offset + u32(world.radius)];
        let p0 = shape_vertices[source.vertex_offset + tri.a].xyz;
        let p1 = shape_vertices[source.vertex_offset + tri.b].xyz;
        let p2 = shape_vertices[source.vertex_offset + tri.c].xyz;
        let s0 = dot(p0, d);
        let s1 = dot(p1, d);
        let s2 = dot(p2, d);
        if (s0 >= s1 && s0 >= s2) {
            return p0;
        }
        if (s1 >= s2) {
            return p1;
        }
        return p2;
    }
    let source = shape_sources[world.source];
    var best_dot = -3.402823466e38;
    var best = vec3f(0.0);
    if (source.vertex_count <= LINEAR_SUPPORT_VERTICES) {
        for (var i = 0u; i < source.vertex_count; i = i + 1u) {
            let v = shape_vertices[source.vertex_offset + i].xyz;
            let s = dot(v, d);
            if (s > best_dot) {
                best_dot = s;
                best = v;
            }
        }
        return best;
    }
    var stack: array<u32, 64>;
    var stack_count = 1u;
    stack[0] = source.node_offset;
    for (var visit = 0u; visit < source.node_count && stack_count > 0u; visit = visit + 1u) {
        stack_count = stack_count - 1u;
        let node = shape_nodes[stack[stack_count]];
        let bound = dot((node.min + node.max) * 0.5, d) + dot((node.max - node.min) * 0.5, abs(d));
        if (bound > best_dot) {
            if (node.leaf == 1u) {
                for (var i = 0u; i < node.right; i = i + 1u) {
                    let tri = shape_triangles[source.triangle_offset + node.left + i];
                    let corners = array<u32, 3>(tri.a, tri.b, tri.c);
                    for (var c = 0u; c < 3u; c = c + 1u) {
                        let v = shape_vertices[source.vertex_offset + corners[c]].xyz;
                        let s = dot(v, d);
                        if (s > best_dot) {
                            best_dot = s;
                            best = v;
                        }
                    }
                }
            } else if (stack_count + 2u <= 64u) {
                stack[stack_count] = node.left;
                stack_count = stack_count + 1u;
                stack[stack_count] = node.right;
                stack_count = stack_count + 1u;
            }
        }
    }
    return best;
}

fn support(world: WorldShape, direction: vec3f) -> vec3f {
    let local_d = quat_rotate(quat_conjugate(world.rotation), direction);
    let scaled_d = local_d * world.scale;
    let local = support_local(world, scaled_d);
    return world.center + quat_rotate(world.rotation, local * world.scale);
}

fn closest_on_triangle(origin: vec3f, a: vec3f, b: vec3f, c: vec3f, out_weights: ptr<function, vec3f>) -> vec3f {
    let ab = b - a;
    let ac = c - a;
    let ap = origin - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        *out_weights = vec3f(1.0, 0.0, 0.0);
        return a;
    }
    let bp = origin - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        *out_weights = vec3f(0.0, 1.0, 0.0);
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let t = d1 / (d1 - d3);
        *out_weights = vec3f(1.0 - t, t, 0.0);
        return a + ab * t;
    }
    let cp = origin - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        *out_weights = vec3f(0.0, 0.0, 1.0);
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let t = d2 / (d2 - d6);
        *out_weights = vec3f(1.0 - t, 0.0, t);
        return a + ac * t;
    }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        *out_weights = vec3f(0.0, 1.0 - t, t);
        return b + (c - b) * t;
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    *out_weights = vec3f(1.0 - v - w, v, w);
    return a + ab * v + ac * w;
}

fn face_span(a: vec3f, b: vec3f, c: vec3f) -> f32 {
    return max(max(length(b - a), length(c - a)), length(c - b));
}

fn tetra_contains(tet: array<SimplexPoint, 4>, origin: vec3f) -> bool {
    for (var face = 0u; face < 4u; face = face + 1u) {
        var a: vec3f;
        var b: vec3f;
        var c: vec3f;
        if (face == 0u) {
            a = tet[0].w;
            b = tet[1].w;
            c = tet[2].w;
        } else if (face == 1u) {
            a = tet[0].w;
            b = tet[2].w;
            c = tet[3].w;
        } else if (face == 2u) {
            a = tet[0].w;
            b = tet[3].w;
            c = tet[1].w;
        } else {
            a = tet[1].w;
            b = tet[3].w;
            c = tet[2].w;
        }
        let normal = cross(b - a, c - a);
        let span = face_span(a, b, c);
        if (length(normal) <= 1e-6 * span * span) {
            return false;
        }
        if (dot(normal, origin - a) > 0.0) {
            return false;
        }
    }
    return true;
}

fn simplex_holds(
    simplex: array<SimplexPoint, 4>,
    count: u32,
    point: vec3f,
    span: f32,
) -> bool {
    let tolerance = (1e-5 * span) * (1e-5 * span);
    for (var i = 0u; i < count; i = i + 1u) {
        let offset = simplex[i].w - point;
        if (dot(offset, offset) <= tolerance) {
            return true;
        }
    }
    return false;
}

fn simplex_reduce(
    simplex: ptr<function, array<SimplexPoint, 4>>,
    lambdas: vec4f,
) -> u32 {
    var kept: array<SimplexPoint, 4>;
    var count = 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (lambdas[i] > 1e-6) {
            kept[count] = (*simplex)[i];
            count = count + 1u;
        }
    }
    if (count == 0u) {
        kept[0] = (*simplex)[0];
        count = 1u;
    }
    for (var i = 0u; i < count; i = i + 1u) {
        (*simplex)[i] = kept[i];
    }
    return count;
}

fn simplex_closest(simplex: array<SimplexPoint, 4>, count: u32) -> SimplexResult {
    var result: SimplexResult;
    result.v = vec3f(3.402823466e38);
    result.lambdas = vec4f(0.0);
    result.penetrating = count == 4u && tetra_contains(simplex, vec3f(0.0));
    result.point_a = vec3f(0.0);
    result.point_b = vec3f(0.0);
    if (result.penetrating) {
        result.v = vec3f(0.0);
        return result;
    }
    for (var i = 0u; i < count; i = i + 1u) {
        let d = length(simplex[i].w);
        if (d < length(result.v)) {
            result.v = simplex[i].w;
            result.lambdas = vec4f(0.0);
            result.lambdas[i] = 1.0;
        }
    }
    for (var i = 0u; i < count; i = i + 1u) {
        for (var j = i + 1u; j < count; j = j + 1u) {
            let a = simplex[i].w;
            let b = simplex[j].w;
            let ab = b - a;
            let denom = dot(ab, ab);
            if (denom < 1e-12) {
                continue;
            }
            let t = clamp(-dot(a, ab) / denom, 0.0, 1.0);
            let point = a + ab * t;
            if (length(point) < length(result.v)) {
                result.v = point;
                result.lambdas = vec4f(0.0);
                result.lambdas[i] = 1.0 - t;
                result.lambdas[j] = t;
            }
        }
    }
    if (count >= 3u) {
        for (var i = 0u; i < count; i = i + 1u) {
            for (var j = i + 1u; j < count; j = j + 1u) {
                for (var k = j + 1u; k < count; k = k + 1u) {
                    var weights = vec3f(0.0);
                    let point = closest_on_triangle(vec3f(0.0), simplex[i].w, simplex[j].w, simplex[k].w, &weights);
                    if (length(point) < length(result.v)) {
                        result.v = point;
                        result.lambdas = vec4f(0.0);
                        result.lambdas[i] = weights.x;
                        result.lambdas[j] = weights.y;
                        result.lambdas[k] = weights.z;
                    }
                }
            }
        }
    }
    for (var i = 0u; i < count; i = i + 1u) {
        result.point_a = result.point_a + result.lambdas[i] * simplex[i].a;
        result.point_b = result.point_b + result.lambdas[i] * simplex[i].b;
    }
    return result;
}

fn support_pair(first: WorldShape, second: WorldShape, direction: vec3f) -> SimplexPoint {
    let a = support(first, direction);
    let b = support(second, -direction);
    return SimplexPoint(a - b, a, b);
}

fn epa_tetrahedron(first: WorldShape, second: WorldShape, simplex: array<SimplexPoint, 4>, count: u32) -> EpaResult {
    var polytope: array<EpaPoint, 64>;
    var new_faces: array<EpaFace, 64>;
    var faces: array<EpaFace, 128>;
    var visible: array<u32, 128>;
    var vertex_count = 0u;
    for (var i = 0u; i < count; i = i + 1u) {
        polytope[vertex_count] = EpaPoint(simplex[i].w, simplex[i].a, simplex[i].b);
        vertex_count = vertex_count + 1u;
    }
    var face_count = 0u;
    if (vertex_count != 4u) {
        var invalid: EpaResult;
        invalid.valid = false;
        invalid.normal = vec3f(0.0, 1.0, 0.0);
        invalid.depth = 0.0;
        invalid.point = vec3f(0.0);
        return invalid;
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        var a: u32;
        var b: u32;
        var c: u32;
        if (i == 0u) {
            a = 0u;
            b = 1u;
            c = 2u;
        } else if (i == 1u) {
            a = 0u;
            b = 2u;
            c = 3u;
        } else if (i == 2u) {
            a = 0u;
            b = 3u;
            c = 1u;
        } else {
            a = 1u;
            b = 3u;
            c = 2u;
        }
        let normal = cross(polytope[b].w - polytope[a].w, polytope[c].w - polytope[a].w);
        let length_n = length(normal);
        if (length_n < 1e-10) {
            var invalid: EpaResult;
            invalid.valid = false;
            invalid.normal = vec3f(0.0, 1.0, 0.0);
            invalid.depth = 0.0;
            invalid.point = vec3f(0.0);
            return invalid;
        }
        let unit = normal / length_n;
        let centroid = (polytope[a].w + polytope[b].w + polytope[c].w) * (1.0 / 3.0);
        faces[face_count] = EpaFace(a, b, c, select(unit, -unit, dot(unit, centroid) > 0.0));
        face_count = face_count + 1u;
    }
    var best_dist = 0.0;
    var best_face = 0u;
    for (var iter = 0u; iter < 32u; iter = iter + 1u) {
        best_dist = 3.402823466e38;
        best_face = 0u;
        for (var i = 0u; i < face_count; i = i + 1u) {
            let dist = dot(faces[i].normal, polytope[faces[i].a].w);
            if (dist < best_dist && length(faces[i].normal) > 0.5) {
                best_dist = dist;
                best_face = i;
            }
        }
        let probe = support_pair(first, second, faces[best_face].normal).w;
        let gain = dot(probe, faces[best_face].normal) - best_dist;
        if (gain < 1e-4) {
            break;
        }
        if (vertex_count >= 64u) {
            break;
        }
        polytope[vertex_count] = EpaPoint(probe, support(first, faces[best_face].normal), support(second, -faces[best_face].normal));
        vertex_count = vertex_count + 1u;
        var visible_count = 0u;
        for (var i = 0u; i < face_count; i = i + 1u) {
            let vert = polytope[faces[i].a].w;
            if (dot(faces[i].normal, probe - vert) > 1e-6) {
                visible[visible_count] = i;
                visible_count = visible_count + 1u;
            }
        }
        var new_count = 0u;
        for (var i = 0u; i < visible_count; i = i + 1u) {
            let face = faces[visible[i]];
            var edges: array<vec2f, 3>;
            edges[0] = vec2f(f32(face.a), f32(face.b));
            edges[1] = vec2f(f32(face.b), f32(face.c));
            edges[2] = vec2f(f32(face.c), f32(face.a));
            for (var e = 0u; e < 3u; e = e + 1u) {
                let start = u32(edges[e].x);
                let end = u32(edges[e].y);
                var is_shared = false;
                for (var j = 0u; j < visible_count; j = j + 1u) {
                    if (j == i) {
                        continue;
                    }
                    let other = faces[visible[j]];
                    if ((other.a == end && other.b == start) || (other.b == end && other.c == start) || (other.c == end && other.a == start)) {
                        is_shared = true;
                    }
                }
                if (!is_shared && new_count < 64u) {
                    let apex = u32(vertex_count - 1u);
                    let normal = cross(polytope[end].w - polytope[apex].w, polytope[start].w - polytope[apex].w);
                    let length_n = length(normal);
                    if (length_n > 1e-10) {
                        let unit = normal / length_n;
                        let centroid = (polytope[apex].w + polytope[start].w + polytope[end].w) * (1.0 / 3.0);
                        new_faces[new_count] = EpaFace(apex, start, end, select(unit, -unit, dot(unit, centroid) > 0.0));
                        new_count = new_count + 1u;
                    }
                }
            }
        }
        var write = 0u;
        var seen = 0u;
        for (var i = 0u; i < face_count; i = i + 1u) {
            if (seen < visible_count && visible[seen] == i) {
                seen = seen + 1u;
                continue;
            }
            faces[write] = faces[i];
            write = write + 1u;
        }
        face_count = write;
        if (face_count + new_count > 128u) {
            break;
        }
        for (var i = 0u; i < new_count; i = i + 1u) {
            faces[face_count] = new_faces[i];
            face_count = face_count + 1u;
        }
    }
    best_dist = 3.402823466e38;
    for (var i = 0u; i < face_count; i = i + 1u) {
        let dist = dot(faces[i].normal, polytope[faces[i].a].w);
        if (dist < best_dist && length(faces[i].normal) > 0.5) {
            best_dist = dist;
            best_face = i;
        }
    }
    var result: EpaResult;
    result.valid = best_dist < 3.402823466e38;
    result.normal = faces[best_face].normal;
    result.depth = best_dist;
    result.point = polytope[faces[best_face].a].a;
    return result;
}

fn shape_scale(world: WorldShape) -> f32 {
    var scale = world.radius + world.half_height;
    scale = max(scale, max(max(world.half_extents.x, world.half_extents.y), world.half_extents.z));
    return max(scale, 1e-4) * max(max(world.scale.x, world.scale.y), world.scale.z);
}

fn simplex_expand_dir(simplex: array<SimplexPoint, 4>, count: u32) -> vec3f {
    if (count == 1u) {
        return vec3f(1.0, 0.0, 0.0);
    }
    if (count == 2u) {
        let edge = simplex[1].w - simplex[0].w;
        var d = cross(edge, vec3f(1.0, 0.0, 0.0));
        if (length(d) < 1e-6) {
            d = cross(edge, vec3f(0.0, 1.0, 0.0));
        }
        return normalize(d);
    }
    let normal = cross(simplex[1].w - simplex[0].w, simplex[2].w - simplex[0].w);
    if (length(normal) > 1e-8) {
        return normalize(normal);
    }
    return vec3f(0.0, 1.0, 0.0);
}

fn convex_closest(first: WorldShape, second: WorldShape, out_simplex: ptr<function, array<SimplexPoint, 4>>, out_count: ptr<function, u32>) -> ConvexClosest {
    var simplex: array<SimplexPoint, 4>;
    var count = 0u;
    var result: ConvexClosest;
    result.distance = 0.0;
    result.point_a = first.center;
    result.point_b = second.center;
    result.normal = sign_normalize(second.center - first.center);
    result.penetrating = false;
    var dir = second.center - first.center;
    if (length(dir) < 1e-8) {
        dir = vec3f(1.0, 0.0, 0.0);
    }
    simplex[0] = support_pair(first, second, dir);
    count = 1u;
    var closest = simplex_closest(simplex, count);
    let tiny = 1e-4 * min(shape_scale(first), shape_scale(second));
    for (var iter = 0u; iter < 24u; iter = iter + 1u) {
        if (closest.penetrating || length(closest.v) < tiny) {
            while (count < 4u) {
                let dir = simplex_expand_dir(simplex, count);
                simplex[count] = support_pair(first, second, dir);
                count = count + 1u;
            }
            result.penetrating = true;
            result.distance = 0.0;
            result.point_a = closest.point_a;
            result.point_b = closest.point_b;
            break;
        }
        let sp = support_pair(first, second, -closest.v);
        if (dot(sp.w, -closest.v) <= dot(closest.v, -closest.v) + 1e-4 * max(1.0, dot(closest.v, closest.v))) {
            result.distance = length(closest.v);
            result.point_a = closest.point_a;
            result.point_b = closest.point_b;
            if (result.distance > 1e-6) {
                result.normal = sign_normalize(closest.point_b - closest.point_a);
            }
            break;
        }
        count = simplex_reduce(&simplex, closest.lambdas);
        if (simplex_holds(simplex, count, sp.w, shape_scale(second))) {
            result.distance = length(closest.v);
            result.point_a = closest.point_a;
            result.point_b = closest.point_b;
            if (result.distance > 1e-6) {
                result.normal = sign_normalize(closest.point_b - closest.point_a);
            }
            break;
        }
        simplex[count] = sp;
        count = count + 1u;
        closest = simplex_closest(simplex, count);
    }
    *out_simplex = simplex;
    *out_count = count;
    return result;
}

fn no_hit() -> ShapeHit {
    return ShapeHit(NO_HIT, vec3f(0.0), vec3f(0.0), NO_TRIANGLE);
}

fn support_projection_depth(first: WorldShape, second: WorldShape, direction: vec3f) -> f32 {
    let n = normalize(direction);
    return dot(support(first, n) - support(second, -n), n);
}

fn convex_penetration_probe(first: WorldShape, second: WorldShape, simplex: array<SimplexPoint, 4>, count: u32) -> EpaResult {
    var best: EpaResult;
    best.valid = false;
    best.normal = vec3f(0.0, 1.0, 0.0);
    best.depth = 3.402823466e38;
    best.point = (first.center + second.center) * 0.5;
    var probed: array<vec3f, 16>;
    var probe_count = 0u;
    for (var i = 0u; i < count; i = i + 1u) {
        for (var j = i + 1u; j < count; j = j + 1u) {
            for (var k = j + 1u; k < count; k = k + 1u) {
                let normal = cross(simplex[j].w - simplex[i].w, simplex[k].w - simplex[i].w);
                if (length(normal) > 1e-8 && probe_count < 16u) {
                    probed[probe_count] = normal;
                    probe_count = probe_count + 1u;
                }
            }
        }
    }
    let center_dir = second.center - first.center;
    if (length(center_dir) > 1e-8 && probe_count < 16u) {
        probed[probe_count] = center_dir;
        probe_count = probe_count + 1u;
    }
    let axes: array<vec3f, 6> = array(
        vec3f(1.0, 0.0, 0.0),
        vec3f(-1.0, 0.0, 0.0),
        vec3f(0.0, 1.0, 0.0),
        vec3f(0.0, -1.0, 0.0),
        vec3f(0.0, 0.0, 1.0),
        vec3f(0.0, 0.0, -1.0),
    );
    for (var i = 0u; i < 6u; i = i + 1u) {
        if (probe_count < 16u) {
            probed[probe_count] = axes[i];
            probe_count = probe_count + 1u;
        }
    }
    for (var i = 0u; i < probe_count; i = i + 1u) {
        let depth = support_projection_depth(first, second, probed[i]);
        if (depth < best.depth) {
            best.depth = depth;
            best.normal = normalize(probed[i]);
        }
    }
    best.valid = best.depth < 3.402823466e38;
    let point_a = support(first, best.normal);
    let point_b = support(second, -best.normal);
    best.point = (point_a + point_b) * 0.5;
    return best;
}

fn convex_hit(first: WorldShape, second: WorldShape) -> ShapeHit {
    var simplex: array<SimplexPoint, 4>;
    var count = 0u;
    let closest = convex_closest(first, second, &simplex, &count);
    if (!closest.penetrating) {
        let probe = convex_penetration_probe(first, second, simplex, count);
        if (probe.depth > 0.0) {
            var normal = probe.normal;
            if (dot(normal, second.center - first.center) < 0.0) {
                normal = -normal;
            }
            return ShapeHit(-probe.depth, (closest.point_a + closest.point_b) * 0.5, normal, NO_TRIANGLE);
        }
        if (closest.distance > 0.0) {
            return ShapeHit(closest.distance, (closest.point_a + closest.point_b) * 0.5, closest.normal, NO_TRIANGLE);
        }
        return no_hit();
    }
    let epa = epa_tetrahedron(first, second, simplex, count);
    var result: ShapeHit;
    if (epa.valid && epa.depth > 0.0 && epa.depth < 3.402823466e38 && length(epa.normal) > 0.5 && epa.depth < support_projection_depth(first, second, epa.normal) + 0.1 * min(shape_scale(first), shape_scale(second))) {
        let point_a = support(first, epa.normal);
        let point_b = support(second, -epa.normal);
        result = ShapeHit(-epa.depth, (point_a + point_b) * 0.5, epa.normal, NO_TRIANGLE);
    } else {
        let probe = convex_penetration_probe(first, second, simplex, count);
        let point_a = support(first, probe.normal);
        let point_b = support(second, -probe.normal);
        result = ShapeHit(-probe.depth, (point_a + point_b) * 0.5, probe.normal, NO_TRIANGLE);
    }
    if (dot(result.normal, second.center - first.center) < 0.0) {
        result.normal = -result.normal;
    }
    return result;
}

fn penetration_hit(probe: WorldShape, solid: WorldShape) -> ShapeHit {
    let hit = convex_hit(probe, solid);
    return ShapeHit(hit.distance, hit.point, -hit.normal, hit.triangle);
}

fn box_corner_id(face_axis: u32, face_sign: f32, first: f32, second: f32) -> u32 {
    var id = 0u;
    id = id | select(0u, 1u << face_axis, face_sign > 0.0);
    id = id | select(0u, 1u << ((face_axis + 1u) % 3u), first > 0.0);
    id = id | select(0u, 1u << ((face_axis + 2u) % 3u), second > 0.0);
    return id;
}

const EMPTY_RING_FACE: u32 = FEATURE_INDEX_LIMIT;
const CAP_ALIGNMENT: f32 = 0.93;
const CYLINDER_CAP_POINTS: u32 = 8u;
const CYLINDER_SIDE_POINTS: u32 = 16u;
const CYLINDER_SIDE_FACE: u32 = 2u;

fn box_feature_ring(
    world: WorldShape,
    direction: vec3f,
    out: ptr<function, array<vec3f, FEATURE_MAX>>,
    out_ids: ptr<function, array<u32, FEATURE_MAX>>,
    out_face: ptr<function, u32>,
) -> u32 {
    let axes = world_rotated_axes(world);
    var best_axis = 0u;
    for (var i = 1u; i < 3u; i = i + 1u) {
        if (abs(dot(axes[i], direction)) > abs(dot(axes[best_axis], direction))) {
            best_axis = i;
        }
    }
    let sign = select(1.0, -1.0, dot(axes[best_axis], direction) < 0.0);
    let face = axes[best_axis] * sign;
    let center = world.center + face * world.half_extents[best_axis];
    let u1 = axes[(best_axis + 1u) % 3u];
    let u2 = axes[(best_axis + 2u) % 3u];
    let e1 = world.half_extents[(best_axis + 1u) % 3u];
    let e2 = world.half_extents[(best_axis + 2u) % 3u];
    (*out)[0] = center + u1 * e1 + u2 * e2;
    (*out_ids)[0] = box_corner_id(best_axis, sign, 1.0, 1.0);
    (*out)[1] = center - u1 * e1 + u2 * e2;
    (*out_ids)[1] = box_corner_id(best_axis, sign, -1.0, 1.0);
    (*out)[2] = center - u1 * e1 - u2 * e2;
    (*out_ids)[2] = box_corner_id(best_axis, sign, -1.0, -1.0);
    (*out)[3] = center + u1 * e1 - u2 * e2;
    (*out_ids)[3] = box_corner_id(best_axis, sign, 1.0, -1.0);
    *out_face = feature_field_face(best_axis * 2u + select(0u, 1u, sign > 0.0));
    return 4u;
}

fn ring_of(
    world: WorldShape,
    direction: vec3f,
    points: ptr<function, array<vec3f, 16>>,
    ids: ptr<function, array<u32, 16>>,
    count: u32,
    out: ptr<function, array<vec3f, FEATURE_MAX>>,
    out_ids: ptr<function, array<u32, FEATURE_MAX>>,
) -> u32 {
    if (count <= 1u) {
        (*out)[0] = support(world, direction);
        (*out_ids)[0] = (*ids)[0];
        return 1u;
    }
    let n = sign_normalize(direction);
    var center = vec3f(0.0);
    for (var i = 0u; i < count; i = i + 1u) {
        center = center + points[i];
    }
    center = center / f32(count);
    var u = vec3f(0.0);
    if (abs(n.x) > 0.9) {
        u = normalize(cross(n, vec3f(0.0, 1.0, 0.0)));
    } else {
        u = normalize(cross(n, vec3f(1.0, 0.0, 0.0)));
    }
    let v = cross(n, u);
    for (var i = 0u; i < count; i = i + 1u) {
        for (var j = i + 1u; j < count; j = j + 1u) {
            let pa = points[i] - center;
            let pb = points[j] - center;
            let angle_a = atan2(dot(pa, v), dot(pa, u));
            let angle_b = atan2(dot(pb, v), dot(pb, u));
            if (angle_b < angle_a) {
                let tmp = points[i];
                points[i] = points[j];
                points[j] = tmp;
                let tmp_id = ids[i];
                ids[i] = ids[j];
                ids[j] = tmp_id;
            }
        }
    }
    var keep = min(count, FEATURE_MAX);
    for (var i = 0u; i < keep; i = i + 1u) {
        (*out)[i] = points[i];
        (*out_ids)[i] = ids[i];
    }
    return keep;
}

fn hull_feature_ring(
    world: WorldShape,
    direction: vec3f,
    out: ptr<function, array<vec3f, FEATURE_MAX>>,
    out_ids: ptr<function, array<u32, FEATURE_MAX>>,
    out_face: ptr<function, u32>,
) -> u32 {
    let source = shape_sources[world.source];
    let d = quat_rotate(quat_conjugate(world.rotation), direction) * world.scale;
    let best_dot = dot(support_local(world, d), d);
    let extent = source.local_max - source.local_min;
    let eps = 1e-3 * max(max(extent.x, extent.y), extent.z) * max(max(world.scale.x, world.scale.y), world.scale.z);
    var points: array<vec3f, 16>;
    var ids: array<u32, 16>;
    var count = 0u;
    var first_id = EMPTY_RING_FACE;
    if (source.vertex_count <= LINEAR_SUPPORT_VERTICES) {
        for (var i = 0u; i < source.vertex_count && count < 16u; i = i + 1u) {
            let v = shape_vertices[source.vertex_offset + i].xyz;
            if (dot(v, d) >= best_dot - eps) {
                points[count] = world.center + quat_rotate(world.rotation, v * world.scale);
                ids[count] = i;
                first_id = min(first_id, i);
                count = count + 1u;
            }
        }
        *out_face = feature_field_face(first_id);
        return ring_of(world, direction, &points, &ids, count, out, out_ids);
    }
    var stack: array<u32, 64>;
    var stack_count = 1u;
    stack[0] = source.node_offset;
    for (var visit = 0u; visit < source.node_count && stack_count > 0u && count < 16u; visit = visit + 1u) {
        stack_count = stack_count - 1u;
        let node = shape_nodes[stack[stack_count]];
        let bound = dot((node.min + node.max) * 0.5, d) + dot((node.max - node.min) * 0.5, abs(d));
        if (bound >= best_dot - eps) {
            if (node.leaf == 1u) {
                for (var i = 0u; i < node.right && count < 16u; i = i + 1u) {
                    let tri = shape_triangles[source.triangle_offset + node.left + i];
                    let corners = array<u32, 3>(tri.a, tri.b, tri.c);
                    for (var c = 0u; c < 3u && count < 16u; c = c + 1u) {
                        let v = shape_vertices[source.vertex_offset + corners[c]].xyz;
                        if (dot(v, d) >= best_dot - eps) {
                            let p = world.center + quat_rotate(world.rotation, v * world.scale);
                            var seen = false;
                            for (var j = 0u; j < count; j = j + 1u) {
                                if (all(p == points[j])) {
                                    seen = true;
                                }
                            }
                            if (!seen) {
                                points[count] = p;
                                ids[count] = corners[c];
                                first_id = min(first_id, corners[c]);
                                count = count + 1u;
                            }
                        }
                    }
                }
            } else if (stack_count + 2u <= 64u) {
                stack[stack_count] = node.left;
                stack_count = stack_count + 1u;
                stack[stack_count] = node.right;
                stack_count = stack_count + 1u;
            }
        }
    }
    *out_face = feature_field_face(first_id);
    return ring_of(world, direction, &points, &ids, count, out, out_ids);
}

fn capsule_feature(
    world: WorldShape,
    direction: vec3f,
    out: ptr<function, array<vec3f, FEATURE_MAX>>,
    out_ids: ptr<function, array<u32, FEATURE_MAX>>,
    out_face: ptr<function, u32>,
) -> u32 {
    let local_direction = quat_rotate(quat_conjugate(world.rotation), direction) * world.scale;
    let cap = select(-1.0, 1.0, local_direction.y > 0.0);
    let radial = sign_normalize(local_direction) * world.radius;
    *out_face = feature_field_face(0u);
    if (abs(local_direction.y) > CAP_ALIGNMENT * length(local_direction)) {
        let local = vec3f(0.0, cap * world.half_height, 0.0) + radial;
        (*out)[0] = world.center + quat_rotate(world.rotation, local * world.scale);
        (*out_ids)[0] = select(0u, 1u, cap > 0.0);
        return 1u;
    }
    let upper = vec3f(0.0, world.half_height, 0.0) + radial;
    let lower = vec3f(0.0, -world.half_height, 0.0) + radial;
    (*out)[0] = world.center + quat_rotate(world.rotation, upper * world.scale);
    (*out)[1] = world.center + quat_rotate(world.rotation, lower * world.scale);
    (*out_ids)[0] = 0u;
    (*out_ids)[1] = 1u;
    return 2u;
}

fn cylinder_feature(
    world: WorldShape,
    direction: vec3f,
    out: ptr<function, array<vec3f, FEATURE_MAX>>,
    out_ids: ptr<function, array<u32, FEATURE_MAX>>,
    out_face: ptr<function, u32>,
) -> u32 {
    let axis = shape_axis(world);
    let alignment = dot(axis, normalize(direction));
    if (abs(alignment) > CAP_ALIGNMENT) {
        let sign = select(1.0, -1.0, alignment < 0.0);
        let cap = select(0u, 1u, sign > 0.0);
        let cap_center = world.center + axis * world.half_height * sign;
        var u = normalize(cross(axis, vec3f(1.0, 0.0, 0.0)));
        if (length(cross(axis, vec3f(1.0, 0.0, 0.0))) < 1e-6) {
            u = normalize(cross(axis, vec3f(0.0, 0.0, 1.0)));
        }
        let v = cross(axis, u);
        for (var i = 0u; i < CYLINDER_CAP_POINTS; i = i + 1u) {
            let angle = f32(i) * 0.78539816;
            (*out)[i] = cap_center + (u * cos(angle) + v * sin(angle)) * world.radius;
            (*out_ids)[i] = cap * CYLINDER_CAP_POINTS + i;
        }
        *out_face = feature_field_face(cap);
        return CYLINDER_CAP_POINTS;
    }
    let side = sign_normalize(direction - axis * dot(direction, axis));
    (*out)[0] = world.center + axis * world.half_height + side * world.radius;
    (*out_ids)[0] = CYLINDER_SIDE_POINTS;
    (*out)[1] = world.center - axis * world.half_height + side * world.radius;
    (*out_ids)[1] = CYLINDER_SIDE_POINTS + 1u;
    *out_face = feature_field_face(CYLINDER_SIDE_FACE);
    return 2u;
}

fn shape_feature(
    world: WorldShape,
    direction: vec3f,
    out: ptr<function, array<vec3f, FEATURE_MAX>>,
    out_ids: ptr<function, array<u32, FEATURE_MAX>>,
    out_face: ptr<function, u32>,
    ring: ptr<function, bool>,
) -> u32 {
    if (world.kind == SHAPE_CUBOID) {
        *ring = true;
        return box_feature_ring(world, direction, out, out_ids, out_face);
    }
    if (world.kind == SHAPE_HULL) {
        let count = hull_feature_ring(world, direction, out, out_ids, out_face);
        *ring = count > 2u;
        return count;
    }
    if (world.kind == SHAPE_CAPSULE) {
        *ring = false;
        return capsule_feature(world, direction, out, out_ids, out_face);
    }
    if (world.kind == SHAPE_CYLINDER) {
        let count = cylinder_feature(world, direction, out, out_ids, out_face);
        *ring = count > 2u;
        return count;
    }
    *ring = false;
    (*out)[0] = support(world, direction);
    (*out_ids)[0] = 0u;
    *out_face = feature_field_face(0u);
    return 1u;
}

fn feature_radius(points: ptr<function, array<vec3f, FEATURE_MAX>>, count: u32) -> f32 {
    var center = vec3f(0.0);
    for (var i = 0u; i < count; i = i + 1u) {
        center = center + (*points)[i];
    }
    center = center / f32(count);
    var radius = 0.0;
    for (var i = 0u; i < count; i = i + 1u) {
        radius = max(radius, length((*points)[i] - center));
    }
    return radius;
}

const CLIP_POINTS: u32 = MANIFOLD_CANDIDATES;

struct ClippedFeature {
    points: array<vec3f, CLIP_POINTS>,
    ids: array<u32, CLIP_POINTS>,
    count: u32,
}

fn clip_feature_to_ring(
    points: array<vec3f, FEATURE_MAX>,
    ids: array<u32, FEATURE_MAX>,
    count: u32,
    ring: array<vec3f, FEATURE_MAX>,
    ring_count: u32,
    ref_dir: vec3f,
) -> ClippedFeature {
    var held_points: array<vec3f, CLIP_POINTS>;
    var held_ids: array<u32, CLIP_POINTS>;
    for (var i = 0u; i < count; i = i + 1u) {
        held_points[i] = points[i];
        held_ids[i] = ids[i];
    }
    var live = count;
    for (var side = 0u; side < ring_count; side = side + 1u) {
        let current = ring[side];
        let next = ring[(side + 1u) % ring_count];
        let plane_normal = normalize(cross(next - current, ref_dir));
        var kept_points: array<vec3f, CLIP_POINTS>;
        var kept_ids: array<u32, CLIP_POINTS>;
        var written = 0u;
        if (live == 2u) {
            let first_point = held_points[0];
            let second_point = held_points[1];
            let first_side = dot(first_point - current, plane_normal) - CLIP_MARGIN;
            let second_side = dot(second_point - current, plane_normal) - CLIP_MARGIN;
            if (first_side <= 0.0) {
                kept_points[written] = first_point;
                kept_ids[written] = held_ids[0];
                written = written + 1u;
            }
            if (first_side * second_side < 0.0 && written < CLIP_POINTS) {
                kept_points[written] = first_point + (second_point - first_point) * (first_side / (first_side - second_side));
                kept_ids[written] = feature_field_clip(side * CLIP_POINTS);
                written = written + 1u;
            }
            if (second_side <= 0.0 && written < CLIP_POINTS) {
                kept_points[written] = second_point;
                kept_ids[written] = held_ids[1];
                written = written + 1u;
            }
        } else {
            for (var i = 0u; i < live; i = i + 1u) {
                let current_point = held_points[i];
                let next_point = held_points[(i + 1u) % live];
                let current_side = dot(current_point - current, plane_normal) - CLIP_MARGIN;
                let next_side = dot(next_point - current, plane_normal) - CLIP_MARGIN;
                if (current_side <= 0.0 && written < CLIP_POINTS) {
                    kept_points[written] = current_point;
                    kept_ids[written] = held_ids[i];
                    written = written + 1u;
                }
                if (current_side * next_side < 0.0 && written < CLIP_POINTS) {
                    kept_points[written] = current_point + (next_point - current_point) * (current_side / (current_side - next_side));
                    kept_ids[written] = feature_field_clip(side * CLIP_POINTS + i);
                    written = written + 1u;
                }
            }
        }
        live = written;
        held_points = kept_points;
        held_ids = kept_ids;
        if (live == 0u) {
            break;
        }
    }
    var clipped: ClippedFeature;
    clipped.points = held_points;
    clipped.ids = held_ids;
    clipped.count = live;
    return clipped;
}

fn convex_pair_manifold(
    first: WorldShape,
    second: WorldShape,
    direction: vec3f,
    contact: ptr<function, Contact>,
) -> bool {
    var first_points: array<vec3f, FEATURE_MAX>;
    var first_ids: array<u32, FEATURE_MAX>;
    var second_points: array<vec3f, FEATURE_MAX>;
    var second_ids: array<u32, FEATURE_MAX>;
    var first_face = 0u;
    var second_face = 0u;
    var first_ring = false;
    var second_ring = false;
    let first_count = shape_feature(first, direction, &first_points, &first_ids, &first_face, &first_ring);
    let second_count = shape_feature(second, -direction, &second_points, &second_ids, &second_face, &second_ring);
    if (!first_ring && !second_ring) {
        return false;
    }
    var first_reference = first_ring;
    if (first_ring && second_ring) {
        first_reference = feature_radius(&first_points, first_count) >= feature_radius(&second_points, second_count);
    }
    var reference: array<vec3f, FEATURE_MAX>;
    var reference_count = 0u;
    var reference_face = second_face;
    var incident: array<vec3f, FEATURE_MAX>;
    var incident_ids: array<u32, FEATURE_MAX>;
    var incident_count = 0u;
    var ref_dir: vec3f;
    if (first_reference) {
        reference = first_points;
        reference_count = first_count;
        reference_face = first_face;
        incident = second_points;
        incident_ids = second_ids;
        incident_count = second_count;
        ref_dir = direction;
    } else {
        reference = second_points;
        reference_count = second_count;
        incident = first_points;
        incident_ids = first_ids;
        incident_count = first_count;
        ref_dir = -direction;
    }
    var center = vec3f(0.0);
    for (var i = 0u; i < reference_count; i = i + 1u) {
        center = center + reference[i];
    }
    center = center / f32(reference_count);
    var deepest = vec3f(0.0);
    var deepest_depth = -1e30;
    for (var i = 0u; i < incident_count; i = i + 1u) {
        let depth = dot(center - incident[i], ref_dir);
        if (depth > deepest_depth) {
            deepest_depth = depth;
            deepest = incident[i];
        }
    }
    let clipped = clip_feature_to_ring(incident, incident_ids, incident_count, reference, reference_count, ref_dir);
    var candidates: array<ManifoldPoint, MANIFOLD_CANDIDATES>;
    var candidate_count = 0u;
    for (var i = 0u; i < clipped.count && candidate_count < MANIFOLD_CANDIDATES; i = i + 1u) {
        let depth = dot(center - clipped.points[i], ref_dir);
        if (depth > 0.0) {
            candidates[candidate_count] = manifold_candidate(
                clipped.points[i] - ref_dir * (depth * 0.5),
                depth,
                select(
                    feature_vertex(clipped.ids[i], reference_face),
                    feature_vertex(reference_face, clipped.ids[i]),
                    first_reference,
                ),
            );
            candidate_count = candidate_count + 1u;
        }
    }
    if (candidate_count == 0u) {
        if (deepest_depth <= 0.0) {
            return false;
        }
        manifold_push(contact, deepest - ref_dir * (deepest_depth * 0.5), deepest_depth, feature_point());
        return true;
    }
    manifold_keep(contact, &candidates, candidate_count);
    return true;
}

