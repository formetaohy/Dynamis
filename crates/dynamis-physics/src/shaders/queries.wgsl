struct Query {
    origin: vec3f,
    kind: u32,
    direction: vec3f,
    extent: f32,
    slot: u32,
    _pad0: u32,
    _pad1: u32,
}

struct QueryResult {
    body_id: u32,
    body_generation: u32,
    distance: f32,
    hit: u32,
}

@group(0) @binding(0) var<storage, read> query_records: array<Query>;
@group(0) @binding(1) var<storage, read> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read_write> query_results: array<QueryResult>;
@group(0) @binding(3) var<uniform> params: SimParams;

var<workgroup> partial_t: array<f32, 64>;
var<workgroup> partial_id: array<u32, 64>;
var<workgroup> partial_generation: array<u32, 64>;

fn ray_hit(query: Query, body: RigidBody) -> f32 {
    let offset = query.origin - body.position;
    let direction = normalize(query.direction);
    let projection = dot(offset, direction);
    let discriminant = projection * projection - (dot(offset, offset) - body.radius * body.radius);
    if (discriminant < 0.0) {
        return 3.402823466e+38;
    }
    let root = sqrt(discriminant);
    var t = -projection - root;
    if (t < 0.0) {
        t = -projection + root;
    }
    if (t < 0.0 || t > query.extent) {
        return 3.402823466e+38;
    }
    return t;
}

fn sphere_overlap(query: Query, body: RigidBody) -> f32 {
    let separation = length(query.origin - body.position) - query.extent - body.radius;
    if (separation > 0.0) {
        return 3.402823466e+38;
    }
    return separation;
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) invocation_id: vec3u,
) {
    let query = query_records[workgroup_id.x];
    var best_t = 3.402823466e+38;
    var best_id = 0xFFFFFFFFu;
    var best_generation = 0xFFFFFFFFu;
    var index = invocation_id.x;
    while (index < params.body_count) {
        let body = bodies[index];
        var distance = 3.402823466e+38;
        if (query.kind == 0u) {
            distance = ray_hit(query, body);
        } else if (query.kind == 1u) {
            distance = sphere_overlap(query, body);
        }
        if (distance < best_t) {
            best_t = distance;
            best_id = body.body_id;
            best_generation = body.generation;
        }
        index = index + 64u;
    }
    partial_t[invocation_id.x] = best_t;
    partial_id[invocation_id.x] = best_id;
    partial_generation[invocation_id.x] = best_generation;
    workgroupBarrier();
    if (invocation_id.x == 0u) {
        var result_t = 3.402823466e+38;
        var result_id = 0xFFFFFFFFu;
        var result_generation = 0xFFFFFFFFu;
        for (var i = 0u; i < 64u; i = i + 1u) {
            if (partial_t[i] < result_t) {
                result_t = partial_t[i];
                result_id = partial_id[i];
                result_generation = partial_generation[i];
            }
        }
        if (result_t == 3.402823466e+38) {
            query_results[query.slot] = QueryResult(0xFFFFFFFFu, 0xFFFFFFFFu, 0.0, 0u);
        } else {
            query_results[query.slot] =
                QueryResult(result_id, result_generation, result_t, 1u);
        }
    }
}
