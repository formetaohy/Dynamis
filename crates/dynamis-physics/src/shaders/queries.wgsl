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

const NO_HIT: f32 = 3.402823466e+38;
const NO_BODY: u32 = 0xFFFFFFFFu;

@group(0) @binding(0) var<storage, read> query_records: array<Query>;
@group(0) @binding(1) var<storage, read> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read_write> query_results: array<QueryResult>;
@group(0) @binding(3) var<uniform> params: SimParams;

var<workgroup> partial_t: array<f32, WORKGROUP_SIZE>;
var<workgroup> partial_id: array<u32, WORKGROUP_SIZE>;
var<workgroup> partial_generation: array<u32, WORKGROUP_SIZE>;

fn ray_hit(query: Query, body: RigidBody) -> f32 {
    let offset = query.origin - body.position;
    let direction = normalize(query.direction);
    let projection = dot(offset, direction);
    let discriminant = projection * projection - (dot(offset, offset) - body.radius * body.radius);
    if (discriminant < 0.0) {
        return NO_HIT;
    }
    let root = sqrt(discriminant);
    var t = -projection - root;
    if (t < 0.0) {
        t = -projection + root;
    }
    if (t < 0.0 || t > query.extent) {
        return NO_HIT;
    }
    return t;
}

fn sphere_overlap(query: Query, body: RigidBody) -> f32 {
    let separation = length(query.origin - body.position) - query.extent - body.radius;
    if (separation > 0.0) {
        return NO_HIT;
    }
    return separation;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) invocation_id: vec3u,
) {
    let query = query_records[workgroup_id.x];
    var best_t = NO_HIT;
    var best_id = NO_BODY;
    var best_generation = NO_BODY;
    var index = invocation_id.x;
    while (index < params.body_count) {
        let body = bodies[index];
        var distance = NO_HIT;
        if (query.kind == QUERY_RAY) {
            distance = ray_hit(query, body);
        } else if (query.kind == QUERY_SPHERE) {
            distance = sphere_overlap(query, body);
        }
        if (distance < best_t) {
            best_t = distance;
            best_id = body.body_id;
            best_generation = body.generation;
        }
        index = index + WORKGROUP_SIZE;
    }
    partial_t[invocation_id.x] = best_t;
    partial_id[invocation_id.x] = best_id;
    partial_generation[invocation_id.x] = best_generation;
    workgroupBarrier();
    if (invocation_id.x == 0u) {
        var result_t = NO_HIT;
        var result_id = NO_BODY;
        var result_generation = NO_BODY;
        for (var i = 0u; i < WORKGROUP_SIZE; i = i + 1u) {
            if (partial_t[i] < result_t) {
                result_t = partial_t[i];
                result_id = partial_id[i];
                result_generation = partial_generation[i];
            }
        }
        if (result_t == NO_HIT) {
            query_results[query.slot] = QueryResult(NO_BODY, NO_BODY, 0.0, 0u);
        } else {
            query_results[query.slot] =
                QueryResult(result_id, result_generation, result_t, 1u);
        }
    }
}
