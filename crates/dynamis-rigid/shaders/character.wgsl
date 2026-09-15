@group(0) @binding(0) var<storage, read> characters: array<Character>;
@group(0) @binding(1) var<storage, read> character_inputs: array<CharacterInput>;
@group(0) @binding(2) var<storage, read_write> character_states: array<CharacterState>;
@group(0) @binding(3) var<storage, read_write> character_sweeps: array<Query>;
@group(0) @binding(4) var<storage, read_write> character_hits: array<QueryResult>;
@group(0) @binding(5) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(6) var<storage, read> row_of_body: array<u32>;
@group(0) @binding(7) var<storage, read_write> wake_flags: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> params: StepParams;

const CHARACTER_SKIN: f32 = 0.05;
const SWEEP_FORWARD: u32 = 0u;
const SWEEP_FORWARD_LOW: u32 = 1u;
const SWEEP_LIFTED_FORWARD: u32 = 2u;
const SWEEP_DOWN: u32 = 3u;
const SWEEP_CEILING: u32 = 4u;

struct CharacterSweep {
    hit: bool,
    distance: f32,
    normal: vec3f,
}

fn character_up(gravity: vec3f) -> vec3f {
    let magnitude = length(gravity);
    if (magnitude > 0.0) {
        return -gravity / magnitude;
    }
    return vec3f(0.0, 1.0, 0.0);
}

fn character_sweep(index: u32, lane: u32) -> CharacterSweep {
    let base = index * CHARACTER_SWEEPS + lane;
    if (atomicLoad(&character_hits[base].header.count) == 0u) {
        return CharacterSweep(false, NO_HIT, vec3f(0.0));
    }
    let hit = character_hits[base].hits[0];
    return CharacterSweep(true, hit.distance, hit.normal);
}

fn floor_like(normal: vec3f, up: vec3f) -> bool {
    return dot(normal, up) > 0.5;
}

fn forward_clearance(forward: CharacterSweep, forward_low: CharacterSweep, up: vec3f) -> f32 {
    var clearance = NO_HIT;
    if (forward.hit && !floor_like(forward.normal, up)) {
        clearance = min(clearance, forward.distance);
    }
    if (forward_low.hit && !floor_like(forward_low.normal, up)) {
        clearance = min(clearance, forward_low.distance);
    }
    return clearance;
}

fn forward_blocked(forward: CharacterSweep, forward_low: CharacterSweep, up: vec3f) -> bool {
    if (!forward.hit && !forward_low.hit) {
        return false;
    }
    if (forward.hit && forward_low.hit) {
        return !(floor_like(forward.normal, up) && floor_like(forward_low.normal, up));
    }
    return !floor_like(select(forward_low.normal, forward.normal, forward.hit), up);
}

fn character_query(character: Character, origin: vec3f, direction: vec3f, extent: f32) -> Query {
    var query: Query;
    query.kind = QUERY_SWEEP;
    query.shape_kind = SHAPE_SPHERE;
    query.filters.flags = FILTER_IGNORE_SENSORS;
    query.filters.targets = QUERY_TARGET_COLLIDERS;
    query.filters.group = 0u;
    query.filters.mask = 0xFFFFFFFFu;
    query.source = 0u;
    query.max_hits = 1u;
    query.filters.exclude_id = character.body_id;
    query.filters.exclude_generation = character.generation;
    query.filters.include_id = NO_BODY;
    query.filters.include_generation = 0u;
    query.origin = origin;
    query.direction = direction;
    query.extent = extent;
    query.radius = character.radius;
    query.half_height = 0.0;
    query.half_extents = vec3f(0.0);
    query.orientation = vec4f(0.0, 0.0, 0.0, 1.0);
    return query;
}

fn work(index: u32) {
    let character = characters[index];
    if (character.body_id == NO_BODY) {
        return;
    }
    let input = character_inputs[index];
    var state = character_states[index];
    let up = character_up(params.gravity.xyz);
    let gravity = length(params.gravity.xyz);
    let input_speed = length(input.direction);
    var horizontal = vec3f(0.0);
    var forward_dir = up;
    if (input_speed > 0.0) {
        horizontal = input.direction / input_speed * character.max_speed;
        forward_dir = horizontal / character.max_speed;
    }
    let horizontal_speed = length(horizontal);
    let forward = character_sweep(index, SWEEP_FORWARD);
    let forward_low = character_sweep(index, SWEEP_FORWARD_LOW);
    let lifted_forward = character_sweep(index, SWEEP_LIFTED_FORWARD);
    let landing = character_sweep(index, SWEEP_DOWN);
    let ceiling = character_sweep(index, SWEEP_CEILING);
    var jumped = false;
    var vertical = state.vertical;
    if (state.grounded != 0u) {
        vertical = 0.0;
        if (input.jump != 0u && gravity > 0.0) {
            vertical = character.jump_speed;
            jumped = true;
        }
    } else {
        vertical = vertical - gravity * params.dt;
    }
    var support = state.position;
    var grounded = false;
    var vertical_after = vertical;
    var horizontal_step = horizontal;
    var press_dir = vec3f(0.0);
    if (!jumped && landing.hit) {
        support = support + up * -(clamp(landing.distance - CHARACTER_SKIN, 0.0, state.down_length));
        grounded = dot(landing.normal, up) > character.cos_slope_limit;
        if (grounded) {
            vertical_after = 0.0;
        }
    }
    if (horizontal_speed > 0.0 && forward_blocked(forward, forward_low, up)) {
        if (!lifted_forward.hit) {
            support = state.position + up * character.step_height;
            grounded = false;
            vertical_after = vertical;
        } else {
            let clearance = forward_clearance(forward, forward_low, up);
            let advance = min(
                max(clearance - CHARACTER_SKIN * 0.5, 0.0),
                horizontal_speed * params.dt,
            );
            horizontal_step = normalize(horizontal) * (advance / params.dt);
            press_dir = normalize(horizontal);
        }
    }
    if (ceiling.hit && vertical_after > 0.0) {
        vertical_after = 0.0;
    }
    let velocity = horizontal + up * vertical_after;
    let position = support + horizontal_step * params.dt + up * (vertical_after * params.dt);
    let down_length = CHARACTER_SKIN + (abs(vertical_after) + horizontal_speed) * params.dt;
    let up_length = CHARACTER_SKIN + max(vertical_after, 0.0) * params.dt;
    let forward_length = horizontal_speed * params.dt + CHARACTER_SKIN;
    state.position = position;
    state.vertical = vertical_after;
    state.down_length = down_length;
    state.grounded = select(0u, 1u, grounded);
    character_states[index] = state;
    let row = row_of_body[character.body_id];
    let pre_move = support + press_dir * (CHARACTER_SKIN * 0.5);
    var body = body_states[row];
    body.position = pre_move;
    body.prev_position = pre_move;
    body.velocity = velocity;
    body_states[row] = body;
    atomicOr(&wake_flags[row], 1u);
    let base = index * CHARACTER_SWEEPS;
    let bottom = position - up * character.half_height;
    let top = position + up * character.half_height;
    let lifted = position + up * character.step_height;
    character_sweeps[base + SWEEP_FORWARD] = character_query(character, position, forward_dir, forward_length);
    character_sweeps[base + SWEEP_FORWARD_LOW] = character_query(character, bottom, forward_dir, forward_length);
    character_sweeps[base + SWEEP_LIFTED_FORWARD] = character_query(character, lifted, forward_dir, forward_length);
    character_sweeps[base + SWEEP_DOWN] = character_query(character, bottom, -up, down_length);
    character_sweeps[base + SWEEP_CEILING] = character_query(character, top, up, up_length);
}
