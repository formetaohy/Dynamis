@group(0) @binding(0) var<storage, read> vehicles: array<Vehicle>;
@group(0) @binding(1) var<storage, read> vehicle_wheels: array<VehicleWheel>;
@group(0) @binding(2) var<storage, read> vehicle_inputs: array<VehicleInput>;
@group(0) @binding(3) var<storage, read_write> vehicle_states: array<VehicleState>;
@group(0) @binding(4) var<storage, read_write> vehicle_sweeps: array<Query>;
@group(0) @binding(5) var<storage, read_write> vehicle_hits: array<QueryHit>;
@group(0) @binding(6) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(7) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(8) var<storage, read> row_of_body: array<u32>;
@group(0) @binding(9) var<storage, read_write> wake_flags: array<atomic<u32>>;
@group(0) @binding(10) var<uniform> params: StepParams;

const TAU: f32 = 6.283185307179586;

struct WheelHit {
    touching: bool,
    body: u32,
    generation: u32,
    point: vec3f,
    distance: f32,
    normal: vec3f,
}

struct WheelLoad {
    force: vec3f,
    torque: vec3f,
    grounded: u32,
}

fn load_body(row: u32) -> Body {
    return Body(body_states[row], body_descs[row]);
}

fn wheel_hit(id: u32) -> WheelHit {
    if (vehicle_sweeps[id].count == 0u) {
        return WheelHit(false, NO_BODY, 0u, vec3f(0.0), NO_HIT, vec3f(0.0, 1.0, 0.0));
    }
    let record = vehicle_hits[id];
    return WheelHit(
        true,
        record.body_id,
        record.body_generation,
        record.point,
        record.distance,
        record.normal,
    );
}

fn wheel_ground_velocity(hit: WheelHit) -> vec3f {
    let row = resolve_row(hit.body, hit.generation);
    if (row == NO_BODY) {
        return vec3f(0.0);
    }
    return point_velocity(load_body(row), hit.point);
}

fn wheel_query(vehicle: Vehicle, base: u32, wheel: VehicleWheel, origin: vec3f, axis: vec3f) -> Query {
    var query: Query;
    query.kind = QUERY_SWEEP;
    query.shape_kind = SHAPE_SPHERE;
    query.filters.flags = FILTER_IGNORE_SENSORS;
    query.filters.targets = QUERY_TARGET_COLLIDERS;
    query.filters.group = 0u;
    query.filters.mask = 0xFFFFFFFFu;
    query.source = 0u;
    query.max_hits = 1u;
    query.hit_base = base;
    query.filters.exclude_id = vehicle.body_id;
    query.filters.exclude_generation = vehicle.generation;
    query.filters.include_id = NO_BODY;
    query.filters.include_generation = 0u;
    query.origin = origin;
    query.direction = axis;
    query.extent = wheel.travel;
    query.radius = wheel.radius;
    query.half_height = 0.0;
    query.half_extents = vec3f(0.0);
    query.orientation = vec4f(0.0, 0.0, 0.0, 1.0);
    return query;
}

fn wheel_load(
    vehicle: Vehicle,
    wheel: VehicleWheel,
    state: BodyState,
    desc: BodyDescriptor,
    mass: f32,
    wheels: u32,
    steer: f32,
    input: VehicleInput,
    hit: WheelHit,
    origin: vec3f,
    axis: vec3f,
) -> WheelLoad {
    var load: WheelLoad;
    load.force = vec3f(0.0);
    load.torque = vec3f(0.0);
    load.grounded = 0u;
    if (!hit.touching || hit.distance > wheel.travel) {
        return load;
    }
    let normal = sign_normalize(hit.normal);
    let contact = origin + axis * hit.distance - normal * wheel.radius;
    let alignment = dot(axis, normal);
    var length = hit.distance;
    if (abs(alignment) > 1e-4) {
        length = (wheel.radius - dot(origin - contact, normal)) / alignment;
    }
    if (length > wheel.travel) {
        return load;
    }
    length = max(length, 0.0);
    let share = mass / f32(wheels);
    let omega = TAU * wheel.frequency;
    let stiffness = share * omega * omega;
    let damping = 2.0 * wheel.damping_ratio * share * omega;
    let lever = contact - body_com_of(state, desc);
    let velocity = state.velocity + cross(state.angular_velocity, lever) - wheel_ground_velocity(hit);
    let compression = wheel.travel - length;
    let normal_force = max(stiffness * compression - damping * dot(velocity, normal), 0.0);
    let grip = wheel.friction * normal_force;
    var forward = quat_rotate(state.orientation, vec3f(0.0, 0.0, 1.0));
    if (wheel.steering != 0u) {
        forward = rotate_about(quat_rotate(state.orientation, vec3f(0.0, 1.0, 0.0)) * steer, forward);
    }
    let tangent = sign_normalize(forward - normal * dot(forward, normal));
    let side = cross(normal, tangent);
    let longitudinal = dot(velocity, tangent);
    let lateral = dot(velocity, side);
    let drive = select(
        0.0,
        input.throttle * vehicle.drive_force / f32(max(vehicle.driving_count, 1u)),
        wheel.driving != 0u,
    );
    let brake = input.brake * vehicle.brake_force / f32(wheels);
    let halt = clamp(-longitudinal * share / params.dt, -brake, brake);
    let traction = clamp(drive + halt, -grip, grip);
    let circle = sqrt(max(grip * grip - traction * traction, 0.0));
    let sideways = clamp(-lateral * share / params.dt, -circle, circle);
    load.force = normal * normal_force + tangent * traction + side * sideways;
    load.torque = cross(lever, load.force);
    load.grounded = 1u;
    return load;
}

fn work(index: u32) {
    let vehicle = vehicles[index];
    if (vehicle.body_id == NO_BODY) {
        return;
    }
    let row = row_of_body[vehicle.body_id];
    var body = body_states[row];
    let desc = body_descs[row];
    if (desc.inverse_mass <= 0.0) {
        return;
    }
    let input = vehicle_inputs[index];
    let wheels = vehicle.wheel_count;
    let mass = 1.0 / desc.inverse_mass;
    let steer = clamp(input.steering, -1.0, 1.0) * vehicle.max_steer;
    let up = quat_rotate(body.orientation, vec3f(0.0, 1.0, 0.0));
    var force = vec3f(0.0);
    var torque = vec3f(0.0);
    var grounded = 0u;
    for (var lane = 0u; lane < wheels; lane = lane + 1u) {
        let wheel = vehicle_wheels[vehicle.wheel_base + lane];
        if (wheel.radius <= 0.0) {
            continue;
        }
        let origin = body.position + quat_rotate(body.orientation, wheel.anchor);
        let axis = -up;
        let lane_load = wheel_load(
            vehicle,
            wheel,
            body,
            desc,
            mass,
            wheels,
            steer,
            input,
            wheel_hit(vehicle.wheel_base + lane),
            origin,
            axis,
        );
        force = force + lane_load.force;
        torque = torque + lane_load.torque;
        grounded = grounded + lane_load.grounded;
        vehicle_sweeps[vehicle.wheel_base + lane] = wheel_query(vehicle, vehicle.wheel_base + lane, wheel, origin, axis);
    }
    body.force = body.force + force;
    body.torque = body.torque + torque;
    body_states[row] = body;
    atomicOr(&wake_flags[row], 1u);
    var state = vehicle_states[index];
    state.position = body.position;
    state.forward_speed = dot(body.velocity, quat_rotate(body.orientation, vec3f(0.0, 0.0, 1.0)));
    state.ground_count = grounded;
    state.ground_count = grounded;
    vehicle_states[index] = state;
}
