const CONTACT_MAX_POINTS: u32 = 4u;
const CONSTRAINT_ACCUMULATOR_SLOTS: u32 = 16u;
const ELEMENT_PARTICLES: u32 = 4u;
const JOINT_DOF: u32 = 6u;
const QUERY_CANDIDATES: u32 = 4096u;
const CHARACTER_SWEEPS: u32 = 5u;

struct StepParams {
    gravity: vec4f,
    dt: f32,
    substep_dt: f32,
    damping: f32,
    angular_damping: f32,
    body_count: u32,
    substeps: u32,
    solve_iterations: u32,
    constraint_count: u32,
    relaxation: f32,
    slop: f32,
    restitution_threshold: f32,
    max_velocity: f32,
    max_angular_velocity: f32,
    dynamic_count: u32,
    sleep_velocity: f32,
    sleep_angular_velocity: f32,
    sleep_time: f32,
    friction_combine: u32,
    restitution_combine: u32,
    position_iterations: u32,
    contact_margin: f32,
    collider_count: u32,
    soft_iterations: u32,
    soft_substeps: u32,
    particle_count: u32,
    element_count: u32,
    soft_substep_dt: f32,
    soft_body_count: u32,
    attachment_count: u32,
    settle_velocity: f32,
    observed_count: u32,
    observed_joint_count: u32,
    character_count: u32,
    vehicle_count: u32,
    vehicle_wheel_count: u32,
    wake_all: u32,
    field_count: u32,
}

struct RowStreams {
    body_edit_runs: u32,
    body_moves: u32,
    constraint_moves: u32,
    soft_edits: u32,
    soft_body_edits: u32,
}

struct SoftAnnouncement {
    scene_target: u32,
    id: u32,
    generation: u32,
    step: u32,
    sensor: u32,
    announced: u32,
    _pad0: array<u32, 2>,
}

struct SoftParticle {
    position: vec4f,
    prev_position: vec4f,
    velocity: vec4f,
    support: f32,
    rest_spacing: f32,
    neighbour_offset: u32,
    neighbour_count: u32,
    owner: u32,
    generation: u32,
    announcement: SoftAnnouncement,
}

struct SoftBody {
    sleep_timer: f32,
    sleeping: u32,
    moving: atomic<u32>,
    wake: atomic<u32>,
    collision_group: u32,
    collision_mask: u32,
    acceleration: vec3f,
    events: u32,
}

struct SoftEdit {
    particle: u32,
    mask: u32,
    inverse_mass: f32,
    radius: f32,
    friction: f32,
    position: vec3f,
    velocity: vec3f,
}

struct SoftBodyEdit {
    acceleration: vec3f,
    mask: u32,
    owner: u32,
}

struct SoftAttachment {
    local: vec3f,
    particle: u32,
    body_id: u32,
    generation: u32,
}

struct SoftElement {
    particles: array<u32, ELEMENT_PARTICLES>,
    rest: f32,
    compliance: f32,
    lambda: f32,
    kind: u32,
    yield_strain: f32,
    break_strain: f32,
    plastic_flow: f32,
}

struct SoftFact {
    scene_target: u32,
    id: u32,
    generation: u32,
}

struct SoftContact {
    normal: vec3f,
    depth: f32,
    point: vec3f,
    friction: f32,
    partner: u32,
    group: u32,
    kind: u32,
    partner_inverse_mass: f32,
    contact: SoftFact,
    sensor: SoftFact,
    support: SoftFact,
}

struct BodyState {
    position: vec3f,
    _pad0: f32,
    prev_position: vec3f,
    _pad1: f32,
    prev_orientation: vec4f,
    orientation: vec4f,
    velocity: vec3f,
    _pad2: f32,
    angular_velocity: vec3f,
    _pad3: f32,
    force: vec3f,
    _pad4: f32,
    torque: vec3f,
    _pad5: f32,
    body_id: u32,
    generation: u32,
    sleep_timer: f32,
    sleeping: u32,
}

struct BodyDescriptor {
    inverse_mass: f32,
    linear_damping: f32,
    angular_damping: f32,
    gravity_scale: f32,
    sleep_velocity: f32,
    sleep_angular_velocity: f32,
    flags: u32,
    _pad0: u32,
    collision_group: u32,
    collision_mask: u32,
    _pad1: u32,
    _pad4: u32,
    com: vec3f,
    _pad2: f32,
    inertia: array<f32, 6>,
    inverse_inertia: array<f32, 6>,
    _pad3: array<f32, 4>,
}

struct RowMove {
    row: u32,
    source: u32,
    fresh: u32,
    _pad: u32,
}

struct BodyEdit {
    kind: u32,
    mask: u32,
    _pad0: u32,
    _pad1: u32,
    state: BodyState,
}

struct BodyEditRun {
    row: u32,
    first: u32,
    len: u32,
    _pad: u32,
}

struct Collider {
    slot: u32,
    kind: u32,
    flags: u32,
    radius: f32,
    half_height: f32,
    relaxation: f32,
    damping_ratio: f32,
    impact_force: f32,
    half_extents: vec3f,
    collision_group: u32,
    local_offset: vec3f,
    collision_mask: u32,
    local_rotation: vec4f,
    friction: f32,
    restitution: f32,
    source: u32,
    rolling_friction: f32,
    scale: vec3f,
    spin_friction: f32,
}

struct Surface {
    friction: f32,
    restitution: f32,
    rolling_friction: f32,
    spin_friction: f32,
}

struct Field {
    position: vec3f,
    pull: f32,
    half_extents: vec3f,
    swirl: f32,
    push: vec3f,
    radius: f32,
    medium: vec3f,
    linear_drag: f32,
    axis: vec3f,
    quadratic_drag: f32,
    orientation: vec4f,
    angular_drag: f32,
    buoyancy: f32,
    region: u32,
    collision_group: u32,
    collision_mask: u32,
}

struct ShapeSource {
    kind: u32,
    vertex_offset: u32,
    vertex_count: u32,
    triangle_offset: u32,
    node_offset: u32,
    node_count: u32,
    cell_offset: u32,
    cell_count: u32,
    grid_rows: u32,
    grid_cols: u32,
    _pad0: u32,
    _pad1: u32,
    local_min: vec3f,
    _pad2: f32,
    local_max: vec3f,
    _pad3: f32,
}

struct BvhNode {
    min: vec3f,
    _pad0: f32,
    max: vec3f,
    _pad1: f32,
    left: u32,
    right: u32,
    leaf: u32,
    _pad2: u32,
}

struct Triangle {
    a: u32,
    b: u32,
    c: u32,
    surface: u32,
    material: Surface,
}

struct Cell {
    surface: u32,
    material: Surface,
}

struct Aabb {
    min: vec3f,
    _pad0: f32,
    max: vec3f,
    _pad1: f32,
}

struct GridEntry {
    min: vec3f,
    group: u32,
    max: vec3f,
    info: u32,
}

struct ManifoldPoint {
    position: vec3f,
    depth: f32,
    local_a: vec3f,
    local_b: vec3f,
    accumulated_normal: f32,
    accumulated_tangent_1: f32,
    accumulated_tangent_2: f32,
    feature: u32,
}

struct CcdImpact {
    normal: vec3f,
    restitution: f32,
    point: vec3f,
    _pad0: f32,
}

struct Contact {
    a: u32,
    b: u32,
    point_count: u32,
    sensor: u32,
    first_body_id: u32,
    second_body_id: u32,
    first_generation: u32,
    second_generation: u32,
    normal: vec3f,
    events: u32,
    surface: u32,
    triangle: u32,
    friction: f32,
    restitution: f32,
    rolling_friction: f32,
    spin_friction: f32,
    carried_normal: f32,
    carried_tangent: f32,
    relaxation: f32,
    damping_ratio: f32,
    points: array<ManifoldPoint, CONTACT_MAX_POINTS>,
}

struct ConstraintDescriptor {
    kind: u32,
    first_body_id: u32,
    second_body_id: u32,
    flags: u32,
    anchor_a: vec3f,
    _pad1: f32,
    anchor_b: vec3f,
    _pad2: f32,
    axis_a: vec3f,
    _pad3: f32,
    axis_b: vec3f,
    _pad4: f32,
    distance: f32,
    limit_min: f32,
    limit_max: f32,
    swing_a: f32,
    swing_b: f32,
    motor_speed: f32,
    motor_max_force: f32,
    spring_frequency: f32,
    spring_damping_ratio: f32,
    break_force: f32,
    break_torque: f32,
    gear_ratio: f32,
    pulley_fixed_a: vec3f,
    _pad_pulley_a: f32,
    pulley_fixed_b: vec3f,
    _pad_pulley_b: f32,
    motor_position: f32,
    motor_stiffness: f32,
    motor_damping: f32,
    cone_angle: f32,
    linear_limit_min: vec3f,
    _pad_lim_min: f32,
    linear_limit_max: vec3f,
    _pad_lim_max: f32,
    angular_limit_min: vec3f,
    _pad_ang_min: f32,
    angular_limit_max: vec3f,
    _pad_ang_max: f32,
    linear_motor_speed: vec3f,
    _pad_lin_speed: f32,
    linear_motor_position: vec3f,
    _pad_lin_position: f32,
    linear_motor_stiffness: vec3f,
    _pad_lin_stiff: f32,
    linear_motor_damping: vec3f,
    _pad_lin_damp: f32,
    linear_motor_force: vec3f,
    _pad_lin_force: f32,
    angular_motor_speed: vec3f,
    _pad_ang_speed: f32,
    angular_motor_position: vec3f,
    _pad_ang_position: f32,
    angular_motor_stiffness: vec3f,
    _pad_ang_stiff: f32,
    angular_motor_damping: vec3f,
    _pad_ang_damp: f32,
    angular_motor_force: vec3f,
    _pad_ang_force: f32,
}

struct ConstraintRuntime {
    reaction: ConstraintReaction,
    reference: vec4f,
    accumulated: array<f32, CONSTRAINT_ACCUMULATOR_SLOTS>,
    broken: u32,
    constraint_id: u32,
    generation: u32,
    _pad0: u32,
}

struct BrokenConstraint {
    constraint_id: u32,
    generation: u32,
}

struct JointState {
    coordinates: array<f32, JOINT_DOF>,
    rates: array<f32, JOINT_DOF>,
    impulses: array<f32, JOINT_DOF>,
    dof_count: u32,
}

struct ConstraintRows {
    first_row: u32,
    second_row: u32,
}

struct ConstraintReaction {
    linear_first: vec3f,
    _pad_linear_first: f32,
    angular_first: vec3f,
    _pad_angular_first: f32,
    linear_second: vec3f,
    _pad_linear_second: f32,
    angular_second: vec3f,
    _pad_angular_second: f32,
}

struct QueryFilter {
    flags: u32,
    targets: u32,
    group: u32,
    mask: u32,
    exclude_id: u32,
    exclude_generation: u32,
    include_id: u32,
    include_generation: u32,
    exclude_soft_id: u32,
    exclude_soft_generation: u32,
    include_soft_id: u32,
    include_soft_generation: u32,
}

struct Query {
    kind: u32,
    shape_kind: u32,
    source: u32,
    max_hits: u32,
    filters: QueryFilter,
    origin: vec3f,
    _pad0: f32,
    direction: vec3f,
    extent: f32,
    radius: f32,
    half_height: f32,
    count: u32,
    hit_base: u32,
    half_extents: vec3f,
    overflow: u32,
    orientation: vec4f,
}

struct QueryHit {
    body_id: u32,
    body_generation: u32,
    distance: f32,
    scene_target: u32,
    point: vec3f,
    triangle: u32,
    normal: vec3f,
    surface: u32,
}

struct ContactEvent {
    point: vec3f,
    _pad0: f32,
    normal: vec3f,
    _pad1: f32,
    kind: u32,
    sensor: u32,
    first_target: u32,
    first_id: u32,
    first_generation: u32,
    second_target: u32,
    second_id: u32,
    second_generation: u32,
    _pad2: u32,
}

struct ImpactEvent {
    first_id: u32,
    first_generation: u32,
    second_id: u32,
    second_generation: u32,
    point: vec3f,
    impulse: f32,
    normal: vec3f,
    friction_impulse: f32,
}

struct Character {
    body_id: u32,
    generation: u32,
    radius: f32,
    half_height: f32,
    step_height: f32,
    cos_slope_limit: f32,
    max_speed: f32,
    jump_speed: f32,
}

struct CharacterInput {
    direction: vec3f,
    jump: u32,
}

struct CharacterState {
    position: vec3f,
    vertical: f32,
    down_length: f32,
    grounded: u32,
    support: u32,
    support_generation: u32,
    owner: u32,
    generation: u32,
    _pad0: u32,
    _pad1: u32,
}

struct VehicleWheel {
    anchor: vec3f,
    radius: f32,
    travel: f32,
    frequency: f32,
    damping_ratio: f32,
    friction: f32,
    steering: u32,
    driving: u32,
}

struct Vehicle {
    body_id: u32,
    generation: u32,
    wheel_base: u32,
    wheel_count: u32,
    driving_count: u32,
    max_steer: f32,
    drive_force: f32,
    brake_force: f32,
}

struct VehicleInput {
    throttle: f32,
    steering: f32,
    brake: f32,
    _pad0: f32,
}

struct VehicleState {
    position: vec3f,
    forward_speed: f32,
    ground_count: u32,
    owner: u32,
    generation: u32,
    _pad0: u32,
}
