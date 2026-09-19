use crate::wgsl::declare_constants;

include!(concat!(env!("OUT_DIR"), "/lengths.rs"));

declare_constants! {
    pub const EDIT_PATCH: u32 = 0;
    pub const EDIT_FORCE: u32 = 1;
    pub const EDIT_TORQUE: u32 = 2;
    pub const EDIT_IMPULSE: u32 = 3;
    pub const EDIT_SLEEP: u32 = 4;
    pub const EDIT_WAKE: u32 = 5;
    pub const EDIT_FORCE_AT_POINT: u32 = 6;
    pub const EDIT_IMPULSE_AT_POINT: u32 = 7;
    pub const EDIT_ANGULAR_IMPULSE: u32 = 8;
    pub const PATCH_POSITION: u32 = 1;
    pub const PATCH_VELOCITY: u32 = 2;
    pub const PATCH_ORIENTATION: u32 = 4;
    pub const PATCH_ANGULAR_VELOCITY: u32 = 8;
    pub const BODY_KINEMATIC: u32 = 1;
    pub const BODY_CCD: u32 = 2;
    pub const OVERRIDE_SLEEP_LINEAR: u32 = 4;
    pub const OVERRIDE_SLEEP_ANGULAR: u32 = 8;
    pub const NO_COLLISION_FILTER: u32 = u32::MAX;
    pub const SOLVER_DELTA_WORDS: u32 = 8;
    pub const SOLVER_ROUNDS_WORDS: u32 = 1;
    pub const REACTION_WORDS: u32 = 6;
    pub const SHAPE_NONE: u32 = 0;
    pub const SHAPE_SPHERE: u32 = 1;
    pub const SHAPE_CUBOID: u32 = 2;
    pub const SHAPE_CAPSULE: u32 = 3;
    pub const SHAPE_CYLINDER: u32 = 4;
    pub const SHAPE_HULL: u32 = 5;
    pub const SHAPE_MESH: u32 = 6;
    pub const SHAPE_HEIGHTFIELD: u32 = 7;
    pub const SHAPE_TRIANGLE: u32 = 8;
    pub const SHAPE_PLANE: u32 = 9;
    pub const COLLIDER_SENSOR: u32 = 1;
    pub const EVENT_MODE_BEGIN_END: u32 = 2;
    pub const EVENT_MODE_PERSIST: u32 = 4;
    pub const CONTACT_ANNOUNCED: u32 = 0x8000_0000;
    pub const ISLAND_WAKE: u32 = 1;
    pub const ELEMENT_ROLE_BITS: u32 = 2;
    pub const ELEMENT_ROLE_MASK: u32 = (1 << ELEMENT_ROLE_BITS) - 1;
    pub const ELEMENT_DISTANCE: u32 = 0;
    pub const ELEMENT_AREA: u32 = 1;
    pub const ELEMENT_BEND: u32 = 2;
    pub const ELEMENT_VOLUME: u32 = 3;
    pub const ELEMENT_KIND_MASK: u32 = 3;
    pub const ELEMENT_BROKEN: u32 = 1 << 31;
    pub const SOFT_EDIT_INVERSE_MASS: u32 = 1;
    pub const SOFT_EDIT_RADIUS: u32 = 2;
    pub const SOFT_EDIT_FRICTION: u32 = 4;
    pub const SOFT_EDIT_POSITION: u32 = 8;
    pub const SOFT_EDIT_VELOCITY: u32 = 16;
    pub const SOFT_BODY_EDIT_ACCELERATION: u32 = 1;
    pub const SOFT_BODY_EDIT_WAKE: u32 = 2;
    pub const FEATURE_POINT: u32 = 0;
    pub const FEATURE_VERTEX: u32 = 1 << 28;
    pub const FEATURE_EDGE: u32 = 2 << 28;
    pub const FEATURE_FACE: u32 = 3 << 28;
    pub const FEATURE_TRIANGLE: u32 = 4 << 28;
    pub const FEATURE_KIND_MASK: u32 = 0xF000_0000;
    pub const FEATURE_FIELD_BITS: u32 = 14;
    pub const FEATURE_FIELD_MASK: u32 = (1 << FEATURE_FIELD_BITS) - 1;
    pub const FEATURE_INDEX_LIMIT: u32 = 1 << 12;
    pub const FEATURE_CLIP: u32 = 1 << 12;
    pub const FEATURE_FACE_BIT: u32 = 1 << 13;
    pub const FEATURE_TRIANGLE_SIDE: u32 = 1 << 27;
    pub const FEATURE_TRIANGLE_MASK: u32 = FEATURE_TRIANGLE_SIDE - 1;
    pub const MAX_CELLS_PER_AXIS: u32 = 2;
    pub const MAX_CELLS_PER_COLLIDER: u32 = 8;
    pub const ENTRY_INDEX_BITS: u32 = 24;
    pub const ENTRY_INDEX_MASK: u32 = (1 << ENTRY_INDEX_BITS) - 1;
    pub const ENTRY_CELL_SHIFT: u32 = ENTRY_INDEX_BITS;
    pub const ENTRY_CELL_MASK: u32 = 7 << ENTRY_CELL_SHIFT;
    pub const ENTRY_KIND_SHIFT: u32 = ENTRY_CELL_SHIFT + 3;
    pub const ENTRY_KIND_MASK: u32 = 3 << ENTRY_KIND_SHIFT;
    pub const ENTRY_MOBILE: u32 = 1 << 29;
    pub const ENTRY_AWAKE: u32 = 1 << 30;
    pub const ENTRY_PRIMARY: u32 = 1 << 31;
    pub const ENTRY_KIND_COLLIDER: u32 = 0;
    pub const ENTRY_KIND_PARTICLE: u32 = 1;
    pub const QUERY_TARGET_COLLIDERS: u32 = 1 << ENTRY_KIND_COLLIDER;
    pub const QUERY_TARGET_PARTICLES: u32 = 1 << ENTRY_KIND_PARTICLE;
    pub const LEVEL_KEY_SHIFT: u32 = 27;
    pub const CELL_HASH_MASK: u32 = 0x07FF_FFFF;
    pub const CONSTRAINT_BALL: u32 = 0;
    pub const CONSTRAINT_DISTANCE: u32 = 1;
    pub const CONSTRAINT_REVOLUTE: u32 = 2;
    pub const CONSTRAINT_PRISMATIC: u32 = 3;
    pub const CONSTRAINT_FIXED: u32 = 4;
    pub const CONSTRAINT_GEAR: u32 = 5;
    pub const CONSTRAINT_PULLEY: u32 = 6;
    pub const CONSTRAINT_CONE: u32 = 7;
    pub const CONSTRAINT_SIXDOF: u32 = 8;
    pub const CONSTRAINT_DISABLE_COLLISIONS: u32 = 1;
    pub const CONSTRAINT_HAS_LIMIT: u32 = 2;
    pub const CONSTRAINT_HAS_MOTOR: u32 = 4;
    pub const CONSTRAINT_IS_SPRING: u32 = 8;
    pub const CONSTRAINT_HAS_SWING: u32 = 16;
    pub const CONSTRAINT_HAS_BREAK: u32 = 32;
    pub const CONSTRAINT_WARM_START: u32 = 64;
    pub const DOF_LOCKED: u32 = 1 << 8;
    pub const DOF_LIMITED: u32 = 1 << 14;
    pub const DOF_DRIVEN: u32 = 1 << 20;
    pub const DOF_LIMIT_ROW_BASE: u32 = 8;
    pub const FIELD_REGION_GLOBAL: u32 = 0;
    pub const FIELD_REGION_SPHERE: u32 = 1;
    pub const FIELD_REGION_CUBOID: u32 = 2;
    pub const QUERY_RAY: u32 = 0;
    pub const QUERY_SPHERE: u32 = 1;
    pub const QUERY_CUBOID: u32 = 2;
    pub const QUERY_SWEEP: u32 = 3;
    pub const QUERY_POINT: u32 = 4;
    pub const QUERY_CONVEX: u32 = 5;
    pub const FILTER_IGNORE_SENSORS: u32 = 1;
    pub const FILTER_IGNORE_SLEEPING: u32 = 2;
    pub const FILTER_IGNORE_STATIC: u32 = 4;
    pub const FILTER_IGNORE_KINEMATIC: u32 = 8;
    pub const EVENT_BEGIN: u32 = 0;
    pub const EVENT_END: u32 = 1;
    pub const EVENT_PERSIST: u32 = 2;
    pub const NO_BODY: u32 = 0xFFFF_FFFF;
    pub const NO_SLOT: u32 = 0xFFFF_FFFF;
    pub const NO_SURFACE: u32 = 0xFFFF_FFFF;
    pub const NO_TRIANGLE: u32 = 0xFFFF_FFFF;
}

const _: () = assert!(
    MAX_CELLS_PER_COLLIDER == MAX_CELLS_PER_AXIS * MAX_CELLS_PER_AXIS * MAX_CELLS_PER_AXIS,
    "a grid entry budget must be the cube of its per axis span"
);
const _: () = assert!(
    FEATURE_FIELD_BITS * 2 + 4 == 32,
    "a pair feature must pack two fields below its kind"
);
const _: () = assert!(
    FEATURE_INDEX_LIMIT <= FEATURE_CLIP && FEATURE_FACE_BIT <= FEATURE_FIELD_MASK,
    "a shape point id must not collide with the shape tags"
);
const _: () = assert!(
    FEATURE_KIND_MASK & FEATURE_FIELD_MASK == 0
        && FEATURE_TRIANGLE & FEATURE_KIND_MASK == FEATURE_TRIANGLE,
    "a feature kind must sit above its shape fields"
);
const _: () = assert!(
    CELL_HASH_MASK == (1 << LEVEL_KEY_SHIFT) - 1,
    "a cell hash must fill every bit below the level"
);
const _: () = assert!(
    1 << ELEMENT_ROLE_BITS == ELEMENT_PARTICLES,
    "an element role must be addressable by a fixed bit width"
);
const _: () = assert!(
    ELEMENT_DISTANCE < ELEMENT_AREA && ELEMENT_AREA < ELEMENT_BEND && ELEMENT_BEND < ELEMENT_VOLUME,
    "element kinds must fill their code space"
);
const _: () = assert!(
    ELEMENT_VOLUME & ELEMENT_KIND_MASK == ELEMENT_VOLUME && ELEMENT_KIND_MASK & ELEMENT_BROKEN == 0,
    "an element failure must sit above its kind code space"
);
const _: () = assert!(
    MAX_CELLS_PER_COLLIDER <= 1 << 3,
    "a grid entry must pack its cell offset into three bits"
);
const _: () = assert!(
    ENTRY_KIND_MASK != 0
        && ENTRY_KIND_MASK & ENTRY_MOBILE == 0
        && ENTRY_MOBILE & ENTRY_AWAKE == 0
        && ENTRY_AWAKE & ENTRY_PRIMARY == 0,
    "a grid entry kind, mobile bit, awake bit and primary bit must be disjoint"
);
const _: () = assert!(
    FIELD_REGION_GLOBAL < FIELD_REGION_SPHERE && FIELD_REGION_SPHERE < FIELD_REGION_CUBOID,
    "field region kinds must fill their code space"
);
const _: () = assert!(
    ENTRY_INDEX_MASK < 1 << ENTRY_CELL_SHIFT
        && ENTRY_CELL_MASK & ENTRY_KIND_MASK == 0
        && ENTRY_KIND_SHIFT + 2 <= 29,
    "a grid entry must pack its index and cell below the kind"
);

pub const DOF_COUNT: u32 = 6;

const _: () = assert!(
    CONSTRAINT_WARM_START < DOF_LOCKED
        && DOF_LOCKED < DOF_LIMITED
        && DOF_LIMITED < DOF_DRIVEN
        && DOF_DRIVEN << (DOF_COUNT - 1) < 1 << 31,
    "dof flags must occupy disjoint bits above the constraint flags"
);
const _: () = assert!(
    DOF_LIMIT_ROW_BASE + DOF_COUNT <= CONSTRAINT_ACCUMULATOR_SLOTS,
    "dof limit rows must fit the constraint accumulator budget"
);
const _: () = assert!(
    DOF_COUNT == JOINT_DOF,
    "the dof budget must be the joint state width"
);

fn dof_field(flags: u32, base: u32, index: u32) -> bool {
    assert!(index < DOF_COUNT, "dof index must be below {DOF_COUNT}");
    flags & (base << index) != 0
}

fn set_dof_field(flags: u32, base: u32, index: u32, on: bool) -> u32 {
    assert!(index < DOF_COUNT, "dof index must be below {DOF_COUNT}");
    let bit = base << index;
    if on { flags | bit } else { flags & !bit }
}

pub fn dof_locked(flags: u32, index: u32) -> bool {
    dof_field(flags, DOF_LOCKED, index)
}

pub fn dof_limited(flags: u32, index: u32) -> bool {
    dof_field(flags, DOF_LIMITED, index)
}

pub fn dof_driven(flags: u32, index: u32) -> bool {
    dof_field(flags, DOF_DRIVEN, index)
}

pub fn set_dof_locked(flags: u32, index: u32, on: bool) -> u32 {
    set_dof_field(flags, DOF_LOCKED, index, on)
}

pub fn set_dof_limited(flags: u32, index: u32, on: bool) -> u32 {
    set_dof_field(flags, DOF_LIMITED, index, on)
}

pub fn set_dof_driven(flags: u32, index: u32, on: bool) -> u32 {
    set_dof_field(flags, DOF_DRIVEN, index, on)
}

pub const NO_HIT: f32 = f32::MAX;

pub const SOLVER_VELOCITY_SCALE: f32 = 262144.0;

pub const SOLVER_POSITION_SCALE: f32 = 16777216.0;

pub fn solve_velocity(word: u32) -> f32 {
    word as i32 as f32 / SOLVER_VELOCITY_SCALE
}
