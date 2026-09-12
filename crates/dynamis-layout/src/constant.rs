use crate::wgsl::declare_constants;

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
    pub const SOLVER_BLOCK_CONTACT: u32 = 0;
    pub const SOLVER_BLOCK_CONSTRAINT: u32 = 1;
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
    pub const COLLIDER_EVENT_BEGIN_END: u32 = 2;
    pub const COLLIDER_EVENT_PERSIST: u32 = 4;
    pub const CONTACT_ANNOUNCED: u32 = 0x8000_0000;
    pub const ISLAND_WAKE: u32 = 1;
    pub const ISLAND_ACTIVE: u32 = 2;
    pub const CONTACT_MAX_POINTS: u32 = 4;
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
    pub const CONSTRAINT_ACCUMULATOR_SLOTS: u32 = 16;
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
    pub const MAX_HITS_PER_QUERY: u32 = 16;
}

const _: () = assert!(
    MAX_CELLS_PER_COLLIDER == MAX_CELLS_PER_AXIS * MAX_CELLS_PER_AXIS * MAX_CELLS_PER_AXIS,
    "a collider budget must be the cube of its per axis span"
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
