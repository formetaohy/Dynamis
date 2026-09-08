pub const COMMAND_ADD: u32 = 0;
pub const COMMAND_REMOVE: u32 = 1;
pub const COMMAND_PATCH: u32 = 2;
pub const COMMAND_FORCE: u32 = 3;
pub const COMMAND_TORQUE: u32 = 4;
pub const COMMAND_IMPULSE: u32 = 5;
pub const COMMAND_CONSTRAINT_ADD: u32 = 6;
pub const COMMAND_CONSTRAINT_REMOVE: u32 = 7;
pub const COMMAND_SLEEP: u32 = 8;
pub const COMMAND_WAKE: u32 = 9;
pub const COMMAND_FORCE_AT_POINT: u32 = 10;
pub const COMMAND_ANGULAR_IMPULSE: u32 = 11;
pub const COMMAND_CONSTRAINT_PATCH: u32 = 12;
pub const COMMAND_SWAP: u32 = 13;

pub const IMPULSE_AT_POINT: u32 = 1;

pub const PATCH_POSITION: u32 = 1;
pub const PATCH_VELOCITY: u32 = 2;
pub const PATCH_MASS: u32 = 4;
pub const PATCH_COLLIDER: u32 = 8;
pub const PATCH_RESTITUTION: u32 = 16;
pub const PATCH_ORIENTATION: u32 = 32;
pub const PATCH_ANGULAR_VELOCITY: u32 = 64;
pub const PATCH_FRICTION: u32 = 128;
pub const PATCH_GROUP: u32 = 256;
pub const PATCH_MASK: u32 = 512;
pub const PATCH_KINEMATIC: u32 = 1024;
pub const PATCH_CCD: u32 = 2048;
pub const PATCH_DYNAMICS: u32 = 4096;

pub const NO_COLLISION_FILTER: u32 = u32::MAX;

pub const OVERFLOW_PAIRS: u32 = 0;
pub const OVERFLOW_EVENTS: u32 = 1;

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

pub const SHAPE_SOURCE_HULL: u32 = 1;
pub const SHAPE_SOURCE_MESH: u32 = 2;
pub const SHAPE_SOURCE_HEIGHTFIELD: u32 = 3;

pub const BODY_KINEMATIC: u32 = 1;
pub const BODY_SLEEPING: u32 = 2;
pub const BODY_CCD: u32 = 4;

pub const COLLIDER_SENSOR: u32 = 1;
pub const COLLIDER_EVENT_BEGIN_END: u32 = 2;
pub const COLLIDER_EVENT_PERSIST: u32 = 4;

pub const ISLAND_WAKE: u32 = 1;
pub const ISLAND_ACTIVE: u32 = 2;

pub const CONTACT_MAX_POINTS: u32 = 4;
pub const MAX_CELLS_PER_COLLIDER: u32 = 8;

pub const CONSTRAINT_BALL: u32 = 0;
pub const CONSTRAINT_DISTANCE: u32 = 1;
pub const CONSTRAINT_REVOLUTE: u32 = 2;
pub const CONSTRAINT_PRISMATIC: u32 = 3;
pub const CONSTRAINT_FIXED: u32 = 4;
pub const CONSTRAINT_GEAR: u32 = 5;
pub const CONSTRAINT_PULLEY: u32 = 6;
pub const CONSTRAINT_CONE: u32 = 7;
pub const CONSTRAINT_SIXDOF: u32 = 8;
pub const CONSTRAINT_INVALID: u32 = 0xFFFF_FFFF;

pub const CONSTRAINT_DISABLE_COLLISIONS: u32 = 1;
pub const CONSTRAINT_HAS_LIMIT: u32 = 2;
pub const CONSTRAINT_HAS_MOTOR: u32 = 4;
pub const CONSTRAINT_IS_SPRING: u32 = 8;
pub const CONSTRAINT_HAS_SWING: u32 = 16;
pub const CONSTRAINT_HAS_BREAK: u32 = 32;
pub const CONSTRAINT_BROKEN: u32 = 64;
pub const CONSTRAINT_WARM_START: u32 = 128;

pub const DOF_FREE: u32 = 0;
pub const DOF_LOCKED: u32 = 1;
pub const DOF_LIMITED: u32 = 2;
pub const DOF_DRIVEN: u32 = 3;

pub fn dof_mode(flags: u32, index: u32) -> u32 {
    (flags >> (8 + index * 2)) & 3
}

pub fn set_dof_mode(flags: u32, index: u32, mode: u32) -> u32 {
    (flags & !(3 << (8 + index * 2))) | (mode << (8 + index * 2))
}

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
pub const NO_HIT: f32 = f32::MAX;

pub const MAX_HITS_PER_QUERY: u32 = 16;
