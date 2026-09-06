use bytemuck::{Pod, Zeroable};
use dynamis_model::{BodyDesc, ConstraintDesc, PhysicsConfig};

const _: () = {
    use std::mem::size_of;
    assert!(size_of::<RigidBodyRecord>() == 160);
    assert!(size_of::<ColliderRecord>() == 64);
    assert!(size_of::<SimParamsRecord>() == 96);
    assert!(size_of::<AabbRecord>() == 32);
    assert!(size_of::<PairRecord>() == 8);
    assert!(size_of::<ManifoldPointRecord>() == 32);
    assert!(size_of::<ContactRecord>() == 160);
    assert!(size_of::<ConstraintRecord>() == 128);
    assert!(size_of::<DispatchArgs>() == 12);
    assert!(size_of::<BodyCommandRecord>() == 240);
    assert!(size_of::<ConstraintCommandRecord>() == 144);
    assert!(size_of::<QueryRecord>() == 48);
    assert!(size_of::<QueryResultRecord>() == 48);
};

pub const COMMAND_ADD: u32 = 0;
pub const COMMAND_REMOVE: u32 = 1;
pub const COMMAND_PATCH: u32 = 2;
pub const COMMAND_FORCE: u32 = 3;
pub const COMMAND_TORQUE: u32 = 4;
pub const COMMAND_IMPULSE: u32 = 5;
pub const COMMAND_CONSTRAINT_ADD: u32 = 6;
pub const COMMAND_CONSTRAINT_REMOVE: u32 = 7;

pub const IMPULSE_AT_POINT: u32 = 1;

pub const PATCH_POSITION: u32 = 1;
pub const PATCH_VELOCITY: u32 = 2;
pub const PATCH_INVERSE_MASS: u32 = 4;
pub const PATCH_COLLIDER: u32 = 8;
pub const PATCH_RESTITUTION: u32 = 16;
pub const PATCH_ORIENTATION: u32 = 32;
pub const PATCH_ANGULAR_VELOCITY: u32 = 64;
pub const PATCH_FRICTION: u32 = 128;
pub const PATCH_GROUP: u32 = 256;
pub const PATCH_MASK: u32 = 512;
pub const PATCH_KINEMATIC: u32 = 1024;

pub const SHAPE_SPHERE: u32 = 0;
pub const SHAPE_BOX: u32 = 1;
pub const SHAPE_CAPSULE: u32 = 2;

pub const BODY_KINEMATIC: u32 = 1;

pub const CONTACT_MAX_POINTS: u32 = 4;

pub const CONSTRAINT_BALL: u32 = 0;
pub const CONSTRAINT_DISTANCE: u32 = 1;
pub const CONSTRAINT_REVOLUTE: u32 = 2;
pub const CONSTRAINT_PRISMATIC: u32 = 3;
pub const CONSTRAINT_FIXED: u32 = 4;
pub const CONSTRAINT_INVALID: u32 = 0xFFFF_FFFF;

pub const QUERY_RAY: u32 = 0;
pub const QUERY_SPHERE: u32 = 1;

pub const NO_BODY: u32 = 0xFFFF_FFFF;
pub const NO_HIT: f32 = f32::MAX;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RigidBodyRecord {
    pub position: [f32; 3],
    _pad0: f32,
    pub orientation: [f32; 4],
    pub velocity: [f32; 3],
    _pad1: f32,
    pub angular_velocity: [f32; 3],
    _pad2: f32,
    pub inverse_mass: f32,
    pub restitution: f32,
    pub friction: f32,
    pub body_id: u32,
    pub generation: u32,
    _pad_collider: u32,
    pub flags: u32,
    pub collision_group: u32,
    pub collision_mask: u32,
    _pad6: u32,
    _pad7: u32,
    _pad8: u32,
    pub inverse_inertia_body: [f32; 3],
    _pad3: f32,
    pub force: [f32; 3],
    _pad4: f32,
    pub torque: [f32; 3],
    _pad5: f32,
}


impl RigidBodyRecord {
    pub fn build(desc: &BodyDesc, body_id: u32, generation: u32) -> Self {
        let inverse_mass = if desc.kinematic || desc.mass > 0.0 {
            if desc.kinematic {
                0.0
            } else {
                1.0 / desc.mass
            }
        } else {
            0.0
        };
        Self {
            position: desc.position,
            _pad0: 0.0,
            orientation: desc.orientation,
            velocity: desc.velocity,
            _pad1: 0.0,
            angular_velocity: desc.angular_velocity,
            _pad2: 0.0,
            inverse_mass,
            restitution: desc.restitution,
            friction: desc.friction,
            body_id,
            generation,
            _pad_collider: 0,
            flags: if desc.kinematic { BODY_KINEMATIC } else { 0 },
            collision_group: desc.collision_group,
            collision_mask: desc.collision_mask,
            _pad6: 0,
            _pad7: 0,
            _pad8: 0,
            inverse_inertia_body: desc.shape.inverse_inertia_diagonal(inverse_mass),
            _pad3: 0.0,
            force: [0.0; 3],
            _pad4: 0.0,
            torque: [0.0; 3],
            _pad5: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ColliderRecord {
    pub shape: u32,
    _pad0: u32,
    pub radius: f32,
    pub half_height: f32,
    pub half_extents: [f32; 3],
    _pad1: f32,
    pub local_offset: [f32; 3],
    _pad2: f32,
    pub local_rotation: [f32; 4],
}

impl ColliderRecord {
    pub fn build(desc: &BodyDesc) -> Self {
        let shape = match desc.shape.kind {
            dynamis_model::ShapeKind::Sphere => SHAPE_SPHERE,
            dynamis_model::ShapeKind::Box => SHAPE_BOX,
            dynamis_model::ShapeKind::Capsule => SHAPE_CAPSULE,
        };
        Self {
            shape,
            _pad0: 0,
            radius: desc.shape.radius,
            half_height: desc.shape.half_height,
            half_extents: desc.shape.half_extents,
            _pad1: 0.0,
            local_offset: [0.0; 3],
            _pad2: 0.0,
            local_rotation: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SimParamsRecord {
    pub gravity: [f32; 4],
    pub dt: f32,
    pub damping: f32,
    pub angular_damping: f32,
    pub body_count: u32,
    pub solve_iterations: u32,
    pub position_iterations: u32,
    pub constraint_count: u32,
    pub relaxation: f32,
    pub slop: f32,
    pub restitution_threshold: f32,
    pub max_velocity: f32,
    pub max_angular_velocity: f32,
    pub grid_cell_size: f32,
    pub max_cells_per_body: u32,
    _pad0: u32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
}

impl SimParamsRecord {
    pub fn new(
        config: &PhysicsConfig,
        dt: f32,
        body_count: u32,
        constraint_count: u32,
    ) -> Self {
        Self {
            gravity: [config.gravity[0], config.gravity[1], config.gravity[2], 0.0],
            dt,
            damping: config.damping,
            angular_damping: config.angular_damping,
            body_count,
            solve_iterations: config.solve_iterations,
            position_iterations: config.position_iterations,
            constraint_count,
            relaxation: config.relaxation,
            slop: config.slop,
            restitution_threshold: config.restitution_threshold,
            max_velocity: config.max_velocity,
            max_angular_velocity: config.max_angular_velocity,
            grid_cell_size: config.broadphase_cell_size,
            max_cells_per_body: 8,
            _pad0: 0,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
            _pad4: 0.0,
            _pad5: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AabbRecord {
    pub min: [f32; 3],
    _pad0: f32,
    pub max: [f32; 3],
    _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PairRecord {
    pub a: u32,
    pub b: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ManifoldPointRecord {
    pub position: [f32; 3],
    pub depth: f32,
    pub accumulated_normal: f32,
    pub accumulated_tangent_1: f32,
    pub accumulated_tangent_2: f32,
    _pad0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ContactRecord {
    pub a: u32,
    pub b: u32,
    pub point_count: u32,
    _pad0: u32,
    pub normal: [f32; 3],
    _pad1: f32,
    pub points: [ManifoldPointRecord; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ConstraintRecord {
    pub kind: u32,
    pub a: u32,
    pub b: u32,
    _pad0: u32,
    pub anchor_a: [f32; 3],
    _pad1: f32,
    pub anchor_b: [f32; 3],
    _pad2: f32,
    pub axis_a: [f32; 3],
    _pad3: f32,
    pub axis_b: [f32; 3],
    _pad4: f32,
    pub distance: f32,
    _pad5: f32,
    _pad6: f32,
    _pad7: f32,
    pub accumulated: [f32; 8],
}

impl ConstraintRecord {
    pub fn build(desc: &ConstraintDesc, a: u32, b: u32) -> Self {
        let kind = match desc.kind {
            dynamis_model::ConstraintKind::Ball => CONSTRAINT_BALL,
            dynamis_model::ConstraintKind::Distance => CONSTRAINT_DISTANCE,
            dynamis_model::ConstraintKind::Revolute => CONSTRAINT_REVOLUTE,
            dynamis_model::ConstraintKind::Prismatic => CONSTRAINT_PRISMATIC,
            dynamis_model::ConstraintKind::Fixed => CONSTRAINT_FIXED,
        };
        Self {
            kind,
            a,
            b,
            _pad0: 0,
            anchor_a: desc.anchor_a,
            _pad1: 0.0,
            anchor_b: desc.anchor_b,
            _pad2: 0.0,
            axis_a: desc.axis,
            _pad3: 0.0,
            axis_b: desc.axis,
            _pad4: 0.0,
            distance: desc.distance,
            _pad5: 0.0,
            _pad6: 0.0,
            _pad7: 0.0,
            accumulated: [0.0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DispatchArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl DispatchArgs {
    pub const fn none() -> Self {
        Self { x: 0, y: 1, z: 1 }
    }

    pub const fn sized(elements: u32) -> Self {
        Self {
            x: elements,
            y: 1,
            z: 1,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BodyCommandRecord {
    pub kind: u32,
    pub slot: u32,
    pub extra: u32,
    _pad: u32,
    pub body: RigidBodyRecord,
    pub collider: ColliderRecord,
}

impl BodyCommandRecord {
    pub fn add(slot: u32, body: RigidBodyRecord, collider: ColliderRecord) -> Self {
        Self {
            kind: COMMAND_ADD,
            slot,
            extra: 0,
            _pad: 0,
            body,
            collider,
        }
    }

    pub fn remove(hole: u32, tail: u32) -> Self {
        Self {
            kind: COMMAND_REMOVE,
            slot: hole,
            extra: tail,
            _pad: 0,
            body: RigidBodyRecord::zeroed(),
            collider: ColliderRecord::zeroed(),
        }
    }

    pub fn patch(slot: u32, mask: u32, body: RigidBodyRecord, collider: ColliderRecord) -> Self {
        Self {
            kind: COMMAND_PATCH,
            slot,
            extra: mask,
            _pad: 0,
            body,
            collider,
        }
    }

    pub fn force(slot: u32, force: [f32; 3]) -> Self {
        let mut body = RigidBodyRecord::zeroed();
        body.force = force;
        Self {
            kind: COMMAND_FORCE,
            slot,
            extra: 0,
            _pad: 0,
            body,
            collider: ColliderRecord::zeroed(),
        }
    }

    pub fn torque(slot: u32, torque: [f32; 3]) -> Self {
        let mut body = RigidBodyRecord::zeroed();
        body.torque = torque;
        Self {
            kind: COMMAND_TORQUE,
            slot,
            extra: 0,
            _pad: 0,
            body,
            collider: ColliderRecord::zeroed(),
        }
    }

    pub fn impulse(slot: u32, impulse: [f32; 3]) -> Self {
        let mut body = RigidBodyRecord::zeroed();
        body.velocity = impulse;
        Self {
            kind: COMMAND_IMPULSE,
            slot,
            extra: 0,
            _pad: 0,
            body,
            collider: ColliderRecord::zeroed(),
        }
    }

    pub fn impulse_at_point(slot: u32, impulse: [f32; 3], point: [f32; 3]) -> Self {
        let mut body = RigidBodyRecord::zeroed();
        body.velocity = impulse;
        body.position = point;
        Self {
            kind: COMMAND_IMPULSE,
            slot,
            extra: IMPULSE_AT_POINT,
            _pad: 0,
            body,
            collider: ColliderRecord::zeroed(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ConstraintCommandRecord {
    pub kind: u32,
    pub slot: u32,
    _pad0: u32,
    _pad1: u32,
    pub constraint: ConstraintRecord,
}

impl ConstraintCommandRecord {
    pub fn add(slot: u32, constraint: ConstraintRecord) -> Self {
        Self {
            kind: COMMAND_CONSTRAINT_ADD,
            slot,
            _pad0: 0,
            _pad1: 0,
            constraint,
        }
    }

    pub fn remove(slot: u32) -> Self {
        Self {
            kind: COMMAND_CONSTRAINT_REMOVE,
            slot,
            _pad0: 0,
            _pad1: 0,
            constraint: ConstraintRecord::zeroed(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct QueryRecord {
    pub origin: [f32; 3],
    pub kind: u32,
    pub direction: [f32; 3],
    pub extent: f32,
    pub slot: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

impl QueryRecord {
    pub fn ray(origin: [f32; 3], direction: [f32; 3], max_t: f32, slot: u32) -> Self {
        Self {
            origin,
            kind: QUERY_RAY,
            direction,
            extent: max_t,
            slot,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }

    pub fn sphere(center: [f32; 3], radius: f32, slot: u32) -> Self {
        Self {
            origin: center,
            kind: QUERY_SPHERE,
            direction: [0.0; 3],
            extent: radius,
            slot,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct QueryResultRecord {
    pub body_id: u32,
    pub body_generation: u32,
    pub distance: f32,
    pub hit: u32,
    pub point: [f32; 3],
    _pad0: f32,
    pub normal: [f32; 3],
    _pad1: f32,
}
