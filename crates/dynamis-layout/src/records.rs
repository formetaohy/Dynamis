use bytemuck::{Pod, Zeroable};
use dynamis_model::{BodyDesc, ColliderDesc, ConstraintDesc, PhysicsConfig};

const _: () = {
    use std::mem::size_of;
    assert!(size_of::<RigidBodyRecord>() == 176);
    assert!(size_of::<ColliderRecord>() == 80);
    assert!(size_of::<ShapeSourceRecord>() == 32);
    assert!(size_of::<BvhNodeRecord>() == 48);
    assert!(size_of::<SimParamsRecord>() == 96);
    assert!(size_of::<AabbRecord>() == 32);
    assert!(size_of::<PairRecord>() == 8);
    assert!(size_of::<ManifoldPointRecord>() == 32);
    assert!(size_of::<ContactRecord>() == 176);
    assert!(size_of::<ContactEventRecord>() == 64);
    assert!(size_of::<ConstraintRecord>() == 144);
    assert!(size_of::<DispatchArgs>() == 12);
    assert!(size_of::<BodyCommandRecord>() == 512);
    assert!(size_of::<ConstraintCommandRecord>() == 160);
    assert!(size_of::<QueryRecord>() == 128);
    assert!(size_of::<QueryResultHeader>() == 16);
    assert!(size_of::<QueryHitRecord>() == 48);
};

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
pub const PATCH_CCD: u32 = 2048;

pub const SHAPE_NONE: u32 = 0;
pub const SHAPE_SPHERE: u32 = 1;
pub const SHAPE_BOX: u32 = 2;
pub const SHAPE_CAPSULE: u32 = 3;
pub const SHAPE_CYLINDER: u32 = 4;
pub const SHAPE_HULL: u32 = 5;
pub const SHAPE_MESH: u32 = 6;
pub const SHAPE_HEIGHTFIELD: u32 = 7;

pub const SHAPE_SOURCE_HULL: u32 = 1;
pub const SHAPE_SOURCE_MESH: u32 = 2;
pub const SHAPE_SOURCE_HEIGHTFIELD: u32 = 3;

pub const BODY_KINEMATIC: u32 = 1;
pub const BODY_SLEEPING: u32 = 2;
pub const BODY_CCD: u32 = 4;

pub const COLLIDER_SENSOR: u32 = 1;

pub const ISLAND_WAKE: u32 = 1;
pub const ISLAND_ACTIVE: u32 = 2;

pub const CONTACT_MAX_POINTS: u32 = 4;

pub const CONSTRAINT_BALL: u32 = 0;
pub const CONSTRAINT_DISTANCE: u32 = 1;
pub const CONSTRAINT_REVOLUTE: u32 = 2;
pub const CONSTRAINT_PRISMATIC: u32 = 3;
pub const CONSTRAINT_FIXED: u32 = 4;
pub const CONSTRAINT_INVALID: u32 = 0xFFFF_FFFF;

pub const CONSTRAINT_DISABLE_COLLISIONS: u32 = 1;
pub const CONSTRAINT_HAS_LIMIT: u32 = 2;
pub const CONSTRAINT_HAS_MOTOR: u32 = 4;
pub const CONSTRAINT_IS_SPRING: u32 = 8;

pub const QUERY_RAY: u32 = 0;
pub const QUERY_SPHERE: u32 = 1;
pub const QUERY_BOX: u32 = 2;
pub const QUERY_SWEEP: u32 = 3;

pub const FILTER_IGNORE_SENSORS: u32 = 1;
pub const FILTER_IGNORE_SLEEPING: u32 = 2;
pub const FILTER_IGNORE_STATIC: u32 = 4;
pub const FILTER_IGNORE_KINEMATIC: u32 = 8;

pub const EVENT_BEGIN: u32 = 0;
pub const EVENT_END: u32 = 1;

pub const NO_BODY: u32 = 0xFFFF_FFFF;
pub const NO_HIT: f32 = f32::MAX;

pub const MAX_COLLIDERS_PER_BODY: u32 = 4;
pub const MAX_HITS_PER_QUERY: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RigidBodyRecord {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub prev_position: [f32; 3],
    pub _prev_pad: f32,
    pub orientation: [f32; 4],
    pub velocity: [f32; 3],
    pub _pad1: f32,
    pub angular_velocity: [f32; 3],
    pub _pad2: f32,
    pub inverse_mass: f32,
    pub restitution: f32,
    pub friction: f32,
    pub body_id: u32,
    pub generation: u32,
    pub collider_count: u32,
    pub flags: u32,
    pub collision_group: u32,
    pub collision_mask: u32,
    pub sleep_timer: f32,
    pub _pad7: u32,
    pub _pad8: u32,
    pub inverse_inertia_body: [f32; 3],
    pub _pad3: f32,
    pub force: [f32; 3],
    pub _pad4: f32,
    pub torque: [f32; 3],
    pub _pad5: f32,
}

impl RigidBodyRecord {
    pub fn build(
        desc: &BodyDesc,
        body_id: u32,
        generation: u32,
        inverse_inertia_body: [f32; 3],
    ) -> Self {
        let inverse_mass = inverse_mass(desc);
        Self {
            position: desc.position,
            _pad0: 0.0,
            prev_position: desc.position,
            _prev_pad: 0.0,
            orientation: desc.orientation,
            velocity: desc.velocity,
            _pad1: 0.0,
            angular_velocity: desc.angular_velocity,
            _pad2: 0.0,
            inverse_mass,
            restitution: desc.colliders[0].restitution,
            friction: desc.colliders[0].friction,
            body_id,
            generation,
            collider_count: desc.colliders.len() as u32,
            flags: matching_flags(desc),
            collision_group: desc.collision_group,
            collision_mask: desc.collision_mask,
            sleep_timer: 0.0,
            _pad7: 0,
            _pad8: 0,
            inverse_inertia_body,
            _pad3: 0.0,
            force: [0.0; 3],
            _pad4: 0.0,
            torque: [0.0; 3],
            _pad5: 0.0,
        }
    }
}

pub fn inverse_mass(desc: &BodyDesc) -> f32 {
    if desc.kinematic || desc.mass > 0.0 {
        if desc.kinematic { 0.0 } else { 1.0 / desc.mass }
    } else {
        0.0
    }
}

pub fn matching_flags(desc: &BodyDesc) -> u32 {
    let mut flags = 0;
    if desc.kinematic {
        flags |= BODY_KINEMATIC;
    }
    if desc.ccd {
        flags |= BODY_CCD;
    }
    flags
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ColliderRecord {
    pub kind: u32,
    pub flags: u32,
    pub radius: f32,
    pub half_height: f32,
    pub half_extents: [f32; 3],
    pub _pad0: f32,
    pub local_offset: [f32; 3],
    pub _pad1: f32,
    pub local_rotation: [f32; 4],
    pub friction: f32,
    pub restitution: f32,
    pub source: u32,
    pub _pad2: u32,
}

impl ColliderRecord {
    pub fn build(collider: &ColliderDesc, source: u32) -> Self {
        let kind = shape_kind(&collider.shape);
        Self {
            kind,
            flags: if collider.sensor { COLLIDER_SENSOR } else { 0 },
            radius: match collider.shape {
                dynamis_model::Shape::Sphere { radius }
                | dynamis_model::Shape::Capsule { radius, .. }
                | dynamis_model::Shape::Cylinder { radius, .. } => radius,
                _ => 0.0,
            },
            half_height: match collider.shape {
                dynamis_model::Shape::Capsule { half_height, .. }
                | dynamis_model::Shape::Cylinder { half_height, .. } => half_height,
                _ => 0.0,
            },
            half_extents: match collider.shape {
                dynamis_model::Shape::Box { half_extents } => half_extents,
                _ => [0.0; 3],
            },
            _pad0: 0.0,
            local_offset: collider.offset,
            _pad1: 0.0,
            local_rotation: collider.rotation,
            friction: collider.friction,
            restitution: collider.restitution,
            source,
            _pad2: 0,
        }
    }
}

fn shape_kind(shape: &dynamis_model::Shape) -> u32 {
    match shape {
        dynamis_model::Shape::Sphere { .. } => SHAPE_SPHERE,
        dynamis_model::Shape::Box { .. } => SHAPE_BOX,
        dynamis_model::Shape::Capsule { .. } => SHAPE_CAPSULE,
        dynamis_model::Shape::Cylinder { .. } => SHAPE_CYLINDER,
        dynamis_model::Shape::Hull(_) => SHAPE_HULL,
        dynamis_model::Shape::Mesh(_) => SHAPE_MESH,
        dynamis_model::Shape::HeightField(_) => SHAPE_HEIGHTFIELD,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ShapeSourceRecord {
    pub kind: u32,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub triangle_offset: u32,
    pub triangle_count: u32,
    pub node_offset: u32,
    pub node_count: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BvhNodeRecord {
    pub min: [f32; 3],
    pub _pad0: f32,
    pub max: [f32; 3],
    pub _pad1: f32,
    pub left: u32,
    pub right: u32,
    pub leaf: u32,
    pub _pad2: u32,
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
    pub max_cells_per_collider: u32,
    pub _pad0: u32,
    pub sleep_velocity: f32,
    pub sleep_angular_velocity: f32,
    pub sleep_time: f32,
    pub wake_velocity: f32,
    pub _pad5: f32,
}

impl SimParamsRecord {
    pub fn new(config: &PhysicsConfig, dt: f32, body_count: u32, constraint_count: u32) -> Self {
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
            max_cells_per_collider: 8,
            _pad0: 0,
            sleep_velocity: config.sleep_velocity,
            sleep_angular_velocity: config.sleep_angular_velocity,
            sleep_time: config.sleep_time,
            wake_velocity: config.wake_velocity,
            _pad5: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AabbRecord {
    pub min: [f32; 3],
    pub _pad0: f32,
    pub max: [f32; 3],
    pub _pad1: f32,
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
    pub _pad0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ContactRecord {
    pub a: u32,
    pub b: u32,
    pub point_count: u32,
    pub sensor: u32,
    pub first_body_id: u32,
    pub second_body_id: u32,
    pub first_generation: u32,
    pub second_generation: u32,
    pub normal: [f32; 3],
    pub _pad2: f32,
    pub points: [ManifoldPointRecord; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ContactEventRecord {
    pub kind: u32,
    pub sensor: u32,
    pub first_id: u32,
    pub first_generation: u32,
    pub second_id: u32,
    pub second_generation: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub point: [f32; 3],
    pub _pad2: f32,
    pub normal: [f32; 3],
    pub _pad3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ConstraintRecord {
    pub kind: u32,
    pub a: u32,
    pub b: u32,
    pub flags: u32,
    pub anchor_a: [f32; 3],
    pub _pad1: f32,
    pub anchor_b: [f32; 3],
    pub _pad2: f32,
    pub axis_a: [f32; 3],
    pub _pad3: f32,
    pub axis_b: [f32; 3],
    pub _pad4: f32,
    pub distance: f32,
    pub limit_min: f32,
    pub limit_max: f32,
    pub motor_speed: f32,
    pub spring_frequency: f32,
    pub spring_damping_ratio: f32,
    pub _pad5: f32,
    pub _pad6: f32,
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
        let mut flags = 0;
        if desc.disable_collisions {
            flags |= CONSTRAINT_DISABLE_COLLISIONS;
        }
        if desc.limit.is_some() {
            flags |= CONSTRAINT_HAS_LIMIT;
        }
        if desc.motor.is_some() {
            flags |= CONSTRAINT_HAS_MOTOR;
        }
        if desc.spring.is_some() {
            flags |= CONSTRAINT_IS_SPRING;
        }
        Self {
            kind,
            a,
            b,
            flags,
            anchor_a: desc.anchor_a,
            _pad1: 0.0,
            anchor_b: desc.anchor_b,
            _pad2: 0.0,
            axis_a: desc.axis,
            _pad3: 0.0,
            axis_b: desc.axis,
            _pad4: 0.0,
            distance: desc.distance,
            limit_min: desc.limit.map_or(0.0, |limit| limit.min),
            limit_max: desc.limit.map_or(0.0, |limit| limit.max),
            motor_speed: desc.motor.map_or(0.0, |motor| motor.speed),
            spring_frequency: desc.spring.map_or(0.0, |spring| spring.frequency),
            spring_damping_ratio: desc.spring.map_or(0.0, |spring| spring.damping_ratio),
            _pad5: 0.0,
            _pad6: 0.0,
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
    pub aux: u32,
    pub body: RigidBodyRecord,
    pub colliders: [ColliderRecord; 4],
}

impl BodyCommandRecord {
    pub fn add(slot: u32, body: RigidBodyRecord, colliders: [ColliderRecord; 4]) -> Self {
        Self {
            kind: COMMAND_ADD,
            slot,
            extra: 0,
            aux: 0,
            body,
            colliders,
        }
    }

    pub fn remove(hole: u32, tail: u32) -> Self {
        Self {
            kind: COMMAND_REMOVE,
            slot: hole,
            extra: tail,
            aux: 0,
            body: RigidBodyRecord::zeroed(),
            colliders: [ColliderRecord::zeroed(); 4],
        }
    }

    pub fn patch(
        slot: u32,
        mask: u32,
        body: RigidBodyRecord,
        colliders: [ColliderRecord; 4],
        collider_index: u32,
    ) -> Self {
        Self {
            kind: COMMAND_PATCH,
            slot,
            extra: mask,
            aux: collider_index,
            body,
            colliders,
        }
    }

    pub fn force(slot: u32, force: [f32; 3]) -> Self {
        let mut body = RigidBodyRecord::zeroed();
        body.force = force;
        Self {
            kind: COMMAND_FORCE,
            slot,
            extra: 0,
            aux: 0,
            body,
            colliders: [ColliderRecord::zeroed(); 4],
        }
    }

    pub fn force_at_point(slot: u32, force: [f32; 3], point: [f32; 3]) -> Self {
        let mut body = RigidBodyRecord::zeroed();
        body.force = force;
        body.position = point;
        Self {
            kind: COMMAND_FORCE_AT_POINT,
            slot,
            extra: 0,
            aux: 0,
            body,
            colliders: [ColliderRecord::zeroed(); 4],
        }
    }

    pub fn torque(slot: u32, torque: [f32; 3]) -> Self {
        let mut body = RigidBodyRecord::zeroed();
        body.torque = torque;
        Self {
            kind: COMMAND_TORQUE,
            slot,
            extra: 0,
            aux: 0,
            body,
            colliders: [ColliderRecord::zeroed(); 4],
        }
    }

    pub fn impulse(slot: u32, impulse: [f32; 3]) -> Self {
        let mut body = RigidBodyRecord::zeroed();
        body.velocity = impulse;
        Self {
            kind: COMMAND_IMPULSE,
            slot,
            extra: 0,
            aux: 0,
            body,
            colliders: [ColliderRecord::zeroed(); 4],
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
            aux: 0,
            body,
            colliders: [ColliderRecord::zeroed(); 4],
        }
    }

    pub fn angular_impulse(slot: u32, impulse: [f32; 3]) -> Self {
        let mut body = RigidBodyRecord::zeroed();
        body.angular_velocity = impulse;
        Self {
            kind: COMMAND_ANGULAR_IMPULSE,
            slot,
            extra: 0,
            aux: 0,
            body,
            colliders: [ColliderRecord::zeroed(); 4],
        }
    }

    pub fn sleep(slot: u32) -> Self {
        Self {
            kind: COMMAND_SLEEP,
            slot,
            extra: 0,
            aux: 0,
            body: RigidBodyRecord::zeroed(),
            colliders: [ColliderRecord::zeroed(); 4],
        }
    }

    pub fn wake(slot: u32) -> Self {
        Self {
            kind: COMMAND_WAKE,
            slot,
            extra: 0,
            aux: 0,
            body: RigidBodyRecord::zeroed(),
            colliders: [ColliderRecord::zeroed(); 4],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ConstraintCommandRecord {
    pub kind: u32,
    pub slot: u32,
    pub _pad0: u32,
    pub _pad1: u32,
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
    pub kind: u32,
    pub shape_kind: u32,
    pub filter_flags: u32,
    pub slot: u32,
    pub group: u32,
    pub mask: u32,
    pub source: u32,
    pub max_hits: u32,
    pub origin: [f32; 3],
    pub _pad0: f32,
    pub direction: [f32; 3],
    pub extent: f32,
    pub radius: f32,
    pub half_height: f32,
    pub _pad1: f32,
    pub _pad1b: f32,
    pub half_extents: [f32; 3],
    pub _pad2: f32,
    pub orientation: [f32; 4],
    pub _pad3: f32,
    pub _pad4: f32,
    pub _pad5: f32,
    pub _pad6: f32,
}

impl QueryRecord {
    pub fn ray(
        origin: [f32; 3],
        direction: [f32; 3],
        max_t: f32,
        filter: &dynamis_model::QueryFilter,
    ) -> Self {
        Self {
            kind: QUERY_RAY,
            filter_flags: filter_flags(filter),
            group: filter.group,
            mask: filter.mask,
            max_hits: filter.max_hits,
            origin,
            direction,
            extent: max_t,
            ..Self::zeroed()
        }
    }

    pub fn sphere(center: [f32; 3], radius: f32, filter: &dynamis_model::QueryFilter) -> Self {
        Self {
            kind: QUERY_SPHERE,
            shape_kind: SHAPE_SPHERE,
            filter_flags: filter_flags(filter),
            group: filter.group,
            mask: filter.mask,
            max_hits: filter.max_hits,
            origin: center,
            extent: radius,
            radius,
            ..Self::zeroed()
        }
    }

    pub fn box_query(
        center: [f32; 3],
        half_extents: [f32; 3],
        filter: &dynamis_model::QueryFilter,
    ) -> Self {
        Self {
            kind: QUERY_BOX,
            shape_kind: SHAPE_BOX,
            filter_flags: filter_flags(filter),
            group: filter.group,
            mask: filter.mask,
            max_hits: filter.max_hits,
            origin: center,
            half_extents,
            ..Self::zeroed()
        }
    }

    pub fn sweep(
        shape: &dynamis_model::Shape,
        orientation: [f32; 4],
        start: [f32; 3],
        direction: [f32; 3],
        length: f32,
        filter: &dynamis_model::QueryFilter,
    ) -> Self {
        let (shape_kind, source) = match shape {
            dynamis_model::Shape::Sphere { .. } => (SHAPE_SPHERE, 0),
            dynamis_model::Shape::Box { .. } => (SHAPE_BOX, 0),
            dynamis_model::Shape::Capsule { .. } => (SHAPE_CAPSULE, 0),
            dynamis_model::Shape::Cylinder { .. } => (SHAPE_CYLINDER, 0),
            dynamis_model::Shape::Hull(handle) => (SHAPE_HULL, handle.id),
            _ => panic!("sweep queries require a convex shape"),
        };
        let mut record = Self::zeroed();
        record.kind = QUERY_SWEEP;
        record.shape_kind = shape_kind;
        record.filter_flags = filter_flags(filter);
        record.group = filter.group;
        record.mask = filter.mask;
        record.source = source;
        record.max_hits = filter.max_hits;
        record.origin = start;
        record.direction = direction;
        record.extent = length;
        record.orientation = orientation;
        match shape {
            dynamis_model::Shape::Sphere { radius }
            | dynamis_model::Shape::Capsule { radius, .. }
            | dynamis_model::Shape::Cylinder { radius, .. } => {
                record.radius = *radius;
            }
            _ => {}
        }
        match shape {
            dynamis_model::Shape::Capsule { half_height, .. }
            | dynamis_model::Shape::Cylinder { half_height, .. } => {
                record.half_height = *half_height;
            }
            _ => {}
        }
        if let dynamis_model::Shape::Box { half_extents } = shape {
            record.half_extents = *half_extents;
        }
        record
    }
}

fn filter_flags(filter: &dynamis_model::QueryFilter) -> u32 {
    let mut flags = 0;
    if filter.ignore_sensors {
        flags |= FILTER_IGNORE_SENSORS;
    }
    if filter.ignore_sleeping {
        flags |= FILTER_IGNORE_SLEEPING;
    }
    if filter.ignore_static {
        flags |= FILTER_IGNORE_STATIC;
    }
    if filter.ignore_kinematic {
        flags |= FILTER_IGNORE_KINEMATIC;
    }
    flags
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct QueryResultHeader {
    pub count: u32,
    pub overflow: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct QueryHitRecord {
    pub body_id: u32,
    pub body_generation: u32,
    pub distance: f32,
    pub _pad0: u32,
    pub point: [f32; 3],
    pub _pad1: f32,
    pub normal: [f32; 3],
    pub _pad2: f32,
}
