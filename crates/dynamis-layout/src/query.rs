use crate::constant::{
    FILTER_IGNORE_KINEMATIC, FILTER_IGNORE_SENSORS, FILTER_IGNORE_SLEEPING, FILTER_IGNORE_STATIC,
    QUERY_CUBOID, QUERY_RAY, QUERY_SPHERE, QUERY_SWEEP, SHAPE_CAPSULE, SHAPE_CUBOID,
    SHAPE_CYLINDER, SHAPE_HULL, SHAPE_SPHERE,
};
use bytemuck::{Pod, Zeroable};
use dynamis_model::{QueryFilter, Shape};

const _: () = {
    use std::mem::size_of;
    assert!(size_of::<QueryRecord>() == 128);
    assert!(size_of::<QueryResultHeader>() == 16);
    assert!(size_of::<QueryHitRecord>() == 48);
};

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
    pub _pad2: f32,
    pub half_extents: [f32; 3],
    pub _pad3: f32,
    pub orientation: [f32; 4],
    pub _pad4: f32,
    pub _pad5: f32,
    pub _pad6: f32,
    pub _pad7: f32,
}

impl QueryRecord {
    pub fn ray(origin: [f32; 3], direction: [f32; 3], max_t: f32, filter: &QueryFilter) -> Self {
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

    pub fn sphere(center: [f32; 3], radius: f32, filter: &QueryFilter) -> Self {
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

    pub fn cuboid(center: [f32; 3], half_extents: [f32; 3], filter: &QueryFilter) -> Self {
        Self {
            kind: QUERY_CUBOID,
            shape_kind: SHAPE_CUBOID,
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
        shape: &Shape,
        orientation: [f32; 4],
        start: [f32; 3],
        direction: [f32; 3],
        length: f32,
        filter: &QueryFilter,
    ) -> Self {
        let (shape_kind, source) = match shape {
            Shape::Sphere { .. } => (SHAPE_SPHERE, 0),
            Shape::Cuboid { .. } => (SHAPE_CUBOID, 0),
            Shape::Capsule { .. } => (SHAPE_CAPSULE, 0),
            Shape::Cylinder { .. } => (SHAPE_CYLINDER, 0),
            Shape::Hull(handle) => (SHAPE_HULL, handle.id),
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
            Shape::Sphere { radius }
            | Shape::Capsule { radius, .. }
            | Shape::Cylinder { radius, .. } => {
                record.radius = *radius;
            }
            _ => {}
        }
        match shape {
            Shape::Capsule { half_height, .. } | Shape::Cylinder { half_height, .. } => {
                record.half_height = *half_height;
            }
            _ => {}
        }
        if let Shape::Cuboid { half_extents } = shape {
            record.half_extents = *half_extents;
        }
        record
    }
}

fn filter_flags(filter: &QueryFilter) -> u32 {
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
