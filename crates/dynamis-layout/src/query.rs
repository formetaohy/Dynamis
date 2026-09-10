use crate::constant::NO_BODY;
use crate::constant::{
    FILTER_IGNORE_KINEMATIC, FILTER_IGNORE_SENSORS, FILTER_IGNORE_SLEEPING, FILTER_IGNORE_STATIC,
    QUERY_CONVEX, QUERY_CUBOID, QUERY_POINT, QUERY_RAY, QUERY_SPHERE, QUERY_SWEEP, SHAPE_CAPSULE,
    SHAPE_CUBOID, SHAPE_CYLINDER, SHAPE_HULL, SHAPE_SPHERE,
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
    pub exclude_id: u32,
    pub exclude_generation: u32,
    pub include_id: u32,
    pub include_generation: u32,
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
}

struct FilterBasis {
    flags: u32,
    group: u32,
    mask: u32,
    exclude_id: u32,
    exclude_generation: u32,
    include_id: u32,
    include_generation: u32,
}

impl FilterBasis {
    fn of(filter: &QueryFilter) -> Self {
        let exclude = filter
            .exclude
            .map_or((NO_BODY, 0), |handle| (handle.id, handle.generation));
        let include = filter
            .include
            .map_or((NO_BODY, 0), |handle| (handle.id, handle.generation));
        Self {
            flags: filter_flags(filter),
            group: filter.group,
            mask: filter.mask,
            exclude_id: exclude.0,
            exclude_generation: exclude.1,
            include_id: include.0,
            include_generation: include.1,
        }
    }
}

fn query_record(kind: u32, filter: &QueryFilter) -> QueryRecord {
    let basis = FilterBasis::of(filter);
    let mut record = QueryRecord::zeroed();
    record.kind = kind;
    record.filter_flags = basis.flags;
    record.group = basis.group;
    record.mask = basis.mask;
    record.max_hits = filter.max_hits;
    record.exclude_id = basis.exclude_id;
    record.exclude_generation = basis.exclude_generation;
    record.include_id = basis.include_id;
    record.include_generation = basis.include_generation;
    record
}

fn shape_fields(record: &mut QueryRecord, shape: &Shape) {
    let (shape_kind, source) = match shape {
        Shape::Sphere { .. } => (SHAPE_SPHERE, 0),
        Shape::Cuboid { .. } => (SHAPE_CUBOID, 0),
        Shape::Capsule { .. } => (SHAPE_CAPSULE, 0),
        Shape::Cylinder { .. } => (SHAPE_CYLINDER, 0),
        Shape::Hull(handle) => (SHAPE_HULL, handle.id),
        _ => panic!("shape queries require a convex shape"),
    };
    record.shape_kind = shape_kind;
    record.source = source;
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
}

impl QueryRecord {
    pub fn ray(origin: [f32; 3], direction: [f32; 3], max_t: f32, filter: &QueryFilter) -> Self {
        let mut record = query_record(QUERY_RAY, filter);
        record.origin = origin;
        record.direction = direction;
        record.extent = max_t;
        record
    }

    pub fn sphere(center: [f32; 3], radius: f32, filter: &QueryFilter) -> Self {
        let mut record = query_record(QUERY_SPHERE, filter);
        record.shape_kind = SHAPE_SPHERE;
        record.origin = center;
        record.extent = radius;
        record.radius = radius;
        record
    }

    pub fn point(origin: [f32; 3], filter: &QueryFilter) -> Self {
        let mut record = query_record(QUERY_POINT, filter);
        record.shape_kind = SHAPE_SPHERE;
        record.origin = origin;
        record
    }

    pub fn cuboid(center: [f32; 3], half_extents: [f32; 3], filter: &QueryFilter) -> Self {
        let mut record = query_record(QUERY_CUBOID, filter);
        record.shape_kind = SHAPE_CUBOID;
        record.origin = center;
        record.half_extents = half_extents;
        record
    }

    pub fn convex(
        shape: &Shape,
        orientation: [f32; 4],
        position: [f32; 3],
        filter: &QueryFilter,
    ) -> Self {
        let mut record = query_record(QUERY_CONVEX, filter);
        record.origin = position;
        record.orientation = orientation;
        shape_fields(&mut record, shape);
        record
    }

    pub fn sweep(
        shape: &Shape,
        orientation: [f32; 4],
        start: [f32; 3],
        direction: [f32; 3],
        length: f32,
        filter: &QueryFilter,
    ) -> Self {
        let mut record = query_record(QUERY_SWEEP, filter);
        record.origin = start;
        record.direction = direction;
        record.extent = length;
        record.orientation = orientation;
        shape_fields(&mut record, shape);
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

/// One query's header followed by its hit lanes, so a batch of queries is one
/// contiguous, exactly-sized readback.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct QueryResultRecord {
    pub header: QueryResultHeader,
    pub hits: [QueryHitRecord; crate::constant::MAX_HITS_PER_QUERY as usize],
}

const _: () = {
    use std::mem::size_of;
    assert!(size_of::<QueryResultRecord>() == 784);
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct QueryHitRecord {
    pub body_id: u32,
    pub body_generation: u32,
    pub distance: f32,
    pub collider_index: u32,
    pub point: [f32; 3],
    pub _pad1: f32,
    pub normal: [f32; 3],
    pub _pad2: f32,
}
