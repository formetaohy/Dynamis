use crate::QueryRecord;
use crate::constant::NO_BODY;
use crate::constant::{
    FILTER_IGNORE_KINEMATIC, FILTER_IGNORE_SENSORS, FILTER_IGNORE_SLEEPING, FILTER_IGNORE_STATIC,
    QUERY_CONVEX, QUERY_CUBOID, QUERY_POINT, QUERY_RAY, QUERY_SPHERE, QUERY_SWEEP,
    QUERY_TARGET_COLLIDERS, QUERY_TARGET_PARTICLES, SHAPE_CAPSULE, SHAPE_CUBOID, SHAPE_CYLINDER,
    SHAPE_HULL, SHAPE_SPHERE,
};
use bytemuck::Zeroable;
use dynamis_model::domain;
use dynamis_model::{QueryFilter, QueryTargets, Shape};

pub fn inert_sweep() -> QueryRecord {
    let mut sweep = QueryRecord::zeroed();
    sweep.kind = QUERY_SWEEP;
    sweep.shape_kind = SHAPE_SPHERE;
    sweep.filters.flags = FILTER_IGNORE_SENSORS;
    sweep.filters.targets = QUERY_TARGET_COLLIDERS;
    sweep.filters.mask = u32::MAX;
    sweep.max_hits = 1;
    sweep.filters.exclude_id = NO_BODY;
    sweep.direction = [0.0, 1.0, 0.0];
    sweep.orientation = [0.0, 0.0, 0.0, 1.0];
    sweep
}
fn target_mask(targets: QueryTargets) -> u32 {
    let mut mask = 0;
    if targets.holds(QueryTargets::COLLIDERS) {
        mask |= QUERY_TARGET_COLLIDERS;
    }
    if targets.holds(QueryTargets::PARTICLES) {
        mask |= QUERY_TARGET_PARTICLES;
    }
    mask
}

fn query_record(kind: u32, filter: &QueryFilter) -> QueryRecord {
    let exclude = filter
        .exclude
        .map_or((NO_BODY, 0), |handle| (handle.id, handle.generation));
    let include = filter
        .include
        .map_or((NO_BODY, 0), |handle| (handle.id, handle.generation));
    let exclude_soft = filter
        .exclude_soft
        .map_or((NO_BODY, 0), |handle| (handle.id, handle.generation));
    let include_soft = filter
        .include_soft
        .map_or((NO_BODY, 0), |handle| (handle.id, handle.generation));
    let mut record = QueryRecord::zeroed();
    record.kind = kind;
    record.filters.flags = filter_flags(filter);
    record.filters.targets = target_mask(filter.targets);
    record.filters.group = filter.group;
    record.filters.mask = filter.mask;
    record.max_hits = filter.max_hits;
    record.filters.exclude_id = exclude.0;
    record.filters.exclude_generation = exclude.1;
    record.filters.include_id = include.0;
    record.filters.include_generation = include.1;
    record.filters.exclude_soft_id = exclude_soft.0;
    record.filters.exclude_soft_generation = exclude_soft.1;
    record.filters.include_soft_id = include_soft.0;
    record.filters.include_soft_generation = include_soft.1;
    record
}

fn shape_fields(record: &mut QueryRecord, shape: &Shape) {
    shape.assert_valid();
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
    pub fn hit_bound(&self) -> u32 {
        self.max_hits.min(crate::QUERY_CANDIDATES)
    }
    pub fn ray(origin: [f32; 3], direction: [f32; 3], max_t: f32, filter: &QueryFilter) -> Self {
        domain::finite_vector(origin, "a query origin");
        domain::finite_vector(direction, "a query direction");
        assert!(direction != [0.0; 3], "a query direction must be non-zero");
        domain::positive(max_t, "a query distance");
        let mut record = query_record(QUERY_RAY, filter);
        record.origin = origin;
        record.direction = direction;
        record.extent = max_t;
        record
    }

    pub fn sphere(center: [f32; 3], radius: f32, filter: &QueryFilter) -> Self {
        domain::finite_vector(center, "a query center");
        domain::positive(radius, "a query radius");
        let mut record = query_record(QUERY_SPHERE, filter);
        record.shape_kind = SHAPE_SPHERE;
        record.origin = center;
        record.extent = radius;
        record.radius = radius;
        record
    }

    pub fn point(origin: [f32; 3], filter: &QueryFilter) -> Self {
        domain::finite_vector(origin, "a query point");
        let mut record = query_record(QUERY_POINT, filter);
        record.shape_kind = SHAPE_SPHERE;
        record.origin = origin;
        record
    }

    pub fn cuboid(center: [f32; 3], half_extents: [f32; 3], filter: &QueryFilter) -> Self {
        domain::finite_vector(center, "a query center");
        for extent in half_extents {
            domain::positive(extent, "a query half extent");
        }
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
        domain::finite_vector(position, "a query position");
        domain::unit_quaternion(orientation, "a query orientation");
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
        domain::finite_vector(start, "a sweep start");
        domain::finite_vector(direction, "a sweep direction");
        assert!(direction != [0.0; 3], "a sweep direction must be non-zero");
        domain::positive(length, "a sweep length");
        domain::unit_quaternion(orientation, "a sweep orientation");
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
