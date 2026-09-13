use dynamis_abi::{
    BodyDescriptorRecord, BodyEditRecord, BodyEditRunRecord, BodyStateRecord,
    BrokenConstraintRecord, BvhNodeRecord, COUNTER_DEVICE_COUNT, COUNTER_STRIDE, ColliderRecord,
    ConstraintDescriptorRecord, ConstraintRuntimeRecord, MAX_HITS_PER_QUERY, QueryHitRecord,
    QueryRecord, QueryResultHeaderRecord, RowMoveRecord, ShapeSourceRecord, StepParamsRecord,
    TriangleRecord,
};
use dynamis_gpu::Contents;
use dynamis_pass::EVENT_SLOTS;
use dynamis_pass::{UNIFORM, streams};
use std::mem::size_of;

pub const DOMAIN: u32 = 0;

pub const VERTEX_BYTES: u64 = size_of::<[f32; 4]>() as u64;
pub const TRIANGLE_BYTES: u64 = size_of::<TriangleRecord>() as u64;

const QUERY_BYTES: u64 = size_of::<QueryRecord>() as u64;
pub const QUERY_RESULT_BYTES: u64 = size_of::<QueryResultHeaderRecord>() as u64
    + MAX_HITS_PER_QUERY as u64 * size_of::<QueryHitRecord>() as u64;

streams! {
    StateStreams, StateStream, StateDemand, DOMAIN, demand,
    demand {
        bodies: u32,
        body_ids: u32,
        colliders: u32,
        constraints: u32,
        body_commands: u32,
        constraint_commands: u32,
        queries: u32,
        shapes: crate::ShapeCapacity,
    }
    streams {
        params, Params: "step params", size_of::<StepParamsRecord>() as u64, Contents::Reset, 1, UNIFORM;
        body_states, BodyStates: "body states", size_of::<BodyStateRecord>() as u64, Contents::Preserve, demand.bodies;
        body_row_of_id, BodyRowOfId: "body row of id", 4, Contents::Preserve, demand.body_ids;
        body_descriptors, BodyDescriptors: "body descriptors", size_of::<BodyDescriptorRecord>() as u64, Contents::Preserve, demand.bodies;
        colliders, Colliders: "colliders", size_of::<ColliderRecord>() as u64, Contents::Preserve, demand.colliders;
        collider_owners, ColliderOwners: "collider owners", 4, Contents::Preserve, demand.colliders;
        body_edits, BodyEdits: "body edits", size_of::<BodyEditRecord>() as u64, Contents::Reset, demand.body_commands;
        body_edit_runs, BodyEditRuns: "body edit runs", size_of::<BodyEditRunRecord>() as u64, Contents::Reset, demand.body_commands;
        body_row_moves, BodyRowMoves: "body row moves", size_of::<RowMoveRecord>() as u64, Contents::Reset, demand.body_moves();
        body_fresh_rows, BodyFreshRows: "fresh body rows", size_of::<BodyStateRecord>() as u64, Contents::Reset, demand.body_commands;
        constraint_descriptors, ConstraintDescriptors: "constraint descriptors", size_of::<ConstraintDescriptorRecord>() as u64, Contents::Preserve, demand.constraints;
        constraint_runtime, ConstraintRuntime: "constraint runtime", size_of::<ConstraintRuntimeRecord>() as u64, Contents::Preserve, demand.constraints;
        wake_flags, WakeFlags: "body wake flags", 4, Contents::Reset, demand.bodies;
        constraint_row_moves, ConstraintRowMoves: "constraint row moves", size_of::<RowMoveRecord>() as u64, Contents::Reset, demand.constraint_moves();
        constraint_fresh_rows, ConstraintFreshRows: "fresh constraint rows", size_of::<ConstraintRuntimeRecord>() as u64, Contents::Reset, demand.constraint_commands;
        constraint_breaks, ConstraintBreaks: "constraint breaks", size_of::<BrokenConstraintRecord>() as u64, Contents::Reset, demand.constraints.saturating_mul(EVENT_SLOTS);
        shape_sources, ShapeSources: "shape sources", size_of::<ShapeSourceRecord>() as u64, Contents::Preserve, demand.shapes.sources;
        shape_vertices, ShapeVertices: "shape vertices", VERTEX_BYTES, Contents::Preserve, demand.shapes.vertices;
        shape_triangles, ShapeTriangles: "shape triangles", TRIANGLE_BYTES, Contents::Preserve, demand.shapes.triangles;
        shape_nodes, ShapeNodes: "shape bvh nodes", size_of::<BvhNodeRecord>() as u64, Contents::Preserve, demand.shapes.nodes;
        query_records, QueryRecords: "queries", QUERY_BYTES, Contents::Reset, demand.queries;
        query_results, QueryResults: "query results", QUERY_RESULT_BYTES, Contents::Reset, demand.queries;
        counters, Counters: "world counters", COUNTER_STRIDE, Contents::Preserve, COUNTER_DEVICE_COUNT as u32;
    }
}

impl StateDemand {
    pub fn body_moves(&self) -> u32 {
        self.body_commands
            .saturating_mul(crate::MOVE_ENTRIES_PER_COMMAND)
    }

    pub fn constraint_moves(&self) -> u32 {
        self.constraint_commands
            .saturating_mul(crate::MOVE_ENTRIES_PER_COMMAND)
    }
}
