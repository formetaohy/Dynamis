use crate::dynamics::EVENT_SLOTS;
use crate::dynamics::engine::{UNIFORM, streams};
use dynamis_gpu::Contents;
use dynamis_layout::{
    BodyDescriptorRecord, BodyEditRecord, BodyEditRunRecord, BodyStateRecord,
    BrokenConstraintRecord, BvhNodeRecord, COUNTER_COUNT, COUNTER_STRIDE, ColliderRecord,
    ConstraintDescriptorRecord, ConstraintRuntimeRecord, MAX_HITS_PER_QUERY, QueryHitRecord,
    QueryRecord, QueryResultHeaderRecord, RowMoveRecord, ShapeSourceRecord, StepParamsRecord,
    TriangleRecord,
};
use std::mem::size_of;

pub(crate) const DOMAIN: u32 = 0;

pub(crate) const VERTEX_BYTES: u64 = size_of::<[f32; 4]>() as u64;
pub(crate) const TRIANGLE_BYTES: u64 = size_of::<TriangleRecord>() as u64;

const QUERY_BYTES: u64 = size_of::<QueryRecord>() as u64;
pub(crate) const QUERY_RESULT_BYTES: u64 = size_of::<QueryResultHeaderRecord>() as u64
    + MAX_HITS_PER_QUERY as u64 * size_of::<QueryHitRecord>() as u64;

streams! {
    SceneStreams, SceneStream, SceneDemand, DOMAIN, demand,
    demand {
        bodies: u32,
        body_ids: u32,
        colliders: u32,
        constraints: u32,
        body_commands: u32,
        constraint_commands: u32,
        queries: u32,
        shapes: crate::dynamics::capacity::ShapeCapacity,
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
        constraint_row_moves, ConstraintRowMoves: "constraint row moves", size_of::<RowMoveRecord>() as u64, Contents::Reset, demand.constraint_moves();
        constraint_fresh_rows, ConstraintFreshRows: "fresh constraint rows", size_of::<ConstraintRuntimeRecord>() as u64, Contents::Reset, demand.constraint_commands;
        constraint_breaks, ConstraintBreaks: "constraint breaks", size_of::<BrokenConstraintRecord>() as u64, Contents::Reset, demand.constraints.saturating_mul(EVENT_SLOTS);
        shape_sources, ShapeSources: "shape sources", size_of::<ShapeSourceRecord>() as u64, Contents::Preserve, demand.shapes.sources;
        shape_vertices, ShapeVertices: "shape vertices", VERTEX_BYTES, Contents::Preserve, demand.shapes.vertices;
        shape_triangles, ShapeTriangles: "shape triangles", TRIANGLE_BYTES, Contents::Preserve, demand.shapes.triangles;
        shape_nodes, ShapeNodes: "shape bvh nodes", size_of::<BvhNodeRecord>() as u64, Contents::Preserve, demand.shapes.nodes;
        query_records, QueryRecords: "queries", QUERY_BYTES, Contents::Reset, demand.queries;
        query_results, QueryResults: "query results", QUERY_RESULT_BYTES, Contents::Reset, demand.queries;
        counters, Counters: "world counters", COUNTER_STRIDE, Contents::Preserve, COUNTER_COUNT as u32;
    }
}

impl SceneDemand {
    pub(crate) fn body_moves(&self) -> u32 {
        self.body_commands
            .saturating_mul(crate::dynamics::capacity::MOVE_ENTRIES_PER_COMMAND)
    }

    pub(crate) fn constraint_moves(&self) -> u32 {
        self.constraint_commands
            .saturating_mul(crate::dynamics::capacity::MOVE_ENTRIES_PER_COMMAND)
    }
}

impl SceneStreams {
    pub(crate) fn shape_resources(&self) -> [(&'static str, crate::dynamics::engine::SlotRef); 4] {
        [
            ("shape_sources", SceneStream::ShapeSources.whole()),
            ("shape_vertices", SceneStream::ShapeVertices.whole()),
            ("shape_triangles", SceneStream::ShapeTriangles.whole()),
            ("shape_nodes", SceneStream::ShapeNodes.whole()),
        ]
    }

    pub(crate) fn counter(&self, slot: usize) -> crate::dynamics::engine::SlotRef {
        assert!(
            slot < self.counters.slots() as usize,
            "counter slot {slot} is outside the counter stream"
        );
        crate::dynamics::engine::SlotRef::range(
            SceneStream::Counters.into(),
            slot as u64 * COUNTER_STRIDE,
            4,
        )
    }

    pub(crate) fn body_row_count(&self) -> u32 {
        self.body_states.slots()
    }

    pub(crate) fn collider_capacity(&self) -> u32 {
        self.collider_owners.slots()
    }

    pub(crate) fn constraint_capacity(&self) -> u32 {
        self.constraint_runtime.slots()
    }

    pub(crate) fn body_words(&self) -> u32 {
        dynamis_sort::key_words(self.body_row_count().max(1))
    }

    pub(crate) fn collider_words(&self) -> u32 {
        dynamis_sort::key_words(self.collider_capacity().max(1))
    }
}
