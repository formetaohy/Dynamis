use crate::StateDomain;
use dynamis_abi::{
    BodyDescriptorRecord, BodyEditRecord, BodyEditRunRecord, BodyStateRecord,
    BrokenConstraintRecord, BvhNodeRecord, COUNTER_DEVICE_COUNT, COUNTER_STRIDE, ColliderRecord,
    ConstraintDescriptorRecord, ConstraintRuntimeRecord, QueryRecord, QueryResultRecord,
    RowMoveRecord, ShapeSourceRecord, StepParamsRecord, TriangleRecord,
};
use dynamis_domain::Domain;
use dynamis_gpu::Contents;
use dynamis_pass::EVENT_SLOTS;
use dynamis_pass::{UNIFORM, streams};
use std::mem::size_of;

pub const VERTEX_BYTES: u64 = size_of::<[f32; 4]>() as u64;
pub const TRIANGLE_BYTES: u64 = size_of::<TriangleRecord>() as u64;
pub const QUERY_RESULT_BYTES: u64 = size_of::<QueryResultRecord>() as u64;

streams! {
    StateStreams, StateStream, StateDemand, StateDomain::ID, demand,
    demand {
        bodies: u32,
        body_ids: u32,
        colliders: u32,
        constraints: u32,
        body_commands: u32,
        constraint_commands: u32,
        queries: u32,
        shapes: crate::ShapeCapacity,
        observed: u32,
    }
    streams {
        params, Params: "step params", StepParamsRecord, 1, Contents::Scratch, 1, UNIFORM;
        body_states, BodyStates: "body states", BodyStateRecord, 1, Contents::Durable, demand.bodies;
        body_row_of_id, BodyRowOfId: "body row of id", u32, 1, Contents::Durable, demand.body_ids;
        body_descriptors, BodyDescriptors: "body descriptors", BodyDescriptorRecord, 1, Contents::Durable, demand.bodies;
        colliders, Colliders: "colliders", ColliderRecord, 1, Contents::Durable, demand.colliders;
        collider_owners, ColliderOwners: "collider owners", u32, 1, Contents::Durable, demand.colliders;
        body_edits, BodyEdits: "body edits", BodyEditRecord, 1, Contents::Scratch, demand.body_commands;
        body_edit_runs, BodyEditRuns: "body edit runs", BodyEditRunRecord, 1, Contents::Scratch, demand.body_commands;
        body_row_moves, BodyRowMoves: "body row moves", RowMoveRecord, 1, Contents::Scratch, demand.body_moves();
        body_fresh_rows, BodyFreshRows: "fresh body rows", BodyStateRecord, 1, Contents::Scratch, demand.body_commands;
        constraint_descriptors, ConstraintDescriptors: "constraint descriptors", ConstraintDescriptorRecord, 1, Contents::Durable, demand.constraints;
        constraint_runtime, ConstraintRuntime: "constraint runtime", ConstraintRuntimeRecord, 1, Contents::Durable, demand.constraints;
        wake_flags, WakeFlags: "body wake flags", u32, 1, Contents::Scratch, demand.bodies;
        constraint_row_moves, ConstraintRowMoves: "constraint row moves", RowMoveRecord, 1, Contents::Scratch, demand.constraint_moves();
        constraint_fresh_rows, ConstraintFreshRows: "fresh constraint rows", ConstraintRuntimeRecord, 1, Contents::Scratch, demand.constraint_commands;
        constraint_breaks, ConstraintBreaks: "constraint breaks", BrokenConstraintRecord, 1, Contents::Scratch, demand.constraints.saturating_mul(EVENT_SLOTS);
        shape_sources, ShapeSources: "shape sources", ShapeSourceRecord, 1, Contents::Durable, demand.shapes.sources;
        shape_vertices, ShapeVertices: "shape vertices", [f32; 4], 1, Contents::Durable, demand.shapes.vertices;
        shape_triangles, ShapeTriangles: "shape triangles", TriangleRecord, 1, Contents::Durable, demand.shapes.triangles;
        shape_nodes, ShapeNodes: "shape bvh nodes", BvhNodeRecord, 1, Contents::Durable, demand.shapes.nodes;
        query_records, QueryRecords: "queries", QueryRecord, 1, Contents::Scratch, demand.queries;
        query_results, QueryResults: "query results", QueryResultRecord, 1, Contents::Scratch, demand.queries;
        observed_ids, ObservedIds: "observed body ids", u32, 1, Contents::Durable, demand.observed;
        observed_states, ObservedStates: "observed body states", BodyStateRecord, 1, Contents::Scratch, demand.observed;
        counters, Counters: "world counters", u32, COUNTER_STRIDE / 4, Contents::Durable, COUNTER_DEVICE_COUNT as u32;
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
