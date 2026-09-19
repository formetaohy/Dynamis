use crate::StateDomain;
use dynamis_abi::{
    BodyDescriptorRecord, BodyEditRecord, BodyEditRunRecord, BodyStateRecord,
    BrokenConstraintRecord, BvhNodeRecord, COUNTER_DEVICE_COUNT, COUNTER_STRIDE, CellRecord,
    ColliderRecord, ConstraintDescriptorRecord, ConstraintRuntimeRecord, FieldRecord,
    JointStateRecord, REACTION_WORDS, RowMoveRecord, RowStreamsRecord, ShapeSourceRecord,
    StepParamsRecord, TriangleRecord,
};
use dynamis_domain::Domain;
use dynamis_domain::StreamWriters;
use dynamis_domain::streams;
use dynamis_gpu::Retention;
use dynamis_gpu::SEGMENT_COUNT;
use std::mem::size_of;

pub const VERTEX_BYTES: u64 = size_of::<[f32; 4]>() as u64;
pub const TRIANGLE_BYTES: u64 = size_of::<TriangleRecord>() as u64;
pub const CELL_BYTES: u64 = size_of::<CellRecord>() as u64;

streams! {
    StateStreams, StateStream, StateDemand, StateDomain::ID, demand,
    demand {
        bodies: u32,
        body_ids: u32,
        colliders: u32,
        constraints: u32,
        constraint_ids: u32,
        fields: u32,
        body_commands: u32,
        constraint_commands: u32,
        shapes: crate::ShapeCapacity,
        observed: u32,
        observed_joints: u32,
    }
    streams {
        params, Params: "step params", StepParamsRecord, 1, Retention::Scratch, StreamWriters::Host, 1, dynamis_gpu::UNIFORM;
        row_streams, RowStreams: "step row streams", RowStreamsRecord, 1, Retention::Scratch, StreamWriters::HostThenDevice, 1;
        body_states, BodyStates: "body states", BodyStateRecord, 1, Retention::Durable, StreamWriters::Device, demand.bodies;
        body_row_of_id, BodyRowOfId: "body row of id", u32, 1, Retention::Durable, StreamWriters::Device, demand.body_ids;
        body_descriptors, BodyDescriptors: "body descriptors", BodyDescriptorRecord, 1, Retention::Durable, StreamWriters::Host, demand.bodies;
        colliders, Colliders: "colliders", ColliderRecord, 1, Retention::Durable, StreamWriters::Host, demand.colliders;
        collider_owners, ColliderOwners: "collider owners", u32, 1, Retention::Durable, StreamWriters::Host, demand.colliders;
        body_edits, BodyEdits: "body edits", BodyEditRecord, 1, Retention::Scratch, StreamWriters::Host, demand.body_commands;
        body_edit_runs, BodyEditRuns: "body edit runs", BodyEditRunRecord, 1, Retention::Scratch, StreamWriters::Host, demand.body_commands;
        body_row_moves, BodyRowMoves: "body row moves", RowMoveRecord, 1, Retention::Scratch, StreamWriters::Host, demand.body_moves();
        body_fresh_rows, BodyFreshRows: "fresh body rows", BodyStateRecord, 1, Retention::Scratch, StreamWriters::Host, demand.body_commands;
        constraint_descriptors, ConstraintDescriptors: "constraint descriptors", ConstraintDescriptorRecord, 1, Retention::Durable, StreamWriters::Host, demand.constraints;
        constraint_runtime, ConstraintRuntime: "constraint runtime", ConstraintRuntimeRecord, 1, Retention::Durable, StreamWriters::Device, demand.constraints;
        constraint_row_of_id, ConstraintRowOfId: "constraint row of id", u32, 1, Retention::Durable, StreamWriters::Device, demand.constraint_ids;
        fields, Fields: "force fields", FieldRecord, 1, Retention::Durable, StreamWriters::Host, demand.fields;
        wake_flags, WakeFlags: "body wake flags", u32, 1, Retention::Scratch, StreamWriters::Device, demand.bodies;
        body_reactions, BodyReactions: "body reactions", u32, REACTION_WORDS, Retention::Scratch, StreamWriters::Device, demand.body_reactions();
        constraint_row_moves, ConstraintRowMoves: "constraint row moves", RowMoveRecord, 1, Retention::Scratch, StreamWriters::Host, demand.constraint_moves();
        constraint_fresh_rows, ConstraintFreshRows: "fresh constraint rows", ConstraintRuntimeRecord, 1, Retention::Scratch, StreamWriters::Host, demand.constraint_commands;
        constraint_breaks, ConstraintBreaks: "constraint breaks", BrokenConstraintRecord, 1, Retention::Scratch, StreamWriters::Device, demand.constraints.saturating_mul(SEGMENT_COUNT);
        shape_sources, ShapeSources: "shape sources", ShapeSourceRecord, 1, Retention::Durable, StreamWriters::Host, demand.shapes.sources;
        shape_vertices, ShapeVertices: "shape vertices", [f32; 4], 1, Retention::Durable, StreamWriters::Host, demand.shapes.vertices;
        shape_triangles, ShapeTriangles: "shape triangles", TriangleRecord, 1, Retention::Durable, StreamWriters::Host, demand.shapes.triangles;
        shape_nodes, ShapeNodes: "shape bvh nodes", BvhNodeRecord, 1, Retention::Durable, StreamWriters::Host, demand.shapes.nodes;
        shape_cells, ShapeCells: "shape cells", CellRecord, 1, Retention::Durable, StreamWriters::Host, demand.shapes.cells;
        observed_ids, ObservedIds: "observed body ids", u32, 1, Retention::Durable, StreamWriters::Host, demand.observed;
        observed_states, ObservedStates: "observed body states", BodyStateRecord, 1, Retention::Scratch, StreamWriters::Device, demand.observed;
        observed_joint_ids, ObservedJointIds: "observed joint ids", u32, 1, Retention::Durable, StreamWriters::Host, demand.observed_joints;
        observed_joint_states, ObservedJointStates: "observed joint states", JointStateRecord, 1, Retention::Scratch, StreamWriters::Device, demand.observed_joints;
        observed_joint_runtimes, ObservedJointRuntimes: "observed joint runtimes", ConstraintRuntimeRecord, 1, Retention::Scratch, StreamWriters::Device, demand.observed_joints;
        counters, Counters: "world counters", u32, COUNTER_STRIDE / 4, Retention::Durable, StreamWriters::Device, COUNTER_DEVICE_COUNT as u32;
        entry_base, EntryBase: "grid entry base", u32, 1, Retention::Scratch, StreamWriters::Host, 1;
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

    pub fn body_reactions(&self) -> u32 {
        self.bodies.saturating_mul(REACTION_WORDS)
    }
}
