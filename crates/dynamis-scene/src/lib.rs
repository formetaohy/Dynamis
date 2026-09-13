mod capacity;
mod streams;

pub use capacity::{Live, ShapeCapacity, demand, floor};
pub use streams::{
    DOMAIN, QUERY_RESULT_BYTES, SceneDemand, SceneStream, SceneStreams, TRIANGLE_BYTES,
    VERTEX_BYTES,
};

use dynamis_abi::{COUNTER_COUNT, COUNTER_STRIDE, StepParamsRecord};
use dynamis_engine::{Resources, SlotRef};

pub const MOVE_ENTRIES_PER_COMMAND: u32 = 2;

#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub params: StepParamsRecord,
    pub query_count: u32,
}

#[derive(Clone, Copy)]
pub enum Count {
    Bodies,
    Dynamic,
    Colliders,
    Constraints,
    EditRuns,
    BodyMoves,
    ConstraintMoves,
}

impl Count {
    pub const fn field(self) -> &'static str {
        match self {
            Self::Bodies => "body_count",
            Self::Dynamic => "dynamic_count",
            Self::Colliders => "collider_count",
            Self::Constraints => "constraint_count",
            Self::EditRuns => "edit_run_count",
            Self::BodyMoves => "body_move_count",
            Self::ConstraintMoves => "constraint_move_count",
        }
    }

    pub fn rows(self, params: &StepParamsRecord) -> u32 {
        match self {
            Self::Bodies => params.body_count,
            Self::Dynamic => params.dynamic_count,
            Self::Colliders => params.collider_count,
            Self::Constraints => params.constraint_count,
            Self::EditRuns => params.edit_run_count,
            Self::BodyMoves => params.body_move_count,
            Self::ConstraintMoves => params.constraint_move_count,
        }
    }
}

pub fn counter(slot: usize) -> SlotRef {
    assert!(
        slot < COUNTER_COUNT,
        "counter slot {slot} is outside the counter stream"
    );
    SlotRef::range(
        SceneStream::Counters.into(),
        slot as u64 * COUNTER_STRIDE,
        4,
    )
}

pub fn shape_resources() -> [(&'static str, SlotRef); 4] {
    [
        ("shape_sources", SceneStream::ShapeSources.whole()),
        ("shape_vertices", SceneStream::ShapeVertices.whole()),
        ("shape_triangles", SceneStream::ShapeTriangles.whole()),
        ("shape_nodes", SceneStream::ShapeNodes.whole()),
    ]
}

pub fn body_row_count<R: Resources>(resources: &R) -> u32 {
    resources.slots(SceneStream::BodyStates.into())
}

pub fn collider_capacity<R: Resources>(resources: &R) -> u32 {
    resources.slots(SceneStream::ColliderOwners.into())
}

pub fn constraint_capacity<R: Resources>(resources: &R) -> u32 {
    resources.slots(SceneStream::ConstraintRuntime.into())
}

pub fn body_words<R: Resources>(resources: &R) -> u32 {
    dynamis_sort::key_words(body_row_count(resources).max(1))
}

pub fn collider_words<R: Resources>(resources: &R) -> u32 {
    dynamis_sort::key_words(collider_capacity(resources).max(1))
}
