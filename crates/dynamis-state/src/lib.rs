mod capacity;
mod streams;

pub use capacity::{Live, ShapeCapacity, floor, plan};
pub use streams::{
    DOMAIN, QUERY_RESULT_BYTES, StateDemand, StateStream, StateStreams, TRIANGLE_BYTES,
    VERTEX_BYTES,
};

use dynamis_abi::{COUNTER_COUNT, COUNTER_STRIDE, StepParamsRecord};
use dynamis_pass::{Resources, SlotRef};

pub const MOVE_ENTRIES_PER_COMMAND: u32 = 2;

#[derive(Clone, Copy, Debug)]
pub struct StepFrame {
    pub params: StepParamsRecord,
    pub query_count: u32,
    pub awake_bodies: Option<u32>,
    pub ccd_bodies: bool,
    pub soft_bodies: bool,
}

impl StepFrame {
    pub fn simulating(&self) -> bool {
        self.awake_bodies != Some(0)
            || self.params.constraint_count > 0
            || self.query_count > 0
            || self.soft_bodies
    }
}

#[derive(Clone, Copy)]
pub enum Count {
    Bodies,
    Dynamic,
    Colliders,
    Constraints,
    Particles,
    Elements,
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
            Self::Particles => "particle_count",
            Self::Elements => "element_count",
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
            Self::Particles => params.particle_count,
            Self::Elements => params.element_count,
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
        StateStream::Counters.into(),
        slot as u64 * COUNTER_STRIDE,
        4,
    )
}

pub fn shape_resources() -> [(&'static str, SlotRef); 4] {
    [
        ("shape_sources", StateStream::ShapeSources.whole()),
        ("shape_vertices", StateStream::ShapeVertices.whole()),
        ("shape_triangles", StateStream::ShapeTriangles.whole()),
        ("shape_nodes", StateStream::ShapeNodes.whole()),
    ]
}

pub fn body_row_count<R: Resources>(resources: &R) -> u32 {
    resources.slots(StateStream::BodyStates.into())
}

pub fn collider_capacity<R: Resources>(resources: &R) -> u32 {
    resources.slots(StateStream::ColliderOwners.into())
}

pub fn constraint_capacity<R: Resources>(resources: &R) -> u32 {
    resources.slots(StateStream::ConstraintRuntime.into())
}

pub fn body_words<R: Resources>(resources: &R) -> u32 {
    dynamis_sort::key_words(body_row_count(resources).max(1))
}

pub fn collider_words<R: Resources>(resources: &R) -> u32 {
    dynamis_sort::key_words(collider_capacity(resources).max(1))
}
