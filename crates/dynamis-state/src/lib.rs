mod capacity;
mod domain;
mod streams;

pub use capacity::{ShapeCapacity, StateInputs, capacity, floor, plan};
pub use domain::StateDomain;
pub use streams::{
    QUERY_RESULT_BYTES, StateDemand, StateStream, StateStreams, TRIANGLE_BYTES, VERTEX_BYTES,
};

use dynamis_abi::{COUNTER_DEVICE_COUNT, COUNTER_STRIDE};
use dynamis_pass::SlotRef;

pub const MOVE_ENTRIES_PER_COMMAND: u32 = 2;

pub fn counter(slot: usize) -> SlotRef {
    assert!(
        slot < COUNTER_DEVICE_COUNT,
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
