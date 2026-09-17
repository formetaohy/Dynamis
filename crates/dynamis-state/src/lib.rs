mod capacity;
mod domain;
mod passes;
mod streams;

pub use capacity::{ShapeCapacity, StateCapacity, StateInputs, capacity, floor, plan};
pub use domain::{StateDomain, StateWork};
pub use passes::{StatePasses, StateRuntime};
pub use streams::{StateDemand, StateStream, StateStreams, TRIANGLE_BYTES, VERTEX_BYTES};

use dynamis_abi::{COUNTER_STRIDE, counter};
use dynamis_gpu::SlotRef;

pub const MOVE_ENTRIES_PER_COMMAND: u32 = 2;

pub fn counter(slot: usize) -> SlotRef {
    counter::spec(slot);
    SlotRef::range(
        StateStream::Counters.into(),
        slot as u64 * COUNTER_STRIDE,
        4,
        StateStream::Counters.element(),
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
