//! The GPU-resident rigid-body world.

mod buffers;
mod capacity;
mod character;
mod pipeline;
mod query_pool;
mod records;
mod reservation;
mod shape_pool;
mod simulation;
mod static_aabb;

pub use character::{Character, CharacterDesc};
pub use dynamis_layout::{
    COUNTER_CONTACTS, COUNTER_COUNT, COUNTER_ENTRIES, COUNTER_EVENTS, COUNTER_PAIRS,
    COUNTER_PREV_CONTACTS, COUNTER_SPILLOVER_ENTRIES, COUNTER_SPILLOVER_EVENTS,
    COUNTER_SPILLOVER_PAIRS, Counters,
};
pub use dynamis_mesh::HullDecomposeSettings;
pub use query_pool::{QueryHandle, QueryHit};
pub use simulation::{ContactManifold, ContactPoint, DebugBuffer, Simulation};

/// The stream planning loop, exposed for integration tests that exercise capacity
/// behaviour without owning a device.
#[doc(hidden)]
pub mod planning {
    pub use crate::capacity::Capacity;
    pub use crate::reservation::{
        Live, Reservation, STREAM_FLOOR, ShapeReservation, StreamDemand, StreamPlan,
    };
}
