//! The GPU-resident rigid-body world.

mod buffers;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamBudget {
    pub pairs_per_body: u32,
    pub events_per_body: u32,
}

impl Default for StreamBudget {
    fn default() -> Self {
        Self {
            pairs_per_body: 128,
            events_per_body: 8,
        }
    }
}
pub use simulation::{ContactManifold, ContactPoint, DebugBuffer, Simulation};
