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

/// How much device storage a world plans per live body before the device has
/// measured anything. Streams grow with what the device actually produces, so this
/// only sets the first plan and the floor it never shrinks below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamBudget {
    /// Pair lanes reserved per live body. One pair is one potential contact: the
    /// broadphase of a dense stack holds far more pairs per body than a loose pile,
    /// so this figure belongs to the scene, not to the engine. The default of 128
    /// covers a dense box pyramid; the device reports an exact figure if a step ever
    /// needs more.
    pub pairs_per_body: u32,
}

impl Default for StreamBudget {
    fn default() -> Self {
        Self {
            pairs_per_body: 128,
        }
    }
}
pub use simulation::{ContactManifold, ContactPoint, DebugBuffer, Simulation};
