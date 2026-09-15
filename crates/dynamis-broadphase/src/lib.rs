mod capacity;
mod domain;
mod passes;
mod streams;

pub use capacity::{BroadphaseCapacity, BroadphaseInputs, Capacity};
pub use domain::BroadphaseDomain;
pub use passes::{Broadphase, BroadphasePasses, BroadphaseRuntime};
pub use streams::{
    BroadphaseDemand, BroadphaseStream, BroadphaseStreams, entry_capacity, pair_capacity,
};

pub fn capacity(streams: &BroadphaseStreams) -> BroadphaseCapacity {
    BroadphaseCapacity {
        entries: streams.entry_keys.slots(),
        pairs: streams.pair_major.slots(),
    }
}
