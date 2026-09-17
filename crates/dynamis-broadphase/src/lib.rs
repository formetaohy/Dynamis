mod capacity;
mod domain;
mod passes;
mod streams;

pub use capacity::{BroadphaseCapacity, BroadphaseInputs, floor, plan};
pub use domain::{BroadphaseDomain, BroadphaseFrame};
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
