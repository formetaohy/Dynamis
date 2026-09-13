use dynamis_abi::GridEntryRecord;
use dynamis_gpu::Contents;
use dynamis_pass::streams;
use std::mem::size_of;

pub const DOMAIN: u32 = 1;

streams! {
    BroadphaseStreams, BroadphaseStream, BroadphaseDemand, DOMAIN, demand,
    demand {
        entries: u32,
        pairs: u32,
        sort: u32,
    }
    streams {
        entry_keys, EntryKeys: "grid entry keys", 4, Contents::Scratch, demand.entries;
        entry_order, EntryOrder: "grid entry order", 4, Contents::Scratch, demand.entries;
        entries, Entries: "grid entries", size_of::<GridEntryRecord>() as u64, Contents::Scratch, demand.entries;
        pair_major, PairMajor: "pairs major", 4, Contents::Scratch, demand.pairs;
        pair_minor, PairMinor: "pairs minor", 4, Contents::Scratch, demand.pairs;
        sort_dummy, SortDummy: "sort key dummy", 4, Contents::Scratch, demand.sort;
        sort_scratch_major, SortScratchMajor: "sort scratch major", 4, Contents::Scratch, demand.sort;
        sort_scratch_minor, SortScratchMinor: "sort scratch minor", 4, Contents::Scratch, demand.sort;
        sort_scratch_payload, SortScratchPayload: "sort scratch payload", 4, Contents::Scratch, demand.sort;
    }
}

pub fn entry_capacity<R: dynamis_pass::Resources>(resources: &R) -> u32 {
    resources.slots(BroadphaseStream::EntryKeys.into())
}

pub fn pair_capacity<R: dynamis_pass::Resources>(resources: &R) -> u32 {
    resources.slots(BroadphaseStream::PairMajor.into())
}

pub fn sort_capacity<R: dynamis_pass::Resources>(resources: &R) -> u32 {
    resources.slots(BroadphaseStream::SortScratchMajor.into())
}
