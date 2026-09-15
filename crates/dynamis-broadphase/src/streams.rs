use crate::BroadphaseDomain;
use dynamis_abi::GridEntryRecord;
use dynamis_domain::Domain;
use dynamis_domain::streams;
use dynamis_gpu::Contents;

streams! {
    BroadphaseStreams, BroadphaseStream, BroadphaseDemand, BroadphaseDomain::ID, demand,
    demand {
        entries: u32,
        pairs: u32,
        sort: u32,
    }
    streams {
        entry_keys, EntryKeys: "grid entry keys", u32, 1, Contents::Scratch, demand.entries;
        entry_order, EntryOrder: "grid entry order", u32, 1, Contents::Scratch, demand.entries;
        entries, Entries: "grid entries", GridEntryRecord, 1, Contents::Scratch, demand.entries;
        pair_major, PairMajor: "pairs major", u32, 1, Contents::Scratch, demand.pairs;
        pair_minor, PairMinor: "pairs minor", u32, 1, Contents::Scratch, demand.pairs;
        sort_dummy, SortDummy: "sort key dummy", u32, 1, Contents::Scratch, demand.sort;
        sort_scratch_major, SortScratchMajor: "sort scratch major", u32, 1, Contents::Scratch, demand.sort;
        sort_scratch_minor, SortScratchMinor: "sort scratch minor", u32, 1, Contents::Scratch, demand.sort;
        sort_scratch_payload, SortScratchPayload: "sort scratch payload", u32, 1, Contents::Scratch, demand.sort;
    }
}

pub fn entry_capacity<R: dynamis_gpu::Resources>(resources: &R) -> u32 {
    resources.slots(BroadphaseStream::EntryKeys.into())
}

pub fn pair_capacity<R: dynamis_gpu::Resources>(resources: &R) -> u32 {
    resources.slots(BroadphaseStream::PairMajor.into())
}
