mod capacity;
mod streams;

pub use capacity::Capacity;
pub use streams::{
    BroadphaseDemand, BroadphaseStream, BroadphaseStreams, DOMAIN, entry_capacity, pair_capacity,
    sort_capacity,
};

use dynamis_abi::COUNTER_ENTRIES;
use dynamis_engine::{Engine, Resources, Stage, domain_passes};
use dynamis_gpu::GpuContext;
use dynamis_kernels::{GRID_INDEX, stream};
use dynamis_scene::SceneStream;
use dynamis_sort::{RadixSort, SortChannels};

domain_passes!(
    BroadphasePasses,
    "broadphase",
    index => "broadphase",
);

pub struct Broadphase {
    passes: BroadphasePasses,
    sort: RadixSort,
    cell_pairs: Stage,
    level_links: Stage,
}

impl Broadphase {
    pub fn new(context: &GpuContext, streams: &impl Resources, passes: BroadphasePasses) -> Self {
        Self {
            passes,
            sort: RadixSort::new(context, "grid sort", sort_capacity(streams)),
            cell_pairs: Stage::build(
                context,
                "cell_pairs",
                stream(
                    context,
                    include_str!("../shaders/cell_pairs.wgsl"),
                    GRID_INDEX,
                    "work",
                    BroadphaseStream::EntryKeys,
                ),
                streams,
                &[
                    ("pair_major", BroadphaseStream::PairMajor.whole()),
                    ("pair_minor", BroadphaseStream::PairMinor.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", SceneStream::Counters.whole()),
                ],
                &[],
            ),
            level_links: Stage::build(
                context,
                "level_links",
                stream(
                    context,
                    include_str!("../shaders/level_links.wgsl"),
                    GRID_INDEX,
                    "work",
                    BroadphaseStream::EntryKeys,
                ),
                streams,
                &[
                    ("pair_major", BroadphaseStream::PairMajor.whole()),
                    ("pair_minor", BroadphaseStream::PairMinor.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", SceneStream::Counters.whole()),
                ],
                &[],
            ),
        }
    }

    pub fn sort_entries(
        &self,
        recorder: &mut dynamis_gpu::ComputeRecorder,
        streams: &impl Resources,
    ) {
        let count = dynamis_scene::counter(COUNTER_ENTRIES).resolve(streams);
        let channels = SortChannels {
            generation: streams.generation(),
            count,
            major: BroadphaseStream::EntryKeys.whole().resolve(streams),
            minor: BroadphaseStream::SortDummy.whole().resolve(streams),
            payload: BroadphaseStream::EntryOrder.whole().resolve(streams),
            scratch_major: BroadphaseStream::SortScratchMajor.whole().resolve(streams),
            scratch_minor: BroadphaseStream::SortScratchMinor.whole().resolve(streams),
            scratch_payload: BroadphaseStream::SortScratchPayload
                .whole()
                .resolve(streams),
        };
        self.sort
            .sort(recorder, &channels, 4, 0, entry_capacity(streams));
    }

    pub fn encode(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        streams: &impl Resources,
    ) {
        let mut index = engine.open(encoder, self.passes.index);
        self.sort_entries(&mut index, streams);
        self.cell_pairs.record_stream(&mut index, streams);
        self.level_links.record_stream(&mut index, streams);
        drop(index);
    }
}
