use crate::domain::BroadphaseFrame;
use crate::streams::BroadphaseStream;
use dynamis_abi::{COUNTER_ENTRIES, COUNTER_IMMOVABLE_ENTRIES};
use dynamis_gpu::{ComputeRecorder, GpuContext, ResourceSource, SlotRef};
use dynamis_pass::{Execution, PassRuntime, Stage, domain_passes};
use dynamis_shader::{Extent, GRID_INDEX, stream};

const PAIR_EMIT: &str = include_str!("../shaders/pair_emit.wgsl");

fn pair_fragments() -> Vec<&'static str> {
    let mut fragments = GRID_INDEX.to_vec();
    fragments.push(PAIR_EMIT);
    fragments
}
use dynamis_sort::{RadixSort, SortChannels};
use dynamis_state::StateStream;

#[derive(Clone, Copy, PartialEq, Eq)]
enum EntryRegion {
    Immovable,
    Moving,
}

impl EntryRegion {
    const fn base(self, frame: &BroadphaseFrame) -> u32 {
        match self {
            Self::Immovable => 0,
            Self::Moving => frame.entry_base,
        }
    }

    const fn slots(self, frame: &BroadphaseFrame) -> u32 {
        match self {
            Self::Immovable => frame.entry_base,
            Self::Moving => frame.moving_slots,
        }
    }

    const fn count(self) -> usize {
        match self {
            Self::Immovable => COUNTER_IMMOVABLE_ENTRIES,
            Self::Moving => COUNTER_ENTRIES,
        }
    }
}

fn region_range(stream: BroadphaseStream, base: u32, slots: u32) -> SlotRef {
    SlotRef::range(
        stream.into(),
        u64::from(base) * stream.element().bytes(),
        u64::from(slots) * stream.element().bytes(),
        stream.element(),
    )
}

pub struct Broadphase {
    sort: RadixSort,
    cell_pairs: Stage,
    level_links: Stage,
}

impl Broadphase {
    fn sort_region(
        &mut self,
        recorder: &mut ComputeRecorder,
        streams: &impl ResourceSource,
        region: EntryRegion,
        frame: &BroadphaseFrame,
    ) {
        let base = region.base(frame);
        let slots = region.slots(frame);
        if slots == 0 {
            return;
        }
        let channels = SortChannels {
            count: dynamis_state::counter(region.count()).resolve(streams),
            major: region_range(BroadphaseStream::EntryKeys, base, slots).resolve(streams),
            minor: region_range(BroadphaseStream::SortDummy, base, slots).resolve(streams),
            payload: region_range(BroadphaseStream::EntryOrder, base, slots).resolve(streams),
            scratch_major: region_range(BroadphaseStream::SortScratchMajor, 0, slots)
                .resolve(streams),
            scratch_minor: region_range(BroadphaseStream::SortScratchMinor, 0, slots)
                .resolve(streams),
            scratch_payload: region_range(BroadphaseStream::SortScratchPayload, 0, slots)
                .resolve(streams),
        };
        self.sort.sort(recorder, &channels, 4, 0);
    }
}

impl PassRuntime<BroadphaseFrame> for Broadphase {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            sort: RadixSort::new(context),
            level_links: Stage::build(
                context,
                "level_links",
                stream(
                    context,
                    include_str!("../shaders/level_links.wgsl"),
                    &pair_fragments(),
                    "work",
                    Extent::of_sum([COUNTER_IMMOVABLE_ENTRIES, COUNTER_ENTRIES], "entries"),
                ),
                streams,
                &[
                    ("pair_major", BroadphaseStream::PairMajor.whole()),
                    ("pair_minor", BroadphaseStream::PairMinor.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", StateStream::Counters.whole()),
                ],
                &[],
            ),
            cell_pairs: Stage::build(
                context,
                "cell_pairs",
                stream(
                    context,
                    include_str!("../shaders/cell_pairs.wgsl"),
                    &pair_fragments(),
                    "work",
                    Extent::of_sum([COUNTER_IMMOVABLE_ENTRIES, COUNTER_ENTRIES], "entries"),
                ),
                streams,
                &[
                    ("pair_major", BroadphaseStream::PairMajor.whole()),
                    ("pair_minor", BroadphaseStream::PairMinor.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", StateStream::Counters.whole()),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &BroadphaseFrame,
    ) {
        if frame.immovable_rebuild {
            self.sort_region(recorder, streams, EntryRegion::Immovable, frame);
        }
        self.sort_region(recorder, streams, EntryRegion::Moving, frame);
        self.level_links.record_stream(recorder, streams);
        self.cell_pairs.record_stream(recorder, streams);
    }
}

domain_passes!(
    BroadphasePasses,
    BroadphaseRuntime,
    BroadphaseFrame,
    broadphase: Broadphase => Execution::INDEXING => &["emit_entries", "emit_soft_entries"],
);
