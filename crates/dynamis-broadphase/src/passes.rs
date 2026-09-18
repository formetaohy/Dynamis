use crate::domain::BroadphaseFrame;
use crate::streams::BroadphaseStream;
use dynamis_abi::{
    COUNTER_AWAKE_BASE, COUNTER_ENTRIES, COUNTER_ENTRY_BASE, COUNTER_IMMOVABLE_ENTRIES,
    COUNTER_RESTING_SORT,
};
use dynamis_gpu::{ComputeRecorder, GpuContext, ResourceSource, TypedSlot};
use dynamis_pass::{Execution, PassRuntime, Stage, domain_passes};
use dynamis_shader::{Extent, GRID_INDEX, stream};

const PAIR_EMIT: &str = include_str!("../shaders/pair_emit.wgsl");

fn pair_fragments() -> Vec<&'static str> {
    let mut fragments = GRID_INDEX.to_vec();
    fragments.push(PAIR_EMIT);
    fragments
}
use dynamis_sort::{RadixSort, SortChannels, units_for};
use dynamis_state::StateStream;

#[derive(Clone, Copy, PartialEq, Eq)]
enum EntryRegion {
    Immovable,
    Resting,
    Awake,
}

impl EntryRegion {
    const fn slots(self, frame: &BroadphaseFrame) -> u32 {
        match self {
            Self::Immovable => frame.entry_base,
            Self::Resting | Self::Awake => frame.moving_slots,
        }
    }

    const fn count(self) -> usize {
        match self {
            Self::Immovable => COUNTER_IMMOVABLE_ENTRIES,
            Self::Resting => COUNTER_RESTING_SORT,
            Self::Awake => COUNTER_ENTRIES,
        }
    }

    const fn origin(self) -> Option<usize> {
        match self {
            Self::Immovable => None,
            Self::Resting => Some(COUNTER_ENTRY_BASE),
            Self::Awake => Some(COUNTER_AWAKE_BASE),
        }
    }
}

fn channels<'a, R: ResourceSource>(streams: &'a R, region: EntryRegion) -> SortChannels<'a> {
    let scratch = |stream: BroadphaseStream| -> TypedSlot<'a> { stream.whole().resolve(streams) };
    SortChannels {
        count: dynamis_state::counter(region.count()).resolve(streams),
        base: region
            .origin()
            .map(|counter| dynamis_state::counter(counter).resolve(streams)),
        major: BroadphaseStream::EntryKeys.whole().resolve(streams),
        minor: BroadphaseStream::SortDummy.whole().resolve(streams),
        payload: BroadphaseStream::EntryOrder.whole().resolve(streams),
        scratch_major: scratch(BroadphaseStream::SortScratchMajor),
        scratch_minor: scratch(BroadphaseStream::SortScratchMinor),
        scratch_payload: scratch(BroadphaseStream::SortScratchPayload),
    }
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
        if region.slots(frame) == 0 {
            return;
        }
        let channels = channels(streams, region);
        let plan = units_for(streams.measured(region.count()).unwrap_or(0));
        self.sort.sort(recorder, &channels, plan, 4, 0);
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
                    Extent::of_offset(COUNTER_AWAKE_BASE, COUNTER_ENTRIES, "entries"),
                ),
                streams,
                &[
                    ("pair_major", BroadphaseStream::PairMajor.whole()),
                    ("pair_minor", BroadphaseStream::PairMinor.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("body_admitted", BroadphaseStream::BodyAdmitted.whole()),
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
                    Extent::of_offset(COUNTER_AWAKE_BASE, COUNTER_ENTRIES, "entries"),
                ),
                streams,
                &[
                    ("pair_major", BroadphaseStream::PairMajor.whole()),
                    ("pair_minor", BroadphaseStream::PairMinor.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("body_admitted", BroadphaseStream::BodyAdmitted.whole()),
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
        if frame.resting_rebuild {
            self.sort_region(recorder, streams, EntryRegion::Resting, frame);
        }
        self.sort_region(recorder, streams, EntryRegion::Awake, frame);
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
