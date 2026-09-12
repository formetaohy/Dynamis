use super::capacity::ShapeCapacity;
use crate::dynamics::engine::{ResourceId, Resources, SlotRef};
use dynamis_gpu::{Contents, GpuBuffer, GpuSlot, ReadbackRing, Stream};
use dynamis_layout::{
    AabbRecord, BodyDescriptorRecord, BodyEditRecord, BodyEditRunRecord, BodyStateRecord,
    BvhNodeRecord, COUNTER_COUNT, COUNTER_STRIDE, ColliderRecord, ConstraintDescriptorRecord,
    ConstraintRowsRecord, ConstraintRuntimeRecord, ContactEventRecord, ContactRecord,
    MAX_HITS_PER_QUERY, QueryHitRecord, QueryRecord, QueryResultHeaderRecord, RowMoveRecord,
    SOLVER_BLOCK_CONSTRAINT, ShapeSourceRecord, StepParamsRecord, TriangleRecord,
};
use dynamis_sort::{SortChannels, key_words};
use std::mem::size_of;
use wgpu::{BufferUsages, Device, Queue};

pub(crate) const COMPACT_BLOCK: u32 = 256;

pub(crate) const EVENT_SLOTS: u32 = ReadbackRing::DEPTH as u32 + 2;

const STREAM: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);
const UNIFORM: BufferUsages = BufferUsages::UNIFORM.union(BufferUsages::COPY_DST);
const PACK: BufferUsages = BufferUsages::COPY_DST.union(BufferUsages::COPY_SRC);

const COUNTER_BYTES: u64 = COUNTER_STRIDE * COUNTER_COUNT as u64;
const SOLVER_BLOCK_KINDS: u32 = SOLVER_BLOCK_CONSTRAINT + 1;
const SOLVER_DELTA_BYTES: u64 = 64;
const SOLVER_CORRECTION_BYTES: u64 = 64;
const SOLVER_RESOLUTION_BYTES: u64 = 16;
const BODY_FRESH_BYTES: u64 = 128;
pub(crate) const VERTEX_BYTES: u64 = size_of::<[f32; 4]>() as u64;
pub(crate) const TRIANGLE_BYTES: u64 = size_of::<TriangleRecord>() as u64;
const QUERY_BYTES: u64 = size_of::<QueryRecord>() as u64;
const CONTACT_TARGET_BYTES: u64 = 4 * dynamis_layout::CONTACT_MAX_POINTS as u64;
const QUERY_RESULT_BYTES: u64 = size_of::<QueryResultHeaderRecord>() as u64
    + MAX_HITS_PER_QUERY as u64 * size_of::<QueryHitRecord>() as u64;

const MOVE_ENTRIES_PER_COMMAND: u32 = 2;

pub(crate) struct Demand {
    pub(crate) bodies: u32,
    pub(crate) body_ids: u32,
    pub(crate) colliders: u32,
    pub(crate) constraints: u32,
    pub(crate) body_commands: u32,
    pub(crate) constraint_commands: u32,
    pub(crate) queries: u32,
    pub(crate) entries: u32,
    pub(crate) pairs: u32,
    pub(crate) events: u32,
    pub(crate) shapes: ShapeCapacity,
    pub(crate) sort: u32,
}

impl Demand {
    pub(crate) fn body_moves(&self) -> u32 {
        self.body_commands.saturating_mul(MOVE_ENTRIES_PER_COMMAND)
    }

    pub(crate) fn constraint_moves(&self) -> u32 {
        self.constraint_commands
            .saturating_mul(MOVE_ENTRIES_PER_COMMAND)
    }

    pub(crate) fn blocks(&self) -> u32 {
        self.pairs.saturating_add(self.constraints)
    }

    pub(crate) fn compact_blocks(&self) -> u32 {
        self.pairs.div_ceil(COMPACT_BLOCK)
    }

    pub(crate) fn sort_slots(entries: u32, pairs: u32, constraints: u32) -> u32 {
        entries.max(pairs).max(pairs.saturating_add(constraints))
    }
}

macro_rules! stream_usage {
    () => {
        STREAM
    };
    ($usage:expr) => {
        $usage
    };
}

macro_rules! world_buffers {
    (
        with ($device:ident, $queue:ident, $demand:ident)
        extras { $( $extra:ident: $extra_ty:ty = $extra_init:expr, )* }
        streams {
            $(
                $name:ident, $variant:ident: $label:literal, $stride:expr, $contents:expr, $slots:expr $(, $usage:expr)?;
            )*
        }
    ) => {
        pub(crate) struct RigidBuffers {
            pub(crate) generation: u64,
            $( pub(crate) $extra: $extra_ty, )*
            $( pub(crate) $name: Stream, )*
        }

        #[derive(Clone, Copy, PartialEq, Eq)]
        pub(crate) enum StreamId {
            $( $variant, )*
        }

        const STREAMS: &[StreamId] = &[ $( StreamId::$variant, )* ];

        impl StreamId {
            pub(crate) const fn whole(self) -> SlotRef {
                SlotRef::whole(self.resource())
            }

            pub(crate) const fn range(self, offset: u64, size: u64) -> SlotRef {
                SlotRef::range(self.resource(), offset, size)
            }

            const fn resource(self) -> ResourceId {
                ResourceId::new(self as u32)
            }

            fn of(resource: ResourceId) -> Self {
                *STREAMS.get(resource.index() as usize).unwrap_or_else(|| {
                    panic!("resource {} is outside the stream table", resource.index())
                })
            }

            fn stream(self, buffers: &RigidBuffers) -> &Stream {
                match self {
                    $( Self::$variant => &buffers.$name, )*
                }
            }
        }

        impl From<StreamId> for ResourceId {
            fn from(id: StreamId) -> Self {
                ResourceId::new(id as u32)
            }
        }

        impl Resources for RigidBuffers {
            fn generation(&self) -> u64 {
                self.generation
            }

            fn slots(&self, resource: ResourceId) -> u32 {
                StreamId::of(resource).stream(self).slots()
            }

            fn whole(&self, resource: ResourceId) -> GpuSlot<'_> {
                StreamId::of(resource).stream(self).slot()
            }

            fn range(&self, resource: ResourceId, offset: u64, size: u64) -> GpuSlot<'_> {
                GpuSlot::range(StreamId::of(resource).stream(self).gpu(), offset, size)
            }
        }

        impl RigidBuffers {
            pub(crate) fn new($device: &Device, $queue: &Queue, $demand: &Demand) -> Self {
                Self {
                    generation: 0,
                    $( $extra: $extra_init, )*
                    $(
                        $name: Stream::new(
                            $device,
                            $queue,
                            $label,
                            $slots,
                            $stride,
                            stream_usage!($($usage)?),
                            $contents,
                        ),
                    )*
                }
            }

            fn reserve_streams(
                &mut self,
                device: &Device,
                encoder: &mut wgpu::CommandEncoder,
                $demand: &Demand,
            ) -> bool {
                let mut changed = false;
                $( changed |= self.$name.reserve(device, encoder, $slots); )*
                changed
            }

            fn streams_match(&self, $demand: &Demand) -> bool {
                true $( && self.$name.slots() == $slots )*
            }

            pub(crate) fn matches(&self, demand: &Demand) -> bool {
                self.streams_match(demand) && self.readback_matches(demand)
            }

            pub(crate) fn readback_matches(&self, demand: &Demand) -> bool {
                self.readback.matches(demand)
            }

            pub(crate) fn reserve(
                &mut self,
                device: &Device,
                encoder: &mut wgpu::CommandEncoder,
                demand: &Demand,
            ) -> bool {
                let streams = self.reserve_streams(device, encoder, demand);
                if streams {
                    self.generation = self
                        .generation
                        .checked_add(1)
                        .expect("a storage generation must not overflow");
                }
                let readback = self.readback.reserve(device, demand);
                streams | readback
            }
        }
    };
}

world_buffers! {
    with (device, queue, demand)
    extras {
        readback: ReadbackBuffers = ReadbackBuffers::new(device, demand),
    }
    streams {
        params, Params: "step params", size_of::<StepParamsRecord>() as u64, Contents::Reset, 1, UNIFORM;
        body_states, BodyStates: "body states", size_of::<BodyStateRecord>() as u64, Contents::Preserve, demand.bodies;
        body_activity, BodyActivity: "body activity", 4, Contents::Reset, demand.bodies;
        body_row_of_id, BodyRowOfId: "body row of id", 4, Contents::Preserve, demand.body_ids;
        body_descriptors, BodyDescriptors: "body descriptors", size_of::<BodyDescriptorRecord>() as u64, Contents::Preserve, demand.bodies;
        colliders, Colliders: "colliders", size_of::<ColliderRecord>() as u64, Contents::Preserve, demand.colliders;
        collider_aabbs, ColliderAabbs: "broadphase aabbs", size_of::<AabbRecord>() as u64, Contents::Reset, demand.colliders;
        collider_owners, ColliderOwners: "collider owners", 4, Contents::Preserve, demand.colliders;
        body_edits, BodyEdits: "body edits", size_of::<BodyEditRecord>() as u64, Contents::Reset, demand.body_commands;
        body_edit_runs, BodyEditRuns: "body edit runs", size_of::<BodyEditRunRecord>() as u64, Contents::Reset, demand.body_commands;
        body_row_moves, BodyRowMoves: "body row moves", size_of::<RowMoveRecord>() as u64, Contents::Reset, demand.body_moves();
        body_fresh_rows, BodyFreshRows: "fresh body rows", BODY_FRESH_BYTES, Contents::Reset, demand.body_commands;
        body_state_scratch, BodyStateScratch: "body state scratch", BODY_FRESH_BYTES, Contents::Reset, demand.bodies;
        ccd_factor, CcdFactor: "ccd retreat factors", 4, Contents::Reset, demand.bodies;
        ccd_impact, CcdImpact: "ccd impacts", 16, Contents::Reset, demand.bodies;
        constraint_descriptors, ConstraintDescriptors: "constraint descriptors", size_of::<ConstraintDescriptorRecord>() as u64, Contents::Preserve, demand.constraints;
        constraint_rows, ConstraintRows: "constraint rows", size_of::<ConstraintRowsRecord>() as u64, Contents::Reset, demand.constraints;
        constraint_runtime, ConstraintRuntime: "constraint runtime", size_of::<ConstraintRuntimeRecord>() as u64, Contents::Preserve, demand.constraints;
        joint_filter_major, JointFilterMajor: "joint filter major", 4, Contents::Reset, demand.constraints;
        joint_filter_minor, JointFilterMinor: "joint filter minor", 4, Contents::Reset, demand.constraints;
        constraint_row_moves, ConstraintRowMoves: "constraint row moves", size_of::<RowMoveRecord>() as u64, Contents::Reset, demand.constraint_moves();
        constraint_fresh_rows, ConstraintFreshRows: "fresh constraint rows", size_of::<ConstraintRuntimeRecord>() as u64, Contents::Reset, demand.constraint_commands;
        constraint_scratch, ConstraintScratch: "constraint state scratch", size_of::<ConstraintRuntimeRecord>() as u64, Contents::Reset, demand.constraints;
        grid_entry_keys, GridEntryKeys: "grid entry keys", 4, Contents::Reset, demand.entries;
        grid_entry_colliders, GridEntryColliders: "grid entry colliders", 4, Contents::Reset, demand.entries;
        pair_major, PairMajor: "pairs major", 4, Contents::Reset, demand.pairs;
        pair_minor, PairMinor: "pairs minor", 4, Contents::Reset, demand.pairs;
        contacts_raw, ContactsRaw: "contacts raw", size_of::<ContactRecord>() as u64, Contents::Reset, demand.pairs;
        contact_valid, ContactValid: "contact valid", 4, Contents::Reset, demand.pairs;
        compact_ranks, CompactRanks: "compact ranks", 4, Contents::Reset, demand.pairs;
        compact_block_sums, CompactBlockSums: "compact block sums", 4, Contents::Reset, demand.compact_blocks();
        compact_block_offsets, CompactBlockOffsets: "compact block offsets", 4, Contents::Reset, demand.compact_blocks();
        contacts, Contacts: "contacts", size_of::<ContactRecord>() as u64, Contents::Reset, demand.pairs;
        contact_target_speeds, ContactTargetSpeeds: "contact target speeds", CONTACT_TARGET_BYTES, Contents::Reset, demand.pairs;
        contact_matched, ContactMatched: "contact matched", 4, Contents::Reset, demand.pairs;
        contact_archive, ContactArchive: "contact archive", size_of::<ContactRecord>() as u64, Contents::Preserve, demand.pairs;
        resting_contacts, RestingContacts: "resting contacts", size_of::<ContactRecord>() as u64, Contents::Preserve, demand.pairs;
        resting_live, RestingLive: "resting contact live", 4, Contents::Preserve, demand.pairs;
        resting_next, RestingNext: "resting contact next", 4, Contents::Preserve, demand.pairs;
        resting_free, RestingFree: "resting contact free", 4, Contents::PreserveSeeded(dynamis_layout::NO_SLOT), 1;
        resting_index_major, RestingIndexMajor: "resting index major", 4, Contents::Preserve, demand.pairs;
        resting_index_minor, RestingIndexMinor: "resting index minor", 4, Contents::Preserve, demand.pairs;
        resting_index_slots, RestingIndexSlots: "resting index slots", 4, Contents::Preserve, demand.pairs;
        solver_segments, SolverSegments: "solver segments", 4, Contents::Reset, SOLVER_BLOCK_KINDS;
        solver_a_bodies, SolverABodies: "solver block bodies", 4, Contents::Reset, demand.blocks();
        solver_a_payload, SolverAPayload: "solver block payload", 4, Contents::Reset, demand.blocks();
        solver_b_bodies, SolverBBodies: "solver block second bodies", 4, Contents::Reset, demand.blocks();
        solver_b_blocks, SolverBBlocks: "solver block second slots", 4, Contents::Reset, demand.blocks();
        solver_block_first_body, SolverBlockFirstBody: "solver block first owner", 4, Contents::Reset, demand.blocks();
        solver_block_second_body, SolverBlockSecondBody: "solver block second owner", 4, Contents::Reset, demand.blocks();
        solver_first_a, SolverFirstA: "solver first block", 4, Contents::Reset, demand.bodies;
        solver_first_b, SolverFirstB: "solver first second block", 4, Contents::Reset, demand.bodies;
        solver_block_counts, SolverBlockCounts: "solver block counts", 4, Contents::Reset, demand.bodies;
        solver_contact_counts, SolverContactCounts: "solver contact counts", 4, Contents::Reset, demand.bodies;
        solver_block_deltas, SolverBlockDeltas: "solver block deltas", SOLVER_DELTA_BYTES, Contents::Reset, demand.blocks();
        solver_block_corrections, SolverBlockCorrections: "solver block corrections", SOLVER_CORRECTION_BYTES, Contents::Reset, demand.blocks();
        solver_resolution, SolverResolution: "solver resolution", SOLVER_RESOLUTION_BYTES, Contents::Reset, demand.bodies;
        solver_contributions, SolverContributions: "solver contributions", 4, Contents::Reset, demand.bodies;
        island_parents, IslandParents: "island parents", 4, Contents::Reset, demand.bodies;
        island_state, IslandState: "island state", 4, Contents::Reset, demand.bodies;
        wake_flags, WakeFlags: "wake flags", 4, Contents::Reset, demand.bodies;
        shape_sources, ShapeSources: "shape sources", size_of::<ShapeSourceRecord>() as u64, Contents::Preserve, demand.shapes.sources;
        shape_vertices, ShapeVertices: "shape vertices", VERTEX_BYTES, Contents::Preserve, demand.shapes.vertices;
        shape_triangles, ShapeTriangles: "shape triangles", TRIANGLE_BYTES, Contents::Preserve, demand.shapes.triangles;
        shape_nodes, ShapeNodes: "shape bvh nodes", size_of::<BvhNodeRecord>() as u64, Contents::Preserve, demand.shapes.nodes;
        query_records, QueryRecords: "queries", QUERY_BYTES, Contents::Reset, demand.queries;
        query_results, QueryResults: "query results", QUERY_RESULT_BYTES, Contents::Reset, demand.queries;
        counters, Counters: "world counters", COUNTER_STRIDE, Contents::Preserve, COUNTER_COUNT as u32;
        events, Events: "contact events", size_of::<ContactEventRecord>() as u64, Contents::Reset, demand.events.saturating_mul(EVENT_SLOTS);
        sort_scratch_major, SortScratchMajor: "sort scratch major", 4, Contents::Reset, demand.sort;
        sort_scratch_minor, SortScratchMinor: "sort scratch minor", 4, Contents::Reset, demand.sort;
        sort_scratch_payload, SortScratchPayload: "sort scratch payload", 4, Contents::Reset, demand.sort;
        sort_dummy_payload, SortDummyPayload: "sort dummy payload", 4, Contents::Reset, demand.sort;
    }
}

impl RigidBuffers {
    pub(crate) fn shape_resources() -> [(&'static str, SlotRef); 4] {
        [
            ("shape_sources", StreamId::ShapeSources.whole()),
            ("shape_vertices", StreamId::ShapeVertices.whole()),
            ("shape_triangles", StreamId::ShapeTriangles.whole()),
            ("shape_nodes", StreamId::ShapeNodes.whole()),
        ]
    }

    pub(crate) fn counter(&self, slot: usize) -> SlotRef {
        assert!(
            slot < self.counters.slots() as usize,
            "counter slot {slot} is outside the counter stream"
        );
        StreamId::Counters.range(slot as u64 * COUNTER_STRIDE, 4)
    }

    pub(crate) fn entry_capacity(&self) -> u32 {
        self.grid_entry_keys.slots()
    }

    pub(crate) fn pair_capacity(&self) -> u32 {
        self.pair_major.slots()
    }

    pub(crate) fn event_capacity(&self) -> u32 {
        self.events.slots() / EVENT_SLOTS
    }

    pub(crate) fn resting_capacity(&self) -> u32 {
        self.resting_contacts.slots()
    }

    pub(crate) fn constraint_capacity(&self) -> u32 {
        self.constraint_runtime.slots()
    }

    pub(crate) fn block_capacity(&self) -> u32 {
        self.solver_a_payload.slots()
    }

    pub(crate) fn body_row_count(&self) -> u32 {
        self.body_states.slots()
    }

    pub(crate) fn collider_capacity(&self) -> u32 {
        self.collider_owners.slots()
    }

    pub(crate) fn sort_capacity(&self) -> u32 {
        self.sort_scratch_major.slots()
    }

    pub(crate) fn body_words(&self) -> u32 {
        key_words(self.body_row_count().max(1))
    }

    pub(crate) fn collider_words(&self) -> u32 {
        key_words(self.collider_capacity().max(1))
    }

    fn scratch(&self) -> [GpuSlot<'_>; 3] {
        [
            StreamId::SortScratchMajor.whole().resolve(self),
            StreamId::SortScratchMinor.whole().resolve(self),
            StreamId::SortScratchPayload.whole().resolve(self),
        ]
    }

    pub(crate) fn sort_lanes<'a>(
        &'a self,
        count: SlotRef,
        major: SlotRef,
        payload: SlotRef,
    ) -> SortChannels<'a> {
        let [scratch_major, scratch_minor, scratch_payload] = self.scratch();
        SortChannels {
            generation: self.generation,
            count: count.resolve(self),
            major: major.resolve(self),
            minor: payload.resolve(self),
            payload: payload.resolve(self),
            scratch_major,
            scratch_minor,
            scratch_payload,
        }
    }

    pub(crate) fn sort_keyed<'a>(
        &'a self,
        count: SlotRef,
        major: SlotRef,
        minor: SlotRef,
        payload: SlotRef,
    ) -> SortChannels<'a> {
        let [scratch_major, scratch_minor, scratch_payload] = self.scratch();
        SortChannels {
            generation: self.generation,
            count: count.resolve(self),
            major: major.resolve(self),
            minor: minor.resolve(self),
            payload: payload.resolve(self),
            scratch_major,
            scratch_minor,
            scratch_payload,
        }
    }

    pub(crate) fn sort_lanes_dual<'a>(
        &'a self,
        count: SlotRef,
        major: SlotRef,
        minor: SlotRef,
    ) -> SortChannels<'a> {
        let [scratch_major, scratch_minor, scratch_payload] = self.scratch();
        SortChannels {
            generation: self.generation,
            count: count.resolve(self),
            major: major.resolve(self),
            minor: minor.resolve(self),
            payload: StreamId::SortDummyPayload.whole().resolve(self),
            scratch_major,
            scratch_minor,
            scratch_payload,
        }
    }
}

pub(crate) struct ReadbackBuffers {
    pub(crate) pack: GpuBuffer,
    pub(crate) step: ReadbackRing,
    pub(crate) events: ReadbackRing,
    pub(crate) queries: ReadbackRing,
}

fn readback_sizes(demand: &Demand) -> (u64, u64, u64) {
    (
        COUNTER_BYTES + u64::from(demand.constraints) * size_of::<ConstraintRuntimeRecord>() as u64,
        u64::from(demand.events) * size_of::<ContactEventRecord>() as u64,
        u64::from(demand.queries) * QUERY_RESULT_BYTES,
    )
}

impl ReadbackBuffers {
    fn new(device: &Device, demand: &Demand) -> Self {
        let (pack_bytes, event_bytes, query_bytes) = readback_sizes(demand);
        Self {
            pack: GpuBuffer::new(device, "world readback pack", pack_bytes, PACK),
            step: ReadbackRing::new(device, "world readback", pack_bytes),
            events: ReadbackRing::new(device, "world events readback", event_bytes),
            queries: ReadbackRing::new(device, "query results readback", query_bytes),
        }
    }

    pub(crate) fn matches(&self, demand: &Demand) -> bool {
        let (pack_bytes, event_bytes, query_bytes) = readback_sizes(demand);
        self.pack.size() == pack_bytes
            && self.step.size() == pack_bytes
            && self.events.size() == event_bytes
            && self.queries.size() == query_bytes
    }

    fn reserve(&mut self, device: &Device, demand: &Demand) -> bool {
        let (pack_bytes, event_bytes, query_bytes) = readback_sizes(demand);
        if self.matches(demand) {
            return false;
        }
        assert!(
            self.step.is_idle() && self.events.is_idle() && self.queries.is_idle(),
            "readback buffers require drained rings before they reallocate"
        );
        if self.pack.size() != pack_bytes {
            self.pack = GpuBuffer::new(device, "world readback pack", pack_bytes, PACK);
            self.step = ReadbackRing::new(device, "world readback", pack_bytes);
        }
        if self.events.size() != event_bytes {
            self.events = ReadbackRing::new(device, "world events readback", event_bytes);
        }
        if self.queries.size() != query_bytes {
            self.queries = ReadbackRing::new(device, "query results readback", query_bytes);
        }
        true
    }
}
