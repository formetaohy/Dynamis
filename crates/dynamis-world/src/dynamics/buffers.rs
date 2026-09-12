use dynamis_gpu::{Contents, GpuBuffer, GpuSlot, ReadbackRing, Stream};
use dynamis_layout::{
    AabbRecord, BodyDescriptorRecord, BodyEditRecord, BodyEditRunRecord, BodyStateRecord,
    BvhNodeRecord, COUNTER_COUNT, COUNTER_STRIDE, ColliderRecord, ConstraintDescriptorRecord,
    ConstraintRuntimeRecord, ContactEventRecord, ContactRecord, MAX_HITS_PER_QUERY, QueryHitRecord,
    QueryRecord, QueryResultHeaderRecord, RowMoveRecord, SOLVER_BLOCK_CONSTRAINT,
    SOLVER_CLASS_BUCKETS, SOLVER_CLASS_COUNT, SOLVER_CLASS_ROW_WORDS, ShapeSourceRecord,
    StepParamsRecord, TriangleRecord,
};
use dynamis_sort::{SortChannels, key_words};
use std::mem::size_of;
use wgpu::{BufferUsages, Device, Queue};

pub(crate) const COMPACT_BLOCK: u32 = 256;

pub(crate) fn class_row_offset(class: u32) -> u64 {
    u64::from(class) * u64::from(SOLVER_CLASS_ROW_WORDS) * 4
}

pub(crate) const EVENT_SLOTS: u32 = ReadbackRing::DEPTH as u32 + 2;

const STREAM: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);
const PACK: BufferUsages = BufferUsages::COPY_DST.union(BufferUsages::COPY_SRC);

const COUNTER_BYTES: u64 = COUNTER_STRIDE * COUNTER_COUNT as u64;
const SOLVER_BLOCK_KINDS: u32 = SOLVER_BLOCK_CONSTRAINT + 1;
const SOLVER_DELTA_BYTES: u64 = 64;
const SOLVER_CORRECTION_BYTES: u64 = 32;
const SOLVER_RESOLUTION_BYTES: u64 = 16;
const BODY_FRESH_BYTES: u64 = 128;
pub(crate) const VERTEX_BYTES: u64 = size_of::<[f32; 4]>() as u64;
pub(crate) const TRIANGLE_BYTES: u64 = size_of::<TriangleRecord>() as u64;
const QUERY_BYTES: u64 = size_of::<QueryRecord>() as u64;
const QUERY_RESULT_BYTES: u64 = size_of::<QueryResultHeaderRecord>() as u64
    + MAX_HITS_PER_QUERY as u64 * size_of::<QueryHitRecord>() as u64;

const MOVE_ENTRIES_PER_COMMAND: u32 = 2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ShapeUse {
    pub(crate) sources: u32,
    pub(crate) vertices: u32,
    pub(crate) triangles: u32,
    pub(crate) nodes: u32,
}

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
    pub(crate) shapes: ShapeUse,
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

macro_rules! world_buffers {
    (
        with ($device:ident, $queue:ident, $demand:ident)
        extras { $( $extra:ident: $extra_ty:ty = $extra_init:expr, )* }
        streams { $( $name:ident: $label:literal, $stride:expr, $contents:expr, $slots:expr; )* }
    ) => {
        pub(crate) struct WorldBuffers {
            $( pub(crate) $extra: $extra_ty, )*
            $( pub(crate) $name: Stream, )*
        }

        impl WorldBuffers {
            pub(crate) fn new($device: &Device, $queue: &Queue, $demand: &Demand) -> Self {
                Self {
                    $( $extra: $extra_init, )*
                    $(
                        $name: Stream::new(
                            $device,
                            $queue,
                            $label,
                            $slots,
                            $stride,
                            STREAM,
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
                self.streams_match(demand) && self.readback.matches(demand)
            }

            pub(crate) fn reserve(
                &mut self,
                device: &Device,
                encoder: &mut wgpu::CommandEncoder,
                demand: &Demand,
            ) -> bool {
                let streams = self.reserve_streams(device, encoder, demand);
                let readback = self.readback.reserve(device, demand);
                streams | readback
            }
        }
    };
}

world_buffers! {
    with (device, queue, demand)
    extras {
        params: GpuBuffer = GpuBuffer::new(
            device,
            "step params",
            size_of::<StepParamsRecord>() as u64,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ),
        class_rounds: GpuBuffer = {
            let row_bytes = u64::from(SOLVER_CLASS_ROW_WORDS) * 4;
            let mut rounds = vec![0u32; (SOLVER_CLASS_COUNT * SOLVER_CLASS_ROW_WORDS) as usize];
            for round in 0..SOLVER_CLASS_COUNT {
                rounds[(round * SOLVER_CLASS_ROW_WORDS) as usize] = round;
            }
            let rounds_buffer = GpuBuffer::new(
                device,
                "solver class rounds",
                u64::from(SOLVER_CLASS_COUNT) * row_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            );
            rounds_buffer.write_at(queue, 0, bytemuck::cast_slice(&rounds));
            rounds_buffer
        },
        readback: ReadbackBuffers = ReadbackBuffers::new(device, demand),
    }
    streams {
        body_states: "body states", size_of::<BodyStateRecord>() as u64, Contents::Preserve, demand.bodies;
        body_activity: "body activity", 4, Contents::Reset, demand.bodies;
        body_row_of_id: "body row of id", 4, Contents::Preserve, demand.body_ids;
        body_descriptors: "body descriptors", size_of::<BodyDescriptorRecord>() as u64, Contents::Reset, demand.bodies;
        colliders: "colliders", size_of::<ColliderRecord>() as u64, Contents::Reset, demand.colliders;
        collider_aabbs: "broadphase aabbs", size_of::<AabbRecord>() as u64, Contents::Reset, demand.colliders;
        collider_owners: "collider owners", 4, Contents::Reset, demand.colliders;
        body_edits: "body edits", size_of::<BodyEditRecord>() as u64, Contents::Reset, demand.body_commands;
        body_edit_runs: "body edit runs", size_of::<BodyEditRunRecord>() as u64, Contents::Reset, demand.body_commands;
        body_row_moves: "body row moves", size_of::<RowMoveRecord>() as u64, Contents::Reset, demand.body_moves();
        body_fresh_rows: "fresh body rows", BODY_FRESH_BYTES, Contents::Reset, demand.body_commands;
        body_state_scratch: "body state scratch", BODY_FRESH_BYTES, Contents::Reset, demand.bodies;
        ccd_factor: "ccd retreat factors", 4, Contents::Reset, demand.bodies;
        ccd_impact: "ccd impacts", 16, Contents::Reset, demand.bodies;
        constraint_descriptors: "constraint descriptors", size_of::<ConstraintDescriptorRecord>() as u64, Contents::Reset, demand.constraints;
        constraint_runtime: "constraint runtime", size_of::<ConstraintRuntimeRecord>() as u64, Contents::Preserve, demand.constraints;
        joint_filter_major: "joint filter major", 4, Contents::Reset, demand.constraints;
        joint_filter_minor: "joint filter minor", 4, Contents::Reset, demand.constraints;
        constraint_row_moves: "constraint row moves", size_of::<RowMoveRecord>() as u64, Contents::Reset, demand.constraint_moves();
        constraint_fresh_rows: "fresh constraint rows", size_of::<ConstraintRuntimeRecord>() as u64, Contents::Reset, demand.constraint_commands;
        constraint_scratch: "constraint state scratch", size_of::<ConstraintRuntimeRecord>() as u64, Contents::Reset, demand.constraints;
        grid_entry_keys: "grid entry keys", 4, Contents::Reset, demand.entries;
        grid_entry_colliders: "grid entry colliders", 4, Contents::Reset, demand.entries;
        pair_major: "pairs major", 4, Contents::Reset, demand.pairs;
        pair_minor: "pairs minor", 4, Contents::Reset, demand.pairs;
        contacts_raw: "contacts raw", size_of::<ContactRecord>() as u64, Contents::Reset, demand.pairs;
        contact_valid: "contact valid", 4, Contents::Reset, demand.pairs;
        compact_ranks: "compact ranks", 4, Contents::Reset, demand.pairs;
        compact_block_sums: "compact block sums", 4, Contents::Reset, demand.compact_blocks();
        compact_block_offsets: "compact block offsets", 4, Contents::Reset, demand.compact_blocks();
        contacts: "contacts", size_of::<ContactRecord>() as u64, Contents::Reset, demand.pairs;
        contact_matched: "contact matched", 4, Contents::Reset, demand.pairs;
        contact_archive: "contact archive", size_of::<ContactRecord>() as u64, Contents::Preserve, demand.pairs;
        resting_contacts: "resting contacts", size_of::<ContactRecord>() as u64, Contents::Preserve, demand.pairs;
        resting_live: "resting contact live", 4, Contents::Preserve, demand.pairs;
        resting_next: "resting contact next", 4, Contents::Preserve, demand.pairs;
        resting_free: "resting contact free", 4, Contents::PreserveSeeded(dynamis_layout::NO_SLOT), 1;
        resting_index_major: "resting index major", 4, Contents::Preserve, demand.pairs;
        resting_index_minor: "resting index minor", 4, Contents::Preserve, demand.pairs;
        resting_index_slots: "resting index slots", 4, Contents::Preserve, demand.pairs;
        solver_segments: "solver segments", 4, Contents::Reset, SOLVER_BLOCK_KINDS;
        solver_overflow: "solver overflow blocks", 4, Contents::Reset, 1;
        solver_a_bodies: "solver block bodies", 4, Contents::Reset, demand.blocks();
        solver_a_payload: "solver block payload", 4, Contents::Reset, demand.blocks();
        solver_b_bodies: "solver block second bodies", 4, Contents::Reset, demand.blocks();
        solver_b_blocks: "solver block second slots", 4, Contents::Reset, demand.blocks();
        solver_block_first_body: "solver block first owner", 4, Contents::Reset, demand.blocks();
        solver_block_second_body: "solver block second owner", 4, Contents::Reset, demand.blocks();
        solver_class_tokens: "solver class tokens", 4, Contents::Reset, demand.blocks();
        solver_class_blocks: "solver class blocks", 4, Contents::Reset, demand.blocks();
        solver_class_counts: "solver class counts", 4, Contents::Reset, SOLVER_CLASS_BUCKETS;
        solver_class_cursors: "solver class cursors", 4, Contents::Reset, SOLVER_CLASS_BUCKETS;
        solver_class_bounds: "solver class bounds", 4, Contents::Reset,
            SOLVER_CLASS_BUCKETS * SOLVER_CLASS_ROW_WORDS;
        solver_first_a: "solver first block", 4, Contents::Reset, demand.bodies;
        solver_first_b: "solver first second block", 4, Contents::Reset, demand.bodies;
        solver_overflow_first_a: "solver overflow first block", 4, Contents::Reset, demand.bodies;
        solver_overflow_first_b: "solver overflow first second block", 4, Contents::Reset, demand.bodies;
        solver_block_counts: "solver block counts", 4, Contents::Reset, demand.bodies;
        solver_contact_counts: "solver contact counts", 4, Contents::Reset, demand.bodies;
        solver_block_deltas: "solver block deltas", SOLVER_DELTA_BYTES, Contents::Reset, demand.blocks();
        solver_block_corrections: "solver block corrections", SOLVER_CORRECTION_BYTES, Contents::Reset, demand.blocks();
        solver_resolution: "solver resolution", SOLVER_RESOLUTION_BYTES, Contents::Reset, demand.bodies;
        solver_contributions: "solver contributions", 4, Contents::Reset, demand.bodies;
        island_parents: "island parents", 4, Contents::Reset, demand.bodies;
        island_state: "island state", 4, Contents::Reset, demand.bodies;
        wake_flags: "wake flags", 4, Contents::Reset, demand.bodies;
        shape_sources: "shape sources", size_of::<ShapeSourceRecord>() as u64, Contents::Reset, demand.shapes.sources;
        shape_vertices: "shape vertices", VERTEX_BYTES, Contents::Reset, demand.shapes.vertices;
        shape_triangles: "shape triangles", TRIANGLE_BYTES, Contents::Reset, demand.shapes.triangles;
        shape_nodes: "shape bvh nodes", size_of::<BvhNodeRecord>() as u64, Contents::Reset, demand.shapes.nodes;
        query_records: "queries", QUERY_BYTES, Contents::Reset, demand.queries;
        query_results: "query results", QUERY_RESULT_BYTES, Contents::Reset, demand.queries;
        counters: "world counters", COUNTER_STRIDE, Contents::Preserve, COUNTER_COUNT as u32;
        events: "contact events", size_of::<ContactEventRecord>() as u64, Contents::Reset, demand.events.saturating_mul(EVENT_SLOTS);
        sort_scratch_major: "sort scratch major", 4, Contents::Reset, demand.sort;
        sort_scratch_minor: "sort scratch minor", 4, Contents::Reset, demand.sort;
        sort_scratch_payload: "sort scratch payload", 4, Contents::Reset, demand.sort;
        sort_dummy_major: "sort dummy major", 4, Contents::Reset, demand.sort;
        sort_dummy_minor: "sort dummy minor", 4, Contents::Reset, demand.sort;
        sort_dummy_payload: "sort dummy payload", 4, Contents::Reset, demand.sort;
    }
}

impl WorldBuffers {
    pub(crate) fn counter(&self, slot: usize) -> GpuSlot<'_> {
        GpuSlot::range(self.counters.gpu(), slot as u64 * COUNTER_STRIDE, 4)
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

    pub(crate) fn contact_capacity(&self) -> u32 {
        self.contacts.slots()
    }

    pub(crate) fn archive_capacity(&self) -> u32 {
        self.contact_archive.slots()
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

    pub(crate) fn overflow_count(&self) -> GpuSlot<'_> {
        self.solver_overflow.slot()
    }

    pub(crate) fn class_round(&self, round: u32) -> GpuSlot<'_> {
        GpuSlot::range(&self.class_rounds, class_row_offset(round), 4)
    }

    pub(crate) fn class_range(&self, class: u32) -> GpuSlot<'_> {
        GpuSlot::range(self.solver_class_bounds.gpu(), class_row_offset(class), 8)
    }

    pub(crate) fn body_words(&self) -> u32 {
        key_words(self.body_row_count().max(1))
    }

    pub(crate) fn collider_words(&self) -> u32 {
        key_words(self.collider_capacity().max(1))
    }

    pub(crate) fn sort_lanes<'a>(
        &'a self,
        count: GpuSlot<'a>,
        major: &'a Stream,
        payload: &'a Stream,
    ) -> SortChannels<'a> {
        SortChannels {
            count,
            major: major.gpu(),
            minor: payload.gpu(),
            payload: payload.gpu(),
            scratch_major: self.sort_scratch_major.gpu(),
            scratch_minor: self.sort_scratch_payload.gpu(),
            scratch_payload: self.sort_scratch_payload.gpu(),
        }
    }

    pub(crate) fn sort_keyed<'a>(
        &'a self,
        count: GpuSlot<'a>,
        major: &'a Stream,
        minor: &'a Stream,
        payload: &'a Stream,
    ) -> SortChannels<'a> {
        SortChannels {
            count,
            major: major.gpu(),
            minor: minor.gpu(),
            payload: payload.gpu(),
            scratch_major: self.sort_scratch_major.gpu(),
            scratch_minor: self.sort_scratch_minor.gpu(),
            scratch_payload: self.sort_scratch_payload.gpu(),
        }
    }

    pub(crate) fn sort_lanes_dual<'a>(
        &'a self,
        count: GpuSlot<'a>,
        major: &'a Stream,
        minor: &'a Stream,
    ) -> SortChannels<'a> {
        SortChannels {
            count,
            major: major.gpu(),
            minor: minor.gpu(),
            payload: self.sort_dummy_payload.gpu(),
            scratch_major: self.sort_scratch_major.gpu(),
            scratch_minor: self.sort_scratch_minor.gpu(),
            scratch_payload: self.sort_scratch_payload.gpu(),
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
