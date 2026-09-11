use crate::capacity::{Reservation, ShapeReservation};
use dynamis_gpu::{DispatchTable, GpuBuffer, GpuSlot, ReadbackRing};
use dynamis_layout::{
    AabbRecord, BodyDescriptorRecord, BodyEditRecord, BodyEditRun, BodyStateRecord, BvhNodeRecord,
    COUNTER_COUNT, COUNTER_STRIDE, ColliderRecord, ConstraintDescriptorRecord,
    ConstraintRuntimeRecord, ContactEventRecord, ContactRecord, MAX_HITS_PER_QUERY, NO_SLOT,
    QueryHitRecord, QueryRecord, QueryResultHeader, RowMoveRecord, SOLVER_BLOCK_CONSTRAINT,
    ShapeSourceRecord, SimParamsRecord,
};
use dynamis_model::MAX_COLLIDERS_PER_BODY;
use dynamis_sort::{SortChannels, key_words};
use std::mem::size_of;
use wgpu::{BufferUsages, Device};

pub(crate) const COMPACT_BLOCK: u32 = 256;

pub(crate) const EVENT_SLOTS: u32 = ReadbackRing::DEPTH as u32;

const STREAM: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);
const PACK: BufferUsages = BufferUsages::COPY_DST.union(BufferUsages::COPY_SRC);

const COUNTER_BYTES: u64 = COUNTER_STRIDE * COUNTER_COUNT as u64;
const DELTA_BYTES: u64 = 64;
const CORRECTION_BYTES: u64 = 32;
const SOLVER_BLOCK_KINDS: u32 = SOLVER_BLOCK_CONSTRAINT + 1;
pub(crate) const VERTEX_BYTES: u64 = size_of::<[f32; 4]>() as u64;
pub(crate) const TRIANGLE_BYTES: u64 = size_of::<[u32; 4]>() as u64;
const QUERY_BYTES: u64 = size_of::<QueryRecord>() as u64;
const QUERY_RESULT_BYTES: u64 = size_of::<QueryResultHeader>() as u64
    + MAX_HITS_PER_QUERY as u64 * size_of::<QueryHitRecord>() as u64;

pub(crate) struct Lanes {
    pub(crate) major: GpuBuffer,
    pub(crate) minor: GpuBuffer,
    pub(crate) payload: GpuBuffer,
}

pub(crate) struct GridLanes {
    pub(crate) cells: GpuBuffer,
    pub(crate) colliders: GpuBuffer,
}

impl GridLanes {
    fn new(device: &Device, label: &str, lanes: u32) -> Self {
        Self {
            cells: GpuBuffer::new(device, &format!("{label} cells"), lanes as u64 * 4, STREAM),
            colliders: GpuBuffer::new(
                device,
                &format!("{label} colliders"),
                lanes as u64 * 4,
                STREAM,
            ),
        }
    }
}

pub(crate) struct PairLanes {
    pub(crate) major: GpuBuffer,
    pub(crate) minor: GpuBuffer,
}

impl PairLanes {
    fn new(device: &Device, label: &str, lanes: u32) -> Self {
        Self {
            major: GpuBuffer::new(device, &format!("{label} major"), lanes as u64 * 4, STREAM),
            minor: GpuBuffer::new(device, &format!("{label} minor"), lanes as u64 * 4, STREAM),
        }
    }
}

impl Lanes {
    fn new(device: &Device, label: &str, lanes: u32) -> Self {
        let lane = |name: &str| {
            GpuBuffer::new(device, &format!("{label} {name}"), lanes as u64 * 4, STREAM)
        };
        Self {
            major: lane("major"),
            minor: lane("minor"),
            payload: lane("payload"),
        }
    }
}

pub(crate) struct BodyBuffers {
    pub(crate) states: GpuBuffer,
    pub(crate) activity: GpuBuffer,
    pub(crate) rows: GpuBuffer,
    pub(crate) descriptors: GpuBuffer,
    pub(crate) colliders: GpuBuffer,
    pub(crate) aabbs: GpuBuffer,
    pub(crate) edits: GpuBuffer,
    pub(crate) edit_runs: GpuBuffer,
    pub(crate) row_moves: GpuBuffer,
    pub(crate) fresh_rows: GpuBuffer,
    pub(crate) state_scratch: GpuBuffer,
}

pub(crate) struct SolverBuffers {
    pub(crate) segments: GpuBuffer,
    pub(crate) a_bodies: GpuBuffer,
    pub(crate) a_payload: GpuBuffer,
    pub(crate) b_bodies: GpuBuffer,
    pub(crate) b_blocks: GpuBuffer,
    pub(crate) first_a: GpuBuffer,
    pub(crate) first_b: GpuBuffer,
    pub(crate) counts: GpuBuffer,
    pub(crate) contact_counts: GpuBuffer,
    pub(crate) deltas: GpuBuffer,
    pub(crate) corrections: GpuBuffer,
}

pub(crate) struct ConstraintBuffers {
    pub(crate) descriptors: GpuBuffer,
    pub(crate) runtime: GpuBuffer,
    pub(crate) joint_major: GpuBuffer,
    pub(crate) joint_minor: GpuBuffer,
    pub(crate) row_moves: GpuBuffer,
    pub(crate) fresh_rows: GpuBuffer,
    pub(crate) scratch: GpuBuffer,
}

pub(crate) struct ContactBuffers {
    pub(crate) entries: GridLanes,
    pub(crate) pairs: PairLanes,
    pub(crate) resting_index: Lanes,
    pub(crate) large_bodies: GpuBuffer,
    pub(crate) raw: GpuBuffer,
    pub(crate) valid: GpuBuffer,
    pub(crate) compact_ranks: GpuBuffer,
    pub(crate) compact_sums: GpuBuffer,
    pub(crate) compact_offsets: GpuBuffer,
    pub(crate) manifolds: GpuBuffer,
    pub(crate) contact_matched: GpuBuffer,
    pub(crate) archive: GpuBuffer,
    pub(crate) resting: GpuBuffer,
    pub(crate) resting_live: GpuBuffer,
    pub(crate) resting_next: GpuBuffer,
    pub(crate) resting_free: GpuBuffer,
}

pub(crate) struct IslandBuffers {
    pub(crate) parents: GpuBuffer,
    pub(crate) state: GpuBuffer,
    pub(crate) wake_flags: GpuBuffer,
}

pub(crate) struct ShapeBuffers {
    pub(crate) sources: GpuBuffer,
    pub(crate) vertices: GpuBuffer,
    pub(crate) triangles: GpuBuffer,
    pub(crate) nodes: GpuBuffer,
}

pub(crate) struct QueryBuffers {
    pub(crate) records: GpuBuffer,
    pub(crate) results: GpuBuffer,
}

pub(crate) struct SortBuffers {
    pub(crate) scratch: Lanes,
    pub(crate) dummy: Lanes,
}

pub(crate) struct ReadbackBuffers {
    pub(crate) pack: GpuBuffer,
    pub(crate) step: ReadbackRing,
    pub(crate) events: ReadbackRing,
    pub(crate) queries: ReadbackRing,
}

pub(crate) struct WorldBuffers {
    pub(crate) params: GpuBuffer,

    pub(crate) counters: GpuBuffer,

    pub(crate) dispatch: DispatchTable,
    pub(crate) bodies: BodyBuffers,
    pub(crate) constraints: ConstraintBuffers,
    pub(crate) contacts: ContactBuffers,
    pub(crate) solver: SolverBuffers,
    pub(crate) islands: IslandBuffers,
    pub(crate) shapes: ShapeBuffers,
    pub(crate) events: GpuBuffer,
    pub(crate) queries: QueryBuffers,
    pub(crate) sort: SortBuffers,
    pub(crate) readback: ReadbackBuffers,
}

impl WorldBuffers {
    pub(crate) fn new(
        device: &Device,
        queue: &wgpu::Queue,
        plan: &Reservation,
        shapes: &ShapeReservation,
    ) -> Self {
        let bodies = plan.bodies;
        let colliders = plan.colliders();
        let constraints = plan.constraints;
        let limits = device.limits();
        let storage_limit = limits.max_storage_buffer_binding_size;
        let buffer_limit = limits.max_buffer_size;
        let checked_size = |label: &str, count: u32, stride: u64| {
            let bytes = u64::from(count)
                .checked_mul(stride)
                .unwrap_or_else(|| panic!("{label} exceeds the device address space"));
            assert!(
                bytes <= storage_limit,
                "{label} requires {bytes} bytes but the device storage binding limit is {storage_limit}"
            );
            bytes
        };
        let lanes = |label: &str, count: u32| {
            GpuBuffer::new(device, label, checked_size(label, count, 4), STREAM)
        };
        let rows = |label: &str, count: u32, stride: u64| {
            GpuBuffer::new(device, label, checked_size(label, count, stride), STREAM)
        };
        let queries = plan.queries;
        let readback_bytes =
            COUNTER_BYTES + constraints as u64 * size_of::<ConstraintRuntimeRecord>() as u64;
        let event_segment_bytes = plan.events as u64 * size_of::<ContactEventRecord>() as u64;
        assert!(
            event_segment_bytes <= buffer_limit,
            "event readback requires {event_segment_bytes} bytes but the device buffer limit is {buffer_limit}"
        );
        let event_bytes = event_segment_bytes * EVENT_SLOTS as u64;
        let query_readback_bytes = queries as u64 * QUERY_RESULT_BYTES;
        assert!(
            query_readback_bytes <= buffer_limit,
            "query readback requires {query_readback_bytes} bytes but the device buffer limit is {buffer_limit}"
        );
        let resting_free = GpuBuffer::new(device, "resting contact free", 4, STREAM);
        resting_free.write(queue, bytemuck::cast_slice(&[NO_SLOT]));
        Self {
            params: GpuBuffer::new(
                device,
                "sim params",
                size_of::<SimParamsRecord>() as u64,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            ),
            counters: GpuBuffer::zeroed(device, "sim counters", COUNTER_BYTES, STREAM),
            dispatch: DispatchTable::new(device, "sim dispatch", crate::pipeline::DISPATCH_SLOTS),
            bodies: BodyBuffers {
                states: rows("body states", bodies, size_of::<BodyStateRecord>() as u64),
                activity: lanes("body activity", bodies),
                rows: lanes("body rows", bodies),
                descriptors: rows(
                    "body descriptors",
                    bodies,
                    size_of::<BodyDescriptorRecord>() as u64,
                ),
                colliders: rows("colliders", colliders, size_of::<ColliderRecord>() as u64),
                aabbs: rows(
                    "broadphase aabbs",
                    colliders,
                    size_of::<AabbRecord>() as u64,
                ),
                edits: rows(
                    "body edits",
                    plan.body_commands,
                    size_of::<BodyEditRecord>() as u64,
                ),
                edit_runs: rows(
                    "body edit runs",
                    plan.body_commands,
                    size_of::<BodyEditRun>() as u64,
                ),
                row_moves: rows(
                    "body row moves",
                    plan.body_moves(),
                    size_of::<RowMoveRecord>() as u64,
                ),
                fresh_rows: rows("fresh body rows", plan.body_commands, 128),
                state_scratch: rows("body state scratch", bodies, 128),
            },
            constraints: ConstraintBuffers {
                descriptors: rows(
                    "constraint descriptors",
                    constraints,
                    size_of::<ConstraintDescriptorRecord>() as u64,
                ),
                runtime: rows(
                    "constraint runtime",
                    constraints,
                    size_of::<ConstraintRuntimeRecord>() as u64,
                ),
                joint_major: lanes("joint filter major", constraints),
                joint_minor: lanes("joint filter minor", constraints),
                row_moves: rows(
                    "constraint row moves",
                    plan.constraint_moves(),
                    size_of::<RowMoveRecord>() as u64,
                ),
                fresh_rows: rows(
                    "fresh constraint rows",
                    plan.constraint_commands,
                    size_of::<ConstraintRuntimeRecord>() as u64,
                ),
                scratch: rows(
                    "constraint state scratch",
                    constraints,
                    size_of::<ConstraintRuntimeRecord>() as u64,
                ),
            },
            solver: SolverBuffers {
                segments: lanes("solver segments", SOLVER_BLOCK_KINDS),
                a_bodies: lanes("solver block bodies", plan.blocks()),
                a_payload: lanes("solver block payload", plan.blocks()),
                b_bodies: lanes("solver block second bodies", plan.blocks()),
                b_blocks: lanes("solver block second slots", plan.blocks()),
                first_a: lanes("solver first block", bodies),
                first_b: lanes("solver first second block", bodies),
                counts: lanes("solver block counts", bodies),
                contact_counts: lanes("solver contact counts", bodies),
                deltas: rows("solver block deltas", plan.blocks(), DELTA_BYTES),
                corrections: rows("solver block corrections", plan.blocks(), CORRECTION_BYTES),
            },
            contacts: ContactBuffers {
                resting_index: Lanes::new(device, "resting index", plan.pairs),
                entries: GridLanes::new(device, "grid entries", plan.entries),
                pairs: PairLanes::new(device, "pairs", plan.pairs),
                large_bodies: lanes("large colliders", colliders),
                raw: rows(
                    "contacts raw",
                    plan.pairs,
                    size_of::<ContactRecord>() as u64,
                ),
                valid: lanes("contact valid", plan.pairs),
                compact_ranks: lanes("compact ranks", plan.pairs),
                compact_sums: lanes("compact block sums", plan.pairs.div_ceil(COMPACT_BLOCK)),
                compact_offsets: lanes("compact block offsets", plan.pairs.div_ceil(COMPACT_BLOCK)),
                manifolds: rows("contacts", plan.pairs, size_of::<ContactRecord>() as u64),
                contact_matched: lanes("contact matched", plan.pairs),
                archive: rows(
                    "contact archive",
                    plan.pairs,
                    size_of::<ContactRecord>() as u64,
                ),
                resting: rows(
                    "resting contacts",
                    plan.pairs,
                    size_of::<ContactRecord>() as u64,
                ),
                resting_live: lanes("resting contact live", plan.pairs),
                resting_next: lanes("resting contact next", plan.pairs),
                resting_free,
            },
            islands: IslandBuffers {
                parents: lanes("island parents", bodies),
                state: lanes("island state", bodies),
                wake_flags: lanes("wake flags", bodies),
            },
            shapes: ShapeBuffers {
                sources: rows(
                    "shape sources",
                    shapes.sources,
                    size_of::<ShapeSourceRecord>() as u64,
                ),
                vertices: rows("shape vertices", shapes.vertices, VERTEX_BYTES),
                triangles: rows("shape triangles", shapes.triangles, TRIANGLE_BYTES),
                nodes: rows(
                    "shape bvh nodes",
                    shapes.nodes,
                    size_of::<BvhNodeRecord>() as u64,
                ),
            },
            events: GpuBuffer::new(device, "contact events", event_bytes, STREAM),
            queries: QueryBuffers {
                records: rows("queries", queries, QUERY_BYTES),
                results: rows("query results", queries, QUERY_RESULT_BYTES),
            },
            sort: SortBuffers {
                scratch: Lanes::new(device, "sort scratch", plan.sort()),
                dummy: Lanes::new(device, "sort dummy", plan.sort()),
            },
            readback: ReadbackBuffers {
                pack: GpuBuffer::new(device, "sim readback pack", readback_bytes, PACK),
                step: ReadbackRing::new(device, "sim readback", readback_bytes),
                events: ReadbackRing::new(device, "sim events readback", event_segment_bytes),
                queries: ReadbackRing::new(device, "query results readback", query_readback_bytes),
            },
        }
    }

    pub(crate) fn counter(&self, slot: usize) -> GpuSlot<'_> {
        GpuSlot::range(&self.counters, slot as u64 * COUNTER_STRIDE, 4)
    }

    pub(crate) fn collider_row(&self) -> u64 {
        size_of::<ColliderRecord>() as u64 * MAX_COLLIDERS_PER_BODY as u64
    }

    pub(crate) fn aabb_row(&self) -> u64 {
        size_of::<AabbRecord>() as u64 * MAX_COLLIDERS_PER_BODY as u64
    }

    pub(crate) fn descriptor_row(&self) -> u64 {
        size_of::<BodyDescriptorRecord>() as u64
    }

    pub(crate) fn constraint_row(&self) -> u64 {
        size_of::<ConstraintDescriptorRecord>() as u64
    }

    pub(crate) fn body_rows(&self) -> u32 {
        (self.bodies.states.size() / size_of::<BodyStateRecord>() as u64) as u32
    }

    pub(crate) fn collider_rows(&self) -> u32 {
        (self.bodies.colliders.size() / size_of::<ColliderRecord>() as u64) as u32
    }

    pub(crate) fn body_words(&self) -> u32 {
        key_words(self.body_rows().max(1))
    }

    pub(crate) fn collider_words(&self) -> u32 {
        key_words(self.collider_rows().max(1))
    }

    pub(crate) fn sort_lanes<'a>(
        &'a self,
        count: GpuSlot<'a>,
        major: &'a GpuBuffer,
        payload: &'a GpuBuffer,
    ) -> SortChannels<'a> {
        SortChannels {
            count,
            major,
            minor: payload,
            payload,
            scratch_major: &self.sort.scratch.major,
            scratch_minor: &self.sort.scratch.payload,
            scratch_payload: &self.sort.scratch.payload,
        }
    }

    pub(crate) fn sort_keyed<'a>(
        &'a self,
        count: GpuSlot<'a>,
        major: &'a GpuBuffer,
        minor: &'a GpuBuffer,
        payload: &'a GpuBuffer,
    ) -> SortChannels<'a> {
        SortChannels {
            count,
            major,
            minor,
            payload,
            scratch_major: &self.sort.scratch.major,
            scratch_minor: &self.sort.scratch.minor,
            scratch_payload: &self.sort.scratch.payload,
        }
    }

    pub(crate) fn sort_lanes_dual<'a>(
        &'a self,
        count: GpuSlot<'a>,
        major: &'a GpuBuffer,
        minor: &'a GpuBuffer,
    ) -> SortChannels<'a> {
        SortChannels {
            count,
            major,
            minor,
            payload: &self.sort.dummy.payload,
            scratch_major: &self.sort.scratch.major,
            scratch_minor: &self.sort.scratch.minor,
            scratch_payload: &self.sort.scratch.payload,
        }
    }
}
