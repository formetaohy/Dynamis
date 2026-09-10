use crate::capacity::{Reservation, ShapeReservation};
use dynamis_gpu::{DispatchTable, GpuBuffer, GpuReadback, GpuSlot};
use dynamis_layout::{
    AabbRecord, BodyCommandRecord, BodyDescriptorRecord, BodyStateRecord, BvhNodeRecord,
    COUNTER_COUNT, COUNTER_STRIDE, ColliderRecord, ConstraintDescriptorRecord,
    ConstraintRuntimeRecord, ContactEventRecord, ContactRecord, MAX_HITS_PER_QUERY, QueryHitRecord,
    QueryRecord, QueryResultHeader, ShapeSourceRecord, SimParamsRecord,
};
use dynamis_model::MAX_COLLIDERS_PER_BODY;
use std::mem::size_of;
use wgpu::{BufferUsages, Device};

pub(crate) const COMPACT_BLOCK: u32 = 256;

pub(crate) const EVENT_SLOTS: u32 = GpuReadback::DEPTH as u32;

const STREAM: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);
const PACK: BufferUsages = BufferUsages::COPY_DST.union(BufferUsages::COPY_SRC);

const COUNTER_BYTES: u64 = COUNTER_STRIDE * COUNTER_COUNT as u64;
const DELTA_BYTES: u64 = 64;
pub(crate) const VERTEX_BYTES: u64 = size_of::<[f32; 4]>() as u64;
pub(crate) const TRIANGLE_BYTES: u64 = size_of::<[u32; 4]>() as u64;
const QUERY_BYTES: u64 = size_of::<QueryRecord>() as u64;
const QUERY_RESULT_BYTES: u64 = size_of::<QueryResultHeader>() as u64
    + MAX_HITS_PER_QUERY as u64 * size_of::<QueryHitRecord>() as u64;

pub(crate) struct ChannelSlots {
    pub(crate) keys_hi: GpuBuffer,
    pub(crate) keys_lo: GpuBuffer,
    pub(crate) values: GpuBuffer,
}

impl ChannelSlots {
    fn new(device: &Device, label: &str, lanes: u32) -> Self {
        let lane = |name: &str| {
            GpuBuffer::new(device, &format!("{label} {name}"), lanes as u64 * 4, STREAM)
        };
        Self {
            keys_hi: lane("keys hi"),
            keys_lo: lane("keys lo"),
            values: lane("values"),
        }
    }
}

pub(crate) struct BodyBuffers {
    pub(crate) states: GpuBuffer,
    pub(crate) descriptors: GpuBuffer,
    pub(crate) colliders: GpuBuffer,
    pub(crate) aabbs: GpuBuffer,
    pub(crate) commands: GpuBuffer,
    pub(crate) command_first: GpuBuffer,
    pub(crate) row_src: GpuBuffer,
    pub(crate) row_fresh: GpuBuffer,
    pub(crate) fresh_states: GpuBuffer,
    pub(crate) state_scratch: GpuBuffer,
}

pub(crate) struct ConstraintBuffers {
    pub(crate) descriptors: GpuBuffer,
    pub(crate) runtime: GpuBuffer,
    pub(crate) a_keys: GpuBuffer,
    pub(crate) a_values: GpuBuffer,
    pub(crate) b_keys: GpuBuffer,
    pub(crate) b_values: GpuBuffer,
    pub(crate) first_a: GpuBuffer,
    pub(crate) first_b: GpuBuffer,
    pub(crate) deltas: GpuBuffer,
    pub(crate) joint_hi: GpuBuffer,
    pub(crate) joint_lo: GpuBuffer,
    pub(crate) row_src: GpuBuffer,
    pub(crate) row_fresh: GpuBuffer,
    pub(crate) fresh: GpuBuffer,
    pub(crate) scratch: GpuBuffer,
}

pub(crate) struct ContactBuffers {
    pub(crate) entries: ChannelSlots,
    pub(crate) pairs: ChannelSlots,
    pub(crate) large_bodies: GpuBuffer,
    pub(crate) raw: GpuBuffer,
    pub(crate) valid: GpuBuffer,
    pub(crate) a_body: GpuBuffer,
    pub(crate) compact_ranks: GpuBuffer,
    pub(crate) compact_sums: GpuBuffer,
    pub(crate) compact_offsets: GpuBuffer,
    pub(crate) manifolds: GpuBuffer,
    pub(crate) previous: GpuBuffer,
    pub(crate) b_keys: GpuBuffer,
    pub(crate) b_values: GpuBuffer,
    pub(crate) first_a: GpuBuffer,
    pub(crate) first_b: GpuBuffer,
    pub(crate) deltas: GpuBuffer,
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
    pub(crate) values: GpuBuffer,
    pub(crate) pad: GpuBuffer,
    pub(crate) scratch: ChannelSlots,
}

pub(crate) struct ReadbackBuffers {
    pub(crate) pack: GpuBuffer,
    pub(crate) step: GpuReadback,
    pub(crate) events: GpuReadback,
    pub(crate) queries: GpuReadback,
}

pub(crate) struct WorldBuffers {
    pub(crate) params: GpuBuffer,

    pub(crate) counters: GpuBuffer,

    pub(crate) dispatch: DispatchTable,
    pub(crate) bodies: BodyBuffers,
    pub(crate) constraints: ConstraintBuffers,
    pub(crate) contacts: ContactBuffers,
    pub(crate) islands: IslandBuffers,
    pub(crate) shapes: ShapeBuffers,
    pub(crate) events: GpuBuffer,
    pub(crate) queries: QueryBuffers,
    pub(crate) sort: SortBuffers,
    pub(crate) readback: ReadbackBuffers,
}

impl WorldBuffers {
    pub(crate) fn new(device: &Device, plan: &Reservation, shapes: &ShapeReservation) -> Self {
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
        Self {
            params: GpuBuffer::new(
                device,
                "sim params",
                size_of::<SimParamsRecord>() as u64,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            ),
            counters: GpuBuffer::new(device, "sim counters", COUNTER_BYTES, STREAM),
            dispatch: DispatchTable::new(device, "sim dispatch", crate::pipeline::DISPATCH_SLOTS),
            bodies: BodyBuffers {
                states: rows("body states", bodies, size_of::<BodyStateRecord>() as u64),
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
                commands: rows(
                    "body edits",
                    plan.body_commands,
                    size_of::<BodyCommandRecord>() as u64,
                ),
                command_first: lanes("body edit first", bodies),
                row_src: lanes("body row src", bodies),
                row_fresh: lanes("body row fresh", bodies),
                fresh_states: rows("fresh body rows", plan.body_commands, 128),
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
                a_keys: lanes("constraint gather a keys", constraints),
                a_values: lanes("constraint gather a values", constraints),
                b_keys: lanes("constraint gather b keys", constraints),
                b_values: lanes("constraint gather b values", constraints),
                first_a: lanes("constraint gather first a", bodies),
                first_b: lanes("constraint gather first b", bodies),
                deltas: rows("constraint solver deltas", constraints, DELTA_BYTES),
                joint_hi: lanes("joint filter keys hi", constraints),
                joint_lo: lanes("joint filter keys lo", constraints),
                row_src: lanes("constraint row src", constraints),
                row_fresh: lanes("constraint row fresh", constraints),
                fresh: rows(
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
            contacts: ContactBuffers {
                entries: ChannelSlots::new(device, "grid entries", plan.entries),
                pairs: ChannelSlots::new(device, "pairs", plan.pairs),
                large_bodies: lanes("large colliders", colliders),
                raw: rows(
                    "contacts raw",
                    plan.pairs,
                    size_of::<ContactRecord>() as u64,
                ),
                valid: lanes("contact valid", plan.pairs),
                a_body: lanes("contact a body", plan.pairs),
                compact_ranks: lanes("compact ranks", plan.pairs),
                compact_sums: lanes("compact block sums", plan.pairs.div_ceil(COMPACT_BLOCK)),
                compact_offsets: lanes("compact block offsets", plan.pairs.div_ceil(COMPACT_BLOCK)),
                manifolds: rows("contacts", plan.pairs, size_of::<ContactRecord>() as u64),
                previous: rows(
                    "previous contacts",
                    plan.pairs,
                    size_of::<ContactRecord>() as u64,
                ),
                b_keys: lanes("contact gather b keys", plan.pairs),
                b_values: lanes("contact gather b values", plan.pairs),
                first_a: lanes("contact gather first a", bodies),
                first_b: lanes("contact gather first b", bodies),
                deltas: rows("contact solver deltas", plan.pairs, DELTA_BYTES),
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
                values: lanes("sort values", plan.sort()),
                pad: lanes("sort pad", plan.sort()),
                scratch: ChannelSlots::new(device, "sort scratch", plan.sort()),
            },
            readback: ReadbackBuffers {
                pack: GpuBuffer::new(device, "sim readback pack", readback_bytes, PACK),
                step: GpuReadback::new(device, "sim readback", readback_bytes),
                events: GpuReadback::new(device, "sim events readback", event_segment_bytes),
                queries: GpuReadback::new(device, "query results readback", query_readback_bytes),
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

    pub(crate) fn contact_rows(&self) -> u32 {
        (self.contacts.manifolds.size() / size_of::<ContactRecord>() as u64) as u32
    }

    pub(crate) fn constraint_rows(&self) -> u32 {
        (self.constraints.runtime.size() / size_of::<ConstraintRuntimeRecord>() as u64) as u32
    }
}
