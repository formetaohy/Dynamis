use crate::reservation::{Reservation, ShapeReservation};
use dynamis_gpu::{DispatchTable, GpuBuffer, GpuReadback, GpuSlot};

/// Event segments: one per in-flight step slot, so a segment is always copied home
/// before the step that would overwrite it is encoded.
pub(crate) const EVENT_SLOTS: u32 = GpuReadback::DEPTH as u32;
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

/// Every stream is written by the device and may be inspected from the host, so all
/// allocations carry the same usages.
const STREAM: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);
const PACK: BufferUsages = BufferUsages::COPY_DST.union(BufferUsages::COPY_SRC);

const COUNTER_BYTES: u64 = COUNTER_STRIDE * COUNTER_COUNT as u64;
const DELTA_BYTES: u64 = 64;
const VERTEX_BYTES: u64 = 16;
const TRIANGLE_BYTES: u64 = 16;
const QUERY_BYTES: u64 = size_of::<QueryRecord>() as u64;
const QUERY_RESULT_BYTES: u64 = size_of::<QueryResultHeader>() as u64
    + MAX_HITS_PER_QUERY as u64 * size_of::<QueryHitRecord>() as u64;

/// The three lanes a sort orders, reserved together.
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

/// Every device allocation the world owns.
pub(crate) struct WorldBuffers {
    pub(crate) params: GpuBuffer,
    /// The one counter vector, laid out so any single slot can be bound alone.
    pub(crate) counters: GpuBuffer,
    /// Indirect argument words, written by the device from counters each step.
    pub(crate) dispatch: DispatchTable,
    pub(crate) body_states: GpuBuffer,
    pub(crate) body_descs: GpuBuffer,
    pub(crate) colliders: GpuBuffer,
    pub(crate) aabbs: GpuBuffer,
    pub(crate) entries: ChannelSlots,
    pub(crate) pairs: ChannelSlots,
    pub(crate) large_bodies: GpuBuffer,
    pub(crate) contacts_raw: GpuBuffer,
    pub(crate) contact_valid: GpuBuffer,
    pub(crate) contact_a_body: GpuBuffer,
    pub(crate) compact_ranks: GpuBuffer,
    pub(crate) compact_block_sums: GpuBuffer,
    pub(crate) compact_block_offsets: GpuBuffer,
    pub(crate) contacts: GpuBuffer,
    pub(crate) prev_contacts: GpuBuffer,
    pub(crate) contact_b_keys: GpuBuffer,
    pub(crate) contact_b_values: GpuBuffer,
    pub(crate) contact_first_a: GpuBuffer,
    pub(crate) contact_first_b: GpuBuffer,
    pub(crate) contact_deltas: GpuBuffer,
    pub(crate) island_parents: GpuBuffer,
    pub(crate) island_state: GpuBuffer,
    pub(crate) wake_flags: GpuBuffer,
    pub(crate) constraint_descs: GpuBuffer,
    pub(crate) constraint_runtime: GpuBuffer,
    pub(crate) constraint_a_keys: GpuBuffer,
    pub(crate) constraint_a_values: GpuBuffer,
    pub(crate) constraint_b_keys: GpuBuffer,
    pub(crate) constraint_b_values: GpuBuffer,
    pub(crate) constraint_first_a: GpuBuffer,
    pub(crate) constraint_first_b: GpuBuffer,
    pub(crate) constraint_deltas: GpuBuffer,
    pub(crate) joint_hi: GpuBuffer,
    pub(crate) joint_lo: GpuBuffer,
    pub(crate) shapes: GpuBuffer,
    pub(crate) shape_vertices: GpuBuffer,
    pub(crate) shape_triangles: GpuBuffer,
    pub(crate) shape_nodes: GpuBuffer,
    pub(crate) events: GpuBuffer,
    /// Content edits, one per touched slot run, separated by slot.
    pub(crate) commands: GpuBuffer,
    /// One-based index of a slot's first edit; 0 means none.
    pub(crate) command_first: GpuBuffer,
    /// Per-slot structural move: source slot, CLEAR sentinel, or fresh lane index.
    pub(crate) row_src: GpuBuffer,
    pub(crate) row_fresh: GpuBuffer,
    /// Fresh row records for structural moves.
    pub(crate) fresh_states: GpuBuffer,
    /// The row layout lands here before it is scattered home.
    pub(crate) state_scratch: GpuBuffer,
    pub(crate) constraint_row_src: GpuBuffer,
    pub(crate) constraint_row_fresh: GpuBuffer,
    pub(crate) constraint_fresh: GpuBuffer,
    pub(crate) constraint_scratch: GpuBuffer,
    pub(crate) queries: GpuBuffer,
    pub(crate) query_results: GpuBuffer,
    /// Payload lane for the sorts that carry no payload of their own.
    pub(crate) sort_values: GpuBuffer,
    pub(crate) sort_pad: GpuBuffer,
    /// One mirrored copy of everything the host tracks, read back once per step.
    pub(crate) readback_pack: GpuBuffer,
    pub(crate) readback: GpuReadback,
    /// Contact events, read back by the effective count once its pack has landed.
    pub(crate) events_readback: GpuReadback,
    pub(crate) queries_readback: GpuReadback,
    pub(crate) sort_scratch: ChannelSlots,
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
        let event_readback_bytes = event_segment_bytes;
        let event_bytes = event_segment_bytes * EVENT_SLOTS as u64;
        assert!(
            event_readback_bytes <= buffer_limit,
            "event readback requires {event_readback_bytes} bytes but the device buffer limit is {buffer_limit}"
        );
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
            body_states: rows("body states", bodies, size_of::<BodyStateRecord>() as u64),
            body_descs: rows(
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
            entries: ChannelSlots::new(device, "grid entries", plan.entries),
            pairs: ChannelSlots::new(device, "pairs", plan.pairs),
            large_bodies: lanes("large colliders", colliders),
            contacts_raw: rows(
                "contacts raw",
                plan.pairs,
                size_of::<ContactRecord>() as u64,
            ),
            contact_valid: lanes("contact valid", plan.pairs),
            contact_a_body: lanes("contact a body", plan.pairs),
            compact_ranks: lanes("compact ranks", plan.pairs),
            compact_block_sums: lanes("compact block sums", plan.pairs.div_ceil(COMPACT_BLOCK)),
            compact_block_offsets: lanes(
                "compact block offsets",
                plan.pairs.div_ceil(COMPACT_BLOCK),
            ),
            contacts: rows("contacts", plan.pairs, size_of::<ContactRecord>() as u64),
            prev_contacts: rows(
                "previous contacts",
                plan.pairs,
                size_of::<ContactRecord>() as u64,
            ),
            contact_b_keys: lanes("contact gather b keys", plan.pairs),
            contact_b_values: lanes("contact gather b values", plan.pairs),
            contact_first_a: lanes("contact gather first a", bodies),
            contact_first_b: lanes("contact gather first b", bodies),
            contact_deltas: rows("contact solver deltas", plan.pairs, DELTA_BYTES),
            island_parents: lanes("island parents", bodies),
            island_state: lanes("island state", bodies),
            wake_flags: lanes("wake flags", bodies),
            constraint_descs: rows(
                "constraint descriptors",
                constraints,
                size_of::<ConstraintDescriptorRecord>() as u64,
            ),
            constraint_runtime: rows(
                "constraint runtime",
                constraints,
                size_of::<ConstraintRuntimeRecord>() as u64,
            ),
            constraint_a_keys: lanes("constraint gather a keys", constraints),
            constraint_a_values: lanes("constraint gather a values", constraints),
            constraint_b_keys: lanes("constraint gather b keys", constraints),
            constraint_b_values: lanes("constraint gather b values", constraints),
            constraint_first_a: lanes("constraint gather first a", bodies),
            constraint_first_b: lanes("constraint gather first b", bodies),
            constraint_deltas: rows("constraint solver deltas", constraints, DELTA_BYTES),
            joint_hi: lanes("joint filter keys hi", constraints),
            joint_lo: lanes("joint filter keys lo", constraints),
            shapes: rows(
                "shape sources",
                shapes.sources,
                size_of::<ShapeSourceRecord>() as u64,
            ),
            shape_vertices: rows("shape vertices", shapes.vertices, VERTEX_BYTES),
            shape_triangles: rows("shape triangles", shapes.triangles, TRIANGLE_BYTES),
            shape_nodes: rows(
                "shape bvh nodes",
                shapes.nodes,
                size_of::<BvhNodeRecord>() as u64,
            ),
            events: GpuBuffer::new(device, "contact events", event_bytes, STREAM),
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
            constraint_row_src: lanes("constraint row src", constraints),
            constraint_row_fresh: lanes("constraint row fresh", constraints),
            constraint_fresh: rows(
                "fresh constraint rows",
                plan.constraint_commands,
                size_of::<ConstraintRuntimeRecord>() as u64,
            ),
            constraint_scratch: rows(
                "constraint state scratch",
                constraints,
                size_of::<ConstraintRuntimeRecord>() as u64,
            ),
            queries: rows("queries", queries, QUERY_BYTES),
            query_results: rows("query results", queries, QUERY_RESULT_BYTES),
            sort_values: lanes("sort values", plan.sort()),
            sort_pad: lanes("sort pad", plan.sort()),
            readback_pack: GpuBuffer::new(device, "sim readback pack", readback_bytes, PACK),
            readback: GpuReadback::new(device, "sim readback", readback_bytes),
            events_readback: GpuReadback::new(device, "sim events readback", event_readback_bytes),
            queries_readback: GpuReadback::new(
                device,
                "query results readback",
                query_readback_bytes,
            ),
            sort_scratch: ChannelSlots::new(device, "sort scratch", plan.sort()),
        }
    }

    /// Binds one counter slot out of the shared vector.
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

    pub(crate) fn bodies(&self) -> u32 {
        (self.body_states.size() / size_of::<BodyStateRecord>() as u64) as u32
    }

    pub(crate) fn colliders(&self) -> u32 {
        (self.colliders.size() / size_of::<ColliderRecord>() as u64) as u32
    }

    pub(crate) fn contacts(&self) -> u32 {
        (self.contacts.size() / size_of::<ContactRecord>() as u64) as u32
    }

    pub(crate) fn constraints(&self) -> u32 {
        (self.constraint_runtime.size() / size_of::<ConstraintRuntimeRecord>() as u64) as u32
    }
}
