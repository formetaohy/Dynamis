use dynamis_gpu::{GpuBuffer, GpuReadback};
use dynamis_layout::{
    AabbRecord, BodyCommandRecord, BvhNodeRecord, ColliderRecord, ConstraintCommandRecord,
    ConstraintRecord, ContactEventRecord, ContactRecord, DispatchArgs, MAX_CELLS_PER_COLLIDER,
    MAX_HITS_PER_QUERY, QueryHitRecord, QueryRecord, QueryResultHeader, RigidBodyRecord,
    ShapeSourceRecord, SimParamsRecord,
};
use std::mem::size_of;
use wgpu::{BufferUsages, Device, Queue};

pub(crate) const SHAPE_VERTICES_PER_SOURCE: u32 = 4096;
pub(crate) const SHAPE_TRIANGLES_PER_SOURCE: u32 = 8192;
pub(crate) const SHAPE_NODES_PER_SOURCE: u32 = 16384;
pub(crate) const COMPACT_BLOCK: u32 = 256;

pub(crate) struct SortSlots {
    pub(crate) keys_hi: GpuBuffer,
    pub(crate) keys_lo: GpuBuffer,
    pub(crate) values: GpuBuffer,
}

impl SortSlots {
    fn new(device: &Device, label: &str, capacity: usize) -> Self {
        let bytes = (capacity * size_of::<u32>()) as u64;
        Self {
            keys_hi: GpuBuffer::new(
                device,
                &format!("{label} keys hi"),
                bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            keys_lo: GpuBuffer::new(
                device,
                &format!("{label} keys lo"),
                bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            values: GpuBuffer::new(
                device,
                &format!("{label} values"),
                bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
        }
    }

    pub(crate) fn capacity_u32(&self) -> u32 {
        (self.keys_hi.size() / 4) as u32
    }
}

pub(crate) struct StageBuffers {
    pub(crate) params: GpuBuffer,
    pub(crate) bodies_current: GpuBuffer,
    pub(crate) colliders: GpuBuffer,
    pub(crate) aabbs: GpuBuffer,
    pub(crate) entries: SortSlots,
    pub(crate) entry_count: GpuBuffer,
    pub(crate) entry_args: GpuBuffer,
    pub(crate) pairs: SortSlots,
    pub(crate) pair_count: GpuBuffer,
    pub(crate) pair_args: GpuBuffer,
    pub(crate) large_bodies: GpuBuffer,
    pub(crate) large_count: GpuBuffer,
    pub(crate) contacts_raw: GpuBuffer,
    pub(crate) contact_valid: GpuBuffer,
    pub(crate) contact_a_body: GpuBuffer,
    pub(crate) compact_ranks: GpuBuffer,
    pub(crate) compact_block_sums: GpuBuffer,
    pub(crate) compact_block_sums_pad: GpuBuffer,
    pub(crate) compact_block_offsets: GpuBuffer,
    pub(crate) contacts: GpuBuffer,
    pub(crate) contact_count: GpuBuffer,
    pub(crate) contact_args: GpuBuffer,
    pub(crate) prev_contacts: GpuBuffer,
    pub(crate) prev_contact_count: GpuBuffer,
    pub(crate) prev_args: GpuBuffer,
    pub(crate) contact_b_keys: GpuBuffer,
    pub(crate) contact_b_values: GpuBuffer,
    pub(crate) contact_b_keys_out: GpuBuffer,
    pub(crate) contact_b_values_out: GpuBuffer,
    pub(crate) contact_first_a: GpuBuffer,
    pub(crate) contact_first_b: GpuBuffer,
    pub(crate) contact_deltas: GpuBuffer,
    pub(crate) island_parents: GpuBuffer,
    pub(crate) island_state: GpuBuffer,
    pub(crate) wake_flags: GpuBuffer,
    pub(crate) constraints: GpuBuffer,
    pub(crate) constraint_gather_a_keys: GpuBuffer,
    pub(crate) constraint_gather_a_values: GpuBuffer,
    pub(crate) constraint_gather_a_keys_out: GpuBuffer,
    pub(crate) constraint_gather_a_values_out: GpuBuffer,
    pub(crate) constraint_gather_b_keys: GpuBuffer,
    pub(crate) constraint_gather_b_values: GpuBuffer,
    pub(crate) constraint_gather_b_keys_out: GpuBuffer,
    pub(crate) constraint_gather_b_values_out: GpuBuffer,
    pub(crate) constraint_first_a: GpuBuffer,
    pub(crate) constraint_first_b: GpuBuffer,
    pub(crate) constraint_deltas: GpuBuffer,
    pub(crate) constraint_joint_count: GpuBuffer,
    pub(crate) constraint_count_state: GpuBuffer,
    pub(crate) constraint_args: GpuBuffer,
    pub(crate) joint_hi: GpuBuffer,
    pub(crate) joint_lo: GpuBuffer,
    pub(crate) joint_count: GpuBuffer,
    pub(crate) joint_args: GpuBuffer,
    pub(crate) shapes: GpuBuffer,
    pub(crate) shape_vertices: GpuBuffer,
    pub(crate) shape_triangles: GpuBuffer,
    pub(crate) shape_nodes: GpuBuffer,
    pub(crate) events: GpuBuffer,
    pub(crate) event_count: GpuBuffer,
    pub(crate) events_pack: GpuBuffer,
    pub(crate) overflow_flags: GpuBuffer,
    pub(crate) commands: GpuBuffer,
    pub(crate) command_count: GpuBuffer,
    pub(crate) constraint_commands: GpuBuffer,
    pub(crate) constraint_command_count: GpuBuffer,
    pub(crate) queries: GpuBuffer,
    pub(crate) query_headers: GpuBuffer,
    pub(crate) query_hits: GpuBuffer,
    pub(crate) query_pack: GpuBuffer,
    pub(crate) bodies_readback: GpuReadback,
    pub(crate) queries_readback: GpuReadback,
    pub(crate) events_readback: GpuReadback,
    pub(crate) constraints_readback: GpuReadback,
    pub(crate) sort_scratch: SortSlots,
}

impl StageBuffers {
    pub(crate) fn new(
        device: &Device,
        shape_sources: usize,
        capacity: usize,
        pair_capacity: usize,
        query_capacity: usize,
        constraint_capacity: usize,
    ) -> Self {
        let collider_capacity = capacity * 4;
        let body_bytes = (capacity * size_of::<RigidBodyRecord>()) as u64;
        let collider_bytes = (collider_capacity * size_of::<ColliderRecord>()) as u64;
        let aabb_bytes = (collider_capacity * size_of::<AabbRecord>()) as u64;
        let contact_bytes = (pair_capacity * size_of::<ContactRecord>()) as u64;
        let contact_key_bytes = (pair_capacity * size_of::<u32>()) as u64;
        let constraint_bytes = (constraint_capacity * size_of::<ConstraintRecord>()) as u64;
        let counter_bytes = size_of::<DispatchArgs>() as u64;
        let state_bytes = (capacity * size_of::<u32>()) as u64;
        let command_bytes = (capacity * size_of::<BodyCommandRecord>()) as u64;
        let constraint_command_bytes =
            (constraint_capacity * size_of::<ConstraintCommandRecord>()) as u64;
        let params_bytes = size_of::<SimParamsRecord>() as u64;
        let query_bytes = (query_capacity * size_of::<QueryRecord>()) as u64;
        let query_header_bytes = (query_capacity * size_of::<QueryResultHeader>()) as u64;
        let query_hit_bytes =
            (query_capacity * MAX_HITS_PER_QUERY as usize * size_of::<QueryHitRecord>()) as u64;
        let entry_capacity = collider_capacity * MAX_CELLS_PER_COLLIDER as usize;
        let joint_bytes = (constraint_capacity * size_of::<u32>()) as u64;
        let shape_bytes = ((shape_sources * size_of::<ShapeSourceRecord>()).max(16)) as u64;
        let vertex_bytes =
            ((shape_sources * SHAPE_VERTICES_PER_SOURCE as usize * 16).max(16)) as u64;
        let triangle_bytes =
            ((shape_sources * SHAPE_TRIANGLES_PER_SOURCE as usize * 16).max(16)) as u64;
        let node_bytes = ((shape_sources
            * SHAPE_NODES_PER_SOURCE as usize
            * size_of::<BvhNodeRecord>())
        .max(16)) as u64;
        let events_bytes = (pair_capacity * size_of::<ContactEventRecord>()) as u64;
        let compact_blocks = (pair_capacity as u64).div_ceil(COMPACT_BLOCK as u64);
        let compact_bytes = (compact_blocks * size_of::<u32>() as u64) as u64;
        let args_bytes: u64 = 32;
        let constraint_args_bytes: u64 = 32;
        Self {
            params: GpuBuffer::new(
                device,
                "sim params",
                params_bytes,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            ),
            bodies_current: GpuBuffer::new(
                device,
                "bodies",
                body_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            colliders: GpuBuffer::new(
                device,
                "colliders",
                collider_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            aabbs: GpuBuffer::new(
                device,
                "broadphase aabbs",
                aabb_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            entries: SortSlots::new(device, "grid entries", entry_capacity),
            entry_count: GpuBuffer::new(
                device,
                "grid entry count",
                counter_bytes,
                BufferUsages::STORAGE
                    | BufferUsages::INDIRECT
                    | BufferUsages::COPY_DST
                    | BufferUsages::COPY_SRC,
            ),
            entry_args: GpuBuffer::new(
                device,
                "grid entry args",
                args_bytes,
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
            ),
            pairs: SortSlots::new(device, "pairs", pair_capacity),
            pair_count: GpuBuffer::new(
                device,
                "pair count",
                counter_bytes,
                BufferUsages::STORAGE
                    | BufferUsages::INDIRECT
                    | BufferUsages::COPY_DST
                    | BufferUsages::COPY_SRC,
            ),
            pair_args: GpuBuffer::new(
                device,
                "pair args",
                args_bytes,
                BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
            ),
            large_bodies: GpuBuffer::new(
                device,
                "large colliders",
                (collider_capacity * size_of::<u32>()) as u64,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            large_count: GpuBuffer::new(
                device,
                "large collider count",
                counter_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            contacts_raw: GpuBuffer::new(
                device,
                "contacts raw",
                contact_bytes,
                BufferUsages::STORAGE,
            ),
            contact_valid: GpuBuffer::new(
                device,
                "contact valid",
                contact_key_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            contact_a_body: GpuBuffer::new(
                device,
                "contact a body",
                contact_key_bytes,
                BufferUsages::STORAGE,
            ),
            compact_ranks: GpuBuffer::new(
                device,
                "compact ranks",
                contact_key_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            compact_block_sums: GpuBuffer::new(
                device,
                "compact block sums",
                compact_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            compact_block_sums_pad: GpuBuffer::new(
                device,
                "compact block sums pad",
                (compact_bytes + 32).max(448),
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            compact_block_offsets: GpuBuffer::new(
                device,
                "compact block offsets",
                compact_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            contacts: GpuBuffer::new(
                device,
                "contacts",
                contact_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            contact_count: GpuBuffer::new(
                device,
                "contact count",
                counter_bytes,
                BufferUsages::STORAGE
                    | BufferUsages::INDIRECT
                    | BufferUsages::COPY_DST
                    | BufferUsages::COPY_SRC,
            ),
            contact_args: GpuBuffer::new(
                device,
                "contact args",
                args_bytes,
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
            ),
            prev_contacts: GpuBuffer::new(
                device,
                "previous contacts",
                contact_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            prev_contact_count: GpuBuffer::new(
                device,
                "previous contact count",
                counter_bytes,
                BufferUsages::STORAGE
                    | BufferUsages::INDIRECT
                    | BufferUsages::COPY_DST
                    | BufferUsages::COPY_SRC,
            ),
            prev_args: GpuBuffer::new(
                device,
                "previous contact args",
                args_bytes,
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
            ),
            contact_b_keys: GpuBuffer::new(
                device,
                "contact gather b keys",
                contact_key_bytes,
                BufferUsages::STORAGE,
            ),
            contact_b_values: GpuBuffer::new(
                device,
                "contact gather b values",
                contact_key_bytes,
                BufferUsages::STORAGE,
            ),
            contact_b_keys_out: GpuBuffer::new(
                device,
                "contact gather b keys out",
                contact_key_bytes,
                BufferUsages::STORAGE,
            ),
            contact_b_values_out: GpuBuffer::new(
                device,
                "contact gather b values out",
                contact_key_bytes,
                BufferUsages::STORAGE,
            ),
            contact_first_a: GpuBuffer::new(
                device,
                "contact gather first a",
                state_bytes,
                BufferUsages::STORAGE,
            ),
            contact_first_b: GpuBuffer::new(
                device,
                "contact gather first b",
                state_bytes,
                BufferUsages::STORAGE,
            ),
            contact_deltas: GpuBuffer::new(
                device,
                "contact solver deltas",
                (pair_capacity * 64) as u64,
                BufferUsages::STORAGE,
            ),
            island_parents: GpuBuffer::new(
                device,
                "island parents",
                state_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            island_state: GpuBuffer::new(
                device,
                "island state",
                state_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            wake_flags: GpuBuffer::new(
                device,
                "wake flags",
                state_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            constraints: GpuBuffer::new(
                device,
                "constraints",
                constraint_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            constraint_gather_a_keys: GpuBuffer::new(
                device,
                "constraint gather a keys",
                (constraint_capacity * size_of::<u32>()) as u64,
                BufferUsages::STORAGE,
            ),
            constraint_gather_a_values: GpuBuffer::new(
                device,
                "constraint gather a values",
                (constraint_capacity * size_of::<u32>()) as u64,
                BufferUsages::STORAGE,
            ),
            constraint_gather_a_keys_out: GpuBuffer::new(
                device,
                "constraint gather a keys out",
                (constraint_capacity * size_of::<u32>()) as u64,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            constraint_gather_a_values_out: GpuBuffer::new(
                device,
                "constraint gather a values out",
                (constraint_capacity * size_of::<u32>()) as u64,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            constraint_gather_b_keys: GpuBuffer::new(
                device,
                "constraint gather b keys",
                (constraint_capacity * size_of::<u32>()) as u64,
                BufferUsages::STORAGE,
            ),
            constraint_gather_b_values: GpuBuffer::new(
                device,
                "constraint gather b values",
                (constraint_capacity * size_of::<u32>()) as u64,
                BufferUsages::STORAGE,
            ),
            constraint_gather_b_keys_out: GpuBuffer::new(
                device,
                "constraint gather b keys out",
                (constraint_capacity * size_of::<u32>()) as u64,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            constraint_gather_b_values_out: GpuBuffer::new(
                device,
                "constraint gather b values out",
                (constraint_capacity * size_of::<u32>()) as u64,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            constraint_first_a: GpuBuffer::new(
                device,
                "constraint gather first a",
                state_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            constraint_first_b: GpuBuffer::new(
                device,
                "constraint gather first b",
                state_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            constraint_deltas: GpuBuffer::new(
                device,
                "constraint solver deltas",
                (constraint_capacity * 64) as u64,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            constraint_joint_count: GpuBuffer::new(
                device,
                "constraint joint count",
                counter_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            constraint_count_state: GpuBuffer::new(
                device,
                "constraint count state",
                counter_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            constraint_args: GpuBuffer::new(
                device,
                "constraint args",
                constraint_args_bytes,
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
            ),
            joint_hi: GpuBuffer::new(
                device,
                "joint filter keys hi",
                joint_bytes,
                BufferUsages::STORAGE,
            ),
            joint_lo: GpuBuffer::new(
                device,
                "joint filter keys lo",
                joint_bytes,
                BufferUsages::STORAGE,
            ),
            joint_count: GpuBuffer::new(
                device,
                "joint filter count",
                counter_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            joint_args: GpuBuffer::new(
                device,
                "joint filter args",
                args_bytes,
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
            ),
            shapes: GpuBuffer::new(
                device,
                "shape sources",
                shape_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            shape_vertices: GpuBuffer::new(
                device,
                "shape vertices",
                vertex_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
            shape_triangles: GpuBuffer::new(
                device,
                "shape triangles",
                triangle_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
            shape_nodes: GpuBuffer::new(
                device,
                "shape bvh nodes",
                node_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
            events: GpuBuffer::new(
                device,
                "contact events",
                events_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            events_pack: GpuBuffer::new(
                device,
                "contact events pack",
                events_bytes + counter_bytes,
                BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            event_count: GpuBuffer::new(
                device,
                "event count",
                counter_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            overflow_flags: GpuBuffer::new(
                device,
                "overflow flags",
                counter_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            commands: GpuBuffer::new(
                device,
                "body commands",
                command_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            command_count: GpuBuffer::new(
                device,
                "command count",
                counter_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
            constraint_commands: GpuBuffer::new(
                device,
                "constraint commands",
                constraint_command_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
            constraint_command_count: GpuBuffer::new(
                device,
                "constraint command count",
                counter_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            queries: GpuBuffer::new(
                device,
                "queries",
                query_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
            query_headers: GpuBuffer::new(
                device,
                "query result headers",
                query_header_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            query_hits: GpuBuffer::new(
                device,
                "query result hits",
                query_hit_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            query_pack: GpuBuffer::new(
                device,
                "query result pack",
                query_header_bytes + query_hit_bytes,
                BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            bodies_readback: GpuReadback::new(device, "bodies readback", body_bytes),
            queries_readback: GpuReadback::new(
                device,
                "query results readback",
                query_header_bytes + query_hit_bytes,
            ),
            events_readback: GpuReadback::new(
                device,
                "events readback",
                events_bytes + counter_bytes,
            ),
            constraints_readback: GpuReadback::new(
                device,
                "constraints readback",
                constraint_bytes,
            ),
            sort_scratch: SortSlots::new(device, "sort scratch", entry_capacity.max(pair_capacity)),
        }
    }

    pub(crate) fn constraint_capacity(&self) -> u32 {
        (self.constraints.size() / size_of::<ConstraintRecord>() as u64) as u32
    }

    pub(crate) fn contact_capacity(&self) -> u32 {
        (self.contacts.size() / size_of::<ContactRecord>() as u64) as u32
    }

    pub(crate) fn reset_counters(&self, queue: &Queue) {
        let idle = DispatchArgs::none();
        let idle = bytemuck::cast_slice(std::slice::from_ref(&idle));
        self.entry_count.write(queue, idle);
        self.pair_count.write(queue, idle);
        self.large_count.write(queue, idle);
        self.contact_count.write(queue, idle);
        self.constraint_joint_count.write(queue, idle);
        self.joint_count.write(queue, idle);
        self.event_count.write(queue, idle);
        self.overflow_flags.write(queue, &[0u8; 12]);
    }
}
