use dynamis_gpu::{GpuBuffer, GpuReadback};
use dynamis_layout::{
    AabbRecord, BodyCommandRecord, ColliderRecord, ConstraintCommandRecord, ConstraintRecord,
    ContactRecord, DispatchArgs, QueryRecord, QueryResultRecord, RigidBodyRecord, SimParamsRecord,
};
use std::mem::size_of;
use wgpu::{BufferUsages, Device, Queue};

pub(crate) const MAX_CELLS_PER_BODY: u32 = 8;

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
    pub(crate) pairs: SortSlots,
    pub(crate) pair_count: GpuBuffer,
    pub(crate) large_bodies: GpuBuffer,
    pub(crate) large_count: GpuBuffer,
    pub(crate) contacts: GpuBuffer,
    pub(crate) contact_count: GpuBuffer,
    pub(crate) constraints: GpuBuffer,
    pub(crate) commands: GpuBuffer,
    pub(crate) command_count: GpuBuffer,
    pub(crate) constraint_commands: GpuBuffer,
    pub(crate) constraint_command_count: GpuBuffer,
    pub(crate) queries: GpuBuffer,
    pub(crate) query_results: GpuBuffer,
    pub(crate) bodies_readback: GpuReadback,
    pub(crate) queries_readback: GpuReadback,
    pub(crate) sort_scratch: SortSlots,
}

impl StageBuffers {
    pub(crate) fn new(
        device: &Device,
        capacity: usize,
        pair_capacity: usize,
        query_capacity: usize,
        constraint_capacity: usize,
    ) -> Self {
        let body_bytes = (capacity * size_of::<RigidBodyRecord>()) as u64;
        let collider_bytes = (capacity * size_of::<ColliderRecord>()) as u64;
        let aabb_bytes = (capacity * size_of::<AabbRecord>()) as u64;
        let contact_bytes = (pair_capacity * size_of::<ContactRecord>()) as u64;
        let constraint_bytes = (constraint_capacity * size_of::<ConstraintRecord>()) as u64;
        let counter_bytes = size_of::<DispatchArgs>() as u64;
        let command_bytes = (capacity * size_of::<BodyCommandRecord>()) as u64;
        let constraint_command_bytes =
            (constraint_capacity * size_of::<ConstraintCommandRecord>()) as u64;
        let params_bytes = size_of::<SimParamsRecord>() as u64;
        let query_bytes = (query_capacity * size_of::<QueryRecord>()) as u64;
        let query_result_bytes = (query_capacity * size_of::<QueryResultRecord>()) as u64;
        let entry_capacity = capacity * MAX_CELLS_PER_BODY as usize;
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
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            colliders: GpuBuffer::new(
                device,
                "colliders",
                collider_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
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
            large_bodies: GpuBuffer::new(
                device,
                "large bodies",
                (capacity * size_of::<u32>()) as u64,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            large_count: GpuBuffer::new(
                device,
                "large body count",
                counter_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
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
            constraints: GpuBuffer::new(
                device,
                "constraints",
                constraint_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
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
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
            queries: GpuBuffer::new(
                device,
                "queries",
                query_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
            query_results: GpuBuffer::new(
                device,
                "query results",
                query_result_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            bodies_readback: GpuReadback::new(device, "bodies readback", body_bytes),
            queries_readback: GpuReadback::new(
                device,
                "query results readback",
                query_result_bytes,
            ),
            sort_scratch: SortSlots::new(device, "sort scratch", entry_capacity.max(pair_capacity)),
        }
    }

    pub(crate) fn constraint_capacity(&self) -> u32 {
        (self.constraints.size() / size_of::<ConstraintRecord>() as u64) as u32
    }

    pub(crate) fn reset_counters(&self, queue: &Queue) {
        let idle = DispatchArgs::none();
        let idle = bytemuck::cast_slice(std::slice::from_ref(&idle));
        self.entry_count.write(queue, idle);
        self.pair_count.write(queue, idle);
        self.large_count.write(queue, idle);
        self.contact_count.write(queue, idle);
    }
}
