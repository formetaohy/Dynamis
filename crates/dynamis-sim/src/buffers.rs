use dynamis_gpu::{GpuBuffer, GpuReadback};
use dynamis_layout::{
    AabbRecord, BodyCommandRecord, ContactRecord, DispatchArgs, PairRecord, QueryRecord,
    QueryResultRecord, RigidBodyRecord, SimParamsRecord,
};
use std::mem::size_of;
use wgpu::{BufferUsages, Device, Queue};

pub(crate) struct StageBuffers {
    pub(crate) params: GpuBuffer,
    pub(crate) bodies: GpuBuffer,
    pub(crate) aabbs: GpuBuffer,
    pub(crate) pairs: GpuBuffer,
    pub(crate) pair_count: GpuBuffer,
    pub(crate) contacts: GpuBuffer,
    pub(crate) contact_count: GpuBuffer,
    pub(crate) commands: GpuBuffer,
    pub(crate) command_count: GpuBuffer,
    pub(crate) queries: GpuBuffer,
    pub(crate) query_results: GpuBuffer,
    pub(crate) bodies_readback: GpuReadback,
    pub(crate) queries_readback: GpuReadback,
}

impl StageBuffers {
    pub(crate) fn new(
        device: &Device,
        capacity: usize,
        pair_capacity: usize,
        query_capacity: usize,
    ) -> Self {
        let body_bytes = (capacity * size_of::<RigidBodyRecord>()) as u64;
        let aabb_bytes = (capacity * size_of::<AabbRecord>()) as u64;
        let pair_bytes = (pair_capacity.max(1) * size_of::<PairRecord>()) as u64;
        let contact_bytes = (pair_capacity.max(1) * size_of::<ContactRecord>()) as u64;
        let counter_bytes = size_of::<DispatchArgs>() as u64;
        let command_bytes = (capacity * size_of::<BodyCommandRecord>()) as u64;
        let params_bytes = size_of::<SimParamsRecord>() as u64;
        let query_bytes = (query_capacity * size_of::<QueryRecord>()) as u64;
        let query_result_bytes = (query_capacity * size_of::<QueryResultRecord>()) as u64;
        Self {
            params: GpuBuffer::new(
                device,
                "sim params",
                params_bytes,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            ),
            bodies: GpuBuffer::new(
                device,
                "bodies",
                body_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            aabbs: GpuBuffer::new(
                device,
                "broadphase aabbs",
                aabb_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            pairs: GpuBuffer::new(
                device,
                "broadphase pairs",
                pair_bytes,
                BufferUsages::STORAGE
                    | BufferUsages::INDIRECT
                    | BufferUsages::COPY_DST
                    | BufferUsages::COPY_SRC,
            ),
            pair_count: GpuBuffer::new(
                device,
                "pair count",
                counter_bytes,
                BufferUsages::STORAGE
                    | BufferUsages::INDIRECT
                    | BufferUsages::COPY_DST
                    | BufferUsages::COPY_SRC,
            ),
            contacts: GpuBuffer::new(
                device,
                "narrowphase contacts",
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
            commands: GpuBuffer::new(
                device,
                "body commands",
                command_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
            command_count: GpuBuffer::new(
                device,
                "command count",
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
        }
    }

    pub(crate) fn reset_counters(&self, queue: &Queue) {
        let idle = DispatchArgs::none();
        let idle = bytemuck::cast_slice(std::slice::from_ref(&idle));
        self.pair_count.write(queue, idle);
        self.contact_count.write(queue, idle);
    }
}
