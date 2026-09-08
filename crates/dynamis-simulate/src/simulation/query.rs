use super::Simulation;
use dynamis_layout::{MAX_HITS_PER_QUERY, QueryRecord, QueryResultHeader};
use dynamis_model::{QueryFilter, Shape};
use dynamis_query::{QueryHandle, QueryHit};

impl Simulation {
    pub fn ray_query(
        &mut self,
        origin: [f32; 3],
        direction: [f32; 3],
        max_t: f32,
        filter: &QueryFilter,
    ) -> QueryHandle {
        assert!(max_t > 0.0, "raycast distance must be positive");
        assert!(direction != [0.0; 3], "raycast direction must be non-zero");
        self.submit_query(QueryRecord::ray(origin, direction, max_t, filter))
    }

    pub fn sphere_query(
        &mut self,
        center: [f32; 3],
        radius: f32,
        filter: &QueryFilter,
    ) -> QueryHandle {
        assert!(radius > 0.0, "sphere query radius must be positive");
        self.submit_query(QueryRecord::sphere(center, radius, filter))
    }

    pub fn cuboid_query(
        &mut self,
        center: [f32; 3],
        half_extents: [f32; 3],
        filter: &QueryFilter,
    ) -> QueryHandle {
        assert!(
            half_extents.iter().all(|extent| *extent > 0.0),
            "cuboid query half extents must be strictly positive"
        );
        self.submit_query(QueryRecord::cuboid(center, half_extents, filter))
    }

    pub fn sweep_query(
        &mut self,
        shape: &Shape,
        orientation: [f32; 4],
        start: [f32; 3],
        direction: [f32; 3],
        length: f32,
        filter: &QueryFilter,
    ) -> QueryHandle {
        assert!(length > 0.0, "sweep length must be positive");
        assert!(direction != [0.0; 3], "sweep direction must be non-zero");
        self.assert_unit(orientation);
        assert!(shape.is_convex(), "sweep queries require a convex shape");
        self.submit_query(QueryRecord::sweep(
            shape,
            orientation,
            start,
            direction,
            length,
            filter,
        ))
    }

    fn submit_query(&mut self, record: QueryRecord) -> QueryHandle {
        let slot = self.query_pool.allocate();
        let generation = self.query_pool.generation(slot);
        let mut record = record;
        record.slot = slot as u32;
        record.max_hits = record.max_hits.min(MAX_HITS_PER_QUERY);
        self.queries.push(record);
        QueryHandle {
            slot: slot as u32,
            generation,
        }
    }

    pub fn query_hit(&self, handle: QueryHandle) -> Option<QueryHit> {
        self.validate_query(handle);
        self.query_pool.hit(handle.slot as usize)
    }

    pub fn query_hits(&self, handle: QueryHandle) -> &[QueryHit] {
        self.validate_query(handle);
        self.query_pool.hits(handle.slot as usize)
    }

    pub fn query_overflow(&self, handle: QueryHandle) -> bool {
        self.validate_query(handle);
        self.query_pool.overflow(handle.slot as usize)
    }

    pub fn flush_queries(&mut self) {
        if self.queries.is_empty() {
            return;
        }
        let step = self.step_index;
        let queue = self.gpu.queue();
        let device = self.gpu.device();
        self.buffers
            .queries
            .write(queue, bytemuck::cast_slice(&self.queries));
        let header_bytes = self.query_pool.capacity() * std::mem::size_of::<QueryResultHeader>();
        self.buffers
            .query_headers
            .write(queue, &vec![0u8; header_bytes]);
        let slots = self.queries.iter().map(|query| query.slot).collect();
        self.query_pool.mark_batch(step, slots);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dynamis query flush encoder"),
        });
        crate::stage::record_queries(&self.stages, &mut encoder, self.queries.len() as u32);
        encoder.copy_buffer_to_buffer(
            self.buffers.query_headers.buffer(),
            0,
            self.buffers.query_pack.buffer(),
            0,
            header_bytes as u64,
        );
        encoder.copy_buffer_to_buffer(
            self.buffers.query_hits.buffer(),
            0,
            self.buffers.query_pack.buffer(),
            header_bytes as u64,
            self.buffers.query_hits.size(),
        );
        let stale_queries = self.buffers.queries_readback.enqueue(
            device,
            &mut encoder,
            self.buffers.query_pack.buffer(),
            step,
        );
        let _ = stale_queries;
        queue.submit([encoder.finish()]);
        self.buffers.queries_readback.arm();
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        self.collect_readbacks();
        self.queries.clear();
    }

    fn validate_query(&self, handle: QueryHandle) {
        let slot = handle.slot as usize;
        assert!(
            slot < self.query_pool.capacity(),
            "query handle {handle:?} is out of range"
        );
        assert_eq!(
            self.query_pool.generation(slot),
            handle.generation,
            "query handle {handle:?} is stale"
        );
    }

    pub(super) fn consume_queries(&mut self, step: u64, bytes: &[u8]) {
        self.query_pool.consume(step, bytes);
    }
}
