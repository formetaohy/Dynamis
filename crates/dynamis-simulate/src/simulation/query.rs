use super::Simulation;
use crate::query_pool::{QueryHandle, QueryHit};
use dynamis_layout::{MAX_HITS_PER_QUERY, QueryRecord};
use dynamis_model::{QueryFilter, Shape};
use std::mem::size_of;

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

    pub fn point_query(&mut self, origin: [f32; 3], filter: &QueryFilter) -> QueryHandle {
        self.submit_query(QueryRecord::point(origin, filter))
    }

    pub fn overlap_query(
        &mut self,
        shape: &Shape,
        orientation: [f32; 4],
        position: [f32; 3],
        filter: &QueryFilter,
    ) -> QueryHandle {
        self.assert_unit(orientation);
        assert!(shape.is_convex(), "overlap queries require a convex shape");
        self.submit_query(QueryRecord::convex(shape, orientation, position, filter))
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
        let mut record = record;
        record.max_hits = record.max_hits.min(MAX_HITS_PER_QUERY);
        let index = self.queries.len() as u32;
        self.queries.push(record);
        QueryHandle {
            batch: self.next_batch,
            index,
        }
    }

    pub fn query_hit(&self, handle: QueryHandle) -> Option<QueryHit> {
        self.validate_query(handle);
        self.query_pool.hit(handle)
    }

    pub fn query_hits(&self, handle: QueryHandle) -> &[QueryHit] {
        self.validate_query(handle);
        self.query_pool.hits(handle)
    }

    pub fn query_overflow(&self, handle: QueryHandle) -> bool {
        self.validate_query(handle);
        self.query_pool.overflow(handle)
    }

    /// Runs the pending query batch without advancing the world.
    pub fn flush_queries(&mut self) {
        if self.queries.is_empty() {
            return;
        }
        self.gpu.assert_alive();
        self.collect_readbacks();
        self.apply_plan();
        self.flush_rows();
        let step = self.step_index;
        let queue = self.gpu.queue().clone();
        let device = self.gpu.device().clone();
        self.buffers
            .queries
            .write(&queue, bytemuck::cast_slice(&self.queries));
        let count = self.queries.len();
        let batch = self.next_batch;
        self.query_pool.submit(batch, step, count);
        self.next_batch += 1;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dynamis query flush"),
        });
        self.pipeline.encode_queries(&mut encoder, count as u32);
        let bytes = count as u64 * size_of::<dynamis_layout::QueryResultRecord>() as u64;
        let arrived = self.buffers.queries_readback.enqueue(
            &device,
            &mut encoder,
            self.buffers.query_results.buffer(),
            bytes,
            batch,
        );
        queue.submit([encoder.finish()]);
        self.buffers.queries_readback.arm();
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        if let Some((batch, bytes)) = arrived {
            self.query_pool.collect(batch, &bytes);
        }
        for (batch, bytes) in self.buffers.queries_readback.poll(&device) {
            self.query_pool.collect(batch, &bytes);
        }
        self.queries.clear();
    }

    fn validate_query(&self, handle: QueryHandle) {
        assert!(
            self.query_pool.is_current(handle),
            "query handle {handle:?} belongs to a batch that has been retired"
        );
        assert!(
            self.query_pool.is_ready(handle),
            "query handle {handle:?} has no results yet; call poll() or wait()"
        );
    }
}
