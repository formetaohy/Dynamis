use super::Simulation;
use crate::query_pool::{QueryHandle, QueryHit, QueryPool};
use dynamis_layout::{MAX_HITS_PER_QUERY, QueryRecord};
use dynamis_model::{QueryFilter, Shape};
use std::mem::size_of;

pub(crate) struct Queries {
    pub(crate) pending: Vec<QueryRecord>,
    pub(crate) next_batch: u64,
    pub(crate) pool: QueryPool,
}

impl Queries {
    pub(crate) fn new() -> Self {
        Self {
            pending: Vec::new(),
            next_batch: 0,
            pool: QueryPool::new(),
        }
    }
}

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
        assert!(
            record.max_hits <= MAX_HITS_PER_QUERY,
            "a query returns at most {MAX_HITS_PER_QUERY} hits"
        );
        let index = self.queries.pending.len() as u32;
        self.queries.pending.push(record);
        QueryHandle {
            batch: self.queries.next_batch,
            index,
        }
    }

    pub fn query_hit(&self, handle: QueryHandle) -> Option<QueryHit> {
        self.validate_query(handle);
        self.queries.pool.hit(handle)
    }

    pub fn query_hits(&self, handle: QueryHandle) -> &[QueryHit] {
        self.validate_query(handle);
        self.queries.pool.hits(handle)
    }

    pub fn query_overflow(&self, handle: QueryHandle) -> bool {
        self.validate_query(handle);
        self.queries.pool.overflow(handle)
    }

    pub fn flush_queries(&mut self) {
        if self.queries.pending.is_empty() {
            self.apply_plan();
            return;
        }
        self.device.gpu.assert_alive();
        self.collect_readbacks();
        self.apply_plan();
        self.flush_rows();
        self.apply_pending_commands();
        let step = self.clock.step;
        let queue = self.device.gpu.queue().clone();
        let device = self.device.gpu.device().clone();
        self.device
            .buffers
            .queries
            .records
            .write(&queue, bytemuck::cast_slice(&self.queries.pending));
        let params = dynamis_layout::SimParamsRecord::new(
            &self.config,
            self.clock.sub_dt,
            self.bodies.dynamic_count as u32,
            self.bodies.alive.len() as u32,
            self.constraints.alive.len() as u32,
            dynamis_layout::RowStreams {
                edit_runs: self.bodies.last_edits,
                body_moves: self.bodies.last_moves,
                constraint_moves: self.constraints.last_moves,
            },
            self.event_slot_of(step),
        );
        self.device
            .buffers
            .params
            .write(&queue, bytemuck::cast_slice(&[params]));
        let count = self.queries.pending.len();
        let batch = self.queries.next_batch;
        self.queries.pool.submit(batch, step, count);
        self.queries.next_batch += 1;
        let frame = crate::pipeline::FrameParams {
            dynamic_count: self.bodies.dynamic_count as u32,
            body_count: self.bodies.alive.len() as u32,
            solve_iterations: self.config.solve_iterations,
            position_iterations: self.config.position_iterations,
            island_rounds: self.island_rounds(),
            query_count: count as u32,
            constraint_count: self.constraints.alive.len() as u32,
            body_move_count: self.bodies.last_moves,
            constraint_move_count: self.constraints.last_moves,
            edit_run_count: self.bodies.last_edits,
        };
        let mut encoder = dynamis_gpu::SubmissionEncoder::new(&device, "dynamis query flush");
        self.device
            .pipeline
            .encode_queries(&mut encoder, &self.device.buffers, &frame);
        let bytes = count as u64 * size_of::<dynamis_layout::QueryResultRecord>() as u64;
        let arrived = self.device.buffers.readback.queries.enqueue(
            &mut encoder,
            self.device.buffers.queries.results.buffer(),
            0,
            bytes,
            batch,
        );
        encoder.submit(&queue);
        if let Some((batch, bytes)) = arrived {
            self.queries.pool.collect(batch, &bytes);
        }
        for (batch, bytes) in self.device.buffers.readback.queries.drain() {
            self.queries.pool.collect(batch, &bytes);
        }
        self.queries.pending.clear();
        self.apply_plan();
    }

    fn validate_query(&self, handle: QueryHandle) {
        assert!(
            self.queries.pool.is_current(handle),
            "query handle {handle:?} belongs to a batch that has been retired"
        );
        assert!(
            self.queries.pool.is_ready(handle),
            "query handle {handle:?} has no results yet; call poll() or wait()"
        );
    }
}
