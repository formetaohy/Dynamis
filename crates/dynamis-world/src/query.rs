use super::World;
use crate::query_pool::{QueryHandle, QueryHit, QueryPool, QueryState};
use dynamis_abi::{MAX_HITS_PER_QUERY, QueryRecord};
use dynamis_model::{QueryFilter, Shape};

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

impl World {
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

    pub fn query_state(&self, handle: QueryHandle) -> QueryState {
        if self.queries.pool.is_ready(handle) {
            QueryState::Retired
        } else if self.queries.pool.is_current(handle) || handle.batch == self.queries.next_batch {
            QueryState::Pending
        } else {
            QueryState::Lapsed
        }
    }

    pub fn resolve_queries(&mut self) {
        if self.queries.pending.is_empty() {
            return;
        }
        self.execute(dynamis_pass::Run::Query);
    }

    pub fn wait_query(&mut self, handle: QueryHandle) {
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
        if self.query_state(handle) != QueryState::Retired {
            for (batch, bytes) in self.backend.readback.queries.drain() {
                self.collect_query_batch(batch, &bytes);
            }
        }
        self.validate_query(handle);
    }

    pub(crate) fn collect_query_batch(&mut self, batch: u64, bytes: &[u8]) {
        let colliders = self.colliders.records();
        let shapes = &self.shapes.pool;
        let bodies = &self.bodies;
        let pool = &self.colliders;
        let soft = &self.soft;
        self.queries.pool.collect(
            batch,
            bytes,
            |slot, index| shapes.source_surface(colliders[slot as usize].source, index),
            |body, slot| crate::colliders::local_collider_of(bodies, pool, body, slot),
            |body, slot| soft.local_particle_of(body, slot),
        );
    }

    fn validate_query(&self, handle: QueryHandle) {
        match self.query_state(handle) {
            QueryState::Retired => {}
            QueryState::Pending => {
                panic!("query handle {handle:?} has not retired; observe it or call wait_query()")
            }
            QueryState::Lapsed => panic!(
                "query handle {handle:?} lapsed; its results retired before the host consumed them"
            ),
        }
    }
}
