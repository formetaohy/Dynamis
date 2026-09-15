use dynamis_abi::{
    ENTRY_INDEX_MASK, ENTRY_KIND_COLLIDER, ENTRY_KIND_PARTICLE, ENTRY_KIND_SHIFT,
    MAX_HITS_PER_QUERY, NO_SURFACE, NO_TRIANGLE, QueryResultHeaderRecord, QueryResultRecord,
};
use dynamis_model::{BodyHandle, SoftBodyHandle, SurfaceDesc};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryHandle {
    pub batch: u64,
    pub index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryState {
    Pending,
    Retired,
    Lapsed,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QueryTarget {
    Collider {
        body: BodyHandle,
        collider: u32,
        triangle: Option<u32>,
        surface: Option<SurfaceDesc>,
    },
    Particle {
        body: SoftBodyHandle,
        particle: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryHit {
    pub target: QueryTarget,
    pub distance: f32,
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub step: u64,
}

impl QueryHit {
    pub fn body(&self) -> BodyHandle {
        match self.target {
            QueryTarget::Collider { body, .. } => body,
            QueryTarget::Particle { .. } => {
                panic!("a query hit against a soft particle carries no rigid body")
            }
        }
    }

    pub fn collider(&self) -> u32 {
        match self.target {
            QueryTarget::Collider { collider, .. } => collider,
            QueryTarget::Particle { .. } => {
                panic!("a query hit against a soft particle carries no collider")
            }
        }
    }

    pub fn triangle(&self) -> Option<u32> {
        match self.target {
            QueryTarget::Collider { triangle, .. } => triangle,
            QueryTarget::Particle { .. } => {
                panic!("a query hit against a soft particle carries no triangle")
            }
        }
    }

    pub fn surface(&self) -> Option<SurfaceDesc> {
        match self.target {
            QueryTarget::Collider { surface, .. } => surface,
            QueryTarget::Particle { .. } => {
                panic!("a query hit against a soft particle carries no surface")
            }
        }
    }

    pub fn soft(&self) -> SoftBodyHandle {
        match self.target {
            QueryTarget::Particle { body, .. } => body,
            QueryTarget::Collider { .. } => {
                panic!("a query hit against a rigid collider carries no soft body")
            }
        }
    }

    pub fn particle(&self) -> u32 {
        match self.target {
            QueryTarget::Particle { particle, .. } => particle,
            QueryTarget::Collider { .. } => {
                panic!("a query hit against a rigid collider carries no particle")
            }
        }
    }
}

pub(crate) struct QueryOutcome {
    pub hits: Vec<Vec<QueryHit>>,
    pub overflow: Vec<bool>,
}

struct QueryBatch {
    batch: u64,
    step: u64,
    width: usize,
    outcome: Option<QueryOutcome>,
}

pub(crate) struct QueryPool {
    batches: VecDeque<QueryBatch>,
}

impl QueryPool {
    pub(crate) fn new() -> Self {
        Self {
            batches: VecDeque::new(),
        }
    }

    pub(crate) fn submit(&mut self, batch: u64, step: u64, width: usize) {
        self.batches.push_back(QueryBatch {
            batch,
            step,
            width,
            outcome: None,
        });
    }

    pub(crate) fn hit(&self, handle: QueryHandle) -> Option<QueryHit> {
        self.hits(handle).first().copied()
    }

    pub(crate) fn hits(&self, handle: QueryHandle) -> &[QueryHit] {
        &self.outcome(handle).hits[handle.index as usize]
    }

    pub(crate) fn overflow(&self, handle: QueryHandle) -> bool {
        self.outcome(handle).overflow[handle.index as usize]
    }

    fn outcome(&self, handle: QueryHandle) -> &QueryOutcome {
        self.batch(handle)
            .and_then(|batch| batch.outcome.as_ref())
            .expect("query outcome read before it arrived")
    }

    pub(crate) fn is_current(&self, handle: QueryHandle) -> bool {
        self.batch(handle).is_some()
    }

    pub(crate) fn is_ready(&self, handle: QueryHandle) -> bool {
        self.batch(handle)
            .is_some_and(|batch| batch.outcome.is_some())
    }

    fn batch(&self, handle: QueryHandle) -> Option<&QueryBatch> {
        self.batches
            .iter()
            .find(|batch| batch.batch == handle.batch)
    }

    pub(crate) fn collect(
        &mut self,
        batch_id: u64,
        bytes: &[u8],
        surface: impl Fn(u32, u32) -> SurfaceDesc,
        collider_of: impl Fn(BodyHandle, u32) -> u32,
        particle_of: impl Fn(SoftBodyHandle, u32) -> u32,
    ) {
        let batch = self
            .batches
            .iter_mut()
            .find(|batch| batch.batch == batch_id)
            .unwrap_or_else(|| panic!("no query batch is registered for batch {batch_id}"));
        let step = batch.step;
        let records = dynamis_abi::decode::<QueryResultRecord>(bytes);
        let mut hits = vec![Vec::new(); batch.width];
        let mut overflow = vec![false; batch.width];
        for (index, result) in records.iter().take(batch.width).enumerate() {
            let QueryResultHeaderRecord {
                count,
                overflow: spilled,
                ..
            } = result.header;
            assert!(
                count <= MAX_HITS_PER_QUERY,
                "GPU query result exceeds the hit lane count"
            );
            hits[index] = result.hits[..count as usize]
                .iter()
                .map(|record| {
                    let target = query_target(record, &surface, &collider_of, &particle_of);
                    QueryHit {
                        target,
                        distance: record.distance,
                        point: record.point,
                        normal: record.normal,
                        step,
                    }
                })
                .collect();
            overflow[index] = spilled != 0;
        }
        batch.outcome = Some(QueryOutcome { hits, overflow });
        while self.batches.len() > dynamis_gpu::FACT_LAG {
            self.batches.pop_front();
        }
    }
}

fn query_target(
    record: &dynamis_abi::QueryHitRecord,
    surface: impl Fn(u32, u32) -> SurfaceDesc,
    collider_of: impl Fn(BodyHandle, u32) -> u32,
    particle_of: impl Fn(SoftBodyHandle, u32) -> u32,
) -> QueryTarget {
    let slot = record.scene_target & ENTRY_INDEX_MASK;
    match record.scene_target >> ENTRY_KIND_SHIFT {
        ENTRY_KIND_COLLIDER => {
            let body = BodyHandle {
                id: record.body_id,
                generation: record.body_generation,
            };
            QueryTarget::Collider {
                body,
                collider: collider_of(body, slot),
                triangle: (record.triangle != NO_TRIANGLE).then_some(record.triangle),
                surface: (record.surface != NO_SURFACE).then(|| surface(slot, record.surface)),
            }
        }
        ENTRY_KIND_PARTICLE => {
            let body = SoftBodyHandle {
                id: record.body_id,
                generation: record.body_generation,
            };
            QueryTarget::Particle {
                body,
                particle: particle_of(body, slot),
            }
        }
        kind => panic!("a query hit reports an unknown scene target kind {kind}"),
    }
}
