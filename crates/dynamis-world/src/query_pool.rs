use dynamis_abi::{MAX_HITS_PER_QUERY, QueryResultHeaderRecord, QueryResultRecord};
use dynamis_model::{BodyHandle, SurfaceDesc};
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
pub struct QueryHit {
    pub body: BodyHandle,
    pub collider: u32,
    pub distance: f32,
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub triangle: Option<u32>,
    pub surface: Option<SurfaceDesc>,
    pub step: u64,
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
                .map(|record| QueryHit {
                    body: BodyHandle {
                        id: record.body_id,
                        generation: record.body_generation,
                    },
                    collider: record.collider_index,
                    distance: record.distance,
                    point: record.point,
                    normal: record.normal,
                    triangle: (record.triangle != dynamis_abi::NO_TRIANGLE)
                        .then_some(record.triangle),
                    surface: (record.surface != dynamis_abi::NO_SURFACE)
                        .then(|| surface(record.collider_index, record.surface)),
                    step,
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
