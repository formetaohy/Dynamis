use dynamis_layout::{MAX_HITS_PER_QUERY, QueryResultHeader, QueryResultRecord};
use dynamis_model::BodyHandle;
use std::collections::VecDeque;

/// Identifies one submitted query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryHandle {
    pub batch: u64,
    pub index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryHit {
    pub body: BodyHandle,
    pub collider: u32,
    pub distance: f32,
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub step: u64,
}

/// One query's answers, as the device left them.
pub struct QueryOutcome {
    pub hits: Vec<Vec<QueryHit>>,
    pub overflow: Vec<bool>,
}

struct QueryBatch {
    batch: u64,
    step: u64,
    width: usize,
    outcome: Option<QueryOutcome>,
}

/// The results of every query batch the device has been asked for and the host has
/// not yet read. A batch is one contiguous run of [`QueryResultRecord`] lanes, so its
/// size follows the number of queries submitted, nothing else.
pub(crate) struct QueryPool {
    batches: VecDeque<QueryBatch>,
}

impl QueryPool {
    pub(crate) fn new() -> Self {
        Self {
            batches: VecDeque::new(),
        }
    }

    /// Registers a submitted batch under the id the host handed out for it.
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

    /// Whether a handle names a batch whose results have already been dropped.
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

    /// Brings a landed batch's results home. The readback sequence is the batch id, so
    /// a step that flushed queries outside the step loop cannot be confused with it.
    pub(crate) fn collect(&mut self, batch_id: u64, bytes: &[u8]) {
        let batch = self
            .batches
            .iter_mut()
            .find(|batch| batch.batch == batch_id)
            .unwrap_or_else(|| panic!("no query batch is registered for batch {batch_id}"));
        let step = batch.step;
        let records = crate::records::records::<QueryResultRecord>(bytes);
        let mut hits = vec![Vec::new(); batch.width];
        let mut overflow = vec![false; batch.width];
        for (index, result) in records.iter().take(batch.width).enumerate() {
            let QueryResultHeader {
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
                    step,
                })
                .collect();
            hits[index].sort_unstable_by(|left, right| left.distance.total_cmp(&right.distance));
            overflow[index] = spilled != 0;
        }
        batch.outcome = Some(QueryOutcome { hits, overflow });
        while self
            .batches
            .front()
            .is_some_and(|oldest| oldest.outcome.is_some() && oldest.step < step)
        {
            self.batches.pop_front();
        }
    }
}
