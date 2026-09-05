use dynamis_layout::QueryResultRecord;
use dynamis_model::BodyHandle;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryHandle {
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryHit {
    pub body: BodyHandle,
    pub distance: f32,
    pub step: u64,
}

pub struct QueryBatch {
    step: u64,
    slots: Vec<u32>,
}

pub struct QueryPool {
    capacity: usize,
    next_slot: usize,
    generations: Vec<u32>,
    pending: Vec<bool>,
    hits: Vec<Option<QueryHit>>,
    batches: VecDeque<QueryBatch>,
}

impl QueryPool {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            next_slot: 0,
            generations: vec![0; capacity],
            pending: vec![false; capacity],
            hits: vec![None; capacity],
            batches: VecDeque::new(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn generation(&self, slot: usize) -> u32 {
        self.generations[slot]
    }

    pub fn hit(&self, slot: usize) -> Option<QueryHit> {
        self.hits[slot]
    }

    pub fn allocate(&mut self) -> usize {
        let slot = self.next_slot;
        self.next_slot = (slot + 1) % self.capacity;
        assert!(
            !self.pending[slot],
            "query slot ring exhausted; collect results with poll() or wait() before submitting more queries"
        );
        self.pending[slot] = true;
        self.generations[slot] += 1;
        slot
    }

    pub fn mark_batch(&mut self, step: u64, slots: Vec<u32>) {
        self.batches.push_back(QueryBatch { step, slots });
    }

    pub fn consume(&mut self, step: u64, records: &[QueryResultRecord]) {
        let batch = self
            .batches
            .pop_front()
            .expect("query readback arrived without a pending batch");
        assert_eq!(batch.step, step, "query readback arrived out of order");
        for slot in batch.slots {
            let index = slot as usize;
            let record = records[index];
            self.hits[index] = if record.hit == 1 {
                Some(QueryHit {
                    body: BodyHandle {
                        id: record.body_id,
                        generation: record.body_generation,
                    },
                    distance: record.distance,
                    step: batch.step,
                })
            } else {
                None
            };
            self.pending[index] = false;
        }
    }
}
