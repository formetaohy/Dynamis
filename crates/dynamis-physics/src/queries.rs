use crate::body::BodyHandle;
use crate::records::{QueryRecord, QueryResultRecord};
use crate::simulation::Simulation;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryHandle {
    pub(crate) slot: u32,
    pub(crate) generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryHit {
    pub body: BodyHandle,
    pub distance: f32,
    pub step: u64,
}

pub(crate) struct QueryBatch {
    step: u64,
    slots: Vec<u32>,
}

pub(crate) struct QueryTracker {
    capacity: usize,
    next_slot: usize,
    generations: Vec<u32>,
    pending: Vec<bool>,
    hits: Vec<Option<QueryHit>>,
    batches: VecDeque<QueryBatch>,
}

impl QueryTracker {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            next_slot: 0,
            generations: vec![0; capacity],
            pending: vec![false; capacity],
            hits: vec![None; capacity],
            batches: VecDeque::new(),
        }
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn generation(&self, slot: usize) -> u32 {
        self.generations[slot]
    }

    pub(crate) fn hit(&self, slot: usize) -> Option<QueryHit> {
        self.hits[slot]
    }

    pub(crate) fn allocate(&mut self) -> usize {
        let slot = self.next_slot;
        self.next_slot = (slot + 1) % self.capacity;
        assert!(
            !self.pending[slot],
            "query result ring exhausted; collect results with poll() or wait() before submitting more queries"
        );
        self.pending[slot] = true;
        self.generations[slot] += 1;
        slot
    }

    pub(crate) fn mark_batch(&mut self, step: u64, slots: Vec<u32>) {
        self.batches.push_back(QueryBatch { step, slots });
    }

    pub(crate) fn consume(&mut self, step: u64, records: &[QueryResultRecord]) {
        let batch = self
            .batches
            .pop_front()
            .expect("query readback arrived without a pending batch");
        assert_eq!(
            batch.step, step,
            "query readback arrived out of order"
        );
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

impl Simulation {
    pub fn raycast(&mut self, origin: [f32; 3], direction: [f32; 3], max_t: f32) -> QueryHandle {
        assert!(max_t > 0.0, "raycast distance must be positive");
        assert!(
            direction != [0.0; 3],
            "raycast direction must be non-zero"
        );
        let slot = self.query_tracker.allocate();
        let generation = self.query_tracker.generation(slot);
        self.queries
            .push(QueryRecord::ray(origin, direction, max_t, slot as u32));
        QueryHandle {
            slot: slot as u32,
            generation,
        }
    }

    pub fn sphere_query(&mut self, center: [f32; 3], radius: f32) -> QueryHandle {
        assert!(radius > 0.0, "sphere query radius must be positive");
        let slot = self.query_tracker.allocate();
        let generation = self.query_tracker.generation(slot);
        self.queries
            .push(QueryRecord::sphere(center, radius, slot as u32));
        QueryHandle {
            slot: slot as u32,
            generation,
        }
    }

    pub fn query_hit(&self, handle: QueryHandle) -> Option<QueryHit> {
        let slot = handle.slot as usize;
        assert!(
            slot < self.query_tracker.capacity(),
            "query handle {handle:?} is out of range"
        );
        assert_eq!(
            self.query_tracker.generation(slot),
            handle.generation,
            "query handle {handle:?} is stale"
        );
        self.query_tracker.hit(slot)
    }

    pub(crate) fn consume_queries(&mut self, step: u64, bytes: &[u8]) {
        let records: &[QueryResultRecord] = bytemuck::cast_slice(bytes);
        self.query_tracker.consume(step, records);
    }
}
