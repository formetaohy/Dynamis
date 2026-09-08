use dynamis_layout::{MAX_HITS_PER_QUERY, QueryHitRecord, QueryResultHeader};
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
    pub collider: u32,
    pub distance: f32,
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub step: u64,
}

struct QueryBatch {
    step: u64,
    slots: Vec<u32>,
}

pub(crate) struct QueryPool {
    capacity: usize,
    next_slot: usize,
    generations: Vec<u32>,
    pending: Vec<bool>,
    hits: Vec<Vec<QueryHit>>,
    overflow: Vec<bool>,
    batches: VecDeque<QueryBatch>,
}

impl QueryPool {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            next_slot: 0,
            generations: vec![0; capacity],
            pending: vec![false; capacity],
            hits: vec![Vec::new(); capacity],
            overflow: vec![false; capacity],
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
        let hits = &self.hits[slot];
        hits.first().copied()
    }

    pub fn hits(&self, slot: usize) -> &[QueryHit] {
        &self.hits[slot]
    }

    pub fn overflow(&self, slot: usize) -> bool {
        self.overflow[slot]
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

    pub fn consume(&mut self, step: u64, bytes: &[u8]) {
        let batch = self
            .batches
            .pop_front()
            .expect("query readback arrived without a pending batch");
        assert_eq!(batch.step, step, "query readback arrived out of order");
        for slot in batch.slots {
            self.consume_slot(step, slot as usize, bytes);
        }
    }

    fn consume_slot(&mut self, step: u64, slot: usize, bytes: &[u8]) {
        let per_header = std::mem::size_of::<QueryResultHeader>();
        let per_hit = std::mem::size_of::<QueryHitRecord>();
        let header_offset = slot * per_header;
        let hits_offset = self.capacity * per_header + slot * MAX_HITS_PER_QUERY as usize * per_hit;
        let header: QueryResultHeader = bytemuck::cast_slice::<u8, QueryResultHeader>(
            &bytes[header_offset..header_offset + per_header],
        )[0];
        let records: &[QueryHitRecord] = bytemuck::cast_slice::<u8, QueryHitRecord>(
            &bytes[hits_offset..hits_offset + MAX_HITS_PER_QUERY as usize * per_hit],
        );
        assert!(
            header.count <= MAX_HITS_PER_QUERY,
            "GPU query result exceeds the slot capacity"
        );
        let mut hits = records[..header.count as usize]
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
            .collect::<Vec<_>>();
        hits.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
        assert!(
            self.pending[slot],
            "query readback arrived for an idle slot"
        );
        self.hits[slot] = hits;
        self.overflow[slot] = header.overflow != 0;
        self.pending[slot] = false;
    }
}
