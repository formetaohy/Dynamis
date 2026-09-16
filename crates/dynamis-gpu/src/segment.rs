use crate::{Publication, SEGMENT_COUNT, Stream, SubmissionEncoder};
use std::collections::VecDeque;
use wgpu::{BufferAddress, Device};

pub struct SegmentRing {
    label: &'static str,
    ring: Publication<(u64, u32)>,
    due: VecDeque<(u64, u32)>,
}

impl SegmentRing {
    pub const fn new(label: &'static str) -> Self {
        Self {
            label,
            ring: Publication::new(label, Publication::<(u64, u32)>::DEPTH),
            due: VecDeque::new(),
        }
    }

    pub fn reserve(&mut self, device: &Device, budget: BufferAddress) -> bool {
        self.ring.reserve(device, budget)
    }

    pub fn pending(&self) -> bool {
        !self.due.is_empty()
    }

    pub fn close(&mut self, step: u64, count: u32) {
        if count > 0 {
            self.due.push_back((step, count));
        }
    }

    pub fn copy(
        &mut self,
        encoder: &mut SubmissionEncoder,
        source: &Stream,
        now: u64,
    ) -> Vec<(u64, u32, Vec<u8>)> {
        let mut displaced = Vec::new();
        while let Some((step, count)) = self.due.pop_front() {
            assert!(
                now <= step + SEGMENT_COUNT as u64,
                "{} for step {step} were overwritten before step {now} could copy them",
                self.label,
            );
            let segment = source.size() / SEGMENT_COUNT as BufferAddress;
            let offset = (step % SEGMENT_COUNT as u64) * segment;
            let bytes = (u64::from(count) * source.stride()).min(segment);
            let regions = [(source.buffer(), offset, bytes)];
            if let Some(((step, count), bytes)) =
                self.ring.declare(encoder, &regions, step, (step, count))
            {
                displaced.push((step, count, bytes));
            }
        }
        displaced
    }

    pub fn collect(&mut self) -> Vec<(u64, u32, Vec<u8>)> {
        self.ring
            .collect()
            .into_iter()
            .map(|((step, count), bytes)| (step, count, bytes))
            .collect()
    }

    pub fn drain(&mut self) -> Vec<(u64, u32, Vec<u8>)> {
        self.ring
            .drain()
            .into_iter()
            .map(|((step, count), bytes)| (step, count, bytes))
            .collect()
    }
}
