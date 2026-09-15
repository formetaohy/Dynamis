use super::World;
use dynamis_abi::{COUNTER_IMPACTS, ImpactEventRecord};
use dynamis_gpu::EVENT_SLOTS;
use dynamis_model::{BodyHandle, ImpactEvent};
use std::collections::VecDeque;
use std::mem::size_of;

pub(crate) struct Impacts {
    pub(crate) impact: Vec<ImpactEvent>,
    pub(crate) due: VecDeque<(u64, u32)>,
}

impl Impacts {
    pub(crate) const fn new() -> Self {
        Self {
            impact: Vec::new(),
            due: VecDeque::new(),
        }
    }
}

impl World {
    pub fn collect_impacts(&mut self) -> Vec<ImpactEvent> {
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
        std::mem::take(&mut self.impacts.impact)
    }

    pub fn drain_impacts(&mut self) -> Vec<ImpactEvent> {
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
        self.sync_impacts();
        std::mem::take(&mut self.impacts.impact)
    }

    pub(crate) fn note_impacts_due(&mut self, step: u64) {
        let count = self.backend.measured[COUNTER_IMPACTS];
        if count > 0 {
            self.impacts.due.push_back((step, count));
        }
    }

    pub(crate) fn copy_impacts(&mut self, encoder: &mut dynamis_gpu::SubmissionEncoder) {
        while let Some((step, count)) = self.impacts.due.pop_front() {
            assert!(
                self.clock.step <= step + EVENT_SLOTS as u64,
                "impact segment for step {step} was overwritten before step {} could copy it",
                self.clock.step
            );
            let segment = self.backend.streams.rigid.impacts.size() / EVENT_SLOTS as u64;
            let offset = self.event_slot_of(step) as u64 * segment;
            let bytes = (count as u64 * size_of::<ImpactEventRecord>() as u64).min(segment);
            let regions = [(self.backend.streams.rigid.impacts.buffer(), offset, bytes)];
            let displaced =
                self.backend
                    .readback
                    .impacts
                    .declare(encoder, &regions, step, (step, count));
            if let Some((manifest, bytes)) = displaced {
                self.consume_impacts(manifest, &bytes);
            }
        }
    }

    pub(crate) fn sync_impacts(&mut self) {
        if self.impacts.due.is_empty() {
            return;
        }
        let device = self.backend.gpu.device().clone();
        let mut encoder = dynamis_gpu::SubmissionEncoder::new(&device, "dynamis impact readback");
        self.copy_impacts(&mut encoder);
        self.submit(encoder);
        for (manifest, bytes) in self.backend.readback.impacts.drain() {
            self.consume_impacts(manifest, &bytes);
        }
    }

    pub(crate) fn consume_impacts(&mut self, (step, count): (u64, u32), bytes: &[u8]) {
        let records = dynamis_abi::decode::<ImpactEventRecord>(bytes);
        assert!(
            records.len() <= count as usize,
            "an impact publication must not carry more impacts than it declared"
        );
        let mut fresh = Vec::with_capacity(records.len());
        for record in &records {
            fresh.push(ImpactEvent {
                first: BodyHandle {
                    id: record.first_id,
                    generation: record.first_generation,
                },
                second: BodyHandle {
                    id: record.second_id,
                    generation: record.second_generation,
                },
                point: record.point,
                normal: record.normal,
                impulse: record.impulse,
                friction_impulse: record.friction_impulse,
                step,
            });
        }
        self.impacts.impact.extend(fresh);
    }
}
