use super::World;
use dynamis_abi::{COUNTER_EVENTS, ContactEventRecord};
use dynamis_gpu::EVENT_SLOTS;
use dynamis_model::{BodyHandle, ContactEvent, ContactEventKind};
use std::collections::VecDeque;
use std::mem::size_of;

pub(crate) struct Events {
    pub(crate) contact: Vec<ContactEvent>,
    pub(crate) due: VecDeque<(u64, u32)>,
}

impl Events {
    pub(crate) const fn new() -> Self {
        Self {
            contact: Vec::new(),
            due: VecDeque::new(),
        }
    }
}

impl World {
    pub fn collect_events(&mut self) -> Vec<ContactEvent> {
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
        std::mem::take(&mut self.events.contact)
    }

    pub fn drain_events(&mut self) -> Vec<ContactEvent> {
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
        self.sync_events();
        std::mem::take(&mut self.events.contact)
    }

    pub(crate) fn note_events_due(&mut self, step: u64) {
        let count = self.backend.measured[COUNTER_EVENTS];
        if count > 0 {
            eprintln!("EVENT due step {step} count {count}");
            self.events.due.push_back((step, count));
        }
    }

    pub(crate) fn copy_events(&mut self, encoder: &mut dynamis_gpu::SubmissionEncoder) {
        while let Some((step, count)) = self.events.due.pop_front() {
            assert!(
                self.clock.step <= step + EVENT_SLOTS as u64,
                "event segment for step {step} was overwritten before step {} could copy it",
                self.clock.step
            );
            let segment = self.backend.streams.rigid.events.size() / EVENT_SLOTS as u64;
            let offset = (step % EVENT_SLOTS as u64) * segment;
            let bytes = (count as u64 * size_of::<ContactEventRecord>() as u64).min(segment);
            let regions = [(self.backend.streams.rigid.events.buffer(), offset, bytes)];
            eprintln!(
                "EVENT declare step {step} count {count} due {}",
                self.events.due.len()
            );
            let displaced = self
                .backend
                .readback
                .events
                .declare(encoder, &regions, step, count);
            if let Some((count, bytes)) = displaced {
                self.consume_events(count, &bytes);
            }
        }
    }

    pub(crate) fn sync_events(&mut self) {
        if self.events.due.is_empty() {
            return;
        }
        let device = self.backend.gpu.device().clone();
        let mut encoder = dynamis_gpu::SubmissionEncoder::new(&device, "dynamis event readback");
        self.copy_events(&mut encoder);
        self.submit(encoder);
        for (count, bytes) in self.backend.readback.events.drain() {
            self.consume_events(count, &bytes);
        }
    }

    pub(crate) fn consume_events(&mut self, count: u32, bytes: &[u8]) {
        let records = dynamis_abi::decode::<ContactEventRecord>(bytes);
        assert!(
            records.len() <= count as usize,
            "an event publication must not carry more events than it declared"
        );
        let mut fresh = Vec::with_capacity(records.len());
        for record in &records {
            let kind = match record.kind {
                dynamis_abi::EVENT_BEGIN => ContactEventKind::Begin,
                dynamis_abi::EVENT_END => ContactEventKind::End,
                dynamis_abi::EVENT_PERSIST => ContactEventKind::Persist,
                other => panic!("GPU event record has an invalid kind {other}"),
            };
            fresh.push(ContactEvent {
                kind,
                first: BodyHandle {
                    id: record.first_id,
                    generation: record.first_generation,
                },
                second: BodyHandle {
                    id: record.second_id,
                    generation: record.second_generation,
                },
                sensor: record.sensor == 1,
                point: record.point,
                normal: record.normal,
            });
        }
        self.events.contact.extend(fresh);
    }
}
