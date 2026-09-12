use super::World;
use crate::dynamics::EVENT_SLOTS;
use dynamis_layout::{COUNTER_EVENTS, ContactEventRecord};
use dynamis_model::{BodyHandle, ContactEvent, ContactEventKind};
use std::collections::VecDeque;
use std::mem::size_of;

pub(crate) struct Events {
    pub(crate) contact: Vec<ContactEvent>,
    pub(crate) due: VecDeque<(u64, u32)>,
    pub(crate) sink: Option<Box<dyn FnMut(ContactEvent)>>,
}

impl Events {
    pub(crate) fn new() -> Self {
        Self {
            contact: Vec::new(),
            due: VecDeque::new(),
            sink: None,
        }
    }
}

impl World {
    pub fn drain_events(&mut self) -> Vec<ContactEvent> {
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
        self.sync_events();
        std::mem::take(&mut self.events.contact)
    }

    pub fn set_event_sink(&mut self, sink: Option<Box<dyn FnMut(ContactEvent)>>) {
        self.events.sink = sink;
    }

    pub(crate) fn note_events_due(&mut self, step: u64) {
        let count = self.backend.measured[COUNTER_EVENTS];
        if count > 0 {
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
            let displaced = self.backend.streams.readback.events.enqueue(
                encoder,
                self.backend.streams.rigid.events.buffer(),
                offset,
                bytes,
                step,
            );
            if let Some((_, bytes)) = displaced {
                self.consume_events(&bytes);
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
        for (_, bytes) in self.backend.streams.readback.events.drain() {
            self.consume_events(&bytes);
        }
    }

    pub(crate) fn consume_events(&mut self, bytes: &[u8]) {
        let records = dynamis_layout::decode::<ContactEventRecord>(bytes);
        let count = records.len();
        let mut fresh = Vec::with_capacity(count);
        for record in &records[..count] {
            let kind = match record.kind {
                dynamis_layout::EVENT_BEGIN => ContactEventKind::Begin,
                dynamis_layout::EVENT_END => ContactEventKind::End,
                dynamis_layout::EVENT_PERSIST => ContactEventKind::Persist,
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
        if let Some(sink) = self.events.sink.as_mut() {
            for event in fresh.iter().copied() {
                sink(event);
            }
        }
        self.events.contact.extend(fresh);
    }
}
