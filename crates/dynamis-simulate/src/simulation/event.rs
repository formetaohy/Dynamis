use super::Simulation;
use dynamis_layout::ContactEventRecord;
use dynamis_model::{BodyHandle, ContactEvent, ContactEventKind};
use std::collections::VecDeque;

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

impl Simulation {
    pub fn drain_events(&mut self) -> Vec<ContactEvent> {
        self.device.gpu.assert_alive();
        self.collect_readbacks();
        self.sync_events();
        std::mem::take(&mut self.events.contact)
    }

    pub fn set_event_sink(&mut self, sink: Option<Box<dyn FnMut(ContactEvent)>>) {
        self.events.sink = sink;
    }

    pub(crate) fn consume_events(&mut self, bytes: &[u8]) {
        let records = crate::records::decode::<ContactEventRecord>(bytes);
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
