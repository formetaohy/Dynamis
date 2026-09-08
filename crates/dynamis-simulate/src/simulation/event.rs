use super::Simulation;
use dynamis_layout::ContactEventRecord;
use dynamis_model::{BodyHandle, ContactEvent, ContactEventKind};

impl Simulation {
    pub fn drain_events(&mut self) -> Vec<ContactEvent> {
        std::mem::take(&mut self.events)
    }

    pub fn set_event_sink(&mut self, sink: Option<Box<dyn FnMut(ContactEvent)>>) {
        self.event_sink = sink;
    }

    pub(super) fn consume_events(&mut self, step: u64, bytes: &[u8]) {
        let event_count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
        let events_bytes =
            &bytes[12..12 + self.event_capacity * std::mem::size_of::<ContactEventRecord>()];
        let records: &[ContactEventRecord] = bytemuck::cast_slice(events_bytes);
        let count = event_count.min(self.event_capacity);
        for record in &records[..count] {
            if record.first_id == u32::MAX || record.first_generation == 0 {
                continue;
            }
            let kind = match record.kind {
                dynamis_layout::EVENT_BEGIN => ContactEventKind::Begin,
                dynamis_layout::EVENT_END => ContactEventKind::End,
                dynamis_layout::EVENT_PERSIST => ContactEventKind::Persist,
                _ => continue,
            };
            self.events.push(ContactEvent {
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
        let _ = step;
        if let Some(sink) = self.event_sink.as_mut() {
            for event in self.events.iter().copied() {
                sink(event);
            }
        }
    }
}
