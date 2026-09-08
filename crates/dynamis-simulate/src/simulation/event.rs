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

    pub(super) fn consume_events(&mut self, bytes: &[u8]) {
        let event_count = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        assert!(
            (event_count as usize) <= self.event_capacity,
            "GPU event count exceeds the event capacity"
        );
        let events_bytes =
            &bytes[12..12 + self.event_capacity * std::mem::size_of::<ContactEventRecord>()];
        let records: &[ContactEventRecord] = bytemuck::cast_slice(events_bytes);
        let mut fresh = Vec::with_capacity(event_count as usize);
        for record in &records[..event_count as usize] {
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
        if let Some(sink) = self.event_sink.as_mut() {
            for event in fresh.iter().copied() {
                sink(event);
            }
        }
        self.events.extend(fresh);
    }
}
