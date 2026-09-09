use super::Simulation;
use dynamis_layout::{COUNTER_EVENTS, ContactEventRecord};
use dynamis_model::{BodyHandle, ContactEvent, ContactEventKind};

impl Simulation {
    pub fn drain_events(&mut self) -> Vec<ContactEvent> {
        std::mem::take(&mut self.events)
    }

    pub fn set_event_sink(&mut self, sink: Option<Box<dyn FnMut(ContactEvent)>>) {
        self.event_sink = sink;
    }

    /// The count travels in the same pack as the records it describes, so the two can
    /// never be paired with different steps.
    pub(crate) fn consume_events(&mut self, _step: u64, bytes: &[u8]) {
        let count = self.observed[COUNTER_EVENTS] as usize;
        let records = crate::records::records::<ContactEventRecord>(bytes);
        assert!(
            count <= records.len(),
            "GPU event count exceeds the event stream"
        );
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
        if let Some(sink) = self.event_sink.as_mut() {
            for event in fresh.iter().copied() {
                sink(event);
            }
        }
        self.events.extend(fresh);
    }
}
