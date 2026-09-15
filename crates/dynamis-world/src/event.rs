use super::World;
use dynamis_abi::ContactEventRecord;
use dynamis_model::{BodyHandle, ContactEvent, ContactEventKind};

#[derive(Clone)]
pub(crate) struct Events {
    pub(crate) contact: Vec<ContactEvent>,
}

impl Events {
    pub(crate) const fn new() -> Self {
        Self {
            contact: Vec::new(),
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
        self.retire_device_facts();
        std::mem::take(&mut self.events.contact)
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
