use super::World;
use super::device::Facts;
use dynamis_abi::ContactEventRecord;
use dynamis_model::{ContactEvent, ContactEventKind};
use dynamis_scene::scene_target;

#[derive(Clone)]
pub(crate) struct EventStore {
    pub(crate) contact: Vec<ContactEvent>,
}

impl EventStore {
    pub(crate) const fn new() -> Self {
        Self {
            contact: Vec::new(),
        }
    }
}

impl World {
    pub fn collect_events(&mut self) -> Vec<ContactEvent> {
        self.sync(Facts::Arrived);
        std::mem::take(&mut self.events.contact)
    }

    pub fn drain_events(&mut self) -> Vec<ContactEvent> {
        self.sync(Facts::Retired);
        std::mem::take(&mut self.events.contact)
    }

    pub(crate) fn consume_events(&mut self, count: u32, bytes: &[u8]) {
        let records = dynamis_abi::decode::<ContactEventRecord>(bytes);
        assert!(
            records.len() <= count as usize,
            "an event publication must not carry more events than it declared"
        );
        let bodies = &self.bodies;
        let pool = &self.colliders;
        let soft = &self.soft;
        let collider_of = |body, slot| crate::collider::local_collider_of(bodies, pool, body, slot);
        let particle_of = |body, slot| soft.local_particle_of(body, slot);
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
                first: scene_target(
                    record.first_target,
                    record.first_id,
                    record.first_generation,
                    collider_of,
                    particle_of,
                ),
                second: scene_target(
                    record.second_target,
                    record.second_id,
                    record.second_generation,
                    collider_of,
                    particle_of,
                ),
                sensor: record.sensor == 1,
                point: record.point,
                normal: record.normal,
            });
        }
        self.events.contact.extend(fresh);
    }
}
