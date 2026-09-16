use super::World;
use dynamis_abi::ImpactEventRecord;
use dynamis_model::{BodyHandle, ImpactEvent};

#[derive(Clone)]
pub(crate) struct ImpactStore {
    pub(crate) impact: Vec<ImpactEvent>,
}

impl ImpactStore {
    pub(crate) const fn new() -> Self {
        Self { impact: Vec::new() }
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
        self.retire_device_facts();
        std::mem::take(&mut self.impacts.impact)
    }

    pub(crate) fn consume_impacts(&mut self, step: u64, count: u32, bytes: &[u8]) {
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
