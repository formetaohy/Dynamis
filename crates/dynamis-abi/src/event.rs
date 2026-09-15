use crate::SoftAnnouncementRecord;
use crate::constant::{EVENT_MODE_BEGIN_END, EVENT_MODE_PERSIST, NO_BODY, NO_SLOT};
use dynamis_model::ContactEventMode;

pub const fn event_flags(mode: ContactEventMode) -> u32 {
    match mode {
        ContactEventMode::None => 0,
        ContactEventMode::BeginEnd => EVENT_MODE_BEGIN_END,
        ContactEventMode::Persist => EVENT_MODE_BEGIN_END | EVENT_MODE_PERSIST,
    }
}

impl SoftAnnouncementRecord {
    pub const fn cleared() -> Self {
        Self {
            scene_target: 0,
            id: NO_BODY,
            generation: 0,
            step: NO_SLOT,
            sensor: 0,
            announced: 0,
            _pad0: [0; 2],
        }
    }

    pub const fn holds(&self, target: u32, id: u32, generation: u32) -> bool {
        self.announced != 0
            && self.scene_target == target
            && self.id == id
            && self.generation == generation
    }
}
