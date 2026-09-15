use crate::CharacterInputRecord;
use crate::CharacterRecord;
use crate::CharacterStateRecord;
use crate::constant::NO_BODY;
use dynamis_model::{CharacterDesc, CharacterState};

impl CharacterRecord {
    pub fn build(desc: &CharacterDesc, body_id: u32, generation: u32) -> Self {
        desc.assert_valid();
        Self {
            body_id,
            generation,
            radius: desc.radius,
            half_height: desc.half_height,
            step_height: desc.step_height,
            cos_slope_limit: desc.slope_cosine(),
            max_speed: desc.max_speed,
            jump_speed: desc.jump_speed,
        }
    }

    pub const fn cleared() -> Self {
        Self {
            body_id: NO_BODY,
            generation: 0,
            radius: 0.0,
            half_height: 0.0,
            step_height: 0.0,
            cos_slope_limit: 0.0,
            max_speed: 0.0,
            jump_speed: 0.0,
        }
    }

    pub const fn is_live(&self) -> bool {
        self.body_id != NO_BODY
    }
}

impl CharacterInputRecord {
    pub const fn idle() -> Self {
        Self {
            direction: [0.0; 3],
            jump: 0,
        }
    }
}

impl CharacterStateRecord {
    pub fn spawn(owner: u32, generation: u32, position: [f32; 3]) -> Self {
        Self {
            position,
            vertical: 0.0,
            down_length: 0.0,
            grounded: 1,
            owner,
            generation,
        }
    }

    pub const fn cleared() -> Self {
        Self {
            position: [0.0; 3],
            vertical: 0.0,
            down_length: 0.0,
            grounded: 0,
            owner: NO_BODY,
            generation: 0,
        }
    }

    pub fn owns(&self, owner: u32, generation: u32) -> bool {
        self.owner == owner && self.generation == generation
    }

    pub fn state(&self) -> CharacterState {
        CharacterState {
            position: self.position,
            vertical_speed: self.vertical,
            grounded: self.grounded != 0,
        }
    }
}
