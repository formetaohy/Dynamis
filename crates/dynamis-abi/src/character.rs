use crate::CharacterInputRecord;
use crate::CharacterRecord;
use crate::CharacterStateRecord;
use crate::constant::NO_BODY;
use dynamis_model::{BodyHandle, CharacterDesc, CharacterState};

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
    /// The state a character answers where it is introduced: at rest, standing on ground its sweeps
    /// have yet to answer and therefore naming no support.
    pub const fn placed(owner: u32, generation: u32, position: [f32; 3]) -> Self {
        Self {
            position,
            vertical: 0.0,
            down_length: 0.0,
            grounded: 1,
            support: NO_BODY,
            support_generation: 0,
            owner,
            generation,
            _pad0: 0,
            _pad1: 0,
        }
    }

    pub const fn cleared() -> Self {
        Self {
            position: [0.0; 3],
            vertical: 0.0,
            down_length: 0.0,
            grounded: 0,
            support: NO_BODY,
            support_generation: 0,
            owner: NO_BODY,
            generation: 0,
            _pad0: 0,
            _pad1: 0,
        }
    }

    pub fn owns(&self, owner: u32, generation: u32) -> bool {
        self.owner == owner && self.generation == generation
    }

    /// The support a character stands on: the body its landing sweep answered, or none while the
    /// sweep answered no floor or the world no longer holds the body it named.
    pub fn support(&self) -> Option<BodyHandle> {
        (self.support != NO_BODY).then_some(BodyHandle {
            id: self.support,
            generation: self.support_generation,
        })
    }

    pub fn state(&self) -> CharacterState {
        CharacterState {
            position: self.position,
            vertical_speed: self.vertical,
            grounded: self.grounded != 0,
            support: self.support(),
        }
    }
}
