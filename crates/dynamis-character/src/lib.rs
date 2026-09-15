pub use dynamis_model::CharacterDesc;
use dynamis_model::{BodyHandle, CharacterHandle, CharacterInput, CharacterState};
use dynamis_world::{Observation, World};

pub struct Character {
    handle: CharacterHandle,
}

impl Character {
    pub fn spawn(world: &mut World, position: [f32; 3], desc: CharacterDesc) -> Self {
        Self {
            handle: world.add_character(position, desc),
        }
    }

    pub const fn handle(&self) -> CharacterHandle {
        self.handle
    }

    pub fn body(&self, world: &World) -> BodyHandle {
        world.character_body(self.handle)
    }

    pub fn drive(&self, world: &mut World, direction: [f32; 3], jump: bool) {
        world.set_character_input(self.handle, CharacterInput { direction, jump });
    }

    pub fn try_state(&self, world: &mut World) -> Option<Observation<CharacterState>> {
        world.try_character_state(self.handle)
    }

    pub fn inspect_state(&self, world: &mut World) -> CharacterState {
        world.inspect_character_state(self.handle)
    }
}
