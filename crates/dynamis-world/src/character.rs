use super::World;
use super::pool::Pool;
use dynamis_abi::{CHARACTER_SWEEPS, CharacterInputRecord, CharacterRecord, CharacterStateRecord};
use dynamis_model::{
    BodyDesc, BodyHandle, CharacterDesc, CharacterHandle, CharacterInput, ColliderDesc, Shape,
};
use dynamis_rigid::RigidStreams;

#[derive(Clone, Copy)]
struct Slot {
    body: Option<BodyHandle>,
    record: CharacterRecord,
    input: CharacterInputRecord,
    state: CharacterStateRecord,
}

impl Slot {
    const RETIRED: Self = Self {
        body: None,
        record: CharacterRecord::cleared(),
        input: CharacterInputRecord::idle(),
        state: CharacterStateRecord::cleared(),
    };
}

#[derive(Clone)]
pub(crate) struct CharacterStore {
    pub(crate) pool: Pool<CharacterHandle>,
    slots: Vec<Slot>,
    pending_inputs: u32,
    pub(crate) last_inputs: u32,
}

impl CharacterStore {
    pub(crate) const fn new() -> Self {
        Self {
            pool: Pool::vacate("character"),
            slots: Vec::new(),
            pending_inputs: 0,
            last_inputs: 0,
        }
    }

    pub(crate) fn count(&self) -> u32 {
        self.pool.len()
    }

    pub(crate) fn live_slots(&self) -> Vec<(CharacterHandle, u32)> {
        let mut slots = self
            .pool
            .alive()
            .iter()
            .map(|handle| (*handle, handle.id))
            .collect::<Vec<_>>();
        slots.sort_unstable_by_key(|(_, slot)| *slot);
        slots
    }

    pub(crate) fn slots(&self) -> u32 {
        self.pool.ids()
    }

    pub(crate) fn owns_body(&self, id: u32) -> bool {
        self.slots
            .iter()
            .any(|slot| slot.body.is_some_and(|body| body.id == id))
    }

    pub(crate) fn pending(&self) -> bool {
        self.pending_inputs > 0
    }

    pub(crate) fn consume(&mut self) {
        self.last_inputs = std::mem::take(&mut self.pending_inputs);
    }

    pub(crate) fn clear_work(&mut self) {
        self.last_inputs = 0;
    }

    pub(crate) fn validate(&self, handle: CharacterHandle) {
        self.pool.validate(handle);
    }

    pub(crate) fn slot_of(&self, handle: CharacterHandle) -> u32 {
        self.pool.row_of(handle)
    }

    pub(crate) fn body_of(&self, handle: CharacterHandle) -> BodyHandle {
        self.slots[self.slot_of(handle) as usize]
            .body
            .expect("a live character owns its body")
    }

    pub(crate) fn spawn(
        &mut self,
        body: BodyHandle,
        record: CharacterRecord,
        position: [f32; 3],
    ) -> CharacterHandle {
        let handle = self.pool.acquire();
        let slot = self.pool.insert(handle);
        if self.slots.len() <= slot as usize {
            self.slots.resize(slot as usize + 1, Slot::RETIRED);
        }
        self.slots[slot as usize] = Slot {
            body: Some(body),
            record,
            input: CharacterInputRecord::idle(),
            state: CharacterStateRecord::spawn(handle.id, handle.generation, position),
        };
        handle
    }

    pub(crate) fn set_input(&mut self, handle: CharacterHandle, input: CharacterInput) {
        let slot = self.slot_of(handle) as usize;
        self.slots[slot].input = CharacterInputRecord {
            direction: input.direction,
            jump: u32::from(input.jump),
        };
        self.pool.mark(handle);
        self.pending_inputs += 1;
    }

    pub(crate) fn retire(&mut self, handle: CharacterHandle) {
        let slot = self.pool.retire(handle).row as usize;
        self.slots[slot] = Slot::RETIRED;
    }

    pub(crate) fn upload(&mut self, queue: &wgpu::Queue, streams: &RigidStreams) {
        for slot in self.pool.take_retired() {
            reset_scratch(queue, streams, slot);
            write_slot(queue, streams, slot, &Slot::RETIRED);
        }
        for slot in self.pool.take_fresh() {
            reset_scratch(queue, streams, slot);
            write_slot(queue, streams, slot, &self.slots[slot as usize]);
        }
        for slot in self.pool.take_dirty() {
            write_input(queue, streams, slot, &self.slots[slot as usize]);
        }
    }
}

fn write_slot(queue: &wgpu::Queue, streams: &RigidStreams, slot: u32, record: &Slot) {
    let at = u64::from(slot);
    streams.characters.write_at(
        queue,
        at * streams.characters.stride(),
        bytemuck::bytes_of(&record.record),
    );
    streams.character_inputs.write_at(
        queue,
        at * streams.character_inputs.stride(),
        bytemuck::bytes_of(&record.input),
    );
    streams.character_states.write_at(
        queue,
        at * streams.character_states.stride(),
        bytemuck::bytes_of(&record.state),
    );
}

fn write_input(queue: &wgpu::Queue, streams: &RigidStreams, slot: u32, record: &Slot) {
    streams.character_inputs.write_at(
        queue,
        u64::from(slot) * streams.character_inputs.stride(),
        bytemuck::bytes_of(&record.input),
    );
}

fn reset_scratch(queue: &wgpu::Queue, streams: &RigidStreams, slot: u32) {
    crate::query::write_inert_queries(
        queue,
        &streams.character_sweeps,
        &streams.character_hits,
        slot * CHARACTER_SWEEPS,
        CHARACTER_SWEEPS,
    );
}

impl World {
    pub fn add_character(&mut self, position: [f32; 3], desc: CharacterDesc) -> CharacterHandle {
        desc.assert_valid();
        assert!(
            position.iter().all(|value| value.is_finite()),
            "a character position must be finite"
        );
        let body = self.spawn(
            BodyDesc::new(ColliderDesc::new(Shape::capsule(
                desc.radius,
                desc.half_height,
            )))
            .position(position)
            .kinematic(true),
        );
        let record = CharacterRecord::build(&desc, body.id, body.generation);
        self.characters.spawn(body, record, position)
    }

    pub fn remove_character(&mut self, handle: CharacterHandle) {
        let body = self.characters.body_of(handle);
        self.characters.retire(handle);
        self.observed.characters.stop_watching(handle.id);
        self.remove(body);
    }

    pub fn set_character_input(&mut self, handle: CharacterHandle, input: CharacterInput) {
        self.characters.validate(handle);
        assert!(
            input.direction.iter().all(|value| value.is_finite()),
            "a character direction must be finite"
        );
        self.characters.set_input(handle, input);
    }

    pub fn characters(&self) -> &[CharacterHandle] {
        self.characters.pool.alive()
    }

    pub fn character_count(&self) -> usize {
        self.characters.pool.len() as usize
    }

    pub fn character_body(&self, handle: CharacterHandle) -> BodyHandle {
        self.characters.body_of(handle)
    }
}
