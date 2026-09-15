use super::World;
use super::ids::IdSpace;
use dynamis_abi::{CHARACTER_SWEEPS, CharacterInputRecord, CharacterRecord, CharacterStateRecord};
use dynamis_model::{
    BodyDesc, BodyHandle, CharacterDesc, CharacterHandle, CharacterInput, ColliderDesc, Shape,
};
use dynamis_rigid::RigidStreams;

#[derive(Clone, Copy)]
struct Slot {
    handle: Option<CharacterHandle>,
    body: Option<BodyHandle>,
    record: CharacterRecord,
    input: CharacterInputRecord,
    state: CharacterStateRecord,
}

impl Slot {
    const RETIRED: Self = Self {
        handle: None,
        body: None,
        record: CharacterRecord::cleared(),
        input: CharacterInputRecord::idle(),
        state: CharacterStateRecord::cleared(),
    };
}

#[derive(Clone)]
pub(crate) struct Characters {
    pub(crate) live: Vec<CharacterHandle>,
    ids: IdSpace,
    slots: Vec<Slot>,
    fresh: Vec<u32>,
    dirty: Vec<u32>,
    retired: Vec<u32>,
    pending_inputs: u32,
    pub(crate) last_inputs: u32,
}

impl Characters {
    pub(crate) const fn new() -> Self {
        Self {
            live: Vec::new(),
            ids: IdSpace::new(),
            slots: Vec::new(),
            fresh: Vec::new(),
            dirty: Vec::new(),
            retired: Vec::new(),
            pending_inputs: 0,
            last_inputs: 0,
        }
    }

    pub(crate) fn count(&self) -> u32 {
        self.live.len() as u32
    }

    pub(crate) fn live_slots(&self) -> Vec<(CharacterHandle, u32)> {
        let mut slots = self
            .live
            .iter()
            .map(|handle| (*handle, handle.id))
            .collect::<Vec<_>>();
        slots.sort_unstable_by_key(|(_, slot)| *slot);
        slots
    }

    pub(crate) fn slots(&self) -> u32 {
        self.ids.len() as u32
    }

    pub(crate) fn owns_body(&self, id: u32) -> bool {
        self.live.iter().any(|handle| {
            self.slots[handle.id as usize]
                .body
                .is_some_and(|body| body.id == id)
        })
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
        let id = handle.id as usize;
        if id >= self.ids.len() {
            panic!("character handle {handle:?} is out of range");
        }
        if self.ids.generation(handle.id) != handle.generation {
            panic!("character handle {handle:?} is stale");
        }
        if self.slots[id].handle != Some(handle) {
            panic!("character handle {handle:?} is not alive");
        }
    }

    pub(crate) fn slot_of(&self, handle: CharacterHandle) -> u32 {
        self.validate(handle);
        handle.id
    }

    pub(crate) fn body_of(&self, handle: CharacterHandle) -> BodyHandle {
        self.slots[self.slot_of(handle) as usize]
            .body
            .expect("a live character owns its body")
    }

    pub(crate) fn spawn(
        &mut self,
        handle: CharacterHandle,
        body: BodyHandle,
        record: CharacterRecord,
        position: [f32; 3],
    ) {
        let slot = handle.id as usize;
        if self.slots.len() <= slot {
            self.slots.resize(slot + 1, Slot::RETIRED);
        }
        self.slots[slot] = Slot {
            handle: Some(handle),
            body: Some(body),
            record,
            input: CharacterInputRecord::idle(),
            state: CharacterStateRecord::spawn(handle.id, handle.generation, position),
        };
        self.live.push(handle);
        self.fresh.push(handle.id);
    }

    pub(crate) fn set_input(&mut self, handle: CharacterHandle, input: CharacterInput) {
        let index = self.slot_of(handle) as usize;
        let slot = &mut self.slots[index];
        slot.input = CharacterInputRecord {
            direction: input.direction,
            jump: u32::from(input.jump),
        };
        self.dirty.push(handle.id);
        self.pending_inputs += 1;
    }

    pub(crate) fn retire(&mut self, handle: CharacterHandle) {
        let slot = self.slot_of(handle) as usize;
        self.slots[slot] = Slot::RETIRED;
        self.live.retain(|live| *live != handle);
        self.retired.push(handle.id);
        self.ids.release(handle.id);
    }

    pub(crate) fn upload(&mut self, queue: &wgpu::Queue, streams: &RigidStreams) {
        let retired = take_sorted(&mut self.retired);
        let fresh = take_sorted(&mut self.fresh);
        let dirty = take_sorted(&mut self.dirty);
        for slot in retired {
            reset_slot(queue, streams, slot);
            write_record(queue, streams, slot, &Slot::RETIRED);
        }
        for slot in fresh {
            reset_slot(queue, streams, slot);
            write_record(queue, streams, slot, &self.slots[slot as usize]);
        }
        for slot in dirty {
            write_input(queue, streams, slot, &self.slots[slot as usize]);
        }
    }
}

fn write_record(queue: &wgpu::Queue, streams: &RigidStreams, slot: u32, slot_record: &Slot) {
    let at = u64::from(slot);
    streams.characters.write_at(
        queue,
        at * streams.characters.stride(),
        bytemuck::bytes_of(&slot_record.record),
    );
    streams.character_inputs.write_at(
        queue,
        at * streams.character_inputs.stride(),
        bytemuck::bytes_of(&slot_record.input),
    );
    streams.character_states.write_at(
        queue,
        at * streams.character_states.stride(),
        bytemuck::bytes_of(&slot_record.state),
    );
}

fn write_input(queue: &wgpu::Queue, streams: &RigidStreams, slot: u32, slot_record: &Slot) {
    streams.character_inputs.write_at(
        queue,
        u64::from(slot) * streams.character_inputs.stride(),
        bytemuck::bytes_of(&slot_record.input),
    );
}

fn reset_slot(queue: &wgpu::Queue, streams: &RigidStreams, slot: u32) {
    let sweeps = [dynamis_abi::inert_sweep(); CHARACTER_SWEEPS as usize];
    streams.character_sweeps.write_at(
        queue,
        u64::from(slot) * u64::from(CHARACTER_SWEEPS) * streams.character_sweeps.stride(),
        bytemuck::cast_slice(&sweeps),
    );
    let hits =
        vec![0u8; u64::from(CHARACTER_SWEEPS) as usize * streams.character_hits.stride() as usize];
    streams.character_hits.write_at(
        queue,
        u64::from(slot) * u64::from(CHARACTER_SWEEPS) * streams.character_hits.stride(),
        &hits,
    );
}

fn take_sorted(slots: &mut Vec<u32>) -> Vec<u32> {
    slots.sort_unstable();
    slots.dedup();
    std::mem::take(slots)
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
        let (id, generation) = self.characters.ids.acquire();
        let handle = CharacterHandle { id, generation };
        let record = CharacterRecord::build(&desc, body.id, body.generation);
        self.characters.spawn(handle, body, record, position);
        handle
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
        &self.characters.live
    }

    pub fn character_count(&self) -> usize {
        self.characters.live.len()
    }

    pub fn character_body(&self, handle: CharacterHandle) -> BodyHandle {
        self.characters.body_of(handle)
    }
}
