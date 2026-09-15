use super::World;
use super::ids::IdSpace;
use dynamis_abi::{
    VEHICLE_WHEELS, VehicleInputRecord, VehicleRecord, VehicleStateRecord, VehicleWheelRecord,
    inert_sweep,
};
use dynamis_model::{BodyHandle, VehicleDesc, VehicleHandle, VehicleInput};
use dynamis_rigid::RigidStreams;

#[derive(Clone, Copy)]
struct Slot {
    handle: Option<VehicleHandle>,
    body: Option<BodyHandle>,
    record: VehicleRecord,
    input: VehicleInputRecord,
    state: VehicleStateRecord,
}

impl Slot {
    const RETIRED: Self = Self {
        handle: None,
        body: None,
        record: VehicleRecord::cleared(),
        input: VehicleInputRecord::idle(),
        state: VehicleStateRecord::cleared(),
    };
}

#[derive(Clone)]
pub(crate) struct Vehicles {
    pub(crate) live: Vec<VehicleHandle>,
    ids: IdSpace,
    slots: Vec<Slot>,
    wheels: Vec<VehicleWheelRecord>,
    fresh: Vec<u32>,
    dirty: Vec<u32>,
    retired: Vec<u32>,
    pending_inputs: u32,
    pub(crate) last_inputs: u32,
}

impl Vehicles {
    pub(crate) const fn new() -> Self {
        Self {
            live: Vec::new(),
            ids: IdSpace::new(),
            slots: Vec::new(),
            wheels: Vec::new(),
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

    pub(crate) fn slots(&self) -> u32 {
        self.ids.len() as u32
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

    pub(crate) fn owns_body(&self, id: u32) -> bool {
        self.live.iter().any(|handle| {
            self.slots[handle.id as usize]
                .body
                .is_some_and(|body| body.id == id)
        })
    }

    pub(crate) fn validate(&self, handle: VehicleHandle) {
        let id = handle.id as usize;
        if id >= self.ids.len() {
            panic!("vehicle handle {handle:?} is out of range");
        }
        if self.ids.generation(handle.id) != handle.generation {
            panic!("vehicle handle {handle:?} is stale");
        }
        if self.slots[id].handle != Some(handle) {
            panic!("vehicle handle {handle:?} is not alive");
        }
    }

    pub(crate) fn slot_of(&self, handle: VehicleHandle) -> u32 {
        self.validate(handle);
        handle.id
    }

    pub(crate) fn body_of(&self, handle: VehicleHandle) -> BodyHandle {
        self.slots[self.slot_of(handle) as usize]
            .body
            .expect("a live vehicle owns its chassis")
    }

    pub(crate) fn live_slots(&self) -> Vec<(VehicleHandle, u32)> {
        let mut slots = self
            .live
            .iter()
            .map(|handle| (*handle, handle.id))
            .collect::<Vec<_>>();
        slots.sort_unstable_by_key(|(_, slot)| *slot);
        slots
    }

    pub(crate) fn spawn(
        &mut self,
        handle: VehicleHandle,
        body: BodyHandle,
        record: VehicleRecord,
        wheels: &[VehicleWheelRecord],
        position: [f32; 3],
    ) {
        assert!(
            wheels.len() <= VEHICLE_WHEELS as usize,
            "a vehicle carries at most {VEHICLE_WHEELS} wheels"
        );
        let slot = handle.id as usize;
        if self.slots.len() <= slot {
            self.slots.resize(slot + 1, Slot::RETIRED);
        }
        if self.wheels.len() < (slot + 1) * VEHICLE_WHEELS as usize {
            self.wheels.resize(
                (slot + 1) * VEHICLE_WHEELS as usize,
                VehicleWheelRecord::cleared(),
            );
        }
        let run = slot * VEHICLE_WHEELS as usize;
        self.wheels[run..run + VEHICLE_WHEELS as usize].fill(VehicleWheelRecord::cleared());
        self.wheels[run..run + wheels.len()].copy_from_slice(wheels);
        self.slots[slot] = Slot {
            handle: Some(handle),
            body: Some(body),
            record,
            input: VehicleInputRecord::idle(),
            state: VehicleStateRecord::spawn(handle.id, handle.generation, position),
        };
        self.live.push(handle);
        self.fresh.push(handle.id);
    }

    pub(crate) fn set_input(&mut self, handle: VehicleHandle, input: VehicleInput) {
        let index = self.slot_of(handle) as usize;
        self.slots[index].input = VehicleInputRecord::of(input);
        self.dirty.push(handle.id);
        self.pending_inputs += 1;
    }

    pub(crate) fn retire(&mut self, handle: VehicleHandle) {
        let slot = self.slot_of(handle) as usize;
        self.slots[slot] = Slot::RETIRED;
        let run = slot * VEHICLE_WHEELS as usize;
        self.wheels[run..run + VEHICLE_WHEELS as usize].fill(VehicleWheelRecord::cleared());
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
            write_wheels(queue, streams, slot, &self.wheels);
        }
        for slot in dirty {
            write_input(queue, streams, slot, &self.slots[slot as usize]);
        }
    }
}

fn write_record(queue: &wgpu::Queue, streams: &RigidStreams, slot: u32, slot_record: &Slot) {
    let at = u64::from(slot);
    streams.vehicles.write_at(
        queue,
        at * streams.vehicles.stride(),
        bytemuck::bytes_of(&slot_record.record),
    );
    streams.vehicle_inputs.write_at(
        queue,
        at * streams.vehicle_inputs.stride(),
        bytemuck::bytes_of(&slot_record.input),
    );
    streams.vehicle_states.write_at(
        queue,
        at * streams.vehicle_states.stride(),
        bytemuck::bytes_of(&slot_record.state),
    );
}

fn write_input(queue: &wgpu::Queue, streams: &RigidStreams, slot: u32, slot_record: &Slot) {
    streams.vehicle_inputs.write_at(
        queue,
        u64::from(slot) * streams.vehicle_inputs.stride(),
        bytemuck::bytes_of(&slot_record.input),
    );
}

fn write_wheels(
    queue: &wgpu::Queue,
    streams: &RigidStreams,
    slot: u32,
    wheels: &[VehicleWheelRecord],
) {
    let run = slot as usize * VEHICLE_WHEELS as usize;
    let at = slot as u64 * u64::from(VEHICLE_WHEELS) * streams.vehicle_wheels.stride();
    streams.vehicle_wheels.write_at(
        queue,
        at,
        bytemuck::cast_slice(&wheels[run..run + VEHICLE_WHEELS as usize]),
    );
}

fn reset_slot(queue: &wgpu::Queue, streams: &RigidStreams, slot: u32) {
    let sweeps = [inert_sweep(); VEHICLE_WHEELS as usize];
    streams.vehicle_sweeps.write_at(
        queue,
        u64::from(slot) * u64::from(VEHICLE_WHEELS) * streams.vehicle_sweeps.stride(),
        bytemuck::cast_slice(&sweeps),
    );
    let hits = vec![0u8; VEHICLE_WHEELS as usize * streams.vehicle_hits.stride() as usize];
    streams.vehicle_hits.write_at(
        queue,
        u64::from(slot) * u64::from(VEHICLE_WHEELS) * streams.vehicle_hits.stride(),
        &hits,
    );
}

fn take_sorted(slots: &mut Vec<u32>) -> Vec<u32> {
    slots.sort_unstable();
    slots.dedup();
    std::mem::take(slots)
}

impl World {
    pub fn add_vehicle(&mut self, desc: VehicleDesc) -> VehicleHandle {
        desc.assert_valid();
        let body = self.spawn(desc.chassis.clone());
        let wheels = desc
            .wheels
            .iter()
            .map(VehicleWheelRecord::build)
            .collect::<Vec<_>>();
        let (id, generation) = self.vehicles.ids.acquire();
        let handle = VehicleHandle { id, generation };
        let record = VehicleRecord::build(&desc, body.id, body.generation);
        self.vehicles
            .spawn(handle, body, record, &wheels, desc.chassis.position);
        handle
    }

    pub fn remove_vehicle(&mut self, handle: VehicleHandle) {
        let body = self.vehicles.body_of(handle);
        self.vehicles.retire(handle);
        self.remove(body);
    }

    pub fn set_vehicle_input(&mut self, handle: VehicleHandle, input: VehicleInput) {
        input.assert_valid();
        self.vehicles.validate(handle);
        self.vehicles.set_input(handle, input);
    }

    pub fn vehicles(&self) -> &[VehicleHandle] {
        &self.vehicles.live
    }

    pub fn vehicle_count(&self) -> usize {
        self.vehicles.live.len()
    }

    pub fn vehicle_body(&self, handle: VehicleHandle) -> BodyHandle {
        self.vehicles.body_of(handle)
    }
}
