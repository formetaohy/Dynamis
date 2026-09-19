use super::World;
use super::arena::{Cleared, Mirror, Run};
use super::pool::Pool;
use dynamis_abi::{VehicleInputRecord, VehicleRecord, VehicleStateRecord, VehicleWheelRecord};
use dynamis_model::{BodyHandle, VehicleDesc, VehicleHandle, VehicleInput};
use dynamis_rigid::RigidStreams;

#[derive(Clone, Copy)]
struct Slot {
    body: Option<BodyHandle>,
    record: VehicleRecord,
    input: VehicleInputRecord,
    state: VehicleStateRecord,
}

impl Slot {
    const RETIRED: Self = Self {
        body: None,
        record: VehicleRecord::cleared(),
        input: VehicleInputRecord::idle(),
        state: VehicleStateRecord::cleared(),
    };
}

impl Cleared for VehicleWheelRecord {
    const CLEARED: Self = Self::cleared();
}

#[derive(Clone)]
pub(crate) struct VehicleStore {
    pub(crate) pool: Pool<VehicleHandle>,
    slots: Vec<Slot>,
    runs: Vec<Run>,
    wheels: Mirror<VehicleWheelRecord>,
    pending_inputs: u32,
    pub(crate) last_inputs: u32,
}

impl VehicleStore {
    pub(crate) const fn new() -> Self {
        Self {
            pool: Pool::vacate("vehicle"),
            slots: Vec::new(),
            runs: Vec::new(),
            wheels: Mirror::new(),
            pending_inputs: 0,
            last_inputs: 0,
        }
    }

    pub(crate) fn count(&self) -> u32 {
        self.pool.len()
    }

    pub(crate) fn slots(&self) -> u32 {
        self.pool.ids()
    }

    pub(crate) fn wheel_count(&self) -> u32 {
        self.wheels.used()
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
        self.slots
            .iter()
            .any(|slot| slot.body.is_some_and(|body| body.id == id))
    }

    pub(crate) fn validate(&self, handle: VehicleHandle) {
        self.pool.validate(handle);
    }

    pub(crate) fn slot_of(&self, handle: VehicleHandle) -> u32 {
        self.pool.row_of(handle)
    }

    pub(crate) fn body_of(&self, handle: VehicleHandle) -> BodyHandle {
        self.slots[self.slot_of(handle) as usize]
            .body
            .expect("a live vehicle owns its chassis")
    }

    pub(crate) fn live_slots(&self) -> Vec<(VehicleHandle, u32)> {
        let mut slots = self
            .pool
            .alive()
            .iter()
            .map(|handle| (*handle, handle.id))
            .collect::<Vec<_>>();
        slots.sort_unstable_by_key(|(_, slot)| *slot);
        slots
    }

    pub(crate) fn spawn(
        &mut self,
        body: BodyHandle,
        desc: &VehicleDesc,
        wheels: &[VehicleWheelRecord],
        position: [f32; 3],
    ) -> VehicleHandle {
        assert!(!wheels.is_empty(), "a vehicle carries at least one wheel");
        let handle = self.pool.acquire();
        let slot = self.pool.insert(handle) as usize;
        let run = self.wheels.take(wheels.len() as u32);
        self.wheels.records_mut()[run.span()].copy_from_slice(wheels);
        if self.slots.len() <= slot {
            self.slots.resize(slot + 1, Slot::RETIRED);
            self.runs.resize(slot + 1, Run::EMPTY);
        }
        self.slots[slot] = Slot {
            body: Some(body),
            record: VehicleRecord::build(desc, body.id, body.generation, run.offset),
            input: VehicleInputRecord::idle(),
            state: VehicleStateRecord::spawn(handle.id, handle.generation, position),
        };
        self.runs[slot] = run;
        handle
    }

    pub(crate) fn set_input(&mut self, handle: VehicleHandle, input: VehicleInput) {
        let slot = self.slot_of(handle) as usize;
        self.slots[slot].input = VehicleInputRecord::of(input);
        self.pool.mark(handle);
        self.pending_inputs += 1;
    }

    pub(crate) fn retire(&mut self, handle: VehicleHandle) {
        let slot = self.pool.retire(handle).row as usize;
        self.wheels.retire(self.runs[slot]);
        self.slots[slot] = Slot::RETIRED;
    }

    pub(crate) fn upload(&mut self, queue: &wgpu::Queue, streams: &RigidStreams) {
        for slot in self.pool.take_retired() {
            reset_scratch(queue, streams, self.runs[slot as usize]);
            write_slot(queue, streams, slot, &Slot::RETIRED);
        }
        for slot in self.pool.take_fresh() {
            reset_scratch(queue, streams, self.runs[slot as usize]);
            write_slot(queue, streams, slot, &self.slots[slot as usize]);
        }
        for slot in self.pool.take_dirty() {
            write_input(queue, streams, slot, &self.slots[slot as usize]);
        }
        self.wheels
            .flush(|offset, records| write_wheels(queue, streams, offset, records));
    }
}

fn write_slot(queue: &wgpu::Queue, streams: &RigidStreams, slot: u32, record: &Slot) {
    let at = u64::from(slot);
    streams.vehicles.write_at(
        queue,
        at * streams.vehicles.stride(),
        bytemuck::bytes_of(&record.record),
    );
    streams.vehicle_inputs.write_at(
        queue,
        at * streams.vehicle_inputs.stride(),
        bytemuck::bytes_of(&record.input),
    );
    streams.vehicle_states.write_at(
        queue,
        at * streams.vehicle_states.stride(),
        bytemuck::bytes_of(&record.state),
    );
}

fn write_input(queue: &wgpu::Queue, streams: &RigidStreams, slot: u32, record: &Slot) {
    streams.vehicle_inputs.write_at(
        queue,
        u64::from(slot) * streams.vehicle_inputs.stride(),
        bytemuck::bytes_of(&record.input),
    );
}

fn write_wheels(
    queue: &wgpu::Queue,
    streams: &RigidStreams,
    offset: u32,
    wheels: &[VehicleWheelRecord],
) {
    streams.vehicle_wheels.write_at(
        queue,
        u64::from(offset) * streams.vehicle_wheels.stride(),
        bytemuck::cast_slice(wheels),
    );
}

fn reset_scratch(queue: &wgpu::Queue, streams: &RigidStreams, run: Run) {
    crate::query::write_inert_queries(
        queue,
        &streams.vehicle_sweeps,
        &streams.vehicle_hits,
        run.offset,
        run.len,
    );
}

impl World {
    pub(crate) fn flush_vehicles(&mut self) {
        let queue = self.backend.gpu.queue().clone();
        self.vehicles.upload(&queue, &self.backend.streams.rigid);
    }

    pub fn add_vehicle(&mut self, desc: VehicleDesc) -> VehicleHandle {
        desc.assert_valid();
        let body = self.spawn(desc.chassis.clone());
        let wheels = desc
            .wheels
            .iter()
            .map(VehicleWheelRecord::build)
            .collect::<Vec<_>>();
        self.vehicles
            .spawn(body, &desc, &wheels, desc.chassis.position)
    }

    pub fn remove_vehicle(&mut self, handle: VehicleHandle) {
        let body = self.vehicles.body_of(handle);
        self.vehicles.retire(handle);
        self.observed.vehicles.stop_watching(handle.id);
        self.remove(body);
    }

    pub fn set_vehicle_input(&mut self, handle: VehicleHandle, input: VehicleInput) {
        input.assert_valid();
        self.vehicles.validate(handle);
        self.vehicles.set_input(handle, input);
    }

    pub fn vehicles(&self) -> &[VehicleHandle] {
        self.vehicles.pool.alive()
    }

    pub fn vehicle_count(&self) -> usize {
        self.vehicles.pool.len() as usize
    }

    pub fn vehicle_body(&self, handle: VehicleHandle) -> BodyHandle {
        self.vehicles.body_of(handle)
    }
}
