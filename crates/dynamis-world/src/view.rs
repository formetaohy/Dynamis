use super::World;
use dynamis_abi::{BodyStateRecord, NO_SLOT};
use dynamis_gpu::SubmissionEncoder;
use dynamis_model::{BodyHandle, BodyState};
use std::mem::size_of;

pub(crate) struct View {
    observed: Vec<u32>,
    slot_of: Vec<u32>,
    dirty: bool,
    epoch: Option<u64>,
}

impl View {
    pub(crate) const fn new() -> Self {
        Self {
            observed: Vec::new(),
            slot_of: Vec::new(),
            dirty: false,
            epoch: None,
        }
    }

    pub(crate) fn len(&self) -> u32 {
        self.observed.len() as u32
    }

    pub(crate) fn ids(&self) -> &[u32] {
        &self.observed
    }

    pub(crate) fn observe(&mut self, id: u32) {
        if self.slot_of.len() <= id as usize {
            self.slot_of.resize(id as usize + 1, NO_SLOT);
        }
        if self.slot_of[id as usize] != NO_SLOT {
            return;
        }
        self.slot_of[id as usize] = self.observed.len() as u32;
        self.observed.push(id);
        self.dirty = true;
    }

    pub(crate) fn forget(&mut self, id: u32) {
        let Some(slot) = self
            .slot_of
            .get(id as usize)
            .copied()
            .filter(|slot| *slot != NO_SLOT)
        else {
            return;
        };
        self.slot_of[id as usize] = NO_SLOT;
        self.observed.swap_remove(slot as usize);
        if let Some(moved) = self.observed.get(slot as usize) {
            self.slot_of[*moved as usize] = slot;
        }
        self.dirty = true;
    }

    fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }
}

impl World {
    pub fn poll(&mut self) {
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
    }

    pub fn wait(&mut self) {
        self.backend.gpu.assert_alive();
        self.resolve_queries();
        loop {
            self.drain_readbacks();
            if self.states_current() {
                return;
            }
            self.submit_states();
        }
    }

    pub fn try_state(&mut self, handle: BodyHandle) -> Option<BodyState> {
        self.validate(handle);
        self.view.observe(handle.id);
        self.view.epoch?;
        self.bodies.states[handle.id as usize]
    }

    pub(crate) fn states_current(&self) -> bool {
        self.bodies
            .alive
            .iter()
            .all(|handle| self.body_current(handle.id))
    }

    pub(crate) fn body_current(&self, id: u32) -> bool {
        self.bodies.covered[id as usize] == self.clock.step
    }

    fn completed_step(&self) -> Option<u64> {
        self.clock.step.checked_sub(1)
    }

    pub(crate) fn flush_observed(&mut self) {
        if !self.view.take_dirty() {
            return;
        }
        let ids = self.view.ids();
        if ids.is_empty() {
            return;
        }
        self.backend
            .streams
            .state
            .observed_ids
            .write(self.backend.gpu.queue(), bytemuck::cast_slice(ids));
    }

    pub(crate) fn copy_observations(&mut self, encoder: &mut SubmissionEncoder, step: u64) {
        let count = self.view.len();
        if count == 0 {
            return;
        }
        let stride = self.backend.streams.state.observed_states.stride();
        let bytes = count as u64 * stride;
        let displaced = self.backend.readback.observations.enqueue(
            encoder,
            self.backend.streams.state.observed_states.buffer(),
            0,
            bytes,
            step,
        );
        if let Some((sequence, bytes)) = displaced {
            self.consume_observations(sequence, &bytes);
        }
    }

    pub(crate) fn consume_observations(&mut self, sequence: u64, bytes: &[u8]) {
        let (records, remainder) = bytes.as_chunks::<{ size_of::<BodyStateRecord>() }>();
        assert!(
            remainder.is_empty(),
            "an observation readback must be a whole number of records"
        );
        for chunk in records {
            self.accept_body(sequence, bytemuck::pod_read_unaligned(chunk));
        }
        self.view.epoch = Some(sequence);
    }

    fn submit_states(&mut self) {
        let step = self
            .completed_step()
            .expect("a body state snapshot requires a completed step");
        let bytes = u64::from(self.bodies.device_count) * size_of::<BodyStateRecord>() as u64;
        if bytes == 0 {
            self.view.epoch = Some(step);
            return;
        }
        let device = self.backend.gpu.device().clone();
        self.backend
            .readback
            .open_states(&device, self.backend.streams.state.body_states.slots());
        let mut encoder = SubmissionEncoder::new(&device, "dynamis body state readback");
        let displaced = self
            .backend
            .readback
            .states
            .as_mut()
            .expect("a body state snapshot opens its readback ring")
            .enqueue(
                &mut encoder,
                self.backend.streams.state.body_states.buffer(),
                0,
                bytes,
                step,
            );
        self.submit(encoder);
        if let Some((sequence, bytes)) = displaced {
            self.consume_states(sequence, &bytes);
        }
    }

    pub(crate) fn consume_states(&mut self, sequence: u64, bytes: &[u8]) {
        let (records, remainder) = bytes.as_chunks::<{ size_of::<BodyStateRecord>() }>();
        assert!(
            remainder.is_empty(),
            "a body state readback must be a whole number of records"
        );
        for chunk in records {
            self.accept_body(sequence, bytemuck::pod_read_unaligned(chunk));
        }
        self.view.epoch = Some(sequence);
    }

    fn accept_body(&mut self, sequence: u64, record: BodyStateRecord) {
        let id = record.body_id as usize;
        assert!(
            id < self.bodies.ids.len(),
            "a state readback returned an out-of-range body id"
        );
        if record.generation != self.bodies.ids.generation(record.body_id)
            || self.bodies.index_of[id] == u32::MAX
        {
            return;
        }
        self.bodies.states[id] = Some(BodyState {
            position: record.position,
            prev_position: record.prev_position,
            orientation: record.orientation,
            velocity: record.velocity,
            angular_velocity: record.angular_velocity,
            inverse_mass: self.bodies.descriptors[id].inverse_mass,
            com: self.bodies.descriptors[id].com,
            sleeping: record.sleeping != 0,
            step: sequence,
        });
        self.bodies.covered[id] = sequence + 1;
    }
}
