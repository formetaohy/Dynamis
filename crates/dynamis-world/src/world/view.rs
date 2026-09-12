use super::World;
use dynamis_gpu::SubmissionEncoder;
use dynamis_layout::BodyStateRecord;
use dynamis_model::{BodyHandle, BodyState};
use std::mem::size_of;

pub(crate) struct View {
    armed: bool,
    arrived: Option<u64>,
}

impl View {
    pub(crate) const fn new() -> Self {
        Self {
            armed: false,
            arrived: None,
        }
    }
}

impl World {
    pub fn poll(&mut self) {
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
    }

    pub fn wait(&mut self) {
        self.backend.gpu.assert_alive();
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
        self.view.armed = true;
        self.view.arrived?;
        self.bodies.states[handle.id as usize]
    }

    pub(crate) fn states_current(&self) -> bool {
        self.view.arrived == self.completed_step()
    }

    fn completed_step(&self) -> Option<u64> {
        self.clock.step.checked_sub(1)
    }

    pub(crate) fn write_states(&mut self, encoder: &mut SubmissionEncoder, step: u64) {
        if !self.view.armed {
            return;
        }
        self.view.armed = false;
        let bytes = self.bodies.alive.len() as u64 * size_of::<BodyStateRecord>() as u64;
        if bytes == 0 {
            self.view.arrived = Some(step);
            return;
        }
        self.record_states(encoder, step, bytes);
    }

    fn submit_states(&mut self) {
        let step = self
            .completed_step()
            .expect("a body state snapshot requires a completed step");
        let bytes = u64::from(self.bodies.device_count) * size_of::<BodyStateRecord>() as u64;
        if bytes == 0 {
            self.view.arrived = Some(step);
            return;
        }
        let device = self.backend.gpu.device().clone();
        let mut encoder = SubmissionEncoder::new(&device, "dynamis body state readback");
        self.record_states(&mut encoder, step, bytes);
        self.submit(encoder);
    }

    fn record_states(&mut self, encoder: &mut SubmissionEncoder, step: u64, bytes: u64) {
        let displaced = self.backend.streams.readback.states.enqueue(
            encoder,
            self.backend.streams.scene.body_states.buffer(),
            0,
            bytes,
            step,
        );
        if let Some((step, bytes)) = displaced {
            self.consume_states(step, &bytes);
        }
    }

    pub(crate) fn consume_states(&mut self, step: u64, bytes: &[u8]) {
        let (records, remainder) = bytes.as_chunks::<{ size_of::<BodyStateRecord>() }>();
        assert!(
            remainder.is_empty(),
            "a body state readback must be a whole number of records"
        );
        for chunk in records {
            self.accept_body(step, bytemuck::pod_read_unaligned(chunk));
        }
        self.view.arrived = Some(step);
    }

    fn accept_body(&mut self, step: u64, record: BodyStateRecord) {
        let id = record.body_id as usize;
        assert!(
            id < self.bodies.ids.len(),
            "GPU readback returned an out-of-range body id"
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
            step,
        });
    }
}
