use crate::buffers::WorldBuffers;
use crate::simulation::Simulation;
use dynamis_layout::{
    BodyStateRecord, COUNTER_BODIES, COUNTER_CONSTRAINTS, COUNTER_COUNT, COUNTER_STRIDE,
    ConstraintRuntimeRecord, Counters,
};
use std::mem::size_of;

/// The per-step readback: the counter vector, then the device-owned rows the host
/// mirrors. One buffer, one mapping, one copy per step.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackLayout {
    pub(crate) counters: u64,
    pub(crate) bodies: u64,
    pub(crate) constraints: u64,
    pub(crate) events: u64,
}

const COUNTER_BYTES: u64 = COUNTER_STRIDE * COUNTER_COUNT as u64;

impl PackLayout {
    pub(crate) fn of(buffers: &WorldBuffers) -> Self {
        let counters = 0;
        let bodies = COUNTER_BYTES;
        let constraints = bodies + buffers.bodies() as u64 * size_of::<BodyStateRecord>() as u64;
        let events = constraints
            + buffers.constraints() as u64 * size_of::<ConstraintRuntimeRecord>() as u64;
        Self {
            counters,
            bodies,
            constraints,
            events,
        }
    }

    fn measured(&self, bytes: &[u8]) -> Counters {
        let stride = COUNTER_STRIDE as usize;
        let region = slice(bytes, self.counters, COUNTER_BYTES);
        let mut counters: Counters = [0; COUNTER_COUNT];
        for (slot, value) in counters.iter_mut().enumerate() {
            let at = slot * stride;
            *value = u32::from_le_bytes(region[at..at + 4].try_into().expect("counter slot"));
        }
        counters
    }
}

impl Simulation {
    pub(crate) fn pack(&self) -> PackLayout {
        PackLayout::of(&self.buffers)
    }

    /// Copies every mirrored region into the pack so one readback carries it all.
    pub(crate) fn pack_step(&self, encoder: &mut wgpu::CommandEncoder) {
        let pack = self.pack();
        let staging = self.buffers.readback_pack.buffer();
        encoder.copy_buffer_to_buffer(
            self.buffers.counters.buffer(),
            0,
            staging,
            pack.counters,
            COUNTER_BYTES,
        );
        encoder.copy_buffer_to_buffer(
            self.buffers.body_states.buffer(),
            0,
            staging,
            pack.bodies,
            self.buffers.body_states.size(),
        );
        encoder.copy_buffer_to_buffer(
            self.buffers.constraint_runtime.buffer(),
            0,
            staging,
            pack.constraints,
            self.buffers.constraint_runtime.size(),
        );
        encoder.copy_buffer_to_buffer(
            self.buffers.events.buffer(),
            0,
            staging,
            pack.events,
            self.buffers.events.size(),
        );
    }

    /// Every region's valid extent is declared by the counters travelling in the same
    /// pack, so a step only ever mirrors the rows that exist.
    pub(crate) fn consume_pack(&mut self, step: u64, bytes: &[u8]) {
        let pack = self.pack();
        let measured = pack.measured(bytes);
        self.accept_measured(step, &measured);
        let bodies = measured[COUNTER_BODIES] as u64 * size_of::<BodyStateRecord>() as u64;
        for record in crate::records::records::<BodyStateRecord>(slice(bytes, pack.bodies, bodies))
        {
            self.accept_body(step, record);
        }
        let constraints =
            measured[COUNTER_CONSTRAINTS] as u64 * size_of::<ConstraintRuntimeRecord>() as u64;
        for record in crate::records::records::<ConstraintRuntimeRecord>(slice(
            bytes,
            pack.constraints,
            constraints,
        )) {
            self.accept_constraint_break(step, record);
        }
        self.consume_events(step, slice(bytes, pack.events, self.buffers.events.size()));
    }
}

fn slice(bytes: &[u8], at: u64, len: u64) -> &[u8] {
    let at = at as usize;
    &bytes[at..at + len as usize]
}
