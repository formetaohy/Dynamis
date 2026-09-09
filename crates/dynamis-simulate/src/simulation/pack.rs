use crate::simulation::Simulation;
use dynamis_layout::{
    COUNTER_CONSTRAINTS, COUNTER_COUNT, COUNTER_STRIDE, ConstraintRuntimeRecord, Counters,
};
use std::mem::size_of;

#[derive(Clone, Copy, Debug)]
pub(crate) struct PackLayout {
    pub(crate) counters: u64,
    pub(crate) constraints: u64,
    pub(crate) events: u64,
}

const COUNTER_BYTES: u64 = COUNTER_STRIDE * COUNTER_COUNT as u64;

impl PackLayout {
    pub(crate) fn for_constraints(constraints: u32) -> Self {
        let counters = 0;
        let constraints_offset = COUNTER_BYTES;
        let events =
            constraints_offset + constraints as u64 * size_of::<ConstraintRuntimeRecord>() as u64;
        Self {
            counters,
            constraints: constraints_offset,
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
    pub(crate) fn pack_step(&self, encoder: &mut wgpu::CommandEncoder) -> u64 {
        let constraints = self.constraint_alive.len() as u32;
        let pack = PackLayout::for_constraints(constraints);
        let staging = self.buffers.readback_pack.buffer();
        encoder.copy_buffer_to_buffer(
            self.buffers.counters.buffer(),
            0,
            staging,
            pack.counters,
            COUNTER_BYTES,
        );
        if constraints > 0 {
            encoder.copy_buffer_to_buffer(
                self.buffers.constraint_runtime.buffer(),
                0,
                staging,
                pack.constraints,
                constraints as u64 * size_of::<ConstraintRuntimeRecord>() as u64,
            );
        }
        encoder.copy_buffer_to_buffer(
            self.buffers.events.buffer(),
            0,
            staging,
            pack.events,
            self.buffers.events.size(),
        );
        pack.events + self.buffers.events.size()
    }

    pub(crate) fn consume_pack(&mut self, step: u64, bytes: &[u8]) {
        let counters = PackLayout::for_constraints(0);
        let measured = counters.measured(bytes);
        let pack = PackLayout::for_constraints(measured[COUNTER_CONSTRAINTS]);
        self.accept_measured(step, &measured);
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
