use super::registry::Plan;
use dynamis_abi::{
    BodyStateRecord, BrokenConstraintRecord, COUNTER_DEVICE_COUNT, COUNTER_STRIDE,
    ContactEventRecord,
};
use dynamis_gpu::{GpuBuffer, Readback};
use dynamis_state::QUERY_RESULT_BYTES;
use std::mem::size_of;
use wgpu::Device;

const COUNTER_BYTES: u64 = COUNTER_STRIDE * COUNTER_DEVICE_COUNT as u64;

pub(crate) struct ReadbackBuffers {
    pub(crate) pack: GpuBuffer,
    pub(crate) step: Readback,
    pub(crate) events: Readback,
    pub(crate) breaks: Readback,
    pub(crate) queries: Readback,
    pub(crate) states: Option<Readback>,
}

fn readback_sizes(plan: &Plan) -> (u64, u64, u64, u64) {
    (
        COUNTER_BYTES,
        u64::from(plan.rigid.events) * size_of::<ContactEventRecord>() as u64,
        u64::from(plan.state.constraints) * size_of::<BrokenConstraintRecord>() as u64,
        u64::from(plan.state.queries) * QUERY_RESULT_BYTES,
    )
}

impl ReadbackBuffers {
    pub(crate) fn new(device: &Device, plan: &Plan) -> Self {
        let (pack_bytes, event_bytes, break_bytes, query_bytes) = readback_sizes(plan);
        Self {
            pack: GpuBuffer::new(device, "world readback pack", pack_bytes, dynamis_gpu::PACK),
            step: Readback::new(device, "world readback", pack_bytes, Readback::DEPTH),
            events: Readback::new(
                device,
                "world events readback",
                event_bytes,
                Readback::DEPTH,
            ),
            breaks: Readback::new(
                device,
                "constraint breaks readback",
                break_bytes,
                Readback::DEPTH,
            ),
            queries: Readback::new(
                device,
                "query results readback",
                query_bytes,
                Readback::DEPTH,
            ),
            states: None,
        }
    }

    pub(crate) fn matches(&self, plan: &Plan) -> bool {
        let (pack_bytes, event_bytes, break_bytes, query_bytes) = readback_sizes(plan);
        self.pack.size() == pack_bytes
            && self.step.size() == pack_bytes
            && self.events.size() == event_bytes
            && self.breaks.size() == break_bytes
            && self.queries.size() == query_bytes
    }

    pub(crate) fn reserve(&mut self, device: &Device, plan: &Plan) -> bool {
        let (pack_bytes, event_bytes, break_bytes, query_bytes) = readback_sizes(plan);
        if self.matches(plan) {
            return false;
        }
        assert!(
            self.step.is_idle()
                && self.events.is_idle()
                && self.breaks.is_idle()
                && self.queries.is_idle(),
            "readback buffers require drained rings before they reallocate"
        );
        if self.pack.size() != pack_bytes {
            self.pack =
                GpuBuffer::new(device, "world readback pack", pack_bytes, dynamis_gpu::PACK);
            self.step = Readback::new(device, "world readback", pack_bytes, Readback::DEPTH);
        }
        if self.events.size() != event_bytes {
            self.events = Readback::new(
                device,
                "world events readback",
                event_bytes,
                Readback::DEPTH,
            );
        }
        if self.breaks.size() != break_bytes {
            self.breaks = Readback::new(
                device,
                "constraint breaks readback",
                break_bytes,
                Readback::DEPTH,
            );
        }
        if self.queries.size() != query_bytes {
            self.queries = Readback::new(
                device,
                "query results readback",
                query_bytes,
                Readback::DEPTH,
            );
        }
        true
    }

    pub(crate) fn open_states(&mut self, device: &Device, slots: u32) {
        let bytes = u64::from(slots) * size_of::<BodyStateRecord>() as u64;
        if let Some(states) = &self.states {
            if states.size() >= bytes {
                return;
            }
            assert!(
                states.is_idle(),
                "the body state readback requires a drained ring before it reallocates"
            );
        }
        self.states = Some(Readback::new(
            device,
            "body state readback",
            bytes,
            Readback::DEPTH,
        ));
    }

    pub(crate) fn collect_states(&mut self) -> Vec<(u64, Vec<u8>)> {
        match &mut self.states {
            Some(states) => states.collect(),
            None => Vec::new(),
        }
    }

    pub(crate) fn drain_states(&mut self) -> Vec<(u64, Vec<u8>)> {
        match &mut self.states {
            Some(states) => states.drain(),
            None => Vec::new(),
        }
    }
}
