use super::scene::QUERY_RESULT_BYTES;
use super::streams::Plan;
use dynamis_gpu::{GpuBuffer, Readback};
use dynamis_layout::{
    BodyStateRecord, COUNTER_COUNT, COUNTER_STRIDE, ConstraintRuntimeRecord, ContactEventRecord,
};
use std::mem::size_of;
use wgpu::Device;

pub(crate) const STATE_DEPTH: usize = 2;

const COUNTER_BYTES: u64 = COUNTER_STRIDE * COUNTER_COUNT as u64;

pub(crate) struct ReadbackBuffers {
    pub(crate) pack: GpuBuffer,
    pub(crate) step: Readback,
    pub(crate) events: Readback,
    pub(crate) queries: Readback,
    pub(crate) states: Readback,
}

fn readback_sizes(plan: &Plan) -> (u64, u64, u64, u64) {
    (
        COUNTER_BYTES
            + u64::from(plan.scene.constraints) * size_of::<ConstraintRuntimeRecord>() as u64,
        u64::from(plan.rigid.events) * size_of::<ContactEventRecord>() as u64,
        u64::from(plan.scene.queries) * QUERY_RESULT_BYTES,
        u64::from(plan.scene.bodies) * size_of::<BodyStateRecord>() as u64,
    )
}

impl ReadbackBuffers {
    pub(crate) fn new(device: &Device, plan: &Plan) -> Self {
        let (pack_bytes, event_bytes, query_bytes, state_bytes) = readback_sizes(plan);
        Self {
            pack: GpuBuffer::new(
                device,
                "world readback pack",
                pack_bytes,
                super::engine::PACK,
            ),
            step: Readback::new(device, "world readback", pack_bytes, Readback::DEPTH),
            events: Readback::new(
                device,
                "world events readback",
                event_bytes,
                Readback::DEPTH,
            ),
            queries: Readback::new(
                device,
                "query results readback",
                query_bytes,
                Readback::DEPTH,
            ),
            states: Readback::new(device, "body state readback", state_bytes, STATE_DEPTH),
        }
    }

    pub(crate) fn matches(&self, plan: &Plan) -> bool {
        let (pack_bytes, event_bytes, query_bytes, state_bytes) = readback_sizes(plan);
        self.pack.size() == pack_bytes
            && self.step.size() == pack_bytes
            && self.events.size() == event_bytes
            && self.queries.size() == query_bytes
            && self.states.size() == state_bytes
    }

    pub(crate) fn reserve(&mut self, device: &Device, plan: &Plan) -> bool {
        let (pack_bytes, event_bytes, query_bytes, state_bytes) = readback_sizes(plan);
        if self.matches(plan) {
            return false;
        }
        assert!(
            self.step.is_idle()
                && self.events.is_idle()
                && self.queries.is_idle()
                && self.states.is_idle(),
            "readback buffers require drained rings before they reallocate"
        );
        if self.pack.size() != pack_bytes {
            self.pack = GpuBuffer::new(
                device,
                "world readback pack",
                pack_bytes,
                super::engine::PACK,
            );
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
        if self.queries.size() != query_bytes {
            self.queries = Readback::new(
                device,
                "query results readback",
                query_bytes,
                Readback::DEPTH,
            );
        }
        if self.states.size() != state_bytes {
            self.states = Readback::new(device, "body state readback", state_bytes, STATE_DEPTH);
        }
        true
    }
}
