use super::registry::Streams;
use dynamis_abi::{COUNTER_DEVICE_COUNT, COUNTER_STRIDE, DeclaredCounters};
use dynamis_gpu::{GpuBuffer, Publication};
use wgpu::Device;

const DEPTH: usize = Publication::<()>::DEPTH;
pub(crate) const COUNTER_BYTES: u64 = COUNTER_STRIDE * COUNTER_DEVICE_COUNT as u64;

pub(crate) struct ReadbackBuffers {
    pub(crate) pack: GpuBuffer,
    pub(crate) counters: Publication<(u64, DeclaredCounters)>,
    pub(crate) queries: Publication<u64>,
}

fn budgets(streams: &Streams) -> [u64; 2] {
    [COUNTER_BYTES, streams.state.query_results.size()]
}

impl ReadbackBuffers {
    pub(crate) fn new(device: &Device, streams: &Streams) -> Self {
        let mut buffers = Self {
            pack: GpuBuffer::new(
                device,
                "world readback pack",
                COUNTER_BYTES,
                dynamis_gpu::PACK,
            ),
            counters: Publication::new("world counters readback", DEPTH),
            queries: Publication::new("query results readback", DEPTH),
        };
        buffers.reserve(device, streams);
        buffers
    }

    pub(crate) fn reserve(&mut self, device: &Device, streams: &Streams) -> bool {
        let [counters, queries] = budgets(streams);
        let mut changed = self.counters.reserve(device, counters);
        changed |= self.queries.reserve(device, queries);
        changed
    }
}
