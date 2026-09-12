mod engine;
pub(crate) mod rigid;

pub(crate) use rigid::Frame;
use rigid::{Pass, Rigid, buffers::RigidBuffers};

use dynamis_gpu::GpuContext;
use wgpu::CommandEncoder;

pub(crate) struct Pipeline {
    engine: engine::Engine,
    rigid: Rigid,
}

impl Pipeline {
    pub(crate) fn new(context: &GpuContext, buffers: &RigidBuffers) -> Self {
        Self {
            engine: engine::Engine::new(
                context,
                Pass::LABELS,
                #[cfg(feature = "profile")]
                "dynamis step",
            ),
            rigid: Rigid::new(context, buffers),
        }
    }

    pub(crate) fn encode(
        &self,
        encoder: &mut CommandEncoder,
        buffers: &RigidBuffers,
        frame: &Frame,
        idle: bool,
    ) {
        self.rigid
            .encode(&self.engine, encoder, buffers, frame, idle);
    }

    pub(crate) fn encode_queries(
        &self,
        encoder: &mut CommandEncoder,
        buffers: &RigidBuffers,
        frame: &Frame,
    ) {
        self.rigid
            .encode_queries(&self.engine, encoder, buffers, frame);
    }

    #[cfg(feature = "profile")]
    pub(crate) fn capture_timings(
        &mut self,
        encoder: &mut dynamis_gpu::SubmissionEncoder,
        sequence: u64,
    ) -> Option<(u64, Vec<dynamis_gpu::GpuPassTiming>)> {
        self.engine.capture_timings(encoder, sequence)
    }

    #[cfg(feature = "profile")]
    pub(crate) fn collect_timings(&mut self) -> Vec<(u64, Vec<dynamis_gpu::GpuPassTiming>)> {
        self.engine.collect_timings()
    }
}
