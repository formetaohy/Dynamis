mod ccd;
mod engine;
pub(crate) mod rigid;

use ccd::{Ccd, CcdPasses};
use engine::Schedule;
pub(crate) use rigid::Frame;
use rigid::{Rigid, RigidPasses, RigidResolutionPasses, buffers::RigidBuffers};

use dynamis_gpu::GpuContext;
use wgpu::CommandEncoder;

pub(crate) struct Pipeline {
    engine: engine::Engine,
    rigid: Rigid,
    ccd: Ccd,
}

impl Pipeline {
    pub(crate) fn new(context: &GpuContext, buffers: &RigidBuffers) -> Self {
        let mut schedule = Schedule::new();
        let rigid = RigidPasses::claim(&mut schedule);
        let ccd = CcdPasses::claim(&mut schedule);
        let resolution = RigidResolutionPasses::claim(&mut schedule);
        Self {
            engine: engine::Engine::new(
                context,
                schedule,
                #[cfg(feature = "profile")]
                "dynamis step",
            ),
            rigid: Rigid::new(context, buffers, rigid, resolution),
            ccd: Ccd::new(context, buffers, ccd),
        }
    }

    pub(crate) fn encode(
        &self,
        encoder: &mut CommandEncoder,
        buffers: &RigidBuffers,
        frame: &Frame,
        idle: bool,
        ccd_active: bool,
    ) {
        self.rigid
            .encode(&self.engine, encoder, buffers, frame, idle);
        if ccd_active && !idle {
            self.ccd.encode(&self.engine, encoder, buffers, frame);
        }
        self.rigid
            .encode_resolution(&self.engine, encoder, buffers, frame, idle);
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
