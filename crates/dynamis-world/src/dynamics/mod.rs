mod capacity;
mod ccd;
mod engine;
mod readback;
pub(crate) mod rigid;
pub(crate) mod scene;
pub(crate) mod shader;
pub(crate) mod soft;
pub(crate) mod streams;

use ccd::{Ccd, CcdPasses};
use engine::Schedule;
use rigid::{Rigid, RigidPasses, RigidResolutionPasses};
use soft::{Soft, SoftPasses};
use streams::Streams;

use dynamis_gpu::{GpuContext, Readback};
use dynamis_layout::StepParamsRecord;
use wgpu::CommandEncoder;

pub(crate) use capacity::Live;
pub use capacity::{ShapeCapacity, SoftCapacity, StreamCapacity};

pub(crate) const EVENT_SLOTS: u32 = Readback::DEPTH as u32 + 2;

pub(crate) struct Frame {
    pub(crate) params: StepParamsRecord,
    pub(crate) query_count: u32,
}

pub(crate) struct Pipeline {
    engine: engine::Engine,
    rigid: Rigid,
    ccd: Ccd,
    soft: Soft,
}

impl Pipeline {
    pub(crate) fn new(context: &GpuContext, streams: &Streams) -> Self {
        let mut schedule = Schedule::new();
        let rigid = RigidPasses::claim(&mut schedule);
        let ccd = CcdPasses::claim(&mut schedule);
        let soft = SoftPasses::claim(&mut schedule);
        let resolution = RigidResolutionPasses::claim(&mut schedule);
        Self {
            engine: engine::Engine::new(
                context,
                schedule,
                #[cfg(feature = "profile")]
                "dynamis step",
            ),
            rigid: Rigid::new(context, streams, rigid, resolution),
            ccd: Ccd::new(context, streams, ccd),
            soft: Soft::new(context, streams, soft),
        }
    }

    pub(crate) fn encode(
        &self,
        encoder: &mut CommandEncoder,
        streams: &Streams,
        frame: &Frame,
        idle: bool,
        ccd_active: bool,
        soft_active: bool,
    ) {
        self.rigid
            .encode(&self.engine, encoder, streams, frame, idle);
        if ccd_active && !idle {
            self.ccd.encode(&self.engine, encoder, streams, frame);
        }
        if soft_active && !idle {
            self.soft.encode(&self.engine, encoder, streams, frame);
        }
        self.rigid
            .encode_resolution(&self.engine, encoder, streams, frame, idle);
    }

    pub(crate) fn encode_queries(
        &self,
        encoder: &mut CommandEncoder,
        streams: &Streams,
        frame: &Frame,
    ) {
        self.rigid
            .encode_queries(&self.engine, encoder, streams, frame);
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
