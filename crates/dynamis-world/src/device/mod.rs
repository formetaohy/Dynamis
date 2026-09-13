mod capacity;
mod readback;
pub(crate) mod streams;

use dynamis_broadphase::{Broadphase, BroadphasePasses};
use dynamis_rigid::{Ccd, CcdPasses, Rigid, RigidPasses, RigidResolutionPasses};
use dynamis_soft::{Soft, SoftPasses};
use streams::Streams;

use dynamis_engine::{Engine, Schedule};
use dynamis_gpu::GpuContext;
use dynamis_scene::Frame;
use wgpu::CommandEncoder;

pub use capacity::StreamCapacity;

pub(crate) struct Pipeline {
    engine: Engine,
    rigid: Rigid,
    broadphase: Broadphase,
    ccd: Ccd,
    soft: Soft,
}

impl Pipeline {
    pub(crate) fn new(context: &GpuContext, streams: &Streams) -> Self {
        let mut schedule = Schedule::new();
        let rigid = RigidPasses::claim(&mut schedule);
        let broadphase = BroadphasePasses::claim(&mut schedule);
        let ccd = CcdPasses::claim(&mut schedule);
        let soft = SoftPasses::claim(&mut schedule);
        let resolution = RigidResolutionPasses::claim(&mut schedule);
        Self {
            engine: Engine::new(
                context,
                schedule,
                #[cfg(feature = "profile")]
                "dynamis step",
            ),
            rigid: Rigid::new(context, streams, rigid, resolution),
            broadphase: Broadphase::new(context, streams, broadphase),
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
            .encode_commands(&self.engine, encoder, streams, frame);
        if idle {
            self.rigid
                .encode_resolution(&self.engine, encoder, streams, frame, true);
            return;
        }
        self.rigid
            .encode_prepare(&self.engine, encoder, streams, frame);
        if soft_active {
            self.soft.encode_bounds(&self.engine, encoder, streams);
        }
        self.rigid
            .encode_entries(&self.engine, encoder, streams, frame);
        if soft_active {
            self.soft.encode_entries(&self.engine, encoder, streams);
        }
        self.broadphase.encode(&self.engine, encoder, streams);
        self.rigid
            .encode_contacts(&self.engine, encoder, streams, frame);
        if ccd_active {
            self.ccd.encode(&self.engine, encoder, streams, frame);
        }
        if soft_active {
            self.soft.encode(&self.engine, encoder, streams, frame);
        }
        self.rigid
            .encode_resolution(&self.engine, encoder, streams, frame, false);
    }

    pub(crate) fn encode_queries(
        &self,
        encoder: &mut CommandEncoder,
        streams: &Streams,
        frame: &Frame,
    ) {
        let mut commands =
            dynamis_gpu::ComputeRecorder::begin(encoder, "query commands", self.engine.per_row());
        self.rigid
            .record_query_commands(&mut commands, streams, frame);
        drop(commands);

        let mut sort =
            dynamis_gpu::ComputeRecorder::begin(encoder, "query sort", self.engine.per_row());
        self.broadphase.sort_entries(&mut sort, streams);
        drop(sort);

        let mut flush =
            dynamis_gpu::ComputeRecorder::begin(encoder, "query flush", self.engine.per_row());
        self.rigid.record_query_flush(&mut flush, streams, frame);
        drop(flush);
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
