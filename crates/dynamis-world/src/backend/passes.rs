use super::frame::StepFrames;
use super::streams::Streams;
use dynamis_broadphase::{Broadphase, BroadphasePasses};
use dynamis_gpu::GpuContext;
use dynamis_pass::{Pass, PassOrder, Phase, Schedule};
use dynamis_rigid::{Ccd, CcdPasses, Rigid, RigidPasses, RigidResolutionPasses};
use dynamis_soft::{Soft, SoftPasses};
use wgpu::CommandEncoder;

pub(crate) struct StepPasses {
    schedule: Schedule,
    rigid: Rigid,
    broadphase: Broadphase,
    ccd: Ccd,
    soft: Soft,
}

impl StepPasses {
    pub(crate) fn new(context: &GpuContext, streams: &Streams) -> Self {
        let mut order = PassOrder::new();
        let rigid = RigidPasses::claim(&mut order);
        let broadphase = BroadphasePasses::claim(&mut order);
        let ccd = CcdPasses::claim(&mut order);
        let soft = SoftPasses::claim(&mut order);
        let resolution = RigidResolutionPasses::claim(&mut order);
        Self {
            schedule: Schedule::new(
                context,
                order,
                #[cfg(feature = "profile")]
                "dynamis step",
            ),
            rigid: Rigid::new(context, streams, rigid, resolution),
            broadphase: Broadphase::new(context, streams, broadphase),
            ccd: Ccd::new(context, streams, ccd),
            soft: Soft::new(context, streams, soft),
        }
    }

    pub(crate) fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        streams: &Streams,
        frames: &StepFrames,
    ) {
        let schedule = &mut self.schedule;
        schedule.begin_step();
        for phase in Phase::ALL {
            self.rigid
                .record(*phase, schedule, encoder, streams, &frames.rigid);
            self.broadphase
                .record(*phase, schedule, encoder, streams, frames.broadphase);
            self.ccd
                .record(*phase, schedule, encoder, streams, &frames.rigid);
            self.soft
                .record(*phase, schedule, encoder, streams, &frames.soft);
        }
    }

    pub(crate) fn declared(&self) -> &[Pass] {
        self.schedule.declared()
    }

    pub(crate) fn encode_queries(
        &self,
        encoder: &mut CommandEncoder,
        streams: &Streams,
        frames: &StepFrames,
    ) {
        let mut commands =
            dynamis_gpu::ComputeRecorder::begin(encoder, "query commands", self.schedule.per_row());
        self.rigid
            .record_query_commands(&mut commands, streams, &frames.rigid);
        drop(commands);

        let mut sort =
            dynamis_gpu::ComputeRecorder::begin(encoder, "query sort", self.schedule.per_row());
        self.broadphase.sort_entries(&mut sort, streams);
        drop(sort);

        let mut flush =
            dynamis_gpu::ComputeRecorder::begin(encoder, "query flush", self.schedule.per_row());
        self.rigid
            .record_query_flush(&mut flush, streams, &frames.rigid);
        drop(flush);
    }

    #[cfg(feature = "profile")]
    pub(crate) fn capture_timings(
        &mut self,
        encoder: &mut dynamis_gpu::SubmissionEncoder,
    ) -> Option<Vec<dynamis_gpu::GpuPassTiming>> {
        self.schedule.capture_timings(encoder)
    }

    #[cfg(feature = "profile")]
    pub(crate) fn collect_timings(&mut self) -> Vec<Vec<dynamis_gpu::GpuPassTiming>> {
        self.schedule.collect_timings()
    }
}
