use super::streams::Streams;
use dynamis_broadphase::{Broadphase, BroadphasePasses};
use dynamis_gpu::GpuContext;
use dynamis_pass::{PassOrder, Schedule};
use dynamis_rigid::{Ccd, CcdPasses, Rigid, RigidPasses, RigidResolutionPasses};
use dynamis_soft::{Soft, SoftPasses};
use dynamis_state::StepFrame;
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

    pub(crate) fn encode(
        &self,
        encoder: &mut CommandEncoder,
        streams: &Streams,
        frame: &StepFrame,
        idle: bool,
        ccd_active: bool,
        soft_active: bool,
    ) {
        self.rigid
            .encode_commands(&self.schedule, encoder, streams, frame);
        if idle {
            self.rigid
                .encode_resolution(&self.schedule, encoder, streams, frame, true);
            return;
        }
        self.rigid
            .encode_prepare(&self.schedule, encoder, streams, frame);
        if soft_active {
            self.soft.encode_bounds(&self.schedule, encoder, streams);
        }
        self.rigid
            .encode_entries(&self.schedule, encoder, streams, frame);
        if soft_active {
            self.soft.encode_entries(&self.schedule, encoder, streams);
        }
        self.broadphase.encode(&self.schedule, encoder, streams);
        self.rigid
            .encode_contacts(&self.schedule, encoder, streams, frame);
        if ccd_active {
            self.ccd.encode(&self.schedule, encoder, streams, frame);
        }
        if soft_active {
            self.soft.encode(&self.schedule, encoder, streams, frame);
        }
        self.rigid
            .encode_resolution(&self.schedule, encoder, streams, frame, false);
    }

    pub(crate) fn encode_queries(
        &self,
        encoder: &mut CommandEncoder,
        streams: &Streams,
        frame: &StepFrame,
    ) {
        let mut commands =
            dynamis_gpu::ComputeRecorder::begin(encoder, "query commands", self.schedule.per_row());
        self.rigid
            .record_query_commands(&mut commands, streams, frame);
        drop(commands);

        let mut sort =
            dynamis_gpu::ComputeRecorder::begin(encoder, "query sort", self.schedule.per_row());
        self.broadphase.sort_entries(&mut sort, streams);
        drop(sort);

        let mut flush =
            dynamis_gpu::ComputeRecorder::begin(encoder, "query flush", self.schedule.per_row());
        self.rigid.record_query_flush(&mut flush, streams, frame);
        drop(flush);
    }

    #[cfg(feature = "profile")]
    pub(crate) fn capture_timings(
        &mut self,
        encoder: &mut dynamis_gpu::SubmissionEncoder,
        sequence: u64,
    ) -> Option<(u64, Vec<dynamis_gpu::GpuPassTiming>)> {
        self.schedule.capture_timings(encoder, sequence)
    }

    #[cfg(feature = "profile")]
    pub(crate) fn collect_timings(&mut self) -> Vec<(u64, Vec<dynamis_gpu::GpuPassTiming>)> {
        self.schedule.collect_timings()
    }
}
