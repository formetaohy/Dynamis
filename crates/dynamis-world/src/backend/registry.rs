use crate::World;
use dynamis_abi::{Counters, FrameCounts, RowStreams, StepParamsRecord};
use dynamis_broadphase::BroadphaseDomain;
use dynamis_domain::StepFacts;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::{Pass, PipelineBuilder, Schedule};
use dynamis_rigid::RigidDomain;
use dynamis_soft::SoftDomain;
use dynamis_state::StateDomain;
use wgpu::CommandEncoder;

dynamis_domain::domains! {
    state: StateDomain,
    broadphase: BroadphaseDomain,
    rigid: RigidDomain,
    soft: SoftDomain,
    [rigid <-> soft]
}

pub(crate) const ACTIVITY_LAG: u32 = dynamis_gpu::Readback::DEPTH as u32 + 2;

#[derive(Clone, Copy)]
pub(crate) struct Rest {
    quiet: u32,
}

impl Rest {
    pub(crate) const IDLE: Self = Self { quiet: 0 };

    pub(crate) fn gate(&mut self, simulating: bool) -> bool {
        if simulating {
            self.quiet = 0;
            return true;
        }
        self.quiet = self.quiet.saturating_add(1);
        self.quiet < ACTIVITY_LAG
    }
}

impl Planning {
    pub(crate) fn plan(&mut self, measured: &Counters, live: &Live, streams: &Streams) -> Plan {
        let (broadphase, idle) =
            self.broadphase
                .plan(measured, &live.broadphase, &streams.broadphase);
        let rigid = self.rigid.plan(
            measured,
            &live.rigid,
            idle,
            broadphase.pairs,
            &streams.rigid,
        );
        let state = dynamis_state::plan(&live.state, idle, &streams.state);
        let soft = dynamis_soft::plan(&live.soft, idle, state.bodies, &streams.soft);
        Plan {
            state,
            broadphase,
            rigid,
            soft,
        }
    }
}

pub(crate) struct StepPasses {
    schedule: Schedule,
    runtimes: PassRuntimes,
}

impl StepPasses {
    pub(crate) fn new(context: &GpuContext, streams: &Streams) -> Self {
        let mut builder = PipelineBuilder::new();
        PassIds::declare(&mut builder);
        let pipeline = builder.resolve();
        let ids = PassIds::resolve(&pipeline);
        Self {
            schedule: Schedule::new(
                context,
                pipeline,
                #[cfg(feature = "profile")]
                "dynamis step",
            ),
            runtimes: PassRuntimes::build(context, streams, ids),
        }
    }

    pub(crate) fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        streams: &Streams,
        frames: &StepFrames,
    ) {
        self.schedule.begin_step();
        let Self { schedule, runtimes } = self;
        for index in 0..schedule.pipeline().len() as u32 {
            let pass = schedule.pass(index);
            runtimes.record(pass, index, schedule, encoder, streams, frames);
        }
    }

    pub(crate) fn encode_queries(
        &self,
        encoder: &mut CommandEncoder,
        streams: &Streams,
        frames: &StepFrames,
    ) {
        let mut commands =
            ComputeRecorder::begin(encoder, "query commands", self.schedule.per_row());
        self.runtimes
            .rigid
            .simulation
            .record_query_commands(&mut commands, streams, &frames.rigid);
        drop(commands);

        let mut sort = ComputeRecorder::begin(encoder, "query sort", self.schedule.per_row());
        self.runtimes.broadphase.sort_entries(&mut sort, streams);
        drop(sort);

        let mut flush = ComputeRecorder::begin(encoder, "query flush", self.schedule.per_row());
        self.runtimes
            .rigid
            .simulation
            .record_query_flush(&mut flush, streams, &frames.rigid);
        drop(flush);
    }

    pub(crate) fn declared(&self) -> &[Pass] {
        self.schedule.pipeline().passes()
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

impl World {
    pub(crate) fn live(&self) -> Live {
        let (particles, elements, attachments, adjacency) = self.soft.used();
        let bodies = self.bodies.alive.len() as u32;
        let colliders = self.colliders.live();
        let collider_pool = self.colliders.used();
        let constraints = self.constraints.alive.len() as u32;
        let body_commands = self.bodies.commands.len() as u32;
        let constraint_commands = self.constraints.commands.len() as u32;
        let queries = self.queries.pending.len() as u32;
        Live {
            state: dynamis_state::StateInputs {
                bodies,
                body_ids: self.bodies.ids.len() as u32,
                collider_pool,
                constraints,
                body_commands,
                constraint_commands,
                queries,
                shapes: self.shapes.pool.used(),
                observed: self.view.len(),
            },
            broadphase: dynamis_broadphase::BroadphaseInputs {
                colliders,
                particles,
                pending_commands: body_commands > 0 || constraint_commands > 0,
            },
            rigid: dynamis_rigid::RigidInputs {
                bodies,
                colliders,
                collider_pool,
                constraints,
                queries,
                observed: self.view.len(),
                ccd: self.ccd_active(),
            },
            soft: dynamis_soft::SoftInputs {
                particles,
                elements,
                attachments,
                adjacency,
                bodies: self.soft.ids_len() as u32,
                material: self.soft.carries_strength(),
            },
        }
    }

    pub(crate) fn host_work(&self) -> HostWork {
        HostWork {
            state: dynamis_state::StateWork {
                queries: self.queries.pending.len() as u32,
                shape_uploads: self.shapes.uploaded,
            },
            broadphase: (),
            rigid: dynamis_rigid::RigidWork {
                body_commands: self.bodies.last_edits + self.bodies.last_moves,
                constraint_commands: self.constraints.last_commands + self.constraints.last_moves,
            },
            soft: dynamis_soft::SoftWork {
                uploads: self.soft.uploaded,
            },
        }
    }

    pub(crate) fn frame_counts(&self) -> FrameCounts {
        FrameCounts {
            dynamic_bodies: self.bodies.dynamic_count as u32,
            bodies: self.bodies.alive.len() as u32,
            body_ids: self.bodies.ids.len() as u32,
            colliders: self.colliders.used(),
            constraints: self.constraints.alive.len() as u32,
            particles: self.soft.used().0,
            elements: self.soft.used().1,
            attachments: self.soft.used().2,
            soft_bodies: self.soft.ids_len() as u32,
        }
    }

    pub(crate) fn step_params(&self, dt: f32) -> StepParamsRecord {
        StepParamsRecord::new(
            &self.config,
            dt,
            self.frame_counts(),
            RowStreams {
                edit_runs: self.bodies.last_edits,
                body_moves: self.bodies.last_moves,
                constraint_moves: self.constraints.last_moves,
                observed: self.view.len(),
            },
            self.event_slot_of(self.clock.step),
        )
    }

    pub(crate) fn busy(&self, work: &HostWork) -> bool {
        self.observed(&self.live(), work).busy()
    }

    fn observed(&self, live: &Live, work: &HostWork) -> Activity {
        match self.backend.measured_step {
            None => Activity::BUSY,
            Some(_) => Activity::of(&self.backend.measured, live, work),
        }
    }

    fn gated(&mut self, live: &Live, work: &HostWork) -> Activity {
        if self.backend.measured_step.is_none() {
            return Activity::BUSY;
        }
        let liveness = Activity::of(&self.backend.measured, live, work);
        let hold = self.backend.rest.gate(liveness.simulating());
        liveness.held(hold)
    }

    pub(crate) fn frames(
        &mut self,
        live: &Live,
        work: &HostWork,
        params: StepParamsRecord,
    ) -> StepFrames {
        let liveness = self.gated(live, work);
        let facts = StepFacts {
            params,
            counts: self.frame_counts(),
        };
        StepFrames::of(&facts, live, liveness)
    }
}
