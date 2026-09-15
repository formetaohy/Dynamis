use crate::World;
use dynamis_abi::{
    Counters, DeclaredCounters, FrameCounts, RowStreams, StepParamsRecord, Subscriptions,
};
use dynamis_broadphase::BroadphaseDomain;
use dynamis_domain::StepFacts;
use dynamis_gpu::GpuContext;
use dynamis_pass::{Pass, PipelineBuilder, Run, Schedule};
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

pub(crate) const ACTIVITY_LAG: u32 = dynamis_gpu::FACT_LAG as u32;

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

impl Plan {
    pub(crate) fn of(measured: &Counters, live: &Live, streams: &Streams, release: bool) -> Self {
        let broadphase =
            dynamis_broadphase::plan(measured, &live.broadphase, &streams.broadphase, release);
        let rigid = dynamis_rigid::plan(
            measured,
            &live.rigid,
            &streams.rigid,
            broadphase.pairs,
            release,
        );
        let state = dynamis_state::plan(&live.state, &streams.state, release);
        let soft = dynamis_soft::plan(&live.soft, &streams.soft, release);
        Self {
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
        run: Run,
    ) {
        let Self { schedule, runtimes } = self;
        schedule.begin(run);
        schedule.record(encoder, |pass, index, recorder| {
            runtimes.record(pass, index, recorder, streams, frames)
        });
        schedule.finish();
    }

    pub(crate) fn declared(&self) -> &[Pass] {
        self.schedule.pipeline().passes()
    }

    pub(crate) fn ran(&self) -> Vec<&'static str> {
        self.schedule.ran_labels()
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
                constraint_ids: self.constraints.ids.len() as u32,
                body_commands,
                constraint_commands,
                queries,
                shapes: self.shapes.pool.used(),
                observed: self.observed.bodies.len(),
                observed_joints: self.observed.joints.demand(),
            },
            broadphase: dynamis_broadphase::BroadphaseInputs {
                colliders,
                particles,
            },
            rigid: dynamis_rigid::RigidInputs {
                bodies,
                colliders,
                collider_pool,
                constraints,
                queries,
                observed: self.observed.bodies.len(),
                observed_joints: self.observed.joints.len(),
                ccd: self.ccd_active(),
                impacts: self.colliders.impact_armed() > 0,
                characters: self.characters.slots(),
                vehicles: self.vehicles.slots(),
            },
            soft: dynamis_soft::SoftInputs {
                particles,
                elements,
                attachments,
                adjacency,
                bodies: self.soft.ids_len() as u32,
                edits: self.soft.pending_edits(),
                body_edits: self.soft.pending_body_edits(),
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
                character_inputs: self.characters.last_inputs > 0,
                vehicle_inputs: self.vehicles.last_inputs > 0,
            },
            soft: dynamis_soft::SoftWork {
                uploads: self.soft.uploaded,
                body_edits: self.soft.last_body_edits,
                edits: self.soft.last_edits,
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
            characters: self.characters.slots(),
            vehicles: self.vehicles.slots(),
        }
    }

    pub(crate) fn step_params(&self, dt: f32) -> StepParamsRecord {
        StepParamsRecord::new(&self.config, dt, self.frame_counts(), self.subscriptions())
    }

    pub(crate) fn subscriptions(&self) -> Subscriptions {
        Subscriptions {
            observed: self.observed.bodies.len(),
            observed_joints: self.observed.joints.len(),
        }
    }

    pub(crate) fn row_streams(&self) -> RowStreams {
        RowStreams {
            body_edit_runs: self.bodies.last_edits,
            body_moves: self.bodies.last_moves,
            constraint_moves: self.constraints.last_moves,
            soft_edits: self.soft.last_edits,
            soft_body_edits: self.soft.last_body_edits,
        }
    }

    pub(crate) fn declared_counters(&self) -> DeclaredCounters {
        DeclaredCounters {
            bodies: self.bodies.alive.len() as u32,
            colliders: self.colliders.live(),
            constraints: self.constraints.alive.len() as u32,
            body_edits: self.bodies.last_edits,
            body_moves: self.bodies.last_moves,
            constraint_commands: self.constraints.last_commands,
            constraint_moves: self.constraints.last_moves,
        }
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
        let facts = self.step_facts(params);
        StepFrames::of(&facts, live, liveness)
    }

    pub(crate) fn query_frames(&self, live: &Live, params: StepParamsRecord) -> StepFrames {
        let facts = self.step_facts(params);
        StepFrames::queries(&facts, live)
    }

    pub(crate) fn publish_frames(&self, live: &Live, params: StepParamsRecord) -> StepFrames {
        let facts = self.step_facts(params);
        StepFrames::publication(&facts, live)
    }

    fn step_facts(&self, params: StepParamsRecord) -> StepFacts {
        StepFacts {
            params,
            counts: self.frame_counts(),
            subscriptions: self.subscriptions(),
            rows: self.row_streams(),
        }
    }
}
