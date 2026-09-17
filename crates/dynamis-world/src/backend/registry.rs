use crate::World;
use dynamis_abi::{Census, Counters, DeclaredCounters, RowStreams};
use dynamis_broadphase::BroadphaseDomain;
use dynamis_domain::StepFacts;
use dynamis_gpu::GpuContext;
use dynamis_pass::{Pass, PipelineBuilder, Run, Schedule};
use dynamis_rigid::RigidDomain;
use dynamis_scene::SceneDomain;
use dynamis_soft::SoftDomain;
use dynamis_state::StateDomain;
use wgpu::CommandEncoder;

dynamis_domain::domains! {
    state: StateDomain,
    broadphase: BroadphaseDomain,
    rigid: RigidDomain,
    soft: SoftDomain,
    scene: SceneDomain,
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
        let soft = dynamis_soft::plan(measured, &live.soft, &streams.soft, release);
        let scene = dynamis_scene::plan(&live.scene, &streams.scene, release);
        Self {
            state,
            broadphase,
            rigid,
            soft,
            scene,
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
    pub(crate) fn census(&self) -> Census {
        let (particles, elements, attachments, adjacency) = self.soft.used();
        Census {
            bodies: self.bodies.pool.len(),
            dynamic_bodies: self.bodies.dynamic,
            body_ids: self.bodies.pool.ids(),
            colliders: self.colliders.used(),
            live_colliders: self.colliders.live(),
            movable_colliders: self.colliders.movable(),
            constraints: self.constraints.pool.len(),
            constraint_ids: self.constraints.pool.ids(),
            particles,
            elements,
            attachments,
            adjacency,
            soft_bodies: self.soft.ids_len() as u32,
            characters: self.characters.slots(),
            live_characters: self.characters.count(),
            vehicles: self.vehicles.slots(),
            vehicle_wheels: self.vehicles.wheel_count(),
            live_vehicles: self.vehicles.count(),
            observed: self.observed.bodies.len(),
            observed_joints: self.observed.joints.len(),
            observed_joint_demand: self.observed.joints.demand(),
            queries: self.queries.pending.len() as u32,
            query_hits: self.queries.pending_hits,
            body_commands: self.bodies.commands.len() as u32,
            constraint_commands: self.constraints.commands.len() as u32,
            pending_soft_edits: self.soft.pending_edits(),
            pending_soft_body_edits: self.soft.pending_body_edits(),
        }
    }

    pub(crate) fn live(&self, census: &Census) -> Live {
        Live {
            state: dynamis_state::StateInputs {
                bodies: census.bodies,
                body_ids: census.body_ids,
                collider_pool: census.colliders,
                constraints: census.constraints,
                constraint_ids: census.constraint_ids,
                body_commands: census.body_commands,
                constraint_commands: census.constraint_commands,
                shapes: self.shapes.pool.used(),
                observed: census.observed,
                observed_joints: census.observed_joint_demand,
            },
            broadphase: dynamis_broadphase::BroadphaseInputs {
                colliders: census.live_colliders,
                movable_colliders: census.movable_colliders,
                particles: census.particles,
            },
            rigid: dynamis_rigid::RigidInputs {
                bodies: census.bodies,
                persist_events: self.colliders.persist_events() > 0,
                movable_colliders: census.movable_colliders,
                collider_pool: census.colliders,
                constraints: census.constraints,
                observed: census.observed,
                observed_joints: census.observed_joints,
                ccd: self.ccd_active(),
                impacts: self.colliders.impact_armed() > 0,
                characters: census.characters,
                vehicles: census.vehicles,
                vehicle_wheels: census.vehicle_wheels,
            },
            soft: dynamis_soft::SoftInputs {
                particles: census.particles,
                elements: census.elements,
                attachments: census.attachments,
                adjacency: census.adjacency,
                bodies: census.soft_bodies,
                edits: census.pending_soft_edits,
                body_edits: census.pending_soft_body_edits,
                material: self.soft.carries_strength(),
                events: self.soft.carries_events(),
            },
            scene: dynamis_scene::SceneInputs {
                queries: census.queries,
                hits: census.query_hits,
            },
        }
    }

    pub(crate) fn host_work(&self) -> HostWork {
        HostWork {
            state: dynamis_state::StateWork {
                shape_uploads: self.shapes.uploaded,
            },
            broadphase: (),
            rigid: dynamis_rigid::RigidWork {
                body_commands: self.bodies.last_edits + self.bodies.last_moves,
                constraint_commands: self.constraints.last_commands + self.constraints.last_moves,
                character_inputs: self.characters.last_inputs > 0,
                vehicle_inputs: self.vehicles.last_inputs > 0,
                wake_all: self.wake_all,
            },
            soft: dynamis_soft::SoftWork {
                uploads: self.soft.uploaded,
                body_edits: self.soft.last_body_edits,
                edits: self.soft.last_edits,
                wake_all: self.wake_all,
            },
            scene: dynamis_scene::SceneWork {
                queries: self.queries.pending.len() as u32,
            },
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

    pub(crate) fn declared_counters(&self, census: &Census, rows: &RowStreams) -> DeclaredCounters {
        DeclaredCounters {
            bodies: census.bodies,
            colliders: census.live_colliders,
            movable_colliders: census.movable_colliders,
            constraints: census.constraints,
            body_edits: rows.body_edit_runs,
            body_moves: rows.body_moves,
            constraint_commands: self.constraints.last_commands,
            constraint_moves: rows.constraint_moves,
        }
    }

    pub(crate) fn busy(&self, work: &HostWork) -> bool {
        let census = self.census();
        self.observed(&self.live(&census), work).busy()
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

    pub(crate) fn frames(&mut self, live: &Live, work: &HostWork, facts: &StepFacts) -> StepFrames {
        let liveness = self.gated(live, work);
        StepFrames::of(facts, live, liveness)
    }

    pub(crate) fn query_frames(&self, live: &Live, facts: &StepFacts) -> StepFrames {
        StepFrames::queries(facts, live)
    }

    pub(crate) fn publish_frames(&self, live: &Live, facts: &StepFacts) -> StepFrames {
        StepFrames::publication(facts, live)
    }
}
