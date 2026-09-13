use crate::World;
use dynamis_abi::{Counters, FrameCounts, RowStreams, StepParamsRecord};
use dynamis_broadphase::BroadphaseDomain;
use dynamis_domain::{Domain, Run, StepFacts};
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

impl HostWork {
    pub(crate) fn pending(&self) -> bool {
        self.state.queries > 0
            || self.state.shape_uploads
            || self.rigid.body_commands > 0
            || self.rigid.constraint_commands > 0
            || self.soft.uploads
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Activity {
    state: bool,
    rigid: bool,
    soft: bool,
}

impl Activity {
    pub(crate) const BUSY: Self = Self {
        state: true,
        rigid: true,
        soft: true,
    };

    fn of(measured: &Counters, work: &HostWork) -> Self {
        let mut live = Self {
            state: StateDomain::active(measured, &work.state),
            rigid: RigidDomain::active(measured, &work.rigid),
            soft: SoftDomain::active(measured, &work.soft),
        };
        live.rigid |= live.soft;
        live.soft |= live.rigid;
        live
    }

    pub(crate) fn simulating(self) -> bool {
        self.rigid || self.soft
    }

    pub(crate) fn busy(self) -> bool {
        self.state || self.simulating()
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
        let (particles, elements, adjacency) = self.soft.used();
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
                ccd: self.ccd_active(),
            },
            soft: dynamis_soft::SoftInputs {
                particles,
                elements,
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
            colliders: self.colliders.used(),
            constraints: self.constraints.alive.len() as u32,
            particles: self.soft.used().0,
            elements: self.soft.used().1,
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
            },
            self.event_slot_of(self.clock.step),
        )
    }

    pub(crate) fn busy(&self, work: &HostWork) -> bool {
        self.observed(work).busy()
    }

    fn observed(&self, work: &HostWork) -> Activity {
        if self.backend.measured_step.is_none() {
            return Activity::BUSY;
        }
        Activity::of(&self.backend.measured, work)
    }

    fn gated(&mut self, work: &HostWork) -> Activity {
        let mut liveness = self.observed(work);
        let lag = self.backend.rest.gate(liveness.simulating());
        liveness.rigid |= lag;
        liveness.soft |= lag;
        liveness
    }

    pub(crate) fn frames(
        &mut self,
        live: &Live,
        work: &HostWork,
        params: StepParamsRecord,
    ) -> StepFrames {
        let liveness = self.gated(work);
        let indexing = liveness.busy();
        let facts = StepFacts {
            params,
            counts: self.frame_counts(),
        };
        StepFrames {
            state: StateDomain::frame(
                &facts,
                &live.state,
                Run {
                    awake: liveness.state,
                    indexing,
                },
            ),
            broadphase: BroadphaseDomain::frame(
                &facts,
                &live.broadphase,
                Run {
                    awake: indexing,
                    indexing,
                },
            ),
            rigid: RigidDomain::frame(
                &facts,
                &live.rigid,
                Run {
                    awake: liveness.rigid,
                    indexing,
                },
            ),
            soft: SoftDomain::frame(
                &facts,
                &live.soft,
                Run {
                    awake: liveness.soft,
                    indexing,
                },
            ),
        }
    }
}
