use super::capacity::StreamCapacity;
use super::readback::ReadbackBuffers;
use crate::World;
use dynamis_abi::{Counters, FrameCounts, RowStreams, StepParamsRecord};
use dynamis_broadphase::BroadphaseDomain;
use dynamis_domain::{Domain, HostWork, Ledger, StepFacts};
use dynamis_gpu::{ComputeRecorder, GpuContext, GpuSlot, Stream};
use dynamis_pass::{Pass, PassOrder, Phase, ResourceId, Resources, Schedule};
use dynamis_rigid::RigidDomain;
use dynamis_soft::SoftDomain;
use dynamis_state::StateDomain;
use wgpu::{CommandEncoder, Device, Queue};

type State = StateDomain;
type Broadphase = BroadphaseDomain;
type Rigid = RigidDomain;
type Soft = SoftDomain;

const _: () = {
    assert!(State::ID != Broadphase::ID);
    assert!(State::ID != Rigid::ID);
    assert!(State::ID != Soft::ID);
    assert!(Broadphase::ID != Rigid::ID);
    assert!(Broadphase::ID != Soft::ID);
    assert!(Rigid::ID != Soft::ID);
};

pub(crate) struct Plan {
    pub(crate) state: <State as Domain>::Demand,
    pub(crate) broadphase: <Broadphase as Domain>::Demand,
    pub(crate) rigid: <Rigid as Domain>::Demand,
    pub(crate) soft: <Soft as Domain>::Demand,
}

pub(crate) struct Live {
    pub(crate) state: <State as Domain>::Inputs,
    pub(crate) broadphase: <Broadphase as Domain>::Inputs,
    pub(crate) rigid: <Rigid as Domain>::Inputs,
    pub(crate) soft: <Soft as Domain>::Inputs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Liveness {
    pub(crate) rigid: bool,
    pub(crate) soft: bool,
    pub(crate) host: bool,
}

impl Liveness {
    pub(crate) const AWAKE: Self = Self {
        rigid: true,
        soft: true,
        host: true,
    };

    pub(crate) fn of(measured: &Counters, work: &HostWork) -> Self {
        let soft = <Soft as Domain>::active(measured, work);
        let rigid = <Rigid as Domain>::active(measured, work) || soft;
        Self {
            rigid,
            soft,
            host: <State as Domain>::active(measured, work),
        }
    }

    pub(crate) const fn indexing(self) -> bool {
        self.rigid || self.soft || self.host
    }
}

pub(crate) const LIVENESS_LAG: u32 = dynamis_gpu::Readback::DEPTH as u32 + 2;

#[derive(Clone, Copy)]
pub(crate) struct Rest {
    quiet: u32,
}

impl Rest {
    pub(crate) const IDLE: Self = Self { quiet: 0 };

    pub(crate) fn gate(&mut self, awake: bool) -> bool {
        if awake {
            self.quiet = 0;
            return true;
        }
        self.quiet = self.quiet.saturating_add(1);
        self.quiet < LIVENESS_LAG
    }
}

pub(crate) struct Planning {
    state: <State as Domain>::Planner,
    broadphase: <Broadphase as Domain>::Planner,
    rigid: <Rigid as Domain>::Planner,
    soft: <Soft as Domain>::Planner,
}

impl Planning {
    pub(crate) const fn new() -> Self {
        Self {
            state: (),
            broadphase: dynamis_broadphase::Capacity::new(),
            rigid: dynamis_rigid::Capacity::new(),
            soft: (),
        }
    }

    pub(crate) fn minimum() -> Plan {
        Plan {
            state: <State as Domain>::minimum(),
            broadphase: <Broadphase as Domain>::minimum(),
            rigid: <Rigid as Domain>::minimum(),
            soft: <Soft as Domain>::minimum(),
        }
    }

    pub(crate) fn plan(&mut self, measured: &Counters, live: &Live, streams: &Streams) -> Plan {
        let mut ledger = Ledger::default();
        let broadphase = <Broadphase as Domain>::plan(
            &mut self.broadphase,
            measured,
            &live.broadphase,
            &mut ledger,
            &streams.broadphase,
        );
        let rigid = <Rigid as Domain>::plan(
            &mut self.rigid,
            measured,
            &live.rigid,
            &mut ledger,
            &streams.rigid,
        );
        let state = <State as Domain>::plan(
            &mut self.state,
            measured,
            &live.state,
            &mut ledger,
            &streams.state,
        );
        let soft = <Soft as Domain>::plan(
            &mut self.soft,
            measured,
            &live.soft,
            &mut ledger,
            &streams.soft,
        );
        ledger.assert_complete();
        Plan {
            state,
            broadphase,
            rigid,
            soft,
        }
    }
}

pub(crate) struct Streams {
    pub(crate) state: <State as Domain>::Streams,
    pub(crate) broadphase: <Broadphase as Domain>::Streams,
    pub(crate) rigid: <Rigid as Domain>::Streams,
    pub(crate) soft: <Soft as Domain>::Streams,
    pub(crate) readback: ReadbackBuffers,
    generation: u64,
}

impl Streams {
    pub(crate) fn new(device: &Device, queue: &Queue, plan: &Plan) -> Self {
        Self {
            state: dynamis_state::StateStreams::new(device, queue, &plan.state),
            broadphase: dynamis_broadphase::BroadphaseStreams::new(device, queue, &plan.broadphase),
            rigid: dynamis_rigid::RigidStreams::new(device, queue, &plan.rigid),
            soft: dynamis_soft::SoftStreams::new(device, queue, &plan.soft),
            readback: ReadbackBuffers::new(device, plan),
            generation: 0,
        }
    }

    pub(crate) fn matches(&self, plan: &Plan) -> bool {
        self.state.matches(&plan.state)
            && self.broadphase.matches(&plan.broadphase)
            && self.rigid.matches(&plan.rigid)
            && self.soft.matches(&plan.soft)
            && self.readback.matches(plan)
    }

    pub(crate) fn readback_matches(&self, plan: &Plan) -> bool {
        self.readback.matches(plan)
    }

    pub(crate) fn reserve(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        plan: &Plan,
    ) -> bool {
        let changed = self.state.reserve(device, encoder, &plan.state)
            | self.broadphase.reserve(device, encoder, &plan.broadphase)
            | self.rigid.reserve(device, encoder, &plan.rigid)
            | self.soft.reserve(device, encoder, &plan.soft)
            | self.readback.reserve(device, plan);
        if changed {
            self.generation = self
                .generation
                .checked_add(1)
                .expect("a storage generation must not overflow");
        }
        changed
    }

    pub(crate) fn durable(&self) -> Vec<(&'static str, &Stream)> {
        let mut streams = Vec::new();
        streams.extend(self.state.durable());
        streams.extend(self.broadphase.durable());
        streams.extend(self.rigid.durable());
        streams.extend(self.soft.durable());
        streams
    }

    pub(crate) fn require(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        floors: impl Fn(&'static str) -> Option<u32>,
    ) -> bool {
        let floors = &floors;
        self.state.require(device, encoder, floors)
            | self.broadphase.require(device, encoder, floors)
            | self.rigid.require(device, encoder, floors)
            | self.soft.require(device, encoder, floors)
    }

    pub(crate) fn stream_capacity(&self) -> StreamCapacity {
        StreamCapacity {
            state: dynamis_state::capacity(&self.state),
            broadphase: dynamis_broadphase::capacity(&self.broadphase),
            rigid: dynamis_rigid::capacity(&self.rigid),
            soft: dynamis_soft::capacity(&self.soft),
        }
    }
}

impl Resources for Streams {
    fn generation(&self) -> u64 {
        self.generation
    }

    fn slots(&self, resource: ResourceId) -> u32 {
        match resource.domain() {
            State::ID => self.state.slots(resource.local()),
            Broadphase::ID => self.broadphase.slots(resource.local()),
            Rigid::ID => self.rigid.slots(resource.local()),
            Soft::ID => self.soft.slots(resource.local()),
            domain => panic!("resource domain {domain} is outside the stream composition"),
        }
    }

    fn whole(&self, resource: ResourceId) -> GpuSlot<'_> {
        match resource.domain() {
            State::ID => self.state.whole(resource.local()),
            Broadphase::ID => self.broadphase.whole(resource.local()),
            Rigid::ID => self.rigid.whole(resource.local()),
            Soft::ID => self.soft.whole(resource.local()),
            domain => panic!("resource domain {domain} is outside the stream composition"),
        }
    }

    fn range(&self, resource: ResourceId, offset: u64, size: u64) -> GpuSlot<'_> {
        match resource.domain() {
            State::ID => self.state.range(resource.local(), offset, size),
            Broadphase::ID => self.broadphase.range(resource.local(), offset, size),
            Rigid::ID => self.rigid.range(resource.local(), offset, size),
            Soft::ID => self.soft.range(resource.local(), offset, size),
            domain => panic!("resource domain {domain} is outside the stream composition"),
        }
    }
}

pub(crate) struct StepPasses {
    schedule: Schedule,
    state: <State as Domain>::Runtime,
    broadphase: <Broadphase as Domain>::Runtime,
    rigid: <Rigid as Domain>::Runtime,
    soft: <Soft as Domain>::Runtime,
}

impl StepPasses {
    pub(crate) fn new(context: &GpuContext, streams: &Streams) -> Self {
        let mut order = PassOrder::new();
        <State as Domain>::claim(&mut order);
        let broadphase = <Broadphase as Domain>::claim(&mut order);
        let rigid = <Rigid as Domain>::claim(&mut order);
        let soft = <Soft as Domain>::claim(&mut order);
        Self {
            schedule: Schedule::new(
                context,
                order,
                #[cfg(feature = "profile")]
                "dynamis step",
            ),
            state: <State as Domain>::build(context, streams, ()),
            broadphase: <Broadphase as Domain>::build(context, streams, broadphase),
            rigid: <Rigid as Domain>::build(context, streams, rigid),
            soft: <Soft as Domain>::build(context, streams, soft),
        }
    }

    pub(crate) fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        streams: &Streams,
        frames: &StepFrames,
    ) {
        self.schedule.begin_step();
        for phase in Phase::ALL {
            <Rigid as Domain>::record(
                &self.rigid,
                *phase,
                &mut self.schedule,
                encoder,
                streams,
                &frames.rigid,
            );
            <Broadphase as Domain>::record(
                &self.broadphase,
                *phase,
                &mut self.schedule,
                encoder,
                streams,
                &frames.broadphase,
            );
            <Soft as Domain>::record(
                &self.soft,
                *phase,
                &mut self.schedule,
                encoder,
                streams,
                &frames.soft,
            );
            <State as Domain>::record(
                &self.state,
                *phase,
                &mut self.schedule,
                encoder,
                streams,
                &frames.state,
            );
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
        self.rigid
            .simulation
            .record_query_commands(&mut commands, streams, &frames.rigid);
        drop(commands);

        let mut sort = ComputeRecorder::begin(encoder, "query sort", self.schedule.per_row());
        self.broadphase.sort_entries(&mut sort, streams);
        drop(sort);

        let mut flush = ComputeRecorder::begin(encoder, "query flush", self.schedule.per_row());
        self.rigid
            .simulation
            .record_query_flush(&mut flush, streams, &frames.rigid);
        drop(flush);
    }

    pub(crate) fn declared(&self) -> &[Pass] {
        self.schedule.declared()
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

pub(crate) struct StepFrames {
    pub(crate) params: StepParamsRecord,
    pub(crate) state: <State as Domain>::Frame,
    pub(crate) broadphase: <Broadphase as Domain>::Frame,
    pub(crate) rigid: <Rigid as Domain>::Frame,
    pub(crate) soft: <Soft as Domain>::Frame,
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
        Live {
            state: dynamis_state::StateInputs {
                bodies,
                body_ids: self.bodies.ids.len() as u32,
                collider_pool,
                constraints,
                body_commands,
                constraint_commands,
                queries: self.queries.pending.len() as u32,
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
                ccd: self.ccd_active(),
            },
            soft: dynamis_soft::SoftInputs {
                particles,
                elements,
                adjacency,
                material: self.soft.carries_strength(),
            },
        }
    }

    pub(crate) fn host_work(&self) -> HostWork {
        HostWork {
            body_commands: self.bodies.last_edits + self.bodies.last_moves,
            constraint_commands: self.constraints.last_commands + self.constraints.last_moves,
            queries: self.queries.pending.len() as u32,
            shape_uploads: self.shapes.uploaded,
            soft_bodies: self.soft.count() as u32,
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
        }
    }

    fn observed(&self, work: &HostWork) -> Liveness {
        if self.backend.measured_step.is_none() {
            return Liveness::AWAKE;
        }
        Liveness::of(&self.backend.measured, work)
    }

    fn gated(&mut self, work: &HostWork) -> Liveness {
        let mut liveness = self.observed(work);
        let awake = self.backend.measured[dynamis_abi::COUNTER_ACTIVE] != 0
            || work.body_commands > 0
            || work.constraint_commands > 0;
        liveness.rigid = self.backend.rest.gate(awake) || liveness.soft;
        liveness
    }

    pub(crate) fn frames(&mut self, live: &Live, work: &HostWork, dt: f32) -> StepFrames {
        let counts = self.frame_counts();
        let params = StepParamsRecord::new(
            &self.config,
            dt,
            counts,
            RowStreams {
                edit_runs: self.bodies.last_edits,
                body_moves: self.bodies.last_moves,
                constraint_moves: self.constraints.last_moves,
            },
            self.event_slot_of(self.clock.step),
        );
        let liveness = self.gated(work);
        let facts = StepFacts {
            rigid: liveness.rigid,
            soft: liveness.soft,
            indexing: liveness.indexing(),
            params,
            counts,
            work: *work,
        };
        StepFrames {
            params,
            state: <State as Domain>::frame(&facts, &live.state),
            broadphase: <Broadphase as Domain>::frame(&facts, &live.broadphase),
            rigid: <Rigid as Domain>::frame(&facts, &live.rigid),
            soft: <Soft as Domain>::frame(&facts, &live.soft),
        }
    }
}
