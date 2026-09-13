use super::capacity::StreamCapacity;
use super::readback::ReadbackBuffers;
use dynamis_abi::Counters;
use dynamis_broadphase as broadphase;
use dynamis_gpu::{GpuSlot, Stream};
use dynamis_pass::{ResourceId, Resources};
use dynamis_rigid as rigid;
use dynamis_soft as soft;
use dynamis_state as state;
use wgpu::{CommandEncoder, Device, Queue};

const _: () = {
    assert!(state::DOMAIN != broadphase::DOMAIN);
    assert!(state::DOMAIN != rigid::DOMAIN);
    assert!(state::DOMAIN != soft::DOMAIN);
    assert!(broadphase::DOMAIN != rigid::DOMAIN);
    assert!(broadphase::DOMAIN != soft::DOMAIN);
    assert!(rigid::DOMAIN != soft::DOMAIN);
};

pub(crate) struct Plan {
    pub(crate) state: state::StateDemand,
    pub(crate) broadphase: broadphase::BroadphaseDemand,
    pub(crate) rigid: rigid::RigidDemand,
    pub(crate) soft: soft::SoftDemand,
}

pub(crate) struct Live {
    pub(crate) state: state::StateInputs,
    pub(crate) broadphase: broadphase::BroadphaseInputs,
    pub(crate) rigid: rigid::RigidInputs,
    pub(crate) soft: soft::SoftInputs,
}

pub(crate) struct Planning {
    broadphase: broadphase::Capacity,
    rigid: rigid::Capacity,
}

pub(crate) struct Streams {
    pub(crate) state: state::StateStreams,
    pub(crate) broadphase: broadphase::BroadphaseStreams,
    pub(crate) rigid: rigid::RigidStreams,
    pub(crate) soft: soft::SoftStreams,
    pub(crate) readback: ReadbackBuffers,
    generation: u64,
}

impl Streams {
    pub(crate) fn new(device: &Device, queue: &Queue, plan: &Plan) -> Self {
        Self {
            state: state::StateStreams::new(device, queue, &plan.state),
            broadphase: broadphase::BroadphaseStreams::new(device, queue, &plan.broadphase),
            rigid: rigid::RigidStreams::new(device, queue, &plan.rigid),
            soft: soft::SoftStreams::new(device, queue, &plan.soft),
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
            state: state::capacity(&self.state),
            broadphase: broadphase::capacity(&self.broadphase),
            rigid: rigid::capacity(&self.rigid),
            soft: soft::capacity(&self.soft),
        }
    }
}

impl Resources for Streams {
    fn generation(&self) -> u64 {
        self.generation
    }

    fn slots(&self, resource: ResourceId) -> u32 {
        match resource.domain() {
            state::DOMAIN => self.state.slots(resource.local()),
            broadphase::DOMAIN => self.broadphase.slots(resource.local()),
            rigid::DOMAIN => self.rigid.slots(resource.local()),
            soft::DOMAIN => self.soft.slots(resource.local()),
            domain => panic!("resource domain {domain} is outside the stream composition"),
        }
    }

    fn whole(&self, resource: ResourceId) -> GpuSlot<'_> {
        match resource.domain() {
            state::DOMAIN => self.state.whole(resource.local()),
            broadphase::DOMAIN => self.broadphase.whole(resource.local()),
            rigid::DOMAIN => self.rigid.whole(resource.local()),
            soft::DOMAIN => self.soft.whole(resource.local()),
            domain => panic!("resource domain {domain} is outside the stream composition"),
        }
    }

    fn range(&self, resource: ResourceId, offset: u64, size: u64) -> GpuSlot<'_> {
        match resource.domain() {
            state::DOMAIN => self.state.range(resource.local(), offset, size),
            broadphase::DOMAIN => self.broadphase.range(resource.local(), offset, size),
            rigid::DOMAIN => self.rigid.range(resource.local(), offset, size),
            soft::DOMAIN => self.soft.range(resource.local(), offset, size),
            domain => panic!("resource domain {domain} is outside the stream composition"),
        }
    }
}

impl Planning {
    pub(crate) const fn new() -> Self {
        Self {
            broadphase: broadphase::Capacity::new(),
            rigid: rigid::Capacity::new(),
        }
    }

    pub(crate) fn minimum() -> Plan {
        let broadphase = broadphase::Capacity::floor();
        Plan {
            state: state::floor(),
            broadphase,
            rigid: rigid::Capacity::floor(broadphase.pairs),
            soft: soft::floor(),
        }
    }

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
        let state = state::plan(&live.state, idle, &streams.state);
        let soft = soft::plan(&live.soft, idle, state.bodies, &streams.soft);
        Plan {
            state,
            broadphase,
            rigid,
            soft,
        }
    }
}
