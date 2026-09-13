use super::capacity::StreamCapacity;
use super::readback::ReadbackBuffers;
use dynamis_abi::Counters;
use dynamis_broadphase as broadphase;
use dynamis_gpu::GpuSlot;
use dynamis_pass::{ResourceId, Resources};
use dynamis_rigid as rigid;
use dynamis_rigid::{RigidDemand, RigidStreams};
use dynamis_soft as soft;
use dynamis_soft::{SoftDemand, SoftStreams};
use dynamis_state as state;
use dynamis_state::{Live, ShapeCapacity, StateDemand, StateStreams};
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
    pub(crate) state: StateDemand,
    pub(crate) broadphase: broadphase::BroadphaseDemand,
    pub(crate) rigid: RigidDemand,
    pub(crate) soft: SoftDemand,
}

pub(crate) struct Planning {
    broadphase: broadphase::Capacity,
    rigid: rigid::Capacity,
}

pub(crate) struct Streams {
    pub(crate) state: StateStreams,
    pub(crate) broadphase: broadphase::BroadphaseStreams,
    pub(crate) rigid: RigidStreams,
    pub(crate) soft: SoftStreams,
    pub(crate) readback: ReadbackBuffers,
    generation: u64,
}

impl Streams {
    pub(crate) fn new(device: &Device, queue: &Queue, plan: &Plan) -> Self {
        Self {
            state: StateStreams::new(device, queue, &plan.state),
            broadphase: broadphase::BroadphaseStreams::new(device, queue, &plan.broadphase),
            rigid: RigidStreams::new(device, queue, &plan.rigid),
            soft: SoftStreams::new(device, queue, &plan.soft),
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

    pub(crate) fn stream_capacity(&self) -> StreamCapacity {
        StreamCapacity {
            entries: broadphase::entry_capacity(self),
            pairs: broadphase::pair_capacity(self),
            events: rigid::event_capacity(self),
            shapes: ShapeCapacity {
                sources: self.state.shape_sources.slots(),
                vertices: self.state.shape_vertices.slots(),
                triangles: self.state.shape_triangles.slots(),
                nodes: self.state.shape_nodes.slots(),
            },
            soft: soft::capacity(self),
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
        let (broadphase, idle) = self.broadphase.plan(measured, live, &streams.broadphase);
        let rigid = self
            .rigid
            .plan(measured, live, idle, broadphase.pairs, &streams.rigid);
        let state = state::plan(live, idle, &streams.state);
        let soft = soft::plan(live, idle, state.bodies, &streams.soft);
        Plan {
            state,
            broadphase,
            rigid,
            soft,
        }
    }
}
