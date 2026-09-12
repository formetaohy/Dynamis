use super::capacity::{Live, ShapeCapacity, SoftCapacity, StreamCapacity};
use super::engine::{ResourceId, Resources, SlotRef};
use super::readback::ReadbackBuffers;
use super::rigid::{self, RigidDemand, RigidStream, RigidStreams};
use super::scene::{self, SceneDemand, SceneStreams};
use super::soft::{self, SoftDemand, SoftStreams};
use dynamis_gpu::GpuSlot;
use dynamis_layout::Counters;
use dynamis_sort::SortChannels;
use wgpu::{CommandEncoder, Device, Queue};

pub(crate) struct Plan {
    pub(crate) scene: SceneDemand,
    pub(crate) rigid: RigidDemand,
    pub(crate) soft: SoftDemand,
}

pub(crate) struct Planning {
    rigid: rigid::Capacity,
}

pub(crate) struct Streams {
    pub(crate) scene: SceneStreams,
    pub(crate) rigid: RigidStreams,
    pub(crate) soft: SoftStreams,
    pub(crate) readback: ReadbackBuffers,
    generation: u64,
}

impl Streams {
    pub(crate) fn new(device: &Device, queue: &Queue, plan: &Plan) -> Self {
        Self {
            scene: SceneStreams::new(device, queue, &plan.scene),
            rigid: RigidStreams::new(device, queue, &plan.rigid),
            soft: SoftStreams::new(device, queue, &plan.soft),
            readback: ReadbackBuffers::new(device, plan),
            generation: 0,
        }
    }

    pub(crate) fn matches(&self, plan: &Plan) -> bool {
        self.scene.matches(&plan.scene)
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
        let changed = self.scene.reserve(device, encoder, &plan.scene)
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
            entries: self.rigid.entry_capacity(),
            pairs: self.rigid.pair_capacity(),
            events: self.rigid.event_capacity(),
            shapes: ShapeCapacity {
                sources: self.scene.shape_sources.slots(),
                vertices: self.scene.shape_vertices.slots(),
                triangles: self.scene.shape_triangles.slots(),
                nodes: self.scene.shape_nodes.slots(),
            },
            soft: SoftCapacity {
                particles: self.soft.particles.slots(),
                links: self.soft.links.slots(),
                adjacency: self.soft.adjacency.slots(),
            },
        }
    }

    pub(crate) fn sort_capacity(&self) -> u32 {
        self.rigid.sort_capacity()
    }

    fn scratch(&self) -> [GpuSlot<'_>; 3] {
        [
            RigidStream::SortScratchMajor.whole().resolve(self),
            RigidStream::SortScratchMinor.whole().resolve(self),
            RigidStream::SortScratchPayload.whole().resolve(self),
        ]
    }

    pub(crate) fn sort_lanes<'a>(
        &'a self,
        count: SlotRef,
        major: SlotRef,
        payload: SlotRef,
    ) -> SortChannels<'a> {
        let [scratch_major, scratch_minor, scratch_payload] = self.scratch();
        SortChannels {
            generation: self.generation,
            count: count.resolve(self),
            major: major.resolve(self),
            minor: payload.resolve(self),
            payload: payload.resolve(self),
            scratch_major,
            scratch_minor,
            scratch_payload,
        }
    }

    pub(crate) fn sort_keyed<'a>(
        &'a self,
        count: SlotRef,
        major: SlotRef,
        minor: SlotRef,
        payload: SlotRef,
    ) -> SortChannels<'a> {
        let [scratch_major, scratch_minor, scratch_payload] = self.scratch();
        SortChannels {
            generation: self.generation,
            count: count.resolve(self),
            major: major.resolve(self),
            minor: minor.resolve(self),
            payload: payload.resolve(self),
            scratch_major,
            scratch_minor,
            scratch_payload,
        }
    }

    pub(crate) fn sort_lanes_dual<'a>(
        &'a self,
        count: SlotRef,
        major: SlotRef,
        minor: SlotRef,
    ) -> SortChannels<'a> {
        let [scratch_major, scratch_minor, scratch_payload] = self.scratch();
        SortChannels {
            generation: self.generation,
            count: count.resolve(self),
            major: major.resolve(self),
            minor: minor.resolve(self),
            payload: RigidStream::SortDummyPayload.whole().resolve(self),
            scratch_major,
            scratch_minor,
            scratch_payload,
        }
    }
}

impl Resources for Streams {
    fn generation(&self) -> u64 {
        self.generation
    }

    fn slots(&self, resource: ResourceId) -> u32 {
        match resource.domain() {
            scene::DOMAIN => self.scene.slots(resource.local()),
            rigid::DOMAIN => self.rigid.slots(resource.local()),
            soft::DOMAIN => self.soft.slots(resource.local()),
            domain => panic!("resource domain {domain} is outside the stream composition"),
        }
    }

    fn whole(&self, resource: ResourceId) -> GpuSlot<'_> {
        match resource.domain() {
            scene::DOMAIN => self.scene.whole(resource.local()),
            rigid::DOMAIN => self.rigid.whole(resource.local()),
            soft::DOMAIN => self.soft.whole(resource.local()),
            domain => panic!("resource domain {domain} is outside the stream composition"),
        }
    }

    fn range(&self, resource: ResourceId, offset: u64, size: u64) -> GpuSlot<'_> {
        match resource.domain() {
            scene::DOMAIN => self.scene.range(resource.local(), offset, size),
            rigid::DOMAIN => self.rigid.range(resource.local(), offset, size),
            soft::DOMAIN => self.soft.range(resource.local(), offset, size),
            domain => panic!("resource domain {domain} is outside the stream composition"),
        }
    }
}

impl Planning {
    pub(crate) const fn new() -> Self {
        Self {
            rigid: rigid::Capacity::new(),
        }
    }

    pub(crate) fn minimum() -> Plan {
        Plan {
            scene: scene::floor(),
            rigid: rigid::Capacity::floor(),
            soft: soft::floor(),
        }
    }

    pub(crate) fn plan(&mut self, measured: &Counters, live: &Live, streams: &Streams) -> Plan {
        let (rigid, idle) = self.rigid.plan(measured, live, &streams.rigid);
        let scene = scene::demand(live, idle, &streams.scene);
        let soft = soft::plan(live, idle, scene.bodies, &streams.soft);
        Plan { scene, rigid, soft }
    }
}
