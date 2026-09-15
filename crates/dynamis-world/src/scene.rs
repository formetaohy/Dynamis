use dynamis_abi::{
    Counters, ENTRY_INDEX_MASK, ENTRY_KIND_COLLIDER, ENTRY_KIND_PARTICLE, ENTRY_KIND_SHIFT,
    QueryRecord, QueryResultRecord, StepParamsRecord,
};
use dynamis_broadphase::BroadphaseStream;
use dynamis_domain::streams;
use dynamis_domain::{Domain, STREAM_FLOOR, StepFacts, settled};
use dynamis_gpu::{ComputeRecorder, Contents, GpuContext, Resources};
use dynamis_model::{BodyHandle, SceneTarget, SoftBodyHandle};
use dynamis_pass::{Execution, PassGroup, PassRuntime, Pipeline, Stage, domain_passes};
use dynamis_shader::{GEOMETRY_INDEX, workgroups};
use dynamis_soft::SoftStream;
use dynamis_state::StateStream;

pub(crate) fn scene_target(
    packed: u32,
    id: u32,
    generation: u32,
    collider_of: impl Fn(BodyHandle, u32) -> u32,
    particle_of: impl Fn(SoftBodyHandle, u32) -> u32,
) -> SceneTarget {
    let slot = packed & ENTRY_INDEX_MASK;
    match packed >> ENTRY_KIND_SHIFT {
        ENTRY_KIND_COLLIDER => {
            let body = BodyHandle { id, generation };
            SceneTarget::Collider {
                body,
                collider: collider_of(body, slot),
            }
        }
        ENTRY_KIND_PARTICLE => {
            let body = SoftBodyHandle { id, generation };
            SceneTarget::Particle {
                body,
                particle: particle_of(body, slot),
            }
        }
        kind => panic!("a scene target reports an unknown kind {kind}"),
    }
}

streams! {
    SceneStreams, SceneStream, SceneDemand, SceneDomain::ID, demand,
    demand {
        queries: u32,
    }
    streams {
        query_records, QueryRecords: "queries", QueryRecord, 1, Contents::Scratch, demand.queries;
        query_results, QueryResults: "query results", QueryResultRecord, 1, Contents::Scratch, demand.queries;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SceneCapacity {
    pub queries: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct SceneInputs {
    pub queries: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SceneWork {
    pub queries: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct SceneFrame {
    pub params: StepParamsRecord,
    pub query_count: u32,
}

pub fn capacity(streams: &SceneStreams) -> SceneCapacity {
    SceneCapacity {
        queries: streams.query_records.slots(),
    }
}

pub fn floor() -> SceneDemand {
    SceneDemand {
        queries: STREAM_FLOOR,
    }
}

pub fn plan(inputs: &SceneInputs, current: &SceneStreams, release: bool) -> SceneDemand {
    SceneDemand {
        queries: settled(
            current.query_records.slots(),
            inputs.queries,
            STREAM_FLOOR,
            release,
        ),
    }
}

pub struct Query {
    kernel: Stage,
}

impl PassRuntime<SceneFrame> for Query {
    fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            kernel: Stage::build(
                context,
                "query",
                workgroups(context, dynamis_shader::SCENE_CAST, GEOMETRY_INDEX),
                streams,
                &[
                    ("queries", SceneStream::QueryRecords.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("colliders", StateStream::Colliders.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", StateStream::Counters.whole()),
                    ("query_results", SceneStream::QueryResults.whole()),
                    ("params", StateStream::Params.whole()),
                    ("collider_owners", StateStream::ColliderOwners.whole()),
                    ("particles", SoftStream::Particles.whole()),
                    ("soft_bodies", SoftStream::BodyStates.whole()),
                ],
                &dynamis_state::shape_resources(),
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
        frame: &SceneFrame,
    ) {
        self.kernel
            .record_workgroups(recorder, streams, frame.query_count);
    }
}

domain_passes!(
    ScenePasses,
    SceneRuntime,
    SceneFrame,
    query: Query => Execution::GRAPH => &["broadphase", "commit"],
);

pub struct SceneDomain;

impl Domain for SceneDomain {
    const ID: u32 = 4;

    const SIMULATES: bool = false;

    const PASS_EDGES: dynamis_pass::PassEdges = &[ScenePasses::EDGES];

    type Demand = SceneDemand;
    type Inputs = SceneInputs;
    type Work = SceneWork;
    type Streams = SceneStreams;
    type Passes = ScenePasses;
    type Runtime = SceneRuntime;
    type Frame = SceneFrame;
    type Capacity = SceneCapacity;

    fn minimum() -> SceneDemand {
        floor()
    }

    fn occupied(inputs: &SceneInputs) -> bool {
        inputs.queries > 0
    }

    fn pending(work: &SceneWork) -> bool {
        work.queries > 0
    }

    fn active(_: &Counters) -> bool {
        false
    }

    fn pass_groups() -> &'static [PassGroup] {
        &[ScenePasses::GROUP]
    }

    fn resolve(pipeline: &Pipeline) -> ScenePasses {
        ScenePasses::resolve(pipeline)
    }

    fn build(context: &GpuContext, streams: &impl Resources, passes: ScenePasses) -> SceneRuntime {
        SceneRuntime::build(context, streams, passes)
    }

    fn gates(_: &SceneFrame) -> u16 {
        0
    }

    fn frame(facts: &StepFacts, inputs: &SceneInputs) -> SceneFrame {
        SceneFrame {
            params: facts.params,
            query_count: inputs.queries,
        }
    }

    fn capacity(streams: &SceneStreams) -> SceneCapacity {
        capacity(streams)
    }

    fn record(
        runtime: &mut SceneRuntime,
        pass: u32,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
        frame: &SceneFrame,
    ) -> bool {
        runtime.record(pass, recorder, streams, frame)
    }
}
