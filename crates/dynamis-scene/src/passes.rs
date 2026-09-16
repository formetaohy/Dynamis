use crate::cast::SceneCast;
use crate::streams::SceneStream;
use dynamis_gpu::{ComputeRecorder, GpuContext, ResourceSource};
use dynamis_pass::{Execution, PassRuntime, domain_passes};

#[derive(Clone, Copy, Debug)]
pub struct SceneFrame {
    pub query_count: u32,
}

pub struct Query {
    cast: SceneCast,
}

impl PassRuntime<SceneFrame> for Query {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            cast: SceneCast::build(
                context,
                streams,
                SceneStream::QueryRecords.whole(),
                SceneStream::QueryHits.whole(),
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &SceneFrame,
    ) {
        self.cast.record(recorder, streams, frame.query_count);
    }
}

domain_passes!(
    ScenePasses,
    SceneRuntime,
    SceneFrame,
    query: Query => Execution::GRAPH => &["broadphase", "commit"],
);
