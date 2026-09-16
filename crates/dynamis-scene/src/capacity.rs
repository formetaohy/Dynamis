use crate::{SceneDemand, SceneStreams};
use dynamis_domain::{STREAM_FLOOR, settled};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SceneCapacity {
    pub queries: u32,
    pub hits: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct SceneInputs {
    pub queries: u32,
    pub hits: u32,
}

pub fn capacity(streams: &SceneStreams) -> SceneCapacity {
    SceneCapacity {
        queries: streams.query_records.slots(),
        hits: streams.query_hits.slots(),
    }
}

pub fn floor() -> SceneDemand {
    SceneDemand {
        queries: STREAM_FLOOR,
        hits: STREAM_FLOOR,
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
        hits: settled(
            current.query_hits.slots(),
            inputs.hits,
            STREAM_FLOOR,
            release,
        ),
    }
}
