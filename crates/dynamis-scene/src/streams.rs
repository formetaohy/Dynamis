use crate::SceneDomain;
use dynamis_abi::{QueryHitRecord, QueryRecord};
use dynamis_domain::Domain;
use dynamis_domain::streams;
use dynamis_gpu::Contents;

streams! {
    SceneStreams, SceneStream, SceneDemand, SceneDomain::ID, demand,
    demand {
        queries: u32,
        hits: u32,
    }
    streams {
        query_records, QueryRecords: "queries", QueryRecord, 1, Contents::Scratch, demand.queries;
        query_hits, QueryHits: "query hits", QueryHitRecord, 1, Contents::Scratch, demand.hits;
    }
}
