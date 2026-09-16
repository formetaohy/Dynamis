use crate::SceneDomain;
use dynamis_abi::{QueryHitRecord, QueryRecord};
use dynamis_domain::Domain;
use dynamis_domain::streams;
use dynamis_gpu::Retention;

streams! {
    SceneStreams, SceneStream, SceneDemand, SceneDomain::ID, demand,
    demand {
        queries: u32,
        hits: u32,
    }
    streams {
        query_records, QueryRecords: "queries", QueryRecord, 1, Retention::Scratch, demand.queries;
        query_hits, QueryHits: "query hits", QueryHitRecord, 1, Retention::Scratch, demand.hits;
    }
}
