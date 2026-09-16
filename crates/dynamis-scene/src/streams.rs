use crate::SceneDomain;
use dynamis_abi::{QueryRecord, QueryResultRecord};
use dynamis_domain::Domain;
use dynamis_domain::streams;
use dynamis_gpu::Contents;

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
