use dynamis_abi::{FrameCounts, RowStreams, StepParamsRecord, Subscriptions};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepFacts {
    pub params: StepParamsRecord,
    pub counts: FrameCounts,
    pub subscriptions: Subscriptions,
    pub rows: RowStreams,
}
