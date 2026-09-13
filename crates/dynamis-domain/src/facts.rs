use dynamis_abi::{FrameCounts, StepParamsRecord};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepFacts {
    pub params: StepParamsRecord,
    pub counts: FrameCounts,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub awake: bool,
    pub indexing: bool,
}
