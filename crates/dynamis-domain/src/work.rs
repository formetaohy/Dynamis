use dynamis_abi::{FrameCounts, StepParamsRecord};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostWork {
    pub body_commands: u32,
    pub constraint_commands: u32,
    pub queries: u32,
    pub shape_uploads: bool,
    pub soft_uploads: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepFacts {
    pub rigid: bool,
    pub soft: bool,
    pub indexing: bool,
    pub params: StepParamsRecord,
    pub counts: FrameCounts,
    pub work: HostWork,
}
