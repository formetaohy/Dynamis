use dynamis_abi::{Census, RowStreams, StepParamsRecord};
use dynamis_model::PhysicsConfig;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepFacts {
    pub params: StepParamsRecord,
    pub census: Census,
    pub rows: RowStreams,
}

impl StepFacts {
    pub fn of(
        config: &PhysicsConfig,
        dt: f32,
        census: Census,
        rows: RowStreams,
        wake_all: bool,
    ) -> Self {
        Self {
            params: StepParamsRecord::new(config, dt, census, wake_all),
            census,
            rows,
        }
    }
}
