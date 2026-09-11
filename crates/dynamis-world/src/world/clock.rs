pub(crate) struct Clock {
    pub(crate) step: u64,
    pub(crate) accumulator: f32,
    pub(crate) time_scale: f32,
    pub(crate) sub_dt: f32,
}

impl Clock {
    pub(crate) const DEFAULT_SUB_DT: f32 = 1.0 / 60.0;

    pub(crate) fn new() -> Self {
        Self {
            step: 0,
            accumulator: 0.0,
            time_scale: 1.0,
            sub_dt: Self::DEFAULT_SUB_DT,
        }
    }
}
