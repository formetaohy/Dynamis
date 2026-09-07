use dynamis_sim::{StageId, STAGE_COUNT};

pub struct StageSampler {
    stage_ns: Vec<[f64; STAGE_COUNT]>,
    recorded: usize,
}

impl StageSampler {
    pub fn new() -> Self {
        Self {
            stage_ns: Vec::new(),
            recorded: 0,
        }
    }

    pub fn record(&mut self, samples: &[f64; STAGE_COUNT]) {
        self.stage_ns.push(*samples);
        self.recorded += 1;
    }

    pub fn stage_series(&self, stage: StageId) -> Vec<f64> {
        self.stage_ns
            .iter()
            .map(|frame| frame[stage as usize])
            .collect()
    }

    pub fn total_series(&self) -> Vec<f64> {
        self.stage_ns
            .iter()
            .map(|frame| frame.iter().sum())
            .collect()
    }

    pub fn recorded(&self) -> usize {
        self.recorded
    }
}

pub fn collect_stage_samples(samples: &[[f64; STAGE_COUNT]], stage: StageId) -> Vec<f64> {
    samples.iter().map(|frame| frame[stage as usize]).collect()
}
