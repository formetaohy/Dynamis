use dynamis_sim::{StageId, STAGE_COUNT};

#[derive(Clone, Copy, Debug)]
pub struct Summary {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub p95: f64,
}

pub fn summary_of(values: &[f64]) -> Summary {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = sorted.iter().sum::<f64>() / sorted.len().max(1) as f64;
    Summary {
        mean,
        min: sorted.first().copied().unwrap_or(0.0),
        max: sorted.last().copied().unwrap_or(0.0),
        p95: sorted[(sorted.len() - 1).min((sorted.len() as f64 * 0.95) as usize).max(0)],
    }
}

#[derive(Clone, Debug)]
pub struct StageSummary {
    pub stage: StageId,
    pub gpu: Summary,
    pub share: f64,
}

#[derive(Clone, Debug)]
pub struct ProfileReport {
    pub adapter: String,
    pub body_count: usize,
    pub steps: usize,
    pub cpu_step: Summary,
    pub gpu_total: Summary,
    pub stages: Vec<StageSummary>,
}

impl ProfileReport {
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("==============================================================\n");
        out.push_str(" dynamis physics profile report\n");
        out.push_str("==============================================================\n");
        out.push_str(&format!(" adapter      : {}\n", self.adapter));
        out.push_str(&format!(" bodies       : {}\n", self.body_count));
        out.push_str(&format!(" steps        : {}\n", self.steps));
        out.push_str(&format!(
            " cpu  mean    : {} /step  (min {}, max {}, p95 {})\n",
            format_micros(self.cpu_step.mean),
            format_micros(self.cpu_step.min),
            format_micros(self.cpu_step.max),
            format_micros(self.cpu_step.p95)
        ));
        out.push_str(&format!(
            " gpu  mean    : {} /step  (min {}, max {}, p95 {})\n",
            format_micros(self.gpu_total.mean),
            format_micros(self.gpu_total.min),
            format_micros(self.gpu_total.max),
            format_micros(self.gpu_total.p95)
        ));
        out.push_str(&format!(
            " headroom     : {:.1}x of 16.6ms frame budget\n",
            16600000.0 / self.cpu_step.mean.max(1.0)
        ));
        out.push_str("--------------------------------------------------------------\n");
        out.push_str(&format!(
            " {:<20} {:>10} {:>10} {:>10} {:>10} {:>8}\n",
            "stage", "mean", "min", "max", "p95", "share"
        ));
        let mut stages = self.stages.clone();
        stages.sort_by(|a, b| b.gpu.mean.partial_cmp(&a.gpu.mean).unwrap());
        for stage in stages {
            out.push_str(&format!(
                " {:<20} {:>10} {:>10} {:>10} {:>10} {:>7.1}%\n",
                stage.stage.name(),
                format_micros(stage.gpu.mean),
                format_micros(stage.gpu.min),
                format_micros(stage.gpu.max),
                format_micros(stage.gpu.p95),
                stage.share * 100.0
            ));
        }
        out.push_str("==============================================================\n");
        out
    }
}

pub fn adapter_label(adapter: &wgpu::AdapterInfo) -> String {
    format!(
        "{} ({:?} / backend {:?})",
        adapter.name, adapter.device_type, adapter.backend
    )
}

pub fn format_micros(ns: f64) -> String {
    if ns >= 1_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else {
        format!("{:.2} us", ns / 1_000.0)
    }
}

pub const STAGE_NAMES: [&str; STAGE_COUNT] = [
    "apply_commands",
    "joint_filter",
    "integrate",
    "broadphase",
    "narrowphase",
    "contact_events",
    "islands",
    "gather",
    "solve",
    "position",
    "queries",
];
