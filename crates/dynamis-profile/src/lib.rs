use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq)]
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
    let p95_index = ((sorted.len() as f64 * 0.95) as usize).min(sorted.len() - 1);
    Summary {
        mean,
        min: sorted.first().copied().unwrap_or(0.0),
        max: sorted.last().copied().unwrap_or(0.0),
        p95: sorted.get(p95_index).copied().unwrap_or(0.0),
    }
}

pub fn format_ns(ns: f64) -> String {
    if ns >= 1_000_000_000.0 {
        format!("{:.3} s", ns / 1_000_000_000.0)
    } else if ns >= 1_000_000.0 {
        format!("{:.3} ms", ns / 1_000_000.0)
    } else if ns >= 1_000.0 {
        format!("{:.2} us", ns / 1_000.0)
    } else {
        format!("{:.1} ns", ns)
    }
}

pub struct PhaseTimer {
    name: &'static str,
    samples: Vec<f64>,
}

impl PhaseTimer {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            samples: Vec::new(),
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn samples(&self) -> &[f64] {
        &self.samples
    }

    pub fn count(&self) -> usize {
        self.samples.len()
    }

    pub fn summary(&self) -> Summary {
        summary_of(&self.samples)
    }

    pub fn total(&self) -> f64 {
        self.samples.iter().sum()
    }
}

pub struct Profiler {
    phases: Vec<PhaseTimer>,
}

impl Profiler {
    pub fn new() -> Self {
        Self { phases: Vec::new() }
    }

    pub fn phase(&self, name: &'static str) -> Option<&PhaseTimer> {
        self.phases.iter().find(|phase| phase.name == name)
    }

    pub fn phases(&self) -> &[PhaseTimer] {
        &self.phases
    }

    pub fn measure<R>(&mut self, name: &'static str, call: impl FnOnce() -> R) -> R {
        let index = self.index_or_push(name);
        let begin = Instant::now();
        let result = call();
        let elapsed = begin.elapsed().as_nanos() as f64;
        self.phases[index].samples.push(elapsed);
        result
    }

    pub fn measure_with<R>(
        &mut self,
        name: &'static str,
        call: impl FnOnce() -> R,
    ) -> (R, Duration) {
        let index = self.index_or_push(name);
        let begin = Instant::now();
        let result = call();
        let elapsed = begin.elapsed();
        self.phases[index].samples.push(elapsed.as_nanos() as f64);
        (result, elapsed)
    }

    fn index_or_push(&mut self, name: &'static str) -> usize {
        if let Some(index) = self.phases.iter().position(|phase| phase.name == name) {
            return index;
        }
        self.phases.push(PhaseTimer::new(name));
        self.phases.len() - 1
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("==============================================================\n");
        out.push_str(" dynamis profile report\n");
        out.push_str("==============================================================\n");
        out.push_str(&format!(
            " {:<18} {:>7} {:>11} {:>11} {:>11} {:>11}\n",
            "phase", "samples", "mean", "min", "max", "p95"
        ));
        for phase in &self.phases {
            let summary = phase.summary();
            out.push_str(&format!(
                " {:<18} {:>7} {:>11} {:>11} {:>11} {:>11}\n",
                phase.name(),
                phase.count(),
                format_ns(summary.mean),
                format_ns(summary.min),
                format_ns(summary.max),
                format_ns(summary.p95),
            ));
        }
        out.push_str("==============================================================\n");
        out
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}
