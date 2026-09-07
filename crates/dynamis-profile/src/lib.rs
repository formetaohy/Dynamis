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

/// External, engine-agnostic profiler that times API calls from the caller side.
///
/// The physics engine itself is never instrumented; every measurement is taken
/// around calls the application makes on its own objects (e.g. `step`, `wait`).
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

    /// Runs `call` while measuring it, appending the elapsed time to `name`.
    pub fn measure<R>(&mut self, name: &'static str, call: impl FnOnce() -> R) -> R {
        let index = self.index_or_push(name);
        let begin = Instant::now();
        let result = call();
        let elapsed = begin.elapsed().as_nanos() as f64;
        self.phases[index].samples.push(elapsed);
        result
    }

    /// Runs `call` while measuring it and returns its value together with the elapsed time.
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

    /// Renders a report of all measured phases.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_aggregates_statistics() {
        let summary = summary_of(&[100.0, 200.0, 300.0, 400.0, 500.0]);
        assert_eq!(summary.mean, 300.0);
        assert_eq!(summary.min, 100.0);
        assert_eq!(summary.max, 500.0);
        assert_eq!(summary.p95, 500.0);
    }

    #[test]
    fn summary_handles_single_sample() {
        let summary = summary_of(&[42.0]);
        assert_eq!(summary.mean, 42.0);
        assert_eq!(summary.min, 42.0);
        assert_eq!(summary.max, 42.0);
        assert_eq!(summary.p95, 42.0);
    }

    #[test]
    fn profiler_groups_samples_by_phase() {
        let mut profiler = Profiler::new();
        profiler.measure("work", || 1 + 1);
        profiler.measure("work", || 2 + 2);
        profiler.measure("other", || 0);
        let work = profiler.phase("work").expect("work phase exists");
        assert_eq!(work.count(), 2);
        assert_eq!(
            profiler.phase("other").expect("other phase exists").count(),
            1
        );
        assert_eq!(profiler.phases().len(), 2);
    }

    #[test]
    fn measure_with_returns_value_and_duration() {
        let mut profiler = Profiler::new();
        let (value, elapsed) = profiler.measure_with("op", || 7);
        assert_eq!(value, 7);
        assert!(elapsed >= Duration::ZERO);
        let op = profiler.phase("op").expect("op phase exists");
        assert_eq!(op.count(), 1);
    }
}
