use crate::{Pass, Pipeline, Run};
#[cfg(feature = "profile")]
use dynamis_gpu::SubmissionEncoder;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use wgpu::CommandEncoder;

const STEP_SCOPE: &str = "dynamis step scope";

pub struct Schedule {
    pipeline: Pipeline,
    per_row: u32,
    opened: Vec<bool>,
    completed: Vec<bool>,
    graph: bool,
    #[cfg(feature = "profile")]
    timer: Option<dynamis_gpu::GpuTimer>,
    timed: bool,
}

impl Schedule {
    pub fn new(
        context: &GpuContext,
        pipeline: Pipeline,
        #[cfg(feature = "profile")] label: &str,
    ) -> Self {
        assert!(
            !pipeline.is_empty(),
            "a step schedule needs at least one pass"
        );
        #[cfg(feature = "profile")]
        let labels = pipeline
            .passes()
            .iter()
            .map(|pass| pass.label)
            .collect::<Vec<_>>();
        Self {
            per_row: context.workgroups_per_row(),
            opened: vec![false; pipeline.len()],
            completed: vec![false; pipeline.len()],
            graph: false,
            pipeline,
            #[cfg(feature = "profile")]
            timer: context.supports_pass_timing().then(|| {
                dynamis_gpu::GpuTimer::new(
                    context.device(),
                    &labels,
                    context.timestamp_period_ns(),
                    label,
                )
            }),
            timed: true,
        }
    }

    pub const fn per_row(&self) -> u32 {
        self.per_row
    }

    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    pub fn ran(&self, index: u32) -> bool {
        self.opened.get(index as usize).copied().unwrap_or(false)
    }

    pub fn ran_labels(&self) -> Vec<&'static str> {
        self.pipeline
            .passes()
            .iter()
            .enumerate()
            .filter(|(index, _)| self.completed[*index])
            .map(|(_, pass)| pass.label)
            .collect()
    }

    pub fn begin(&mut self, run: Run) {
        self.reset(run.graphs(), run.timed());
    }

    fn reset(&mut self, graph: bool, timed: bool) {
        self.opened.fill(false);
        self.graph = graph;
        self.timed = timed;
    }

    pub fn finish(&mut self) {
        if self.graph {
            self.completed.copy_from_slice(&self.opened);
        }
    }

    pub fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        passes: impl FnMut(Pass, u32, &mut ComputeRecorder<'_>) -> bool,
    ) {
        #[cfg(feature = "profile")]
        if self.timed && self.timer.is_some() {
            self.record_per_pass_scopes(encoder, passes);
            return;
        }
        self.record_one_scope(encoder, passes);
    }

    fn record_one_scope(
        &mut self,
        encoder: &mut CommandEncoder,
        mut passes: impl FnMut(Pass, u32, &mut ComputeRecorder<'_>) -> bool,
    ) {
        let mut recorder = ComputeRecorder::begin(&mut *encoder, STEP_SCOPE, self.per_row);
        for index in 0..self.pipeline.len() as u32 {
            let declared = self.pipeline.pass(index);
            self.opened[index as usize] = passes(declared, index, &mut recorder);
        }
    }

    #[cfg(feature = "profile")]
    fn record_per_pass_scopes(
        &mut self,
        encoder: &mut CommandEncoder,
        mut passes: impl FnMut(Pass, u32, &mut ComputeRecorder<'_>) -> bool,
    ) {
        let per_row = self.per_row;
        for index in 0..self.pipeline.len() as u32 {
            let declared = self.pipeline.pass(index);
            let timer = self.timer.as_ref().expect("a timed frame owns its timer");
            let mut recorder = ComputeRecorder::begin_timed(
                &mut *encoder,
                declared.label,
                Some(timer.writes(index as usize)),
                per_row,
            );
            self.opened[index as usize] = passes(declared, index, &mut recorder);
        }
    }

    #[cfg(feature = "profile")]
    pub fn capture_timings(
        &mut self,
        encoder: &mut SubmissionEncoder,
    ) -> Option<Vec<dynamis_gpu::GpuPassTiming>> {
        if !self.timed {
            return None;
        }
        let ran = self.opened.clone();
        self.timer
            .as_mut()
            .and_then(|timer| timer.capture(encoder, &ran))
    }

    #[cfg(feature = "profile")]
    pub fn collect_timings(&mut self) -> Vec<Vec<dynamis_gpu::GpuPassTiming>> {
        match &mut self.timer {
            Some(timer) => timer.collect(),
            None => Vec::new(),
        }
    }
}
