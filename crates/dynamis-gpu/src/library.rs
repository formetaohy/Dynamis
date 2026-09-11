use crate::pipeline::{ComputeLayout, ComputeProgram, PipelineHandle};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use wgpu::Device;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarmupBudget {
    All,
    Within(Duration),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WarmupProgress {
    pub ready: usize,
    pub total: usize,
}

impl WarmupProgress {
    pub fn complete(&self) -> bool {
        self.ready == self.total
    }
}

pub(crate) struct PipelineLibrary {
    entries: HashMap<Arc<ComputeProgram>, PipelineHandle>,
    pending: VecDeque<PipelineHandle>,
}

impl PipelineLibrary {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            pending: VecDeque::new(),
        }
    }

    pub(crate) fn declare(&mut self, device: &Device, program: ComputeProgram) -> PipelineHandle {
        let program = Arc::new(program);
        if let Some(handle) = self.entries.get(&program) {
            return handle.clone();
        }
        let handle = PipelineHandle::new(program.clone(), ComputeLayout::new(device, &program));
        self.entries.insert(program, handle.clone());
        self.pending.push_back(handle.clone());
        handle
    }

    pub(crate) fn drain_pending(&mut self) -> Vec<PipelineHandle> {
        self.pending.drain(..).collect()
    }

    pub(crate) fn take_pending(&mut self) -> Option<PipelineHandle> {
        self.pending.pop_front()
    }

    pub(crate) fn progress(&self) -> WarmupProgress {
        WarmupProgress {
            ready: self
                .entries
                .values()
                .filter(|handle| handle.is_warmed())
                .count(),
            total: self.entries.len(),
        }
    }
}
