use super::arena::{Arena, Run};
use dynamis_layout::ColliderRecord;

pub(crate) struct ColliderPool {
    arena: Arena,
    records: Vec<ColliderRecord>,
    runs: Vec<Run>,
    cleared: Vec<Run>,
}

impl ColliderPool {
    pub(crate) const fn new() -> Self {
        Self {
            arena: Arena::new(),
            records: Vec::new(),
            runs: Vec::new(),
            cleared: Vec::new(),
        }
    }

    pub(crate) fn used(&self) -> u32 {
        self.arena.used()
    }

    pub(crate) fn live(&self) -> u32 {
        self.runs.iter().map(|run| run.len).sum()
    }

    pub(crate) fn records(&self) -> &[ColliderRecord] {
        &self.records
    }

    pub(crate) fn take_cleared(&mut self) -> Vec<Run> {
        std::mem::take(&mut self.cleared)
    }

    pub(crate) fn run_of(&self, id: u32) -> Option<Run> {
        self.runs
            .get(id as usize)
            .copied()
            .filter(|run| run.len > 0)
    }

    pub(crate) fn assign(&mut self, id: u32, records: &[ColliderRecord]) {
        let len = records.len() as u32;
        assert!(len > 0, "a collider run must hold at least one collider");
        if self.runs.len() <= id as usize {
            self.runs.resize(id as usize + 1, Run::EMPTY);
        }
        if self.runs[id as usize].len != len {
            self.release(id);
            let run = self.take(len);
            self.runs[id as usize] = run;
        }
        let run = self.runs[id as usize];
        for (slot, record) in records.iter().enumerate() {
            self.records[run.offset as usize + slot] = *record;
        }
    }

    fn take(&mut self, len: u32) -> Run {
        let run = self.arena.take(len);
        self.records
            .resize(self.arena.used() as usize, ColliderRecord::cleared());
        run
    }

    pub(crate) fn release(&mut self, id: u32) {
        let Some(run) = self.run_of(id) else {
            return;
        };
        self.runs[id as usize] = Run::EMPTY;
        for index in run.span() {
            self.records[index] = ColliderRecord::cleared();
        }
        self.arena.release(run);
        self.records.truncate(self.arena.used() as usize);
        self.cleared.push(run);
    }
}
