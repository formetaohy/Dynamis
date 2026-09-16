use super::arena::{Arena, Run, merged};
use super::body::BodyStore;
use dynamis_abi::{ColliderRecord, ENTRY_INDEX_MASK};
use dynamis_model::BodyHandle;

#[derive(Clone)]
pub(crate) struct ColliderStore {
    arena: Arena,
    records: Vec<ColliderRecord>,
    runs: Vec<Run>,
    cleared: Vec<Run>,
    live: u32,
    armed: u32,
}

impl ColliderStore {
    pub(crate) const fn new() -> Self {
        Self {
            arena: Arena::new(),
            records: Vec::new(),
            runs: Vec::new(),
            cleared: Vec::new(),
            live: 0,
            armed: 0,
        }
    }

    pub(crate) fn used(&self) -> u32 {
        self.arena.used()
    }

    pub(crate) fn live(&self) -> u32 {
        self.live
    }

    pub(crate) fn impact_armed(&self) -> u32 {
        self.armed
    }

    pub(crate) fn records(&self) -> &[ColliderRecord] {
        &self.records
    }

    pub(crate) fn take_cleared(&mut self) -> Vec<Run> {
        merged(std::mem::take(&mut self.cleared), self.arena.used())
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
        let span = run.span();
        self.armed -= armed(&self.records[span.clone()]);
        for (slot, record) in records.iter().enumerate() {
            self.records[run.offset as usize + slot] = *record;
        }
        self.armed += armed(records);
    }

    fn take(&mut self, len: u32) -> Run {
        let run = self.arena.take(len);
        assert!(
            self.arena.used() <= ENTRY_INDEX_MASK,
            "a collider slot must fit a grid entry index"
        );
        self.records
            .resize(self.arena.used() as usize, ColliderRecord::cleared());
        self.live += len;
        run
    }

    pub(crate) fn release(&mut self, id: u32) {
        let Some(run) = self.run_of(id) else {
            return;
        };
        let released = armed(&self.records[run.span()]);
        self.armed -= released;
        self.runs[id as usize] = Run::EMPTY;
        for index in run.span() {
            self.records[index] = ColliderRecord::cleared();
        }
        self.arena.release(run);
        self.records.truncate(self.arena.used() as usize);
        self.cleared.push(run);
        self.live -= run.len;
    }
}

pub(crate) fn local_collider_of(
    bodies: &BodyStore,
    pool: &ColliderStore,
    body: BodyHandle,
    slot: u32,
) -> u32 {
    if bodies.handle_of(body.id) != Some(body) {
        return slot;
    }
    let run = pool.run_of(body.id).unwrap_or(Run::EMPTY);
    slot.checked_sub(run.offset).unwrap_or_else(|| {
        panic!("body {body:?} cannot hold the collider a query hit reported at slot {slot}")
    })
}

fn armed(records: &[ColliderRecord]) -> u32 {
    records
        .iter()
        .filter(|record| record.impact_force.is_finite())
        .count() as u32
}
