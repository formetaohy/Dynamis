use dynamis_layout::{ColliderRecord, NO_BODY};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ColliderRun {
    pub(crate) offset: u32,
    pub(crate) len: u32,
}

pub(crate) struct ColliderPool {
    records: Vec<ColliderRecord>,
    owners: Vec<u32>,
    runs: Vec<ColliderRun>,
    free: Vec<ColliderRun>,
    cleared: Vec<ColliderRun>,
    used: u32,
}

impl ColliderPool {
    pub(crate) const fn new() -> Self {
        Self {
            records: Vec::new(),
            owners: Vec::new(),
            runs: Vec::new(),
            free: Vec::new(),
            cleared: Vec::new(),
            used: 0,
        }
    }

    pub(crate) fn used(&self) -> u32 {
        self.used
    }

    pub(crate) fn live(&self) -> u32 {
        self.runs.iter().map(|run| run.len).sum()
    }

    pub(crate) fn records(&self) -> &[ColliderRecord] {
        &self.records
    }

    pub(crate) fn owners(&self) -> &[u32] {
        &self.owners
    }

    pub(crate) fn take_cleared(&mut self) -> Vec<ColliderRun> {
        std::mem::take(&mut self.cleared)
    }

    pub(crate) fn run_of(&self, id: u32) -> Option<ColliderRun> {
        self.runs
            .get(id as usize)
            .copied()
            .filter(|run| run.len > 0)
    }

    fn grow(&mut self, rows: u32) {
        if self.records.len() >= rows as usize {
            return;
        }
        self.records
            .resize(rows as usize, ColliderRecord::cleared());
        self.owners.resize(rows as usize, NO_BODY);
    }

    fn take(&mut self, len: u32) -> u32 {
        let placed = self
            .free
            .iter()
            .position(|block| block.len >= len)
            .map(|index| {
                let block = self.free.swap_remove(index);
                if block.len > len {
                    self.free.push(ColliderRun {
                        offset: block.offset + len,
                        len: block.len - len,
                    });
                }
                block.offset
            });
        match placed {
            Some(offset) => offset,
            None => {
                let offset = self.used;
                self.used += len;
                self.grow(self.used);
                offset
            }
        }
    }

    pub(crate) fn assign(&mut self, id: u32, records: &[ColliderRecord]) {
        let len = records.len() as u32;
        assert!(len > 0, "a collider run must hold at least one collider");
        if self.runs.len() <= id as usize {
            self.runs
                .resize(id as usize + 1, ColliderRun { offset: 0, len: 0 });
        }
        if self.runs[id as usize].len != len {
            self.release(id);
            let offset = self.take(len);
            self.runs[id as usize] = ColliderRun { offset, len };
        }
        let run = self.runs[id as usize];
        assert!(
            (run.offset + run.len) as usize <= self.records.len(),
            "collider run escapes the pool"
        );
        for slot in 0..len {
            let index = (run.offset + slot) as usize;
            self.records[index] = records[slot as usize];
            self.owners[index] = id;
        }
    }

    pub(crate) fn release(&mut self, id: u32) {
        let Some(run) = self.run_of(id) else {
            return;
        };
        self.runs[id as usize] = ColliderRun { offset: 0, len: 0 };
        for index in run.offset..run.offset + run.len {
            let index = index as usize;
            self.records[index] = ColliderRecord::cleared();
            self.owners[index] = NO_BODY;
        }
        self.cleared.push(run);
        self.coalesce(run);
    }

    fn coalesce(&mut self, released: ColliderRun) {
        let mut block = released;
        self.free.retain(|candidate| {
            if candidate.offset + candidate.len == block.offset {
                block = ColliderRun {
                    offset: candidate.offset,
                    len: candidate.len + block.len,
                };
                false
            } else if block.offset + block.len == candidate.offset {
                block = ColliderRun {
                    offset: block.offset,
                    len: block.len + candidate.len,
                };
                false
            } else {
                true
            }
        });
        self.free.push(block);
    }
}
