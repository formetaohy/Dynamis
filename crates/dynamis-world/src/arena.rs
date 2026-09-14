#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Run {
    pub(crate) offset: u32,
    pub(crate) len: u32,
}

impl Run {
    pub(crate) const EMPTY: Self = Self { offset: 0, len: 0 };

    pub(crate) fn span(self) -> std::ops::Range<usize> {
        self.offset as usize..(self.offset + self.len) as usize
    }
}

pub(crate) trait Cleared: Copy {
    const CLEARED: Self;
}

#[derive(Clone)]
pub(crate) struct Mirror<T> {
    storage: Vec<T>,
    arena: Arena,
    pending: Vec<Run>,
}

impl<T: Cleared> Mirror<T> {
    pub(crate) const fn new() -> Self {
        Self {
            storage: Vec::new(),
            arena: Arena::new(),
            pending: Vec::new(),
        }
    }

    pub(crate) fn used(&self) -> u32 {
        self.arena.used()
    }

    pub(crate) fn take(&mut self, len: u32) -> Run {
        if len == 0 {
            return Run::EMPTY;
        }
        let run = self.arena.take(len);
        let used = self.arena.used() as usize;
        if self.storage.len() < used {
            self.storage.resize(used, T::CLEARED);
        }
        self.pending.push(run);
        run
    }

    pub(crate) fn retire(&mut self, run: Run) {
        if run.len == 0 {
            return;
        }
        for slot in run.span() {
            self.storage[slot] = T::CLEARED;
        }
        self.arena.release(run);
        self.pending.push(run);
    }

    pub(crate) fn records(&self) -> &[T] {
        &self.storage
    }

    pub(crate) fn records_mut(&mut self) -> &mut [T] {
        &mut self.storage
    }

    pub(crate) fn flush(&mut self, mut write: impl FnMut(u32, &[T])) {
        let pending = std::mem::take(&mut self.pending);
        for run in pending {
            if run.len == 0 {
                continue;
            }
            write(run.offset, &self.storage[run.span()]);
        }
    }
}

#[derive(Clone)]
pub(crate) struct Arena {
    free: Vec<Run>,
    used: u32,
}

pub(crate) fn merged(mut runs: Vec<Run>, live: u32) -> Vec<Run> {
    runs.retain_mut(|run| {
        if run.offset >= live {
            return false;
        }
        run.len = run.len.min(live - run.offset);
        true
    });
    runs.sort_unstable_by_key(|run| run.offset);
    let mut coalesced: Vec<Run> = Vec::with_capacity(runs.len());
    for run in runs {
        match coalesced.last_mut() {
            Some(previous) if previous.offset + previous.len >= run.offset => {
                previous.len = previous.offset.max(run.offset + run.len) - previous.offset;
            }
            _ => coalesced.push(run),
        }
    }
    coalesced
}

impl Arena {
    pub(crate) const fn new() -> Self {
        Self {
            free: Vec::new(),
            used: 0,
        }
    }

    pub(crate) fn used(&self) -> u32 {
        self.used
    }

    pub(crate) fn take(&mut self, len: u32) -> Run {
        assert!(len > 0, "an arena run must hold at least one slot");
        let recycled = self
            .free
            .iter()
            .position(|run| run.len >= len)
            .map(|index| {
                let run = self.free.swap_remove(index);
                if run.len > len {
                    self.free.push(Run {
                        offset: run.offset + len,
                        len: run.len - len,
                    });
                }
                run.offset
            });
        let offset = recycled.unwrap_or_else(|| {
            let offset = self.used;
            self.used += len;
            offset
        });
        self.used = self.used.max(offset + len);
        Run { offset, len }
    }

    pub(crate) fn release(&mut self, run: Run) {
        assert!(run.len > 0, "an arena run must hold at least one slot");
        assert!(
            run.offset + run.len <= self.used,
            "an arena run must sit below the arena watermark"
        );
        let mut merged = run;
        self.free.retain(|candidate| {
            if candidate.offset + candidate.len == merged.offset {
                merged = Run {
                    offset: candidate.offset,
                    len: candidate.len + merged.len,
                };
                false
            } else if merged.offset + merged.len == candidate.offset {
                merged = Run {
                    offset: merged.offset,
                    len: merged.len + candidate.len,
                };
                false
            } else {
                true
            }
        });
        self.free.push(merged);
        self.trim();
    }

    fn trim(&mut self) {
        while let Some(index) = self
            .free
            .iter()
            .position(|run| run.offset + run.len == self.used)
        {
            let run = self.free.swap_remove(index);
            self.used = run.offset;
        }
    }
}
