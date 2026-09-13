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

pub(crate) struct Arena {
    free: Vec<Run>,
    used: u32,
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
