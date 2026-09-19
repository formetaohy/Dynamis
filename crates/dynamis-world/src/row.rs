use super::pool::Identity;
use dynamis_abi::RowMoveRecord;
use std::collections::HashMap;

const NO_ROW: u32 = u32::MAX;

/// The rows a store holds on the device, the state each of them answers, and the rows whose
/// occupant the store has moved since the device last held them. The moves the device owes are the
/// difference between the rows at hand and the rows the device holds at the rows the store moved,
/// so a row that already answers the identity at hand is no difference and the store does not
/// answer for it again: a run that repeated a row move would undo a swap and answer a removal with
/// a row that no longer holds the state it moved.
#[derive(Clone)]
pub(crate) struct RowLayout<H: Identity> {
    held: Vec<H>,
    held_of: Vec<u32>,
    moved: Vec<u32>,
}

impl<H: Identity> RowLayout<H> {
    pub(crate) const fn new() -> Self {
        Self {
            held: Vec::new(),
            held_of: Vec::new(),
            moved: Vec::new(),
        }
    }

    /// The rows the device holds, which the streams they live in must keep readable while the
    /// difference that reaches the rows at hand is outstanding.
    pub(crate) fn rows(&self) -> u32 {
        self.held.len() as u32
    }

    /// Whether the device holds the rows at hand: every row whose occupant the store moved owes a
    /// move until the handover that carries the moves records where each of them landed.
    pub(crate) fn settled(&self) -> bool {
        self.moved.is_empty()
    }

    /// Declares that `row` answers another identity than it did when the device last held the rows.
    pub(crate) fn touch(&mut self, row: u32) {
        self.moved.push(row);
    }

    /// Records that the device holds the rows at hand at every row the handover that composed the
    /// moves carried, which is where they were moved to.
    pub(crate) fn settle(&mut self, rows: &[H]) {
        let moved = std::mem::take(&mut self.moved);
        for row in moved {
            let Some(handle) = rows.get(row as usize).copied() else {
                continue;
            };
            if self.held.len() <= row as usize {
                self.held.resize(row as usize + 1, handle);
            }
            self.held[row as usize] = handle;
            let id = handle.id() as usize;
            if self.held_of.len() <= id {
                self.held_of.resize(id + 1, NO_ROW);
            }
            self.held_of[id] = row;
        }
        self.held.truncate(rows.len());
    }

    /// The moves that reach `rows` from the rows the device holds, and, for every row whose state
    /// the device does not hold, the record `seed` answers for it.
    pub(crate) fn difference<T>(
        &self,
        rows: &[H],
        mut seed: impl FnMut(H) -> T,
    ) -> (Vec<RowMoveRecord>, Vec<T>) {
        let mut moved = self.moved.clone();
        moved.sort_unstable();
        moved.dedup();
        let mut moves = Vec::new();
        let mut fresh = Vec::new();
        for row in moved {
            let Some(handle) = rows.get(row as usize).copied() else {
                continue;
            };
            if self.held.get(row as usize) == Some(&handle) {
                continue;
            }
            let source = self
                .held_of
                .get(handle.id() as usize)
                .copied()
                .filter(|source| self.held.get(*source as usize) == Some(&handle));
            let record = match source {
                Some(source) => RowMoveRecord::source(row, source),
                None => {
                    fresh.push(seed(handle));
                    RowMoveRecord::fresh(row, fresh.len() as u32 - 1)
                }
            };
            moves.push(record);
        }
        (moves, fresh)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RowIdentity {
    Slot(u32),
    Fresh(u32),
    Empty,
}

impl RowIdentity {
    fn is_self_at(self, row: u32) -> bool {
        self == Self::Slot(row)
    }
}

pub(crate) struct RowMap {
    identities: Vec<(u32, RowIdentity)>,
    index_of: HashMap<u32, usize>,
    row_of: HashMap<RowIdentity, u32>,
    fresh: u32,
}

impl RowMap {
    pub(crate) fn new() -> Self {
        Self {
            identities: Vec::new(),
            index_of: HashMap::new(),
            row_of: HashMap::new(),
            fresh: 0,
        }
    }

    pub(crate) fn identity_of(&self, row: u32) -> RowIdentity {
        match self.index_of.get(&row) {
            Some(index) => self.identities[*index].1,
            None => RowIdentity::Slot(row),
        }
    }

    fn store(&mut self, row: u32, identity: RowIdentity) {
        match self.index_of.get(&row) {
            Some(index) => self.identities[*index] = (row, identity),
            None => {
                self.index_of.insert(row, self.identities.len());
                self.identities.push((row, identity));
            }
        }
    }

    fn detach(&mut self, row: u32) -> RowIdentity {
        let identity = self.identity_of(row);
        if !identity.is_self_at(row) {
            self.row_of.remove(&identity);
        }
        identity
    }

    fn attach(&mut self, row: u32, identity: RowIdentity) {
        if !identity.is_self_at(row) {
            self.row_of.insert(identity, row);
        }
    }

    pub(crate) fn add(&mut self, row: u32) {
        self.detach(row);
        let identity = RowIdentity::Fresh(self.fresh);
        self.fresh += 1;
        self.store(row, identity);
        self.attach(row, identity);
    }

    pub(crate) fn remove(&mut self, hole: u32, tail: u32) {
        if hole == tail {
            self.detach(hole);
            self.store(hole, RowIdentity::Empty);
            return;
        }
        let inherited = self.detach(tail);
        self.detach(hole);
        self.store(hole, inherited);
        self.store(tail, RowIdentity::Empty);
        self.attach(hole, inherited);
    }

    pub(crate) fn swap(&mut self, first: u32, second: u32) {
        let first_identity = self.detach(first);
        let second_identity = self.detach(second);
        self.store(first, second_identity);
        self.store(second, first_identity);
        self.attach(first, second_identity);
        self.attach(second, first_identity);
    }

    pub(crate) fn row_of(&self, identity: RowIdentity) -> Option<u32> {
        match identity {
            RowIdentity::Empty => None,
            RowIdentity::Fresh(_) => self.row_of.get(&identity).copied(),
            RowIdentity::Slot(slot) => match self.row_of.get(&identity).copied() {
                Some(row) => Some(row),
                None if !self.index_of.contains_key(&slot) => Some(slot),
                None => None,
            },
        }
    }
}
