use dynamis_layout::RowMoveRecord;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RowIdentity {
    Slot(u32),
    Fresh(u32),
    Empty,
}

impl RowIdentity {
    fn record(self, row: u32) -> RowMoveRecord {
        match self {
            Self::Slot(source) => RowMoveRecord::source(row, source),
            Self::Fresh(index) => RowMoveRecord::fresh(row, index),
            Self::Empty => RowMoveRecord::clear(row),
        }
    }

    fn is_self_at(self, row: u32) -> bool {
        self == Self::Slot(row)
    }
}

pub(crate) struct RowMap {
    identities: Vec<(u32, RowIdentity)>,
    index_of: HashMap<u32, usize>,
    row_of: HashMap<RowIdentity, u32>,
}

impl RowMap {
    pub(crate) fn new() -> Self {
        Self {
            identities: Vec::new(),
            index_of: HashMap::new(),
            row_of: HashMap::new(),
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

    pub(crate) fn add(&mut self, row: u32, fresh: u32) {
        self.detach(row);
        let identity = RowIdentity::Fresh(fresh);
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

    pub(crate) fn moves(&self, rows: u32) -> Vec<RowMoveRecord> {
        self.identities
            .iter()
            .filter_map(|(row, identity)| {
                if *row >= rows {
                    return None;
                }
                (!identity.is_self_at(*row)).then(|| identity.record(*row))
            })
            .collect()
    }
}

pub(crate) struct RowJournal<T> {
    buckets: Vec<(RowIdentity, Vec<T>)>,
    index_of: HashMap<RowIdentity, usize>,
}

impl<T> RowJournal<T> {
    pub(crate) fn new() -> Self {
        Self {
            buckets: Vec::new(),
            index_of: HashMap::new(),
        }
    }

    pub(crate) fn push(&mut self, identity: RowIdentity, entry: T) {
        match self.index_of.get(&identity) {
            Some(index) => self.buckets[*index].1.push(entry),
            None => {
                self.index_of.insert(identity, self.buckets.len());
                self.buckets.push((identity, vec![entry]));
            }
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (RowIdentity, &[T])> {
        self.buckets
            .iter()
            .map(|(identity, entries)| (*identity, entries.as_slice()))
    }
}
