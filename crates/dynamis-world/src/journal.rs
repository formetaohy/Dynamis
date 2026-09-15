use std::collections::HashMap;
use std::hash::Hash;

#[derive(Clone)]
pub(crate) struct EditJournal<K, T> {
    buckets: Vec<(K, Vec<T>)>,
    index_of: HashMap<K, usize>,
}

impl<K: Copy + Eq + Hash, T> EditJournal<K, T> {
    pub(crate) fn new() -> Self {
        Self {
            buckets: Vec::new(),
            index_of: HashMap::new(),
        }
    }

    pub(crate) fn push(&mut self, key: K, entry: T) {
        match self.index_of.get(&key) {
            Some(index) => self.buckets[*index].1.push(entry),
            None => {
                self.index_of.insert(key, self.buckets.len());
                self.buckets.push((key, vec![entry]));
            }
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (K, &[T])> {
        self.buckets
            .iter()
            .map(|(key, entries)| (*key, entries.as_slice()))
    }

    pub(crate) fn remove(&mut self, key: K) {
        let Some(index) = self.index_of.remove(&key) else {
            return;
        };
        self.buckets.swap_remove(index);
        if let Some((moved, _)) = self.buckets.get(index) {
            self.index_of.insert(*moved, index);
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.buckets.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.buckets.clear();
        self.index_of.clear();
    }
}
