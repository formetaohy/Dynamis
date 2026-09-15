use dynamis_gpu::{Publication, SubmissionEncoder};
use std::collections::{BTreeMap, HashMap};
use wgpu::{Buffer, Device};

pub(crate) trait Kind {
    type Manifest;
    type Context<'a>: Copy;
    type Value;

    fn step(manifest: &Self::Manifest) -> u64;

    fn decode(
        context: Self::Context<'_>,
        manifest: &Self::Manifest,
        bytes: &[u8],
    ) -> Vec<(u32, Self::Value)>;
}

struct Watch {
    keys: Vec<u32>,
    slots: HashMap<u32, usize>,
    dirty: bool,
}

impl Watch {
    fn new() -> Self {
        Self {
            keys: Vec::new(),
            slots: HashMap::new(),
            dirty: false,
        }
    }

    fn watch(&mut self, id: u32) -> bool {
        if self.slots.contains_key(&id) {
            return false;
        }
        self.slots.insert(id, self.keys.len());
        self.keys.push(id);
        self.dirty = true;
        true
    }

    fn forget(&mut self, id: u32) {
        let Some(slot) = self.slots.remove(&id) else {
            return;
        };
        self.keys.swap_remove(slot);
        if let Some(moved) = self.keys.get(slot) {
            self.slots.insert(*moved, slot);
        }
        self.dirty = true;
    }

    fn len(&self) -> u32 {
        self.keys.len() as u32
    }

    fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    fn clear(&mut self) {
        self.keys.clear();
        self.slots.clear();
        self.dirty = true;
    }
}

struct Stored<V> {
    covered: u64,
    value: V,
}

pub(crate) struct Facts<K: Kind> {
    watch: Watch,
    declared: Vec<u32>,
    sequence: u64,
    publication: Publication<K::Manifest>,
    stored: BTreeMap<u32, Stored<K::Value>>,
    age: Option<u64>,
}

impl<K: Kind> Facts<K> {
    pub(crate) fn new(label: &'static str, depth: usize) -> Self {
        Self {
            watch: Watch::new(),
            declared: Vec::new(),
            sequence: 0,
            publication: Publication::new(label, depth),
            stored: BTreeMap::new(),
            age: None,
        }
    }

    pub(crate) fn watch(&mut self, id: u32) -> bool {
        self.watch.watch(id)
    }

    pub(crate) fn watch_all(&mut self, ids: impl IntoIterator<Item = u32>) {
        for id in ids {
            self.watch.watch(id);
        }
    }

    pub(crate) fn forget(&mut self, id: u32) {
        self.watch.forget(id);
    }

    pub(crate) fn keys(&self) -> &[u32] {
        &self.watch.keys
    }

    pub(crate) fn len(&self) -> u32 {
        self.watch.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.watch.is_empty()
    }

    pub(crate) fn take_dirty(&mut self) -> bool {
        self.watch.take_dirty()
    }

    pub(crate) fn declared(&self) -> &[u32] {
        &self.declared
    }

    pub(crate) fn set_declared(&mut self, ids: Vec<u32>) {
        self.declared = ids;
    }

    pub(crate) fn demand(&self) -> u32 {
        self.watch.len().max(self.declared.len() as u32)
    }

    pub(crate) fn age(&self) -> Option<u64> {
        self.age
    }

    pub(crate) fn needs_publication(&self, step: u64) -> bool {
        !self.watch.is_empty() && self.age != step.checked_sub(1)
    }

    pub(crate) fn get(&self, id: u32) -> Option<&K::Value> {
        self.stored.get(&id).map(|stored| &stored.value)
    }

    pub(crate) fn covered(&self, id: u32) -> Option<u64> {
        self.stored.get(&id).map(|stored| stored.covered)
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = (u32, &K::Value)> {
        self.stored.iter().map(|(id, stored)| (*id, &stored.value))
    }

    pub(crate) fn insert(&mut self, id: u32, covered: u64, value: K::Value) {
        self.stored.insert(id, Stored { covered, value });
    }

    pub(crate) fn patch(&mut self, id: u32, edit: impl FnOnce(&mut K::Value)) {
        if let Some(stored) = self.stored.get_mut(&id) {
            edit(&mut stored.value);
        }
    }

    pub(crate) fn stop_watching(&mut self, id: u32) {
        self.watch.forget(id);
        self.stored.remove(&id);
    }

    pub(crate) fn stop(&mut self) {
        self.watch.clear();
        self.stored.clear();
        self.age = None;
    }

    pub(crate) fn reset(&mut self) {
        self.stop();
        self.declared.clear();
        self.publication.clear();
    }

    pub(crate) fn publish(
        &mut self,
        context: K::Context<'_>,
        device: &Device,
        encoder: &mut SubmissionEncoder,
        budget: u64,
        regions: &[(&Buffer, u64, u64)],
        manifest: K::Manifest,
    ) {
        self.publication.reserve(device, budget);
        self.sequence += 1;
        if let Some((manifest, bytes)) =
            self.publication
                .declare(encoder, regions, self.sequence, manifest)
        {
            self.accept(context, &manifest, &bytes);
        }
    }

    pub(crate) fn collect(&mut self, context: K::Context<'_>) {
        for (manifest, bytes) in self.publication.collect() {
            self.accept(context, &manifest, &bytes);
        }
    }

    pub(crate) fn drain(&mut self, context: K::Context<'_>) {
        for (manifest, bytes) in self.publication.drain() {
            self.accept(context, &manifest, &bytes);
        }
    }

    fn accept(&mut self, context: K::Context<'_>, manifest: &K::Manifest, bytes: &[u8]) {
        let step = K::step(manifest);
        for (id, value) in K::decode(context, manifest, bytes) {
            self.stored.insert(
                id,
                Stored {
                    covered: step + 1,
                    value,
                },
            );
        }
        self.age = Some(self.age.map_or(step, |age| age.max(step)));
    }
}
