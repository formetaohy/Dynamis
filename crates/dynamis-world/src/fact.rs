use dynamis_gpu::{Publication, SubmissionEncoder};
use wgpu::{Buffer, Device};

pub(crate) trait Kind {
    const LABEL: &'static str;

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

const ABSENT: u32 = u32::MAX;

#[derive(Default)]
struct Watch {
    keys: Vec<u32>,
    slots: Vec<u32>,
    revision: u64,
}

impl Watch {
    fn slot(&self, id: u32) -> Option<usize> {
        self.slots
            .get(id as usize)
            .copied()
            .filter(|slot| *slot != ABSENT)
            .map(|slot| slot as usize)
    }

    fn hold(&mut self, id: u32) {
        let rows = id as usize + 1;
        if rows > self.slots.len() {
            self.slots.resize(rows, ABSENT);
        }
    }

    fn declare(&mut self, id: u32) -> bool {
        if self.slot(id).is_some() {
            return false;
        }
        self.hold(id);
        self.slots[id as usize] = self.keys.len() as u32;
        self.keys.push(id);
        self.revision += 1;
        true
    }

    fn forget(&mut self, id: u32) {
        let Some(slot) = self.slot(id) else {
            return;
        };
        self.keys.swap_remove(slot);
        self.slots[id as usize] = ABSENT;
        if let Some(moved) = self.keys.get(slot) {
            self.slots[*moved as usize] = slot as u32;
        }
        self.revision += 1;
    }

    fn clear(&mut self) {
        if self.keys.is_empty() {
            return;
        }
        self.keys.clear();
        self.slots.fill(ABSENT);
        self.revision += 1;
    }
}

struct Stored<V> {
    covered: u64,
    value: V,
}

pub(crate) struct FactStore<K: Kind> {
    watch: Watch,
    mirrors: Vec<Option<Stored<K::Value>>>,
    sequence: u64,
    publication: Publication<K::Manifest>,
    published: Option<(u64, u64)>,
    age: Option<u64>,
}

impl<K: Kind> FactStore<K> {
    pub(crate) fn new(depth: usize) -> Self {
        Self {
            watch: Watch::default(),
            mirrors: Vec::new(),
            sequence: 0,
            publication: Publication::new(K::LABEL, depth),
            published: None,
            age: None,
        }
    }

    pub(crate) fn watch(&mut self, id: u32) -> bool {
        self.watch.declare(id)
    }

    pub(crate) fn watch_all(&mut self, ids: impl IntoIterator<Item = u32>) {
        for id in ids {
            self.watch.declare(id);
        }
    }

    pub(crate) fn keys(&self) -> &[u32] {
        &self.watch.keys
    }

    pub(crate) fn len(&self) -> u32 {
        self.watch.keys.len() as u32
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.watch.keys.is_empty()
    }

    /// Whether the device still holds a watch set another revision than the one at hand: the ids
    /// the pass that publishes this kind reads are the ids of the watch set at hand, so a watch set
    /// that moved owes the device the list it moved to.
    pub(crate) fn watch_moved(&self) -> bool {
        !self.watch.keys.is_empty()
            && self.published.map(|(_, revision)| revision) != Some(self.watch.revision)
    }

    pub(crate) fn age(&self) -> Option<u64> {
        self.age
    }

    /// Whether the host still owes an observation of this kind: a kind owes one while it watches
    /// something and the publication it last declared did not carry the watch set at hand at the
    /// step this one follows.
    pub(crate) fn needs_publication(&self, step: u64) -> bool {
        !self.watch.keys.is_empty()
            && self.published != step.checked_sub(1).map(|step| (step, self.watch.revision))
    }

    pub(crate) fn get(&self, id: u32) -> Option<&K::Value> {
        self.entry(id).map(|stored| &stored.value)
    }

    pub(crate) fn covered(&self, id: u32) -> Option<u64> {
        self.entry(id).map(|stored| stored.covered)
    }

    fn entry(&self, id: u32) -> Option<&Stored<K::Value>> {
        self.mirrors.get(id as usize).and_then(Option::as_ref)
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = (u32, &K::Value)> {
        self.mirrors
            .iter()
            .enumerate()
            .filter_map(|(id, entry)| entry.as_ref().map(|stored| (id as u32, &stored.value)))
    }

    pub(crate) fn insert(&mut self, id: u32, covered: u64, value: K::Value) {
        let rows = id as usize + 1;
        if rows > self.mirrors.len() {
            self.mirrors.resize_with(rows, || None);
        }
        self.mirrors[id as usize] = Some(Stored { covered, value });
    }

    pub(crate) fn patch(&mut self, id: u32, edit: impl FnOnce(&mut K::Value)) {
        if let Some(stored) = self.mirrors.get_mut(id as usize).and_then(Option::as_mut) {
            edit(&mut stored.value);
        }
    }

    pub(crate) fn stop_watching(&mut self, id: u32) {
        self.watch.forget(id);
        if let Some(slot) = self.mirrors.get_mut(id as usize) {
            *slot = None;
        }
    }

    pub(crate) fn stop(&mut self) {
        self.watch.clear();
        self.mirrors.clear();
        self.published = None;
        self.age = None;
    }

    pub(crate) fn reset(&mut self) {
        self.stop();
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
        self.published = Some((K::step(&manifest), self.watch.revision));
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
            self.insert(id, step + 1, value);
        }
        self.age = Some(self.age.map_or(step, |age| age.max(step)));
    }
}
