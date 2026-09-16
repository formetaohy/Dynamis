use super::id::IdSpace;

pub(crate) trait Identity: Copy + PartialEq + std::fmt::Debug {
    fn id(self) -> u32;

    fn generation(self) -> u32;

    fn revealed(id: u32, generation: u32) -> Self;
}

macro_rules! identity {
    ($($handle:ty),* $(,)?) => {
        $(
            impl Identity for $handle {
                fn id(self) -> u32 {
                    self.id
                }

                fn generation(self) -> u32 {
                    self.generation
                }

                fn revealed(id: u32, generation: u32) -> Self {
                    Self { id, generation }
                }
            }
        )*
    };
}

identity!(
    dynamis_model::BodyHandle,
    dynamis_model::CharacterHandle,
    dynamis_model::ConstraintHandle,
    dynamis_model::SoftBodyHandle,
    dynamis_model::ShapeSourceHandle,
    dynamis_model::VehicleHandle,
);

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Removal {
    Compact,
    Vacate,
}

pub(crate) struct Retired<H> {
    pub(crate) row: u32,
    pub(crate) moved: Option<H>,
}

#[derive(Clone)]
pub(crate) struct Pool<H: Identity> {
    label: &'static str,
    ids: IdSpace,
    alive: Vec<H>,
    rows: Vec<u32>,
    removal: Removal,
    fresh: Vec<u32>,
    dirty: Vec<u32>,
    retired: Vec<u32>,
}

impl<H: Identity> Pool<H> {
    pub(crate) const fn compact(label: &'static str) -> Self {
        Self::new(label, Removal::Compact)
    }

    pub(crate) const fn vacate(label: &'static str) -> Self {
        Self::new(label, Removal::Vacate)
    }

    const fn new(label: &'static str, removal: Removal) -> Self {
        Self {
            label,
            ids: IdSpace::new(),
            alive: Vec::new(),
            rows: Vec::new(),
            removal,
            fresh: Vec::new(),
            dirty: Vec::new(),
            retired: Vec::new(),
        }
    }

    pub(crate) fn len(&self) -> u32 {
        self.alive.len() as u32
    }

    pub(crate) fn alive(&self) -> &[H] {
        &self.alive
    }

    pub(crate) fn ids(&self) -> u32 {
        self.ids.len() as u32
    }

    pub(crate) fn generation(&self, id: u32) -> u32 {
        self.ids.generation(id)
    }

    pub(crate) fn identities(&self) -> &IdSpace {
        &self.ids
    }

    pub(crate) fn pending(&self) -> bool {
        !self.fresh.is_empty() || !self.dirty.is_empty() || !self.retired.is_empty()
    }

    pub(crate) fn acquire(&mut self) -> H {
        let (id, generation) = self.ids.acquire();
        if self.rows.len() <= id as usize {
            self.rows.resize(id as usize + 1, u32::MAX);
        }
        H::revealed(id, generation)
    }

    pub(crate) fn insert(&mut self, handle: H) -> u32 {
        let row = match self.removal {
            Removal::Compact => self.alive.len() as u32,
            Removal::Vacate => handle.id(),
        };
        self.rows[handle.id() as usize] = row;
        self.alive.push(handle);
        self.fresh.push(handle.id());
        row
    }

    pub(crate) fn contains(&self, handle: H) -> bool {
        self.row_of_id(handle.id()) != u32::MAX
            && self.ids.generation(handle.id()) == handle.generation()
    }

    pub(crate) fn handle_of(&self, id: u32) -> Option<H> {
        if self.row_of_id(id) == u32::MAX {
            return None;
        }
        Some(H::revealed(id, self.ids.generation(id)))
    }

    pub(crate) fn validate(&self, handle: H) {
        assert!(
            (handle.id() as usize) < self.rows.len(),
            "a {} handle must be inside its identity space",
            self.label,
        );
        assert!(
            self.ids.generation(handle.id()) == handle.generation(),
            "a {} handle must not be stale",
            self.label,
        );
        assert!(
            self.row_of_id(handle.id()) != u32::MAX,
            "a {} handle must be alive",
            self.label,
        );
    }

    pub(crate) fn row_of(&self, handle: H) -> u32 {
        self.validate(handle);
        self.rows[handle.id() as usize]
    }

    pub(crate) fn row_of_id(&self, id: u32) -> u32 {
        self.rows.get(id as usize).copied().unwrap_or(u32::MAX)
    }

    pub(crate) fn handle_of_row(&self, row: u32) -> H {
        self.alive[row as usize]
    }

    pub(crate) fn mark(&mut self, handle: H) {
        self.dirty.push(handle.id());
    }

    pub(crate) fn mark_row(&mut self, row: u32) {
        self.dirty.push(self.alive[row as usize].id());
    }

    pub(crate) fn swap_rows(&mut self, first: u32, second: u32) {
        self.alive.swap(first as usize, second as usize);
        let first_id = self.alive[first as usize].id();
        let second_id = self.alive[second as usize].id();
        self.rows[first_id as usize] = first;
        self.rows[second_id as usize] = second;
        self.dirty.push(first_id);
        self.dirty.push(second_id);
    }

    pub(crate) fn retire(&mut self, handle: H) -> Retired<H> {
        self.validate(handle);
        let id = handle.id();
        let row = self.rows[id as usize];
        let moved = match self.removal {
            Removal::Compact => {
                let tail = self.alive.len() - 1;
                self.alive.swap(row as usize, tail);
                let moved = self.alive[row as usize];
                self.alive.pop();
                (row != tail as u32).then_some(moved)
            }
            Removal::Vacate => {
                self.alive.retain(|live| live.id() != id);
                None
            }
        };
        if let Some(moved) = moved {
            self.rows[moved.id() as usize] = row;
            self.dirty.push(moved.id());
        }
        self.rows[id as usize] = u32::MAX;
        self.ids.release(id);
        self.retired.push(id);
        Retired { row, moved }
    }

    pub(crate) fn take_fresh(&mut self) -> Vec<u32> {
        take_sorted(&mut self.fresh)
    }

    pub(crate) fn take_dirty(&mut self) -> Vec<u32> {
        take_sorted(&mut self.dirty)
    }

    pub(crate) fn take_retired(&mut self) -> Vec<u32> {
        take_sorted(&mut self.retired)
    }

    pub(crate) fn changed(&mut self) -> Vec<u32> {
        let mut changed = self.take_fresh();
        changed.extend(self.take_dirty());
        changed.sort_unstable();
        changed.dedup();
        changed
    }
}

fn take_sorted(slots: &mut Vec<u32>) -> Vec<u32> {
    slots.sort_unstable();
    slots.dedup();
    std::mem::take(slots)
}
