use super::arena::{Arena, Run, merged};
use super::body::BodyStore;
use dynamis_abi::{COLLIDER_SENSOR, ColliderRecord, ENTRY_INDEX_MASK, EVENT_MODE_PERSIST};
use dynamis_model::BodyHandle;

#[derive(Clone, Copy)]
struct ColliderRun {
    run: Run,
    movable: bool,
}

/// The part of a collider record the grid resolution is a maximum over: the shape, the scale, and
/// the source bounds the collider is sized by. The resolution is pose independent, so a record edit
/// that leaves this alone cannot move the cell size every entry is keyed at.
#[derive(Clone, Copy, PartialEq)]
struct ColliderSizing {
    kind: u32,
    source: u32,
    radius: f32,
    half_height: f32,
    half_extents: [f32; 3],
    scale: [f32; 3],
}

/// The part of a collider record a grid entry is built from: the sizing, plus where the collider
/// sits inside its body. An entry's box and cell answer exactly these.
#[derive(Clone, Copy, PartialEq)]
struct ColliderPlacement {
    sizing: ColliderSizing,
    local_offset: [f32; 3],
    local_rotation: [f32; 4],
}

/// The part of a collider record that decides which bodies it reaches and how it answers them:
/// the placement a grid entry is built from, and whether it answers contacts at all. A record that
/// is replaced with the same answering owes the bodies it holds nothing, whatever its materials
/// carry.
#[derive(Clone, Copy, PartialEq)]
struct ColliderAnswering {
    placement: ColliderPlacement,
    sensor: bool,
}

fn sizing_of(record: &ColliderRecord) -> ColliderSizing {
    ColliderSizing {
        kind: record.kind,
        source: record.source,
        radius: record.radius,
        half_height: record.half_height,
        half_extents: record.half_extents,
        scale: record.scale,
    }
}

fn placement_of(record: &ColliderRecord) -> ColliderPlacement {
    ColliderPlacement {
        sizing: sizing_of(record),
        local_offset: record.local_offset,
        local_rotation: record.local_rotation,
    }
}

fn answering_of(record: &ColliderRecord) -> ColliderAnswering {
    ColliderAnswering {
        placement: placement_of(record),
        sensor: record.flags & COLLIDER_SENSOR != 0,
    }
}

fn replaced<T: PartialEq>(
    held: &[ColliderRecord],
    records: &[ColliderRecord],
    key: impl Fn(&ColliderRecord) -> T,
) -> bool {
    held.len() != records.len()
        || held
            .iter()
            .zip(records)
            .any(|(held, record)| key(held) != key(record))
}

/// How replacing or retiring a body's collider block moved the facts the broadphase grid indices
/// derive from: the grid resolution is a maximum over every block's sizing, and each half of the
/// grid holds the entries of the bodies that reach it. A block that is replaced at the same
/// sizing and placement owes neither, whatever else its records carry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ColliderDelta {
    /// The block now belongs to the other half of the partition: its entries moved between the
    /// moving and the immovable halves of the grid.
    pub(crate) crossed: bool,
    /// The sizing the grid resolution is a maximum over changed, or a block appeared or vanished.
    pub(crate) sizing: bool,
    /// The placement an entry is built from changed, or a block's records were replaced.
    pub(crate) placement: bool,
    /// The block now answers another shape or another placement, so the bodies it reaches no
    /// longer answer the collider the device held.
    pub(crate) answering: bool,
}

impl ColliderDelta {
    /// Whether the entries a half of the grid holds from this body must be built again.
    pub(crate) const fn entries(self) -> bool {
        self.crossed || self.sizing || self.placement
    }
}

impl ColliderRun {
    const EMPTY: Self = Self {
        run: Run::EMPTY,
        movable: false,
    };

    const fn movable_length(self) -> u32 {
        if self.movable { self.run.len } else { 0 }
    }
}

#[derive(Clone)]
pub(crate) struct ColliderStore {
    arena: Arena,
    records: Vec<ColliderRecord>,
    runs: Vec<ColliderRun>,
    cleared: Vec<Run>,
    live: u32,
    movable: u32,
    armed: u32,
    persistent: u32,
}

impl ColliderStore {
    pub(crate) const fn new() -> Self {
        Self {
            arena: Arena::new(),
            records: Vec::new(),
            runs: Vec::new(),
            cleared: Vec::new(),
            live: 0,
            movable: 0,
            armed: 0,
            persistent: 0,
        }
    }

    pub(crate) fn used(&self) -> u32 {
        self.arena.used()
    }

    pub(crate) fn live(&self) -> u32 {
        self.live
    }

    pub(crate) fn movable(&self) -> u32 {
        self.movable
    }

    pub(crate) fn impact_armed(&self) -> u32 {
        self.armed
    }

    pub(crate) fn persist_events(&self) -> u32 {
        self.persistent
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
            .filter(|entry| entry.run.len > 0)
            .map(|entry| entry.run)
    }

    pub(crate) fn assign(
        &mut self,
        id: u32,
        movable: bool,
        records: &[ColliderRecord],
    ) -> ColliderDelta {
        let len = records.len() as u32;
        assert!(len > 0, "a collider run must hold at least one collider");
        let delta = self.delta_of(id, movable, records);
        if self.runs.len() <= id as usize {
            self.runs.resize(id as usize + 1, ColliderRun::EMPTY);
        }
        let entry = if self.runs[id as usize].run.len == len {
            ColliderRun {
                movable,
                ..self.runs[id as usize]
            }
        } else {
            self.clear(id);
            ColliderRun {
                run: self.take(len),
                movable,
            }
        };
        self.hold_run(id, entry);
        let span = entry.run.span();
        self.armed -= armed(&self.records[span.clone()]);
        self.persistent -= persisting(&self.records[span.clone()]);
        for (slot, record) in records.iter().enumerate() {
            self.records[entry.run.offset as usize + slot] = *record;
        }
        self.armed += armed(records);
        self.persistent += persisting(records);
        delta
    }

    fn delta_of(&self, id: u32, movable: bool, records: &[ColliderRecord]) -> ColliderDelta {
        let Some(run) = self.run_of(id) else {
            return ColliderDelta {
                crossed: false,
                sizing: true,
                placement: true,
                answering: true,
            };
        };
        let held = &self.records[run.span()];
        ColliderDelta {
            crossed: self.runs[id as usize].movable != movable,
            sizing: replaced(held, records, sizing_of),
            placement: replaced(held, records, placement_of),
            answering: replaced(held, records, answering_of),
        }
    }

    fn hold_run(&mut self, id: u32, entry: ColliderRun) {
        let previous = std::mem::replace(&mut self.runs[id as usize], entry);
        self.movable -= previous.movable_length();
        self.movable += entry.movable_length();
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

    /// Retires a body's collider block, answering how the retirement moved the facts the grid
    /// indices derive. A body without a block retires nothing.
    pub(crate) fn release(&mut self, id: u32) -> ColliderDelta {
        if !self.clear(id) {
            return ColliderDelta::default();
        }
        ColliderDelta {
            crossed: false,
            sizing: true,
            placement: true,
            answering: true,
        }
    }

    fn clear(&mut self, id: u32) -> bool {
        let Some(run) = self.run_of(id) else {
            return false;
        };
        let released = armed(&self.records[run.span()]);
        self.armed -= released;
        self.persistent -= persisting(&self.records[run.span()]);
        self.hold_run(id, ColliderRun::EMPTY);
        for index in run.span() {
            self.records[index] = ColliderRecord::cleared();
        }
        self.arena.release(run);
        self.records.truncate(self.arena.used() as usize);
        self.cleared.push(run);
        self.live -= run.len;
        true
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

fn persisting(records: &[ColliderRecord]) -> u32 {
    records
        .iter()
        .filter(|record| (record.flags & EVENT_MODE_PERSIST) != 0)
        .count() as u32
}

fn armed(records: &[ColliderRecord]) -> u32 {
    records
        .iter()
        .filter(|record| record.impact_force.is_finite())
        .count() as u32
}
