//! The body and constraint command stream, compiled once per flush into parallel
//! work: a row re-map for structural commands (add/remove/swap) and per-slot
//! command groups for content edits (patch/force/impulse/sleep/wake).
//!
//! Host mutations never touch the device directly; they are recorded as a command
//! sequence, then compiled here into the two things the device can do in parallel:
//! a re-map of rows (which slot carries which row of body state) and one ordered
//! run of edits per touched slot.

use crate::simulation::Simulation;
use dynamis_layout::{
    BodyCommandRecord, BodyStateRecord, COMMAND_ADD, COMMAND_CONSTRAINT_ADD, COMMAND_REMOVE,
    COMMAND_SWAP, ConstraintRuntimeRecord,
};

/// A row's identity through the structural shuffle: its own slot, a freshly
/// recorded row, or nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowIdentity {
    /// The row that started life in this slot.
    Slot(u32),
    /// A fresh row recorded by an add command.
    Fresh(u32),
    /// No row; the slot is cleared.
    Empty,
}

/// One slot's row destination after every structural command ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowMove {
    /// The row stays where it is; no read or write happens for it.
    Keep,
    /// The final slot takes the row currently held at the source slot.
    MoveSource(u32),
    /// The final slot is written from a fresh host record.
    Fresh(u32),
    /// The final slot holds nothing; the row is cleared.
    Clear,
}

/// The device lanes a move list encodes to.
pub(crate) struct MoveLanes {
    pub(crate) src: Vec<u32>,
    pub(crate) fresh: Vec<u32>,
}

impl MoveLanes {
    pub(crate) fn of(moves: &[RowMove]) -> Self {
        let mut lanes = Self {
            src: Vec::with_capacity(moves.len()),
            fresh: Vec::with_capacity(moves.len()),
        };
        for (slot, move_) in moves.iter().enumerate() {
            let (source, fresh) = move_.encode(slot as u32);
            lanes.src.push(source);
            lanes.fresh.push(fresh);
        }
        lanes
    }
}

impl RowMove {
    /// The device encoding: the source slot for a plain move, the CLEAR sentinel,
    /// or the slot itself for a keep; the fresh lane index lives in the companion
    /// lane.
    pub(crate) fn encode(&self, slot: u32) -> (u32, u32) {
        match self {
            Self::Keep => (slot, u32::MAX),
            Self::MoveSource(source) => (*source, u32::MAX),
            Self::Fresh(index) => (slot, *index),
            Self::Clear => (u32::MAX, u32::MAX),
        }
    }

    pub(crate) fn is_structural(&self) -> bool {
        !matches!(self, Self::Keep)
    }
}

/// Which row every device slot holds as a command sequence is replayed, and which
/// slots the sequence touched.
struct RowMap {
    contents: Vec<RowIdentity>,
    moved: Vec<bool>,
}

impl RowMap {
    fn new(slots: usize) -> Self {
        Self {
            contents: (0..slots as u32).map(RowIdentity::Slot).collect(),
            moved: vec![false; slots],
        }
    }

    fn add(&mut self, slot: u32, fresh: u32) {
        self.contents[slot as usize] = RowIdentity::Fresh(fresh);
        self.moved[slot as usize] = true;
    }

    fn remove(&mut self, hole: u32, tail: u32) {
        self.contents[hole as usize] = self.contents[tail as usize];
        self.contents[tail as usize] = RowIdentity::Empty;
        self.moved[hole as usize] = true;
        self.moved[tail as usize] = true;
    }

    fn swap(&mut self, first: u32, second: u32) {
        self.contents.swap(first as usize, second as usize);
        self.moved[first as usize] = true;
        self.moved[second as usize] = true;
    }

    /// The row destination of every alive slot, in slot order.
    fn moves(&self, rows: usize) -> Vec<RowMove> {
        self.contents[..rows]
            .iter()
            .enumerate()
            .map(|(slot, identity)| match identity {
                RowIdentity::Slot(source) if !self.moved[slot] => RowMove::Keep,
                RowIdentity::Slot(source) => RowMove::MoveSource(*source),
                RowIdentity::Fresh(index) => RowMove::Fresh(*index),
                RowIdentity::Empty => RowMove::Clear,
            })
            .collect()
    }

    fn identity_of(&self, slot: u32) -> RowIdentity {
        self.contents[slot as usize]
    }
}

/// The stable identity of a row's content, used to keep edits attached to the row
/// they were recorded for: initial rows keep their birth slot, fresh rows take the
/// lane index after it.
fn identity_index(identity: RowIdentity, slots: u32) -> Option<u32> {
    match identity {
        RowIdentity::Slot(slot) => Some(slot),
        RowIdentity::Fresh(index) => Some(slots + index),
        RowIdentity::Empty => None,
    }
}

/// The compiled body command stream.
pub(crate) struct CompiledBodyCommands {
    /// The row destination of every alive slot, in slot order.
    pub moves: Vec<RowMove>,
    /// Fresh row records, the `Fresh` index in `moves` points into this.
    pub fresh: Vec<BodyStateRecord>,
    /// Content edits re-ordered so each slot's commands run in their original
    /// relative order, contiguous per slot.
    pub edits: Vec<BodyCommandRecord>,
    /// One-based index into `edits` of each slot's first command; 0 means none.
    pub edit_first: Vec<u32>,
    /// Whether any structural work is needed at all.
    pub structurally_dirty: bool,
}

pub(crate) struct CompiledConstraintCommands {
    pub moves: Vec<RowMove>,
    pub fresh: Vec<ConstraintRuntimeRecord>,
}

impl Simulation {
    /// Compiles the pending body command sequence.
    ///
    /// A content edit belongs to the row it was recorded against, not to the slot
    /// that row happened to occupy then: the edit follows the row through every
    /// later move and dies with it if the row is removed. Edits are emitted on the
    /// row's final slot, preserving their original order, so the device can run one
    /// slot's whole edit run per lane.
    pub(crate) fn compile_commands(&self) -> CompiledBodyCommands {
        let mut map = RowMap::new(self.slots);
        let mut fresh: Vec<BodyStateRecord> = Vec::new();
        let mut content_edits: Vec<Vec<BodyCommandRecord>> = Vec::new();
        for command in &self.bodies.commands {
            match command.kind {
                COMMAND_ADD => {
                    map.add(command.slot, fresh.len() as u32);
                    fresh.push(command.state);
                }
                COMMAND_REMOVE => map.remove(command.slot, command.mask),
                COMMAND_SWAP => map.swap(command.slot, command.mask),
                _ => {
                    let identity = map.identity_of(command.slot);
                    if let Some(id) = identity_index(identity, self.slots as u32) {
                        if content_edits.len() <= id as usize {
                            content_edits.resize_with(id as usize + 1, Vec::new);
                        }
                        content_edits[id as usize].push(*command);
                    }
                }
            }
        }
        let moves = map.moves(self.bodies.alive.len());
        let mut ordered = Vec::new();
        let mut edit_first = vec![0u32; self.bodies.alive.len()];
        for (slot, identity) in map.contents[..self.bodies.alive.len()].iter().enumerate() {
            let Some(id) = identity_index(*identity, self.slots as u32) else {
                continue;
            };
            let Some(group) = content_edits.get(id as usize) else {
                continue;
            };
            if group.is_empty() {
                continue;
            }
            edit_first[slot] = ordered.len() as u32 + 1;
            ordered.extend(group.iter().map(|command| BodyCommandRecord {
                slot: slot as u32,
                ..*command
            }));
        }
        let structurally_dirty = moves.iter().any(RowMove::is_structural);
        CompiledBodyCommands {
            moves,
            fresh,
            edits: ordered,
            edit_first,
            structurally_dirty,
        }
    }

    /// Compiles the pending constraint command sequence into a re-map; constraints
    /// only ever add or swap rows.
    pub(crate) fn compile_constraint_commands(&self) -> CompiledConstraintCommands {
        let mut map = RowMap::new(self.slots);
        let mut fresh: Vec<ConstraintRuntimeRecord> = Vec::new();
        for command in &self.constraints.commands {
            match command.kind {
                COMMAND_CONSTRAINT_ADD => {
                    map.add(command.slot, fresh.len() as u32);
                    fresh.push(ConstraintRuntimeRecord::fresh(
                        command.constraint_id,
                        command.generation,
                    ));
                }
                _ => map.swap(command.slot, command.tail),
            }
        }
        CompiledConstraintCommands {
            moves: map.moves(self.constraints.alive.len()),
            fresh,
        }
    }
}
