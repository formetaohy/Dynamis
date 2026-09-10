//! The body and constraint command stream, compiled once per flush into parallel
//! work: a row re-map for structural commands (add/remove/swap) and per-slot
//! command groups for content edits (patch/force/impulse/sleep/wake).
//!
//! Host mutations never touch the device directly; they are recorded as a command
//! sequence, then compiled here into the two things the device can do in parallel:
//! a re-map of rows (which slot carries which row of body state) and one ordered
//! run of edits per touched slot. The single-lane command loop is gone; a batch of
//! spawns or force applications costs one linear host pass and a few parallel
//! dispatches, with the exact same per-slot ordering the old loop guaranteed.

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

impl Simulation {
    /// Compiles the pending body command sequence.
    ///
    /// Slot contents start as identity (slot i holds row i) and follow every
    /// add/remove/swap in order. A content edit belongs to the row it was recorded
    /// against, not to the slot that row happened to occupy then: the edit follows
    /// the row through every later move and dies with it if the row is removed.
    /// Edits are emitted on the row's final slot, preserving their original order,
    /// so the device can run one slot's whole edit run per lane.
    pub(crate) fn compile_commands(&self) -> CompiledBodyCommands {
        let mut contents: Vec<RowIdentity> =
            (0..self.slots as u32).map(RowIdentity::Slot).collect();
        let mut moved_flags = vec![false; self.slots];
        let mut fresh: Vec<BodyStateRecord> = Vec::new();
        let mut content_edits: Vec<Vec<BodyCommandRecord>> = Vec::new();
        for command in &self.commands {
            let slot = command.slot;
            match command.kind {
                COMMAND_ADD => {
                    contents[slot as usize] = RowIdentity::Fresh(fresh.len() as u32);
                    fresh.push(command.state);
                    moved_flags[slot as usize] = true;
                }
                COMMAND_REMOVE => {
                    contents[slot as usize] = contents[command.mask as usize];
                    contents[command.mask as usize] = RowIdentity::Empty;
                    moved_flags[slot as usize] = true;
                    moved_flags[command.mask as usize] = true;
                }
                COMMAND_SWAP => {
                    contents.swap(slot as usize, command.mask as usize);
                    moved_flags[slot as usize] = true;
                    moved_flags[command.mask as usize] = true;
                }
                _ => {
                    let identity = contents[slot as usize];
                    if let Some(id) = identity_index(identity, self.slots as u32) {
                        if content_edits.len() <= id as usize {
                            content_edits.resize_with(id as usize + 1, Vec::new);
                        }
                        content_edits[id as usize].push(*command);
                    }
                }
            }
        }
        let moves: Vec<RowMove> = contents[..self.alive.len()]
            .iter()
            .enumerate()
            .map(|(slot, identity)| move_of(identity, &moved_flags[slot]))
            .collect();
        let mut ordered = Vec::new();
        let mut edit_first = vec![0u32; self.alive.len()];
        for (slot, identity) in contents[..self.alive.len()].iter().enumerate() {
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
        let structurally_dirty = moves.iter().any(|move_| !matches!(move_, RowMove::Keep));
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
        let mut contents: Vec<RowIdentity> =
            (0..self.slots as u32).map(RowIdentity::Slot).collect();
        let mut moved_flags = vec![false; self.slots];
        let mut fresh: Vec<ConstraintRuntimeRecord> = Vec::new();
        for command in &self.constraint_commands {
            match command.kind {
                COMMAND_CONSTRAINT_ADD => {
                    contents[command.slot as usize] = RowIdentity::Fresh(fresh.len() as u32);
                    fresh.push(ConstraintRuntimeRecord::fresh(
                        command.constraint_id,
                        command.generation,
                    ));
                    moved_flags[command.slot as usize] = true;
                }
                _ => {
                    contents.swap(command.slot as usize, command.tail as usize);
                    moved_flags[command.slot as usize] = true;
                    moved_flags[command.tail as usize] = true;
                }
            }
        }
        let moves = contents[..self.constraint_alive.len()]
            .iter()
            .enumerate()
            .map(|(slot, identity)| constraint_move_of(identity, &moved_flags[slot]))
            .collect();
        CompiledConstraintCommands { moves, fresh }
    }
}

pub(crate) struct CompiledConstraintCommands {
    pub moves: Vec<ConstraintMove>,
    pub fresh: Vec<ConstraintRuntimeRecord>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConstraintMove {
    Keep,
    MoveSource(u32),
    Fresh(u32),
    Clear,
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

fn move_of(identity: &RowIdentity, moved: &bool) -> RowMove {
    match identity {
        RowIdentity::Slot(slot) if !moved => RowMove::Keep,
        RowIdentity::Slot(slot) => RowMove::MoveSource(*slot),
        RowIdentity::Fresh(index) => RowMove::Fresh(*index),
        RowIdentity::Empty => RowMove::Clear,
    }
}

fn constraint_move_of(identity: &RowIdentity, moved: &bool) -> ConstraintMove {
    match identity {
        RowIdentity::Slot(slot) if !moved => ConstraintMove::Keep,
        RowIdentity::Slot(slot) => ConstraintMove::MoveSource(*slot),
        RowIdentity::Fresh(index) => ConstraintMove::Fresh(*index),
        RowIdentity::Empty => ConstraintMove::Clear,
    }
}

/// The device encoding of one body row move: the source slot for a plain move,
/// the CLEAR sentinel, or the slot itself for a keep; the fresh lane index lives
/// in the companion lane.
pub(crate) fn encode_move(move_: &RowMove, slot: u32) -> (u32, u32) {
    match move_ {
        RowMove::Keep => (slot, u32::MAX),
        RowMove::MoveSource(source) => (*source, u32::MAX),
        RowMove::Fresh(index) => (slot, *index),
        RowMove::Clear => (u32::MAX, u32::MAX),
    }
}

pub(crate) fn encode_constraint_move(move_: &ConstraintMove, slot: u32) -> (u32, u32) {
    match move_ {
        ConstraintMove::Keep => (slot, u32::MAX),
        ConstraintMove::MoveSource(source) => (*source, u32::MAX),
        ConstraintMove::Fresh(index) => (slot, *index),
        ConstraintMove::Clear => (u32::MAX, u32::MAX),
    }
}

impl ConstraintMove {
    pub(crate) fn is_structural(&self) -> bool {
        !matches!(self, ConstraintMove::Keep)
    }
}
