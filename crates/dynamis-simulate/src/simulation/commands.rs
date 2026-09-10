use crate::simulation::Simulation;
use dynamis_layout::{BodyEditRecord, BodyEditRun, BodyStateRecord};

#[derive(Clone, Copy)]
pub(crate) enum BodyCommand {
    Add {
        row: u32,
        state: BodyStateRecord,
    },
    Remove {
        hole: u32,
        tail: u32,
    },
    Swap {
        first: u32,
        second: u32,
    },
    Patch {
        row: u32,
        mask: u32,
        state: BodyStateRecord,
    },
    Force {
        row: u32,
        force: [f32; 3],
    },
    ForceAtPoint {
        row: u32,
        force: [f32; 3],
        point: [f32; 3],
    },
    Torque {
        row: u32,
        torque: [f32; 3],
    },
    Impulse {
        row: u32,
        impulse: [f32; 3],
    },
    ImpulseAtPoint {
        row: u32,
        impulse: [f32; 3],
        point: [f32; 3],
    },
    AngularImpulse {
        row: u32,
        impulse: [f32; 3],
    },
    Sleep {
        row: u32,
    },
    Wake {
        row: u32,
    },
}

impl BodyCommand {
    pub(crate) fn row(&self) -> u32 {
        match *self {
            Self::Add { row, .. }
            | Self::Patch { row, .. }
            | Self::Force { row, .. }
            | Self::ForceAtPoint { row, .. }
            | Self::Torque { row, .. }
            | Self::Impulse { row, .. }
            | Self::ImpulseAtPoint { row, .. }
            | Self::AngularImpulse { row, .. }
            | Self::Sleep { row }
            | Self::Wake { row } => row,
            Self::Remove { hole, .. } => hole,
            Self::Swap { first, .. } => first,
        }
    }

    fn edit(&self) -> Option<BodyEditRecord> {
        Some(match *self {
            Self::Patch { mask, state, .. } => BodyEditRecord::patch(mask, state),
            Self::Force { force, .. } => BodyEditRecord::force(force),
            Self::ForceAtPoint { force, point, .. } => BodyEditRecord::force_at_point(force, point),
            Self::Torque { torque, .. } => BodyEditRecord::torque(torque),
            Self::Impulse { impulse, .. } => BodyEditRecord::impulse(impulse),
            Self::ImpulseAtPoint { impulse, point, .. } => {
                BodyEditRecord::impulse_at_point(impulse, point)
            }
            Self::AngularImpulse { impulse, .. } => BodyEditRecord::angular_impulse(impulse),
            Self::Sleep { .. } => BodyEditRecord::sleep(),
            Self::Wake { .. } => BodyEditRecord::wake(),
            Self::Add { .. } | Self::Remove { .. } | Self::Swap { .. } => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowIdentity {
    Slot(u32),
    Fresh(u32),
    Empty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowMove {
    Keep,
    MoveSource(u32),
    Fresh(u32),
    Clear,
}

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

fn identity_index(identity: RowIdentity, slots: u32) -> Option<u32> {
    match identity {
        RowIdentity::Slot(slot) => Some(slot),
        RowIdentity::Fresh(index) => Some(slots + index),
        RowIdentity::Empty => None,
    }
}

pub(crate) struct CompiledBodyEdits {
    pub moves: Vec<RowMove>,
    pub fresh: Vec<BodyStateRecord>,
    pub edits: Vec<BodyEditRecord>,
    pub runs: Vec<BodyEditRun>,
    pub structurally_dirty: bool,
}

pub(crate) struct CompiledConstraintEdits {
    pub moves: Vec<RowMove>,
    pub fresh: Vec<dynamis_layout::ConstraintRuntimeRecord>,
}

impl Simulation {
    pub(crate) fn compile_body_edits(&self) -> CompiledBodyEdits {
        let mut map = RowMap::new(self.slots);
        let mut fresh: Vec<BodyStateRecord> = Vec::new();
        let mut journaled: Vec<Vec<BodyCommand>> = Vec::new();
        for command in &self.bodies.commands {
            match command {
                BodyCommand::Add { row, state } => {
                    map.add(*row, fresh.len() as u32);
                    fresh.push(*state);
                }
                BodyCommand::Remove { hole, tail } => map.remove(*hole, *tail),
                BodyCommand::Swap { first, second } => map.swap(*first, *second),
                _ => {
                    let Some(id) =
                        identity_index(map.identity_of(command.row()), self.slots as u32)
                    else {
                        continue;
                    };
                    if journaled.len() <= id as usize {
                        journaled.resize_with(id as usize + 1, Vec::new);
                    }
                    journaled[id as usize].push(*command);
                }
            }
        }
        let moves = map.moves(self.bodies.alive.len());
        let mut edits = Vec::new();
        let mut runs = Vec::new();
        for (row, identity) in map.contents[..self.bodies.alive.len()].iter().enumerate() {
            let Some(id) = identity_index(*identity, self.slots as u32) else {
                continue;
            };
            let Some(commands) = journaled.get(id as usize) else {
                continue;
            };
            let first = edits.len();
            for command in commands {
                if let Some(edit) = command.edit() {
                    edits.push(edit);
                }
            }
            if edits.len() > first {
                runs.push(BodyEditRun::new(row, first, edits.len() - first));
            }
        }
        let structurally_dirty = moves.iter().any(RowMove::is_structural);
        CompiledBodyEdits {
            moves,
            fresh,
            edits,
            runs,
            structurally_dirty,
        }
    }

    pub(crate) fn compile_constraint_edits(&self) -> CompiledConstraintEdits {
        let mut map = RowMap::new(self.slots);
        let mut fresh: Vec<dynamis_layout::ConstraintRuntimeRecord> = Vec::new();
        for command in &self.constraints.commands {
            match command.kind {
                dynamis_layout::COMMAND_CONSTRAINT_ADD => {
                    map.add(command.slot, fresh.len() as u32);
                    fresh.push(dynamis_layout::ConstraintRuntimeRecord::fresh(
                        command.constraint_id,
                        command.generation,
                    ));
                }
                _ => map.swap(command.slot, command.tail),
            }
        }
        CompiledConstraintEdits {
            moves: map.moves(self.constraints.alive.len()),
            fresh,
        }
    }
}
