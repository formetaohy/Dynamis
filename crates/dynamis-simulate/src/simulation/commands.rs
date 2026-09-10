use super::Simulation;
use super::rows::{RowJournal, RowMap};
use dynamis_layout::{BodyEditRecord, BodyEditRun, BodyStateRecord, RowMoveRecord};

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

pub(crate) struct CompiledBodyCommands {
    pub(crate) moves: Vec<RowMoveRecord>,
    pub(crate) fresh: Vec<BodyStateRecord>,
    pub(crate) edits: Vec<BodyEditRecord>,
    pub(crate) runs: Vec<BodyEditRun>,
}

pub(crate) struct CompiledConstraintCommands {
    pub(crate) moves: Vec<RowMoveRecord>,
    pub(crate) fresh: Vec<dynamis_layout::ConstraintRuntimeRecord>,
}

impl Simulation {
    pub(crate) fn compile_body_commands(&self) -> CompiledBodyCommands {
        let mut map = RowMap::new();
        let mut journal: RowJournal<BodyCommand> = RowJournal::new();
        let mut fresh: Vec<BodyStateRecord> = Vec::new();
        for command in &self.bodies.commands {
            match command {
                BodyCommand::Add { row, state } => {
                    map.add(*row, fresh.len() as u32);
                    fresh.push(*state);
                }
                BodyCommand::Remove { hole, tail } => map.remove(*hole, *tail),
                BodyCommand::Swap { first, second } => map.swap(*first, *second),
                _ => journal.push(map.identity_of(command.row()), *command),
            }
        }
        let rows = self.bodies.alive.len() as u32;
        let moves = map.moves(rows);
        let mut edits = Vec::new();
        let mut runs = Vec::new();
        for (identity, commands) in journal.iter() {
            let Some(row) = map.row_of(identity) else {
                continue;
            };
            assert!(row < rows, "an edit run escapes the live rows");
            let first = edits.len();
            for command in commands {
                if let Some(edit) = command.edit() {
                    edits.push(edit);
                }
            }
            if edits.len() > first {
                runs.push(BodyEditRun::new(row as usize, first, edits.len() - first));
            }
        }
        CompiledBodyCommands {
            moves,
            fresh,
            edits,
            runs,
        }
    }

    pub(crate) fn compile_constraint_commands(&self) -> CompiledConstraintCommands {
        let mut map = RowMap::new();
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
        CompiledConstraintCommands {
            moves: map.moves(self.constraints.alive.len() as u32),
            fresh,
        }
    }
}
