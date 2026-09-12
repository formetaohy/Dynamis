mod broadphase;
pub(crate) mod buffers;
pub(crate) mod capacity;
mod ccd;
mod commands;
mod commit;
mod grid;
mod integrate;
mod islands;
mod narrowphase;
mod shader;
mod sleep;
mod solver;

use crate::dynamics::engine::Engine;
use broadphase::Broadphase;
use buffers::RigidBuffers;
use buffers::StreamId;
use ccd::Ccd;
use commands::Commands;
use commit::Commit;
use dynamis_gpu::GpuContext;
use dynamis_layout::{COUNTER_ENTRIES, StepParamsRecord};
use dynamis_sort::RadixSort;
use grid::Grid;
use integrate::Integrate;
use islands::Islands;
use narrowphase::Narrowphase;
use sleep::Sleep;
use solver::Solver;

macro_rules! passes {
    ($($variant:ident => $label:literal),+ $(,)?) => {
        #[derive(Clone, Copy)]
        pub(crate) enum Pass {
            $($variant,)+
        }

        impl Pass {
            pub(crate) const LABELS: &'static [&'static str] = &[$($label,)+];

            pub(crate) const fn index(self) -> usize {
                self as usize
            }
        }
    };
}

passes!(
    Commands => "commands",
    Integrate => "integrate",
    Grid => "grid",
    Broadphase => "broadphase",
    Narrowphase => "narrowphase",
    Islands => "islands",
    SolverPrepare => "solver_prepare",
    Solver => "solver",
    Impact => "impact",
    SolverPosition => "solver_position",
    Sleep => "sleep",
    Commit => "commit",
    RestingGather => "resting_gather",
    RestingIndex => "resting_index",
);

#[derive(Clone, Copy)]
pub(crate) enum Count {
    Bodies,
    Dynamic,
    Colliders,
    Constraints,
    EditRuns,
    BodyMoves,
    ConstraintMoves,
}

impl Count {
    pub(crate) const fn field(self) -> &'static str {
        match self {
            Self::Bodies => "body_count",
            Self::Dynamic => "dynamic_count",
            Self::Colliders => "collider_count",
            Self::Constraints => "constraint_count",
            Self::EditRuns => "edit_run_count",
            Self::BodyMoves => "body_move_count",
            Self::ConstraintMoves => "constraint_move_count",
        }
    }

    pub(crate) fn rows(self, params: &StepParamsRecord) -> u32 {
        match self {
            Self::Bodies => params.body_count,
            Self::Dynamic => params.dynamic_count,
            Self::Colliders => params.collider_count,
            Self::Constraints => params.constraint_count,
            Self::EditRuns => params.edit_run_count,
            Self::BodyMoves => params.body_move_count,
            Self::ConstraintMoves => params.constraint_move_count,
        }
    }
}

pub(crate) struct Frame {
    pub(crate) params: StepParamsRecord,
    pub(crate) query_count: u32,
    pub(crate) island_rounds: u32,
}

pub(crate) struct Rigid {
    commands: Commands,
    integrate: Integrate,
    grid: Grid,
    broadphase: Broadphase,
    narrowphase: Narrowphase,
    islands: Islands,
    sleep: Sleep,
    ccd: Ccd,
    solver: Solver,
    commit: Commit,
    sort: RadixSort,
}

impl Rigid {
    pub(crate) fn new(context: &GpuContext, buffers: &RigidBuffers) -> Self {
        let sort = RadixSort::new(context, "world sort", buffers.sort_capacity());
        Self {
            commands: Commands::build(context, buffers),
            integrate: Integrate::build(context, buffers),
            grid: Grid::build(context, buffers),
            broadphase: Broadphase::build(context, buffers),
            narrowphase: Narrowphase::build(context, buffers),
            islands: Islands::build(context, buffers),
            sleep: Sleep::build(context, buffers),
            ccd: Ccd::build(context, buffers),
            solver: Solver::build(context, buffers),
            commit: Commit::build(context, buffers),
            sort,
        }
    }

    pub(crate) fn encode(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        buffers: &RigidBuffers,
        frame: &Frame,
        idle: bool,
    ) {
        let mut commands = engine.open(encoder, Pass::Commands.index());
        self.commands.record(&mut commands, buffers, frame);
        drop(commands);

        if !idle {
            let mut integrate = engine.open(encoder, Pass::Integrate.index());
            self.integrate
                .record(&mut integrate, buffers, frame, &self.sort);
            drop(integrate);

            let mut grid = engine.open(encoder, Pass::Grid.index());
            self.grid.record(&mut grid, buffers, frame);
            drop(grid);

            let mut broadphase = engine.open(encoder, Pass::Broadphase.index());
            self.broadphase
                .record(&mut broadphase, buffers, frame, &self.sort);
            drop(broadphase);

            let mut narrowphase = engine.open(encoder, Pass::Narrowphase.index());
            self.narrowphase
                .record(&mut narrowphase, buffers, &self.sort);
            drop(narrowphase);

            let mut islands = engine.open(encoder, Pass::Islands.index());
            self.islands.record(&mut islands, buffers, frame);
            drop(islands);

            let mut prepare = engine.open(encoder, Pass::SolverPrepare.index());
            self.solver.record_prepare(&mut prepare, buffers, frame);
            drop(prepare);

            let mut solver = engine.open(encoder, Pass::Solver.index());
            self.solver.record(&mut solver, buffers, frame, &self.sort);
            drop(solver);

            let mut impact = engine.open(encoder, Pass::Impact.index());
            self.integrate.record_advance(&mut impact, buffers, frame);
            self.ccd.record(&mut impact, buffers, frame);
            drop(impact);

            let mut position = engine.open(encoder, Pass::SolverPosition.index());
            self.solver
                .record_position_iterations(&mut position, buffers, frame);
            drop(position);

            let mut sleep = engine.open(encoder, Pass::Sleep.index());
            self.sleep.record(&mut sleep, buffers, frame);
            drop(sleep);
        }
        let mut commit = engine.open(encoder, Pass::Commit.index());
        self.commit.record(&mut commit, buffers, frame);
        drop(commit);
        if !idle {
            let mut gather = engine.open(encoder, Pass::RestingGather.index());
            self.commit.record_gather(&mut gather, buffers);
            drop(gather);
            let mut index = engine.open(encoder, Pass::RestingIndex.index());
            self.commit.record_index(&mut index, buffers, &self.sort);
            drop(index);
        }
    }

    pub(crate) fn encode_queries(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        buffers: &RigidBuffers,
        frame: &Frame,
    ) {
        let mut commands =
            dynamis_gpu::ComputeRecorder::begin(encoder, "query commands", engine.per_row());
        self.commands.record_moves(&mut commands, buffers, frame);
        self.commands.record_edits(&mut commands, buffers, frame);
        self.commands.reset(&mut commands, buffers);
        self.integrate
            .record_broadphase(&mut commands, buffers, frame);
        self.grid.record(&mut commands, buffers, frame);
        drop(commands);

        let mut flush =
            dynamis_gpu::ComputeRecorder::begin(encoder, "query flush", engine.per_row());
        let channels = buffers.sort_lanes(
            buffers.counter(COUNTER_ENTRIES),
            StreamId::GridEntryKeys.whole(),
            StreamId::GridEntryColliders.whole(),
        );
        self.sort
            .sort(&mut flush, &channels, 4, 0, buffers.entry_capacity());
        self.commit.record_query(&mut flush, buffers, frame);
    }
}
