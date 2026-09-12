mod broadphase;
mod capacity;
mod commands;
mod commit;
mod grid;
mod integrate;
mod islands;
mod narrowphase;
mod sleep;
mod solver;
mod streams;

use crate::dynamics::Frame;
use crate::dynamics::engine::{Engine, domain_passes};
use crate::dynamics::streams::Streams;
use broadphase::Broadphase;
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

pub(crate) use capacity::Capacity;
pub(crate) use streams::{DOMAIN, RigidDemand, RigidStream, RigidStreams};

domain_passes!(
    RigidPasses,
    "rigid",
    commands => "commands",
    integrate => "integrate",
    grid => "grid",
    broadphase => "broadphase",
    narrowphase => "narrowphase",
    islands => "islands",
    solver_prepare => "solver_prepare",
    solver => "solver",
    advance => "advance",
);

domain_passes!(
    RigidResolutionPasses,
    "rigid resolution",
    solver_position => "solver_position",
    sleep => "sleep",
    commit => "commit",
    resting_gather => "resting_gather",
    resting_index => "resting_index",
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

fn island_rounds(streams: &Streams) -> u32 {
    streams.scene.body_row_count().max(2).ilog2() + 1
}

pub(crate) struct Rigid {
    passes: RigidPasses,
    resolution: RigidResolutionPasses,
    commands: Commands,
    integrate: Integrate,
    grid: Grid,
    broadphase: Broadphase,
    narrowphase: Narrowphase,
    islands: Islands,
    sleep: Sleep,
    solver: Solver,
    commit: Commit,
    sort: RadixSort,
}

impl Rigid {
    pub(crate) fn new(
        context: &GpuContext,
        streams: &Streams,
        passes: RigidPasses,
        resolution: RigidResolutionPasses,
    ) -> Self {
        let sort = RadixSort::new(context, "world sort", streams.sort_capacity());
        Self {
            passes,
            resolution,
            commands: Commands::build(context, streams),
            integrate: Integrate::build(context, streams),
            grid: Grid::build(context, streams),
            broadphase: Broadphase::build(context, streams),
            narrowphase: Narrowphase::build(context, streams),
            islands: Islands::build(context, streams),
            sleep: Sleep::build(context, streams),
            solver: Solver::build(context, streams),
            commit: Commit::build(context, streams),
            sort,
        }
    }

    pub(crate) fn encode(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        streams: &Streams,
        frame: &Frame,
        idle: bool,
    ) {
        let mut commands = engine.open(encoder, self.passes.commands);
        self.commands.record(&mut commands, streams, frame);
        drop(commands);

        if idle {
            return;
        }
        let mut integrate = engine.open(encoder, self.passes.integrate);
        self.integrate
            .record(&mut integrate, streams, frame, &self.sort);
        drop(integrate);

        let mut grid = engine.open(encoder, self.passes.grid);
        self.grid.record(&mut grid, streams, frame);
        drop(grid);

        let mut broadphase = engine.open(encoder, self.passes.broadphase);
        self.broadphase
            .record(&mut broadphase, streams, frame, &self.sort);
        drop(broadphase);

        let mut narrowphase = engine.open(encoder, self.passes.narrowphase);
        self.narrowphase
            .record(&mut narrowphase, streams, &self.sort);
        drop(narrowphase);

        let mut islands = engine.open(encoder, self.passes.islands);
        self.islands
            .record(&mut islands, streams, frame, island_rounds(streams));
        drop(islands);

        let mut prepare = engine.open(encoder, self.passes.solver_prepare);
        self.solver.record_prepare(&mut prepare, streams, frame);
        drop(prepare);

        let mut solver = engine.open(encoder, self.passes.solver);
        self.solver.record(&mut solver, streams, frame, &self.sort);
        drop(solver);

        let mut advance = engine.open(encoder, self.passes.advance);
        self.integrate.record_advance(&mut advance, streams, frame);
        drop(advance);
    }

    pub(crate) fn encode_resolution(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        streams: &Streams,
        frame: &Frame,
        idle: bool,
    ) {
        if !idle {
            let mut position = engine.open(encoder, self.resolution.solver_position);
            self.solver
                .record_position_iterations(&mut position, streams, frame);
            drop(position);

            let mut sleep = engine.open(encoder, self.resolution.sleep);
            self.sleep.record(&mut sleep, streams, frame);
            drop(sleep);
        }
        let mut commit = engine.open(encoder, self.resolution.commit);
        self.commit.record(&mut commit, streams, frame);
        drop(commit);
        if !idle {
            let mut gather = engine.open(encoder, self.resolution.resting_gather);
            self.commit.record_gather(&mut gather, streams);
            drop(gather);
            let mut index = engine.open(encoder, self.resolution.resting_index);
            self.commit.record_index(&mut index, streams, &self.sort);
            drop(index);
        }
    }

    pub(crate) fn encode_queries(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        streams: &Streams,
        frame: &Frame,
    ) {
        let mut commands =
            dynamis_gpu::ComputeRecorder::begin(encoder, "query commands", engine.per_row());
        self.commands.record_moves(&mut commands, streams, frame);
        self.commands.record_edits(&mut commands, streams, frame);
        self.commands.reset(&mut commands, streams);
        self.integrate
            .record_broadphase(&mut commands, streams, frame);
        self.grid.record(&mut commands, streams, frame);
        drop(commands);

        let mut flush =
            dynamis_gpu::ComputeRecorder::begin(encoder, "query flush", engine.per_row());
        let channels = streams.sort_lanes(
            streams.scene.counter(COUNTER_ENTRIES),
            RigidStream::GridEntryKeys.whole(),
            RigidStream::GridEntryColliders.whole(),
        );
        self.sort
            .sort(&mut flush, &channels, 4, 0, streams.rigid.entry_capacity());
        self.commit.record_query(&mut flush, streams, frame);
    }
}
