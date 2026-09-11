mod broadphase;
mod ccd;
mod commands;
mod commit;
mod dispatch;
mod grid;
mod integrate;
mod islands;
mod narrowphase;
mod shader;
mod sleep;
mod solver;
mod stage;
#[cfg(feature = "profile")]
mod timing;

use crate::buffers::WorldBuffers;
use crate::capacity::Reservation;
#[cfg(feature = "profile")]
use dynamis_gpu::GpuTimer;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::COUNTER_ENTRIES;
use dynamis_sort::RadixSort;
use wgpu::CommandEncoder;

use broadphase::Broadphase;
use ccd::Ccd;
use commands::Commands;
use commit::Commit;
use dispatch::{Dispatch, SORT_ENTRIES};
use grid::Grid;
use integrate::Integrate;
use islands::Islands;
use narrowphase::Narrowphase;
use sleep::Sleep;
use solver::Solver;

pub(crate) use dispatch::DISPATCH_SLOTS;
use dispatch::{
    BROADPHASE_BATCH, COMMANDS_BATCH, COMMIT_BATCH, GRID_BATCH, ISLANDS_BATCH,
    RESTING_GATHER_BATCH, RESTING_SORT_BATCH, SOLVER_BATCH,
};

macro_rules! passes {
    ($($variant:ident => $label:literal),+ $(,)?) => {
        #[derive(Clone, Copy)]
        enum Pass {
            $($variant,)+
        }

        impl Pass {
            const LABELS: &'static [&'static str] = &[$($label,)+];

            const fn label(self) -> &'static str {
                Self::LABELS[self.index()]
            }

            const fn index(self) -> usize {
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
    Sleep => "sleep",
    Commit => "commit",
    RestingGather => "resting_gather",
    RestingIndex => "resting_index",
);

pub(crate) struct FrameParams {
    pub(crate) dynamic_count: u32,
    pub(crate) body_count: u32,
    pub(crate) solve_iterations: u32,
    pub(crate) island_rounds: u32,
    pub(crate) query_count: u32,
    pub(crate) constraint_count: u32,
    pub(crate) edit_run_count: u32,
    pub(crate) body_move_count: u32,
    pub(crate) constraint_move_count: u32,
}

pub(crate) struct Pipeline {
    per_row: u32,
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
    dispatch: Dispatch,
    sort: RadixSort,
    #[cfg(feature = "profile")]
    timer: Option<GpuTimer>,
}

impl Pipeline {
    pub(crate) fn new(context: &GpuContext, buffers: &WorldBuffers, plan: &Reservation) -> Self {
        let per_row = context.workgroups_per_row();
        let sort = RadixSort::new(context, "sim sort", plan.sort());
        #[cfg(feature = "profile")]
        let timer = context.supports_pass_timing().then(|| {
            GpuTimer::new(
                context.device(),
                Pass::LABELS,
                context.timestamp_period_ns(),
                "dynamis step",
            )
        });
        Self {
            per_row,
            commands: Commands::build(context, buffers, per_row),
            integrate: Integrate::build(context, buffers, per_row),
            grid: Grid::build(context, buffers, per_row),
            broadphase: Broadphase::build(context, buffers, per_row),
            narrowphase: Narrowphase::build(context, buffers, per_row),
            islands: Islands::build(context, buffers, per_row),
            sleep: Sleep::build(context, buffers, per_row),
            ccd: Ccd::build(context, buffers, per_row),
            solver: Solver::build(context, buffers, per_row),
            commit: Commit::build(context, buffers, per_row),
            dispatch: Dispatch::build(context, buffers, per_row),
            sort,
            #[cfg(feature = "profile")]
            timer,
        }
    }

    fn open<'a>(&'a self, encoder: &'a mut CommandEncoder, pass: Pass) -> ComputeRecorder<'a> {
        #[cfg(feature = "profile")]
        if let Some(timer) = &self.timer {
            return ComputeRecorder::begin_timed(
                encoder,
                pass.label(),
                Some(timer.writes(pass.index())),
                self.per_row,
            );
        }
        ComputeRecorder::begin(encoder, pass.label(), self.per_row)
    }

    pub(crate) fn encode(
        &self,
        encoder: &mut CommandEncoder,
        buffers: &WorldBuffers,
        params: &FrameParams,
        idle: bool,
    ) {
        let mut commands = self.open(encoder, Pass::Commands);
        self.commands.record(&mut commands, params);
        drop(commands);
        self.dispatch.write(encoder, COMMANDS_BATCH);

        if idle {
            self.dispatch.write(encoder, ISLANDS_BATCH);
            self.dispatch.write(encoder, COMMIT_BATCH);
        } else {
            let mut integrate = self.open(encoder, Pass::Integrate);
            self.integrate
                .record(&mut integrate, buffers, params, &self.sort);
            drop(integrate);

            let mut grid = self.open(encoder, Pass::Grid);
            self.grid.record(&mut grid, params.body_count);
            drop(grid);
            self.dispatch.write(encoder, GRID_BATCH);

            let mut broadphase = self.open(encoder, Pass::Broadphase);
            self.broadphase
                .record(&mut broadphase, buffers, params, &self.sort);
            drop(broadphase);
            self.dispatch.write(encoder, BROADPHASE_BATCH);

            let mut narrowphase = self.open(encoder, Pass::Narrowphase);
            self.narrowphase
                .record(&mut narrowphase, buffers, &self.sort);
            drop(narrowphase);
            self.dispatch.write(encoder, ISLANDS_BATCH);

            let mut islands = self.open(encoder, Pass::Islands);
            self.islands.record(&mut islands, buffers, params);
            drop(islands);

            let mut prepare = self.open(encoder, Pass::SolverPrepare);
            self.solver.record_prepare(&mut prepare, params.body_count);
            drop(prepare);
            self.dispatch.write(encoder, SOLVER_BATCH);

            let mut solver = self.open(encoder, Pass::Solver);
            self.solver.record(&mut solver, buffers, params, &self.sort);
            drop(solver);

            let mut impact = self.open(encoder, Pass::Impact);
            self.integrate
                .record_advance(&mut impact, params.dynamic_count);
            self.ccd.record(&mut impact, buffers);
            self.solver.record_position(&mut impact, buffers, params);
            drop(impact);

            let mut sleep = self.open(encoder, Pass::Sleep);
            self.sleep.record(&mut sleep, params);
            drop(sleep);
            self.dispatch.write(encoder, COMMIT_BATCH);
        }
        let mut commit = self.open(encoder, Pass::Commit);
        self.commit.record(&mut commit, buffers, params);
        drop(commit);
        if !idle {
            self.dispatch.write(encoder, RESTING_GATHER_BATCH);
            let mut gather = self.open(encoder, Pass::RestingGather);
            self.commit.record_gather(&mut gather, buffers);
            drop(gather);
            self.dispatch.write(encoder, RESTING_SORT_BATCH);
            let mut index = self.open(encoder, Pass::RestingIndex);
            self.commit.record_index(&mut index, buffers, &self.sort);
            drop(index);
        }
    }

    pub(crate) fn encode_queries(
        &self,
        encoder: &mut CommandEncoder,
        buffers: &WorldBuffers,
        frame: &FrameParams,
    ) {
        let mut commands = ComputeRecorder::begin(encoder, "query commands", self.per_row);
        self.commands.record_moves(&mut commands, frame);
        self.commands
            .record_edits(&mut commands, frame.edit_run_count);
        self.commands.reset(&mut commands);
        self.integrate
            .record_broadphase(&mut commands, frame.dynamic_count);
        self.grid.record(&mut commands, frame.body_count);
        drop(commands);
        self.dispatch.write(encoder, GRID_BATCH);

        let mut flush = ComputeRecorder::begin(encoder, "query flush", self.per_row);
        let channels = buffers.sort_lanes(
            buffers.counter(COUNTER_ENTRIES),
            &buffers.contacts.entries.cells,
            &buffers.contacts.entries.colliders,
        );
        self.sort
            .sort(&mut flush, &channels, 4, 0, &buffers.dispatch, SORT_ENTRIES);
        self.commit.record_query(&mut flush, frame.query_count);
    }
}
