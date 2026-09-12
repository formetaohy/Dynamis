pub(crate) mod buffers;
pub(crate) mod capacity;

mod broadphase;
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
mod stage;
#[cfg(feature = "profile")]
mod timing;

use buffers::WorldBuffers;
#[cfg(feature = "profile")]
use dynamis_gpu::GpuTimer;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{COUNTER_ENTRIES, StepParamsRecord};
use dynamis_sort::RadixSort;
use wgpu::CommandEncoder;

use broadphase::Broadphase;
use ccd::Ccd;
use commands::Commands;
use commit::Commit;
use grid::Grid;
use integrate::Integrate;
use islands::Islands;
use narrowphase::Narrowphase;
use sleep::Sleep;
use solver::Solver;

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
    SolverPosition => "solver_position",
    Sleep => "sleep",
    Commit => "commit",
    RestingGather => "resting_gather",
    RestingIndex => "resting_index",
);

pub(crate) struct Frame {
    pub(crate) params: StepParamsRecord,
    pub(crate) query_count: u32,
    pub(crate) island_rounds: u32,
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
    sort: RadixSort,
    #[cfg(feature = "profile")]
    timer: Option<GpuTimer>,
}

impl Pipeline {
    pub(crate) fn new(context: &GpuContext, buffers: &WorldBuffers) -> Self {
        let per_row = context.workgroups_per_row();
        let sort = RadixSort::new(context, "world sort", buffers.sort_capacity());
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
        frame: &Frame,
        idle: bool,
    ) {
        let mut commands = self.open(encoder, Pass::Commands);
        self.commands.record(&mut commands, buffers, frame);
        drop(commands);

        if !idle {
            let mut integrate = self.open(encoder, Pass::Integrate);
            self.integrate
                .record(&mut integrate, buffers, frame, &self.sort);
            drop(integrate);

            let mut grid = self.open(encoder, Pass::Grid);
            self.grid.record(&mut grid, buffers, frame);
            drop(grid);

            let mut broadphase = self.open(encoder, Pass::Broadphase);
            self.broadphase
                .record(&mut broadphase, buffers, frame, &self.sort);
            drop(broadphase);

            let mut narrowphase = self.open(encoder, Pass::Narrowphase);
            self.narrowphase
                .record(&mut narrowphase, buffers, frame, &self.sort);
            drop(narrowphase);

            let mut islands = self.open(encoder, Pass::Islands);
            self.islands.record(&mut islands, buffers, frame);
            drop(islands);

            let mut prepare = self.open(encoder, Pass::SolverPrepare);
            self.solver.record_prepare(&mut prepare, buffers, frame);
            drop(prepare);

            let mut solver = self.open(encoder, Pass::Solver);
            self.solver.record(&mut solver, buffers, frame, &self.sort);
            drop(solver);

            let mut impact = self.open(encoder, Pass::Impact);
            self.integrate.record_advance(&mut impact, buffers, frame);
            self.ccd.record(&mut impact, buffers, frame);
            drop(impact);

            let mut position = self.open(encoder, Pass::SolverPosition);
            self.solver
                .record_position_iterations(&mut position, buffers, frame);
            drop(position);

            let mut sleep = self.open(encoder, Pass::Sleep);
            self.sleep.record(&mut sleep, buffers, frame);
            drop(sleep);
        }
        let mut commit = self.open(encoder, Pass::Commit);
        self.commit.record(&mut commit, buffers, frame);
        drop(commit);
        if !idle {
            let mut gather = self.open(encoder, Pass::RestingGather);
            self.commit.record_gather(&mut gather, buffers, frame);
            drop(gather);
            let mut index = self.open(encoder, Pass::RestingIndex);
            self.commit.record_index(&mut index, buffers, &self.sort);
            drop(index);
        }
    }

    pub(crate) fn encode_queries(
        &self,
        encoder: &mut CommandEncoder,
        buffers: &WorldBuffers,
        frame: &Frame,
    ) {
        let mut commands = ComputeRecorder::begin(encoder, "query commands", self.per_row);
        self.commands.record_moves(&mut commands, buffers, frame);
        self.commands.record_edits(&mut commands, buffers, frame);
        self.commands.reset(&mut commands);
        self.integrate
            .record_broadphase(&mut commands, buffers, frame);
        self.grid.record(&mut commands, buffers, frame);
        drop(commands);

        let mut flush = ComputeRecorder::begin(encoder, "query flush", self.per_row);
        let channels = buffers.sort_lanes(
            buffers.counter(COUNTER_ENTRIES),
            &buffers.grid_entry_keys,
            &buffers.grid_entry_colliders,
        );
        self.sort
            .sort(&mut flush, &channels, 4, 0, buffers.entry_capacity());
        self.commit.record_query(&mut flush, frame);
    }
}
