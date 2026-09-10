mod broadphase;
mod commands;
mod dispatch;
mod grid;
mod integrate;
mod islands;
mod narrowphase;
mod position_solve;
mod shader;
mod sleep;
mod sort;
mod stage;
mod tail;
#[cfg(feature = "profile")]
mod timing;
mod velocity_solve;

use crate::buffers::WorldBuffers;
use crate::capacity::Reservation;
#[cfg(feature = "profile")]
use dynamis_gpu::GpuTimer;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::COUNTER_ENTRIES;
use dynamis_sort::RadixSort;
use wgpu::CommandEncoder;

use broadphase::Broadphase;
use commands::Commands;
use dispatch::{Dispatch, SORT_ENTRIES};
use grid::Grid;
use integrate::Integrate;
use islands::Islands;
use narrowphase::Narrowphase;
use position_solve::PositionSolve;
use sleep::Sleep;
use tail::Tail;
use velocity_solve::VelocitySolve;

pub(crate) use dispatch::DISPATCH_SLOTS;
use dispatch::{
    BROADPHASE_BATCH, COMMANDS_BATCH, GRID_BATCH, ISLANDS_BATCH, RESTING_GATHER_BATCH,
    RESTING_SORT_BATCH, TAIL_BATCH,
};

enum Pass {
    Commands,
    Integrate,
    Grid,
    Broadphase,
    Narrowphase,
    Islands,
    Sleep,
    VelocitySolve,
    PositionSolve,
    Tail,
    RestingGather,
    RestingIndex,
}

impl Pass {
    const LABELS: &[&str] = &[
        "commands",
        "integrate",
        "grid",
        "broadphase",
        "narrowphase",
        "islands",
        "sleep",
        "velocity_solve",
        "position_solve",
        "tail",
        "resting_gather",
        "resting_index",
    ];

    const fn index(self) -> usize {
        self as usize
    }
}

pub(crate) struct FrameParams {
    pub(crate) dynamic_count: u32,
    pub(crate) body_count: u32,
    pub(crate) solve_iterations: u32,
    pub(crate) position_iterations: u32,
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
    velocity_solve: VelocitySolve,
    position_solve: PositionSolve,
    tail: Tail,
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
            velocity_solve: VelocitySolve::build(context, buffers, per_row),
            position_solve: PositionSolve::build(context, buffers, per_row),
            tail: Tail::build(context, buffers, per_row),
            dispatch: Dispatch::build(context, buffers, per_row),
            sort,
            #[cfg(feature = "profile")]
            timer,
        }
    }

    fn open<'a>(&'a self, encoder: &'a mut CommandEncoder, pass: Pass) -> ComputeRecorder<'a> {
        let index = pass.index();
        #[cfg(feature = "profile")]
        if let Some(timer) = &self.timer {
            return ComputeRecorder::begin_timed(
                encoder,
                Pass::LABELS[index],
                Some(timer.writes(index)),
                self.per_row,
            );
        }
        ComputeRecorder::begin(encoder, Pass::LABELS[index], self.per_row)
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
            self.dispatch.write(encoder, TAIL_BATCH);
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
            self.islands
                .record(&mut islands, buffers, params, &self.sort);
            drop(islands);

            let mut sleep = self.open(encoder, Pass::Sleep);
            self.sleep.record(&mut sleep, params);
            drop(sleep);
            self.dispatch.write(encoder, TAIL_BATCH);

            let mut velocity_solve = self.open(encoder, Pass::VelocitySolve);
            self.velocity_solve
                .record(&mut velocity_solve, buffers, params);
            drop(velocity_solve);

            let mut position_solve = self.open(encoder, Pass::PositionSolve);
            self.position_solve
                .record(&mut position_solve, buffers, params);
            drop(position_solve);
        }
        let mut tail = self.open(encoder, Pass::Tail);
        self.tail.record(&mut tail, buffers, params);
        drop(tail);
        if !idle {
            self.dispatch.write(encoder, RESTING_GATHER_BATCH);
            let mut gather = self.open(encoder, Pass::RestingGather);
            self.tail.record_gather(&mut gather, buffers);
            drop(gather);
            self.dispatch.write(encoder, RESTING_SORT_BATCH);
            let mut index = self.open(encoder, Pass::RestingIndex);
            self.tail.record_index(&mut index, buffers, &self.sort);
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
        self.tail.record_query(&mut flush, frame.query_count);
    }
}
