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
use dynamis_sort::{RadixSort, key_words};
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

/// The step, in the order its passes read and write each other's counters.
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
    /// Whether structural row moves were compiled into the command stream.
    pub(crate) body_structural: bool,
    pub(crate) constraint_structural: bool,
    /// Whether any per-slot body edit was compiled.
    pub(crate) has_body_edits: bool,
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

    /// One step, recorded from the host-owned counts and the dispatch table alike:
    /// stages whose work the device counted run by indirect arguments, stages whose
    /// work the host knows run by direct counts, and a batch of arguments is written
    /// at every point of the step where its counters become final.
    pub(crate) fn encode(
        &self,
        encoder: &mut CommandEncoder,
        buffers: &WorldBuffers,
        params: &FrameParams,
    ) {
        let mut commands = self.open(encoder, Pass::Commands);
        self.commands.record(&mut commands, params);
        drop(commands);
        self.dispatch.write(encoder, 0);

        let mut integrate = self.open(encoder, Pass::Integrate);
        self.integrate
            .record(&mut integrate, buffers, params, &self.sort);
        drop(integrate);

        let mut grid = self.open(encoder, Pass::Grid);
        self.grid.record(&mut grid, params.body_count);
        drop(grid);
        self.dispatch.write(encoder, 1);

        let mut broadphase = self.open(encoder, Pass::Broadphase);
        self.broadphase
            .record(&mut broadphase, buffers, params, &self.sort);
        drop(broadphase);
        self.dispatch.write(encoder, 2);

        let mut narrowphase = self.open(encoder, Pass::Narrowphase);
        self.narrowphase
            .record(&mut narrowphase, buffers, &self.sort);
        drop(narrowphase);
        self.dispatch.write(encoder, 3);

        let mut islands = self.open(encoder, Pass::Islands);
        self.islands
            .record(&mut islands, buffers, params, &self.sort);
        drop(islands);

        let mut sleep = self.open(encoder, Pass::Sleep);
        self.sleep.record(&mut sleep, params);
        drop(sleep);

        let mut velocity_solve = self.open(encoder, Pass::VelocitySolve);
        self.velocity_solve
            .record(&mut velocity_solve, buffers, params);
        drop(velocity_solve);

        let mut position_solve = self.open(encoder, Pass::PositionSolve);
        self.position_solve
            .record(&mut position_solve, buffers, params);
        drop(position_solve);

        let mut tail = self.open(encoder, Pass::Tail);
        self.tail.record(&mut tail, buffers, params);
        drop(tail);
    }

    /// Refreshes the broadphase scene the queries run against, then runs them.
    ///
    /// A query outside a step must see the world as the device holds it now, so the
    /// grid is rebuilt from the current states rather than trusting the last step's
    /// cells: stale entries would let a moved body slip through a ray unanswered.
    pub(crate) fn encode_queries(
        &self,
        encoder: &mut CommandEncoder,
        buffers: &WorldBuffers,
        frame: &FrameParams,
    ) {
        let mut commands = ComputeRecorder::begin(encoder, "query commands", self.per_row);
        self.commands.record_moves(&mut commands, frame);
        self.commands.reset(&mut commands);
        self.integrate
            .record_broadphase(&mut commands, frame.dynamic_count);
        self.grid.record(&mut commands, frame.body_count);
        drop(commands);
        self.dispatch.write(encoder, 1);

        let mut flush = ComputeRecorder::begin(encoder, "query flush", self.per_row);
        let channels = sort::lanes(
            buffers,
            buffers.counter(COUNTER_ENTRIES),
            &buffers.contacts.entries.keys_lo,
            &buffers.contacts.entries.keys_hi,
            &buffers.sort.values,
        );
        self.sort.sort(
            &mut flush,
            &channels,
            key_words(buffers.collider_rows()),
            4,
            &buffers.dispatch,
            SORT_ENTRIES,
        );
        self.tail.record_query(&mut flush, frame.query_count);
    }
}
