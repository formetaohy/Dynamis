mod capacity;
mod ccd;
mod commands;
mod commit;
mod entries;
mod integrate;
mod islands;
mod narrowphase;
mod solver;
mod sort;
mod streams;

use dynamis_pass::Resources;
use dynamis_state::StepFrame;
const IDENTITY: &[&str] = &[include_str!("../shaders/identity.wgsl")];
const CONTACT: &[&str] = &[
    include_str!("../shaders/identity.wgsl"),
    include_str!("../shaders/events.wgsl"),
];

use commands::Commands;
use commit::Commit;
use dynamis_gpu::GpuContext;
use dynamis_pass::{Phase, Schedule, domain_passes};
use dynamis_sort::RadixSort;
use entries::Entries;
use integrate::Integrate;
use islands::Islands;
use islands::Sleep;
use narrowphase::Narrowphase;
use solver::Solver;

pub use capacity::Capacity;
pub use ccd::{Ccd, CcdPasses};
pub use streams::{
    DOMAIN, RigidDemand, RigidStream, RigidStreams, block_capacity, event_capacity,
    resting_capacity, sort_capacity,
};

domain_passes!(
    RigidPasses,
    "rigid",
    commands: Phase::Commands => "commands",
    integrate: Phase::Prepare => "integrate",
    entries: Phase::Entries => "entries",
    narrowphase: Phase::Contacts => "narrowphase",
    islands: Phase::Contacts => "islands",
    wake: Phase::Contacts => "wake",
    solver_prepare: Phase::Contacts => "solver_prepare",
    solver: Phase::Contacts => "solver",
    advance: Phase::Contacts => "advance",
);

domain_passes!(
    RigidResolutionPasses,
    "rigid resolution",
    solver_position: Phase::Project => "solver_position",
    sleep: Phase::Project => "sleep",
    commit: Phase::Project => "commit",
    resting_gather: Phase::Project => "resting_gather",
    resting_index: Phase::Project => "resting_index",
);

pub struct Rigid {
    passes: RigidPasses,
    resolution: RigidResolutionPasses,
    commands: Commands,
    integrate: Integrate,
    entries: Entries,
    narrowphase: Narrowphase,
    islands: Islands,
    sleep: Sleep,
    solver: Solver,
    commit: Commit,
    sort: RadixSort,
}

impl Rigid {
    pub fn new(
        context: &GpuContext,
        streams: &impl Resources,
        passes: RigidPasses,
        resolution: RigidResolutionPasses,
    ) -> Self {
        Self {
            passes,
            resolution,
            commands: Commands::build(context, streams),
            integrate: Integrate::build(context, streams),
            entries: Entries::build(context, streams),
            narrowphase: Narrowphase::build(context, streams),
            islands: Islands::build(context, streams),
            sleep: Sleep::build(context, streams),
            solver: Solver::build(context, streams),
            commit: Commit::build(context, streams),
            sort: RadixSort::new(context, "rigid sort", sort_capacity(streams)),
        }
    }

    pub fn record(
        &self,
        phase: Phase,
        schedule: &mut Schedule,
        encoder: &mut wgpu::CommandEncoder,
        streams: &impl Resources,
        frame: &StepFrame,
    ) {
        let simulating = frame.simulating();
        match phase {
            Phase::Commands => {
                let mut commands = schedule.open(encoder, self.passes.commands);
                self.commands.record(&mut commands, streams, frame);
                drop(commands);
            }
            Phase::Prepare if simulating => {
                let mut integrate = schedule.open(encoder, self.passes.integrate);
                self.integrate
                    .record(&mut integrate, streams, frame, &self.sort);
                drop(integrate);
            }
            Phase::Entries if simulating => {
                let mut entries = schedule.open(encoder, self.passes.entries);
                self.entries.record(&mut entries, streams, frame);
                drop(entries);
            }
            Phase::Contacts if simulating => {
                let mut narrowphase = schedule.open(encoder, self.passes.narrowphase);
                self.narrowphase
                    .record(&mut narrowphase, streams, &self.sort);
                drop(narrowphase);

                let mut islands = schedule.open(encoder, self.passes.islands);
                self.islands.record(&mut islands, streams, frame);
                drop(islands);

                let mut wake = schedule.open(encoder, self.passes.wake);
                self.islands.record_wake(&mut wake, streams, frame);
                drop(wake);

                let mut prepare = schedule.open(encoder, self.passes.solver_prepare);
                self.solver.record_prepare(&mut prepare, streams, frame);
                drop(prepare);

                let mut solver = schedule.open(encoder, self.passes.solver);
                self.solver.record(&mut solver, streams, frame, &self.sort);
                drop(solver);

                let mut advance = schedule.open(encoder, self.passes.advance);
                self.integrate.record_advance(&mut advance, streams, frame);
                drop(advance);
            }
            Phase::Project => {
                if simulating {
                    let mut position = schedule.open(encoder, self.resolution.solver_position);
                    self.solver
                        .record_position_iterations(&mut position, streams, frame);
                    drop(position);

                    let mut sleep = schedule.open(encoder, self.resolution.sleep);
                    self.sleep.record(&mut sleep, streams, frame);
                    drop(sleep);
                }
                let mut commit = schedule.open(encoder, self.resolution.commit);
                self.commit.record(&mut commit, streams, frame);
                drop(commit);
                if simulating {
                    let mut gather = schedule.open(encoder, self.resolution.resting_gather);
                    self.commit.record_gather(&mut gather, streams);
                    drop(gather);
                    let mut index = schedule.open(encoder, self.resolution.resting_index);
                    self.commit.record_index(&mut index, streams, &self.sort);
                    drop(index);
                }
            }
            _ => {}
        }
    }

    pub fn record_query_commands(
        &self,
        recorder: &mut dynamis_gpu::ComputeRecorder,
        streams: &impl Resources,
        frame: &StepFrame,
    ) {
        self.commands.record_moves(recorder, streams, frame);
        self.commands.record_edits(recorder, streams, frame);
        self.commands.reset(recorder, streams);
        self.integrate.record_broadphase(recorder, streams, frame);
        self.entries.record(recorder, streams, frame);
    }

    pub fn record_query_flush(
        &self,
        recorder: &mut dynamis_gpu::ComputeRecorder,
        streams: &impl Resources,
        frame: &StepFrame,
    ) {
        self.commit.record_query(recorder, streams, frame);
    }
}
