mod capacity;
mod ccd;
mod commands;
mod commit;
mod entries;
mod integrate;
mod islands;
mod narrowphase;
mod sleep;
mod solver;
mod sort;
mod streams;

use dynamis_engine::Resources;
use dynamis_scene::Frame;
const IDENTITY: &[&str] = &[include_str!("../shaders/identity.wgsl")];
const CONTACT: &[&str] = &[
    include_str!("../shaders/identity.wgsl"),
    include_str!("../shaders/events.wgsl"),
];

use commands::Commands;
use commit::Commit;
use dynamis_engine::{Engine, domain_passes};
use dynamis_gpu::GpuContext;
use dynamis_sort::RadixSort;
use entries::Entries;
use integrate::Integrate;
use islands::Islands;
use narrowphase::Narrowphase;
use sleep::Sleep;
use solver::Solver;

pub use capacity::Capacity;
pub use ccd::{Ccd, CcdPasses};
pub use sort::{keyed, lanes, lanes_dual};
pub use streams::{
    DOMAIN, RigidDemand, RigidStream, RigidStreams, block_capacity, event_capacity,
    resting_capacity, sort_capacity,
};

domain_passes!(
    RigidPasses,
    "rigid",
    commands => "commands",
    integrate => "integrate",
    entries => "entries",
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

fn island_rounds(streams: &impl Resources) -> u32 {
    dynamis_scene::body_row_count(streams).max(2).ilog2() + 1
}

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

    pub fn encode_commands(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        streams: &impl Resources,
        frame: &Frame,
    ) {
        let mut commands = engine.open(encoder, self.passes.commands);
        self.commands.record(&mut commands, streams, frame);
        drop(commands);
    }

    pub fn encode_prepare(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        streams: &impl Resources,
        frame: &Frame,
    ) {
        let mut integrate = engine.open(encoder, self.passes.integrate);
        self.integrate
            .record(&mut integrate, streams, frame, &self.sort);
        drop(integrate);
    }

    pub fn encode_entries(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        streams: &impl Resources,
        frame: &Frame,
    ) {
        let mut entries = engine.open(encoder, self.passes.entries);
        self.entries.record(&mut entries, streams, frame);
        drop(entries);
    }

    pub fn encode_contacts(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        streams: &impl Resources,
        frame: &Frame,
    ) {
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

    pub fn encode_resolution(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        streams: &impl Resources,
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

    pub fn record_query_commands(
        &self,
        recorder: &mut dynamis_gpu::ComputeRecorder,
        streams: &impl Resources,
        frame: &Frame,
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
        frame: &Frame,
    ) {
        self.commit.record_query(recorder, streams, frame);
    }
}
