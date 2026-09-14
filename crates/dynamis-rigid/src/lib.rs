mod capacity;
mod ccd;
mod commands;
mod commit;
mod domain;
mod entries;
mod integrate;
mod islands;
mod live;
mod narrowphase;
mod queries;
mod solver;
mod sort;
mod streams;

use dynamis_abi::{FrameCounts, StepParamsRecord};
use dynamis_gpu::Resources;

const IDENTITY: &[&str] = &[include_str!("../shaders/identity.wgsl")];
const CONTACT: &[&str] = &[
    include_str!("../shaders/identity.wgsl"),
    include_str!("../shaders/events.wgsl"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RigidShape {
    pub island_rounds: u32,
    pub body_row_words: u32,
    pub body_id_words: u32,
    pub collider_slot_words: u32,
}

impl RigidShape {
    pub fn of(counts: &FrameCounts) -> Self {
        Self {
            island_rounds: propagation_rounds(counts.dynamic_bodies),
            body_row_words: dynamis_sort::key_words(counts.bodies.max(1)),
            body_id_words: dynamis_sort::key_words(counts.body_ids.max(1)),
            collider_slot_words: dynamis_sort::key_words(counts.colliders.max(1)),
        }
    }
}

fn propagation_rounds(bodies: u32) -> u32 {
    bodies.max(2).ilog2() + 1
}

#[derive(Clone, Copy, Debug)]
pub struct RigidFrame {
    pub params: StepParamsRecord,
    pub shape: RigidShape,
    pub query_count: u32,
    pub observed_count: u32,
    pub ccd: bool,
}

use commands::Commands;
use commit::Commit;
use dynamis_gpu::GpuContext;
use dynamis_pass::{Execution, Schedule, domain_passes};
use dynamis_sort::RadixSort;
use entries::Entries;
use integrate::Integrate;
use islands::Islands;
use islands::Sleep;
use live::Live;
use narrowphase::Narrowphase;
use queries::Queries;
use solver::Solver;

pub use capacity::{Capacity, RigidCapacity, RigidInputs, capacity};
pub use ccd::{Ccd, CcdPasses};
pub use domain::{RigidDomain, RigidDomainPasses, RigidDomainRuntime, RigidWork};
pub use streams::{RigidDemand, RigidStream, RigidStreams, event_capacity, sort_capacity};

domain_passes!(
    RigidPasses,
    commands => Execution::ALWAYS => &[],
    prepare => Execution::INDEXING.and(Execution::STEP) => &["commands"],
    query_aabbs => Execution::QUERY => &["commands"],
    entries => Execution::INDEXING => &["prepare", "query_aabbs", "soft_bounds"],
    narrowphase => Execution::AWAKE => &["broadphase"],
    islands => Execution::AWAKE => &["narrowphase"],
    wake => Execution::AWAKE => &["islands"],
    live => Execution::AWAKE => &["wake"],
    solver_prepare => Execution::AWAKE => &["live"],
    substeps => Execution::AWAKE => &["solver_prepare"],
);

domain_passes!(
    RigidResolutionPasses,
    sleep => Execution::AWAKE => &["soft_apply"],
    commit => Execution::STEP => &["sleep"],
    resting_gather => Execution::AWAKE => &["commit"],
    resting_index => Execution::AWAKE => &["resting_gather"],
    query => Execution::ALWAYS => &["broadphase", "commit"],
);

pub struct Rigid {
    passes: RigidPasses,
    resolution: RigidResolutionPasses,
    commands: Commands,
    integrate: Integrate,
    entries: Entries,
    narrowphase: Narrowphase,
    queries: Queries,
    islands: Islands,
    sleep: Sleep,
    live: Live,
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
            queries: Queries::build(context, streams),
            islands: Islands::build(context, streams),
            sleep: Sleep::build(context, streams),
            live: Live::build(context, streams),
            solver: Solver::build(context, streams),
            commit: Commit::build(context, streams),
            sort: RadixSort::new(context, "rigid sort", sort_capacity(streams)),
        }
    }

    pub fn record(
        &self,
        pass: u32,
        schedule: &mut Schedule,
        encoder: &mut wgpu::CommandEncoder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        if pass == self.passes.commands {
            let mut commands = schedule.open(encoder, pass);
            self.commands.record(&mut commands, streams, frame);
            drop(commands);
        } else if pass == self.passes.prepare {
            let mut prepare = schedule.open(encoder, pass);
            self.integrate
                .record(&mut prepare, streams, frame, &self.sort);
            drop(prepare);
        } else if pass == self.passes.query_aabbs {
            let mut aabbs = schedule.open(encoder, pass);
            self.integrate.record_broadphase(&mut aabbs, streams, frame);
            drop(aabbs);
        } else if pass == self.passes.entries {
            let mut entries = schedule.open(encoder, pass);
            self.entries.record(&mut entries, streams, frame);
            drop(entries);
        } else if pass == self.passes.narrowphase {
            let mut narrowphase = schedule.open(encoder, pass);
            self.narrowphase
                .record(&mut narrowphase, streams, frame, &self.sort);
            drop(narrowphase);
        } else if pass == self.passes.islands {
            let mut islands = schedule.open(encoder, pass);
            self.islands.record(&mut islands, streams, frame);
            drop(islands);
        } else if pass == self.passes.wake {
            let mut wake = schedule.open(encoder, pass);
            self.islands.record_wake(&mut wake, streams, frame);
            drop(wake);
        } else if pass == self.passes.live {
            let mut live = schedule.open(encoder, pass);
            self.live.record(&mut live, streams, frame);
            drop(live);
        } else if pass == self.passes.solver_prepare {
            let mut prepare = schedule.open(encoder, pass);
            self.solver.record_prepare(&mut prepare, streams, frame);
            drop(prepare);
        } else if pass == self.passes.substeps {
            let mut substeps = schedule.open(encoder, pass);
            self.solver.record_topology(&mut substeps, streams);
            for substep in 0..frame.params.substeps {
                self.integrate.record_substep(&mut substeps, streams);
                if substep == 0 {
                    self.solver.record_warm(&mut substeps, streams);
                }
                self.solver.record_iterations(&mut substeps, streams, frame);
                self.integrate
                    .record_substep_advance(&mut substeps, streams);
                self.solver
                    .record_position_iterations(&mut substeps, streams, frame);
            }
            drop(substeps);
        } else if pass == self.resolution.sleep {
            let mut sleep = schedule.open(encoder, pass);
            self.sleep.record(&mut sleep, streams, frame);
            drop(sleep);
        } else if pass == self.resolution.commit {
            let mut commit = schedule.open(encoder, pass);
            self.commit.record(&mut commit, streams, frame);
            drop(commit);
        } else if pass == self.resolution.resting_gather {
            let mut gather = schedule.open(encoder, pass);
            self.commit.record_gather(&mut gather, streams);
            drop(gather);
        } else if pass == self.resolution.resting_index {
            let mut index = schedule.open(encoder, pass);
            self.commit
                .record_index(&mut index, streams, frame, &self.sort);
            drop(index);
        } else if pass == self.resolution.query {
            let mut query = schedule.open(encoder, pass);
            self.queries.record(&mut query, streams, frame);
            drop(query);
        }
    }
}
