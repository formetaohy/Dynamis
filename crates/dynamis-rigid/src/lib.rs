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
mod reactions;
mod solver;
mod sort;
mod streams;

use dynamis_abi::{FrameCounts, StepParamsRecord};
use dynamis_gpu::Resources;

const IDENTITY_FRAGMENT: &str = include_str!("../shaders/identity.wgsl");
const EVENTS_FRAGMENT: &str = include_str!("../shaders/events.wgsl");
const ISLAND_LINK_FRAGMENT: &str = include_str!("../shaders/island_link.wgsl");
const JOINED_PAIRS_FRAGMENT: &str = include_str!("../shaders/joined_pairs.wgsl");
const BODY_ROW_FRAGMENT: &str = include_str!("../shaders/body_row.wgsl");

const IDENTITY: &[&str] = &[IDENTITY_FRAGMENT];
const CONTACT: &[&str] = &[IDENTITY_FRAGMENT, EVENTS_FRAGMENT];
const CONTACT_ROW: &[&str] = &[IDENTITY_FRAGMENT, EVENTS_FRAGMENT, BODY_ROW_FRAGMENT];
const IDENTITY_LINK: &[&str] = &[IDENTITY_FRAGMENT, ISLAND_LINK_FRAGMENT];
const IDENTITY_LINK_ROW: &[&str] = &[IDENTITY_FRAGMENT, ISLAND_LINK_FRAGMENT, BODY_ROW_FRAGMENT];
const CONSTRAINT_LINK: &[&str] = &[ISLAND_LINK_FRAGMENT];

pub(crate) fn geometry_fragments() -> Vec<&'static str> {
    let mut fragments = dynamis_shader::GEOMETRY.to_vec();
    fragments.push(JOINED_PAIRS_FRAGMENT);
    fragments
}

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
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::{Execution, domain_passes};
use dynamis_sort::RadixSort;
use entries::Entries;
use integrate::Integrate;
use islands::Islands;
use islands::Sleep;
use live::Live;
use narrowphase::Narrowphase;
use queries::Queries;
use reactions::Reactions;
use solver::Solver;

pub use capacity::{Capacity, RigidCapacity, RigidInputs, capacity};
pub use ccd::{Ccd, CcdPasses};
pub use domain::{RigidDomain, RigidDomainPasses, RigidDomainRuntime, RigidWork};
pub use streams::{RigidDemand, RigidStream, RigidStreams, event_capacity};

domain_passes!(
    RigidPasses,
    commands => Execution::GRAPH => &[],
    prepare => Execution::INDEXING.and(Execution::STEP) => &["commands"],
    query_aabbs => Execution::QUERY => &["commands"],
    entries => Execution::INDEXING => &["prepare", "query_aabbs", "soft_bounds"],
    narrowphase => Execution::AWAKE => &["broadphase"],
    islands => Execution::AWAKE => &["narrowphase"],
    wake => Execution::AWAKE => &["islands"],
    live => Execution::AWAKE => &["wake"],
    solver_prepare => Execution::AWAKE => &["live"],
    substeps => Execution::AWAKE => &["solver_prepare"],
    reactions => Execution::AWAKE => &["soft_substeps"],
);

domain_passes!(
    RigidResolutionPasses,
    sleep => Execution::AWAKE => &["reactions"],
    commit => Execution::STEP => &["sleep"],
    observe => Execution::PUBLISH => &["commit"],
    resting_gather => Execution::AWAKE => &["commit"],
    resting_index => Execution::AWAKE => &["resting_gather"],
    query => Execution::GRAPH => &["broadphase", "commit"],
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
    reactions: Reactions,
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
            reactions: Reactions::build(context, streams),
            live: Live::build(context, streams),
            solver: Solver::build(context, streams),
            commit: Commit::build(context, streams),
            sort: RadixSort::new(context, "rigid sort"),
        }
    }

    pub fn record(
        &mut self,
        pass: u32,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) -> bool {
        if pass == self.passes.commands {
            self.commands.record(recorder, streams, frame);
        } else if pass == self.passes.prepare {
            self.integrate
                .record(recorder, streams, frame, &mut self.sort);
        } else if pass == self.passes.query_aabbs {
            self.integrate.record_broadphase(recorder, streams, frame);
        } else if pass == self.passes.entries {
            self.entries.record(recorder, streams, frame);
        } else if pass == self.passes.narrowphase {
            self.narrowphase
                .record(recorder, streams, frame, &mut self.sort);
        } else if pass == self.passes.islands {
            self.islands.record(recorder, streams, frame);
        } else if pass == self.passes.wake {
            self.islands.record_wake(recorder, streams, frame);
        } else if pass == self.passes.live {
            self.live.record(recorder, streams, frame);
        } else if pass == self.passes.solver_prepare {
            self.solver.record_prepare(recorder, streams, frame);
        } else if pass == self.passes.substeps {
            self.solver.record_topology(recorder, streams);
            for substep in 0..frame.params.substeps {
                self.integrate.record_substep(recorder, streams);
                if substep == 0 {
                    self.solver.record_warm(recorder, streams);
                }
                self.solver.record_iterations(recorder, streams, frame);
                self.integrate.record_substep_advance(recorder, streams);
                self.solver
                    .record_position_iterations(recorder, streams, frame);
            }
        } else if pass == self.passes.reactions {
            self.reactions.record(recorder, streams, frame);
        } else if pass == self.resolution.sleep {
            self.sleep.record(recorder, streams, frame);
        } else if pass == self.resolution.commit {
            self.commit.record(recorder, streams, frame);
        } else if pass == self.resolution.observe {
            self.commit
                .record_observe(recorder, streams, frame.observed_count);
        } else if pass == self.resolution.resting_gather {
            self.commit.record_gather(recorder, streams);
        } else if pass == self.resolution.resting_index {
            self.commit
                .record_index(recorder, streams, frame, &mut self.sort);
        } else if pass == self.resolution.query {
            self.queries.record(recorder, streams, frame);
        } else {
            return false;
        }
        true
    }
}
