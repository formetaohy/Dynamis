mod capacity;
mod ccd;
mod character;
mod commands;
mod commit;
mod domain;
mod entries;
mod impacts;
mod integrate;
mod islands;
mod live;
mod narrowphase;
mod reactions;
mod solver;
mod sort;
mod streams;
mod vehicle;

use commit::OBSERVED_JOINTS_EXECUTION;
use dynamis_abi::{Census, RowStreams, StepParamsRecord};

const IDENTITY_FRAGMENT: &str = include_str!("../shaders/identity.wgsl");
const EVENTS_FRAGMENT: &str = include_str!("../shaders/events.wgsl");
const COUNTERS_FRAGMENT: &str = dynamis_shader::COUNTER_ACCESS;
const ISLAND_LINK_FRAGMENT: &str = include_str!("../shaders/island_link.wgsl");
const JOINED_PAIRS_FRAGMENT: &str = include_str!("../shaders/joined_pairs.wgsl");
const BODY_ROW_FRAGMENT: &str = include_str!("../shaders/body_row.wgsl");

const IDENTITY: &[&str] = &[IDENTITY_FRAGMENT];
const CONTACT: &[&str] = &[COUNTERS_FRAGMENT, IDENTITY_FRAGMENT, EVENTS_FRAGMENT];
const CONTACT_COUNTERS: &[&str] = &[COUNTERS_FRAGMENT, IDENTITY_FRAGMENT];
const CONTACT_ROW: &[&str] = &[
    COUNTERS_FRAGMENT,
    IDENTITY_FRAGMENT,
    EVENTS_FRAGMENT,
    BODY_ROW_FRAGMENT,
];
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
    pub fn of(census: &Census) -> Self {
        Self {
            island_rounds: propagation_rounds(census.dynamic_bodies),
            body_row_words: dynamis_sort::key_words(census.bodies.max(1)),
            body_id_words: dynamis_sort::key_words(census.body_ids.max(1)),
            collider_slot_words: dynamis_sort::key_words(census.colliders.max(1)),
        }
    }
}

fn propagation_rounds(bodies: u32) -> u32 {
    bodies.max(2).ilog2() + 1
}

#[derive(Clone, Copy, Debug)]
pub struct RigidFrame {
    pub params: StepParamsRecord,
    pub rows: RowStreams,
    pub shape: RigidShape,
    pub observed_count: u32,
    pub observed_joints: u32,
    pub ccd: bool,
    pub impacts: bool,
}

use character::{Character, CharacterSweeps};
use commands::Commands;
use commit::{Commit, Observe, ObserveJoints, RestingGather, RestingIndex};
use dynamis_pass::{Execution, domain_passes};
use entries::Entries;
use integrate::{Prepare, QueryAabbs};
use islands::{Islands, Sleep, Wake};
use live::Live;
use narrowphase::Narrowphase;
use reactions::Reactions;
use solver::{SolverPrepare, Substeps};
use vehicle::{Vehicle, VehicleSweeps};

pub use capacity::{RigidCapacity, RigidInputs, capacity, floor, plan};
pub use ccd::{CcdApply, CcdPasses, CcdRuntime, CcdSweep};
pub use domain::{RigidDomain, RigidDomainPasses, RigidDomainRuntime, RigidWork};
pub use streams::{RigidDemand, RigidStream, RigidStreams, event_capacity, impact_capacity};

domain_passes!(
    RigidPasses,
    RigidRuntime,
    RigidFrame,
    commands: Commands => Execution::GRAPH => &[],
    character: Character => Execution::STEP.and(Execution::AWAKE) => &["commands"],
    prepare: Prepare => Execution::INDEXING.and(Execution::STEP) => &["commands", "character"],
    query_aabbs: QueryAabbs => Execution::QUERY => &["commands"],
    vehicle: Vehicle => Execution::STEP.and(Execution::AWAKE) => &["commands"],
    entries: Entries => Execution::INDEXING => &["prepare", "query_aabbs", "soft_bounds"],
    narrowphase: Narrowphase => Execution::AWAKE => &["broadphase"],
    islands: Islands => Execution::AWAKE => &["narrowphase"],
    wake: Wake => Execution::AWAKE => &["islands"],
    live: Live => Execution::AWAKE => &["wake"],
    solver_prepare: SolverPrepare => Execution::AWAKE => &["live"],
    substeps: Substeps => Execution::AWAKE => &["solver_prepare"],
    reactions: Reactions => Execution::AWAKE => &["soft_substeps"],
);

domain_passes!(
    RigidResolutionPasses,
    RigidResolutionRuntime,
    RigidFrame,
    sleep: Sleep => Execution::AWAKE => &["reactions"],
    commit: Commit => Execution::STEP => &["sleep"],
    observe: Observe => Execution::PUBLISH => &["commit"],
    observe_joints: ObserveJoints => OBSERVED_JOINTS_EXECUTION => &["observe"],
    resting_gather: RestingGather => Execution::AWAKE => &["commit"],
    resting_index: RestingIndex => Execution::AWAKE => &["resting_gather"],
    character_sweeps: CharacterSweeps => Execution::STEP.and(Execution::AWAKE) => &["query"],
    vehicle_sweeps: VehicleSweeps => Execution::STEP.and(Execution::AWAKE) => &["query"],
);
