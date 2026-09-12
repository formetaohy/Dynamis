use crate::dynamics::engine::streams;
use dynamis_gpu::{Contents, ReadbackRing};
use dynamis_layout::{
    AabbRecord, BodyStateRecord, CONTACT_MAX_POINTS, ConstraintRowsRecord, ConstraintRuntimeRecord,
    ContactEventRecord, ContactRecord, NO_SLOT, SOLVER_BLOCK_CONSTRAINT,
};
use std::mem::size_of;

pub(crate) const DOMAIN: u32 = 1;

pub(crate) const EVENT_SLOTS: u32 = ReadbackRing::DEPTH as u32 + 2;
pub(crate) const COMPACT_BLOCK: u32 = 256;

const SOLVER_BLOCK_KINDS: u32 = SOLVER_BLOCK_CONSTRAINT + 1;
const SOLVER_DELTA_BYTES: u64 = 64;
const SOLVER_CORRECTION_BYTES: u64 = 64;
const SOLVER_RESOLUTION_BYTES: u64 = 16;
const CONTACT_TARGET_BYTES: u64 = 4 * CONTACT_MAX_POINTS as u64;

streams! {
    RigidStreams, RigidStream, RigidDemand, DOMAIN, demand,
    demand {
        bodies: u32,
        colliders: u32,
        constraints: u32,
        entries: u32,
        pairs: u32,
        events: u32,
        sort: u32,
    }
    streams {
        collider_aabbs, ColliderAabbs: "broadphase aabbs", size_of::<AabbRecord>() as u64, Contents::Reset, demand.colliders;
        body_activity, BodyActivity: "body activity", 4, Contents::Reset, demand.bodies;
        body_state_scratch, BodyStateScratch: "body state scratch", size_of::<BodyStateRecord>() as u64, Contents::Reset, demand.bodies;
        constraint_rows, ConstraintRows: "constraint rows", size_of::<ConstraintRowsRecord>() as u64, Contents::Reset, demand.constraints;
        constraint_scratch, ConstraintScratch: "constraint state scratch", size_of::<ConstraintRuntimeRecord>() as u64, Contents::Reset, demand.constraints;
        joint_filter_major, JointFilterMajor: "joint filter major", 4, Contents::Reset, demand.constraints;
        joint_filter_minor, JointFilterMinor: "joint filter minor", 4, Contents::Reset, demand.constraints;
        ccd_factor, CcdFactor: "ccd retreat factors", 4, Contents::Reset, demand.bodies;
        ccd_impact, CcdImpact: "ccd impacts", 16, Contents::Reset, demand.bodies;
        grid_entry_keys, GridEntryKeys: "grid entry keys", 4, Contents::Reset, demand.entries;
        grid_entry_colliders, GridEntryColliders: "grid entry colliders", 4, Contents::Reset, demand.entries;
        pair_major, PairMajor: "pairs major", 4, Contents::Reset, demand.pairs;
        pair_minor, PairMinor: "pairs minor", 4, Contents::Reset, demand.pairs;
        contacts_raw, ContactsRaw: "contacts raw", size_of::<ContactRecord>() as u64, Contents::Reset, demand.pairs;
        contact_valid, ContactValid: "contact valid", 4, Contents::Reset, demand.pairs;
        compact_ranks, CompactRanks: "compact ranks", 4, Contents::Reset, demand.pairs;
        compact_block_sums, CompactBlockSums: "compact block sums", 4, Contents::Reset, demand.compact_blocks();
        compact_block_offsets, CompactBlockOffsets: "compact block offsets", 4, Contents::Reset, demand.compact_blocks();
        contacts, Contacts: "contacts", size_of::<ContactRecord>() as u64, Contents::Reset, demand.pairs;
        contact_target_speeds, ContactTargetSpeeds: "contact target speeds", CONTACT_TARGET_BYTES, Contents::Reset, demand.pairs;
        contact_matched, ContactMatched: "contact matched", 4, Contents::Reset, demand.pairs;
        contact_archive, ContactArchive: "contact archive", size_of::<ContactRecord>() as u64, Contents::Preserve, demand.pairs;
        resting_contacts, RestingContacts: "resting contacts", size_of::<ContactRecord>() as u64, Contents::Preserve, demand.pairs;
        resting_live, RestingLive: "resting contact live", 4, Contents::Preserve, demand.pairs;
        resting_next, RestingNext: "resting contact next", 4, Contents::Preserve, demand.pairs;
        resting_free, RestingFree: "resting contact free", 4, Contents::PreserveSeeded(NO_SLOT), 1;
        resting_index_major, RestingIndexMajor: "resting index major", 4, Contents::Preserve, demand.pairs;
        resting_index_minor, RestingIndexMinor: "resting index minor", 4, Contents::Preserve, demand.pairs;
        resting_index_slots, RestingIndexSlots: "resting index slots", 4, Contents::Preserve, demand.pairs;
        solver_segments, SolverSegments: "solver segments", 4, Contents::Reset, SOLVER_BLOCK_KINDS;
        solver_a_bodies, SolverABodies: "solver block bodies", 4, Contents::Reset, demand.blocks();
        solver_a_payload, SolverAPayload: "solver block payload", 4, Contents::Reset, demand.blocks();
        solver_b_bodies, SolverBBodies: "solver block second bodies", 4, Contents::Reset, demand.blocks();
        solver_b_blocks, SolverBBlocks: "solver block second slots", 4, Contents::Reset, demand.blocks();
        solver_block_first_body, SolverBlockFirstBody: "solver block first owner", 4, Contents::Reset, demand.blocks();
        solver_block_second_body, SolverBlockSecondBody: "solver block second owner", 4, Contents::Reset, demand.blocks();
        solver_first_a, SolverFirstA: "solver first block", 4, Contents::Reset, demand.bodies;
        solver_first_b, SolverFirstB: "solver first second block", 4, Contents::Reset, demand.bodies;
        solver_block_counts, SolverBlockCounts: "solver block counts", 4, Contents::Reset, demand.bodies;
        solver_contact_counts, SolverContactCounts: "solver contact counts", 4, Contents::Reset, demand.bodies;
        solver_block_deltas, SolverBlockDeltas: "solver block deltas", SOLVER_DELTA_BYTES, Contents::Reset, demand.blocks();
        solver_block_corrections, SolverBlockCorrections: "solver block corrections", SOLVER_CORRECTION_BYTES, Contents::Reset, demand.blocks();
        solver_resolution, SolverResolution: "solver resolution", SOLVER_RESOLUTION_BYTES, Contents::Reset, demand.bodies;
        solver_contributions, SolverContributions: "solver contributions", 4, Contents::Reset, demand.bodies;
        island_parents, IslandParents: "island parents", 4, Contents::Reset, demand.bodies;
        island_state, IslandState: "island state", 4, Contents::Reset, demand.bodies;
        wake_flags, WakeFlags: "wake flags", 4, Contents::Reset, demand.bodies;
        events, Events: "contact events", size_of::<ContactEventRecord>() as u64, Contents::Reset, demand.events.saturating_mul(EVENT_SLOTS);
        sort_scratch_major, SortScratchMajor: "sort scratch major", 4, Contents::Reset, demand.sort;
        sort_scratch_minor, SortScratchMinor: "sort scratch minor", 4, Contents::Reset, demand.sort;
        sort_scratch_payload, SortScratchPayload: "sort scratch payload", 4, Contents::Reset, demand.sort;
        sort_dummy_payload, SortDummyPayload: "sort dummy payload", 4, Contents::Reset, demand.sort;
    }
}

impl RigidDemand {
    pub(crate) fn blocks(&self) -> u32 {
        self.pairs.saturating_add(self.constraints)
    }

    pub(crate) fn compact_blocks(&self) -> u32 {
        self.pairs.div_ceil(COMPACT_BLOCK)
    }

    pub(crate) fn sort_slots(entries: u32, pairs: u32, constraints: u32) -> u32 {
        entries.max(pairs).max(pairs.saturating_add(constraints))
    }
}

impl RigidStreams {
    pub(crate) fn entry_capacity(&self) -> u32 {
        self.grid_entry_keys.slots()
    }

    pub(crate) fn pair_capacity(&self) -> u32 {
        self.pair_major.slots()
    }

    pub(crate) fn event_capacity(&self) -> u32 {
        self.events.slots() / EVENT_SLOTS
    }

    pub(crate) fn resting_capacity(&self) -> u32 {
        self.resting_contacts.slots()
    }

    pub(crate) fn block_capacity(&self) -> u32 {
        self.solver_a_payload.slots()
    }

    pub(crate) fn sort_capacity(&self) -> u32 {
        self.sort_scratch_major.slots()
    }
}
