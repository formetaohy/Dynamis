use crate::RigidDomain;
use dynamis_abi::{
    AabbRecord, BodyStateRecord, CHARACTER_SWEEPS, CONTACT_MAX_POINTS, CharacterInputRecord,
    CharacterRecord, CharacterStateRecord, ConstraintRowsRecord, ConstraintRuntimeRecord,
    ContactEventRecord, ContactRecord, ImpactEventRecord, NO_SLOT, QueryHitRecord, QueryRecord,
    SOLVER_BLOCK_CONSTRAINT, VehicleInputRecord, VehicleRecord, VehicleStateRecord,
    VehicleWheelRecord,
};
use dynamis_domain::Domain;
use dynamis_domain::streams;
use dynamis_gpu::Retention;
use dynamis_gpu::SEGMENT_COUNT;

pub const COMPACT_BLOCK: u32 = 256;

const SOLVER_BLOCK_KINDS: u32 = SOLVER_BLOCK_CONSTRAINT + 1;

pub const DELTA_WORDS: u32 = dynamis_abi::SOLVER_DELTA_WORDS;
pub const BLOCK_LANES: u32 = 2;

streams! {
    RigidStreams, RigidStream, RigidDemand, RigidDomain::ID, demand,
    demand {
        bodies: u32,
        colliders: u32,
        constraints: u32,
        pairs: u32,
        contacts: u32,
        resting: u32,
        events: u32,
        impacts: u32,
        sort: u32,
        characters: u32,
        vehicles: u32,
        wheels: u32,
    }
    streams {
        collider_aabbs, ColliderAabbs: "broadphase aabbs", AabbRecord, 1, Retention::Scratch, demand.colliders;
        live_bodies, LiveBodies: "live body rows", u32, 1, Retention::Scratch, demand.bodies;
        body_activity, BodyActivity: "body activity", u32, 1, Retention::Scratch, demand.bodies;
        body_motion, BodyMotion: "body motion", u32, 1, Retention::Scratch, demand.bodies;
        body_state_scratch, BodyStateScratch: "body state scratch", BodyStateRecord, 1, Retention::Scratch, demand.bodies;
        constraint_rows, ConstraintRows: "constraint rows", ConstraintRowsRecord, 1, Retention::Scratch, demand.constraints;
        constraint_scratch, ConstraintScratch: "constraint state scratch", ConstraintRuntimeRecord, 1, Retention::Scratch, demand.constraints;
        joint_filter_major, JointFilterMajor: "joint filter major", u32, 1, Retention::Scratch, demand.constraints;
        joint_filter_minor, JointFilterMinor: "joint filter minor", u32, 1, Retention::Scratch, demand.constraints;
        ccd_factor, CcdFactor: "ccd retreat factors", u32, 1, Retention::Scratch, demand.bodies;
        ccd_impact, CcdImpact: "ccd impacts", [f32; 4], 1, Retention::Scratch, demand.bodies;
        contacts_raw, ContactsRaw: "contacts raw", ContactRecord, 1, Retention::Scratch, demand.pairs;
        contact_valid, ContactValid: "contact valid", u32, 1, Retention::Scratch, demand.pairs;
        compact_ranks, CompactRanks: "compact ranks", u32, 1, Retention::Scratch, demand.pairs;
        compact_block_sums, CompactBlockSums: "compact block sums", u32, 1, Retention::Scratch, demand.compact_blocks();
        compact_block_offsets, CompactBlockOffsets: "compact block offsets", u32, 1, Retention::Scratch, demand.compact_blocks();
        contacts, Contacts: "contacts", ContactRecord, 1, Retention::Scratch, demand.contacts;
        contact_target_speeds, ContactTargetSpeeds: "contact target speeds", f32, CONTACT_MAX_POINTS, Retention::Scratch, demand.contacts;
        contact_matched, ContactMatched: "contact matched", u32, 1, Retention::Scratch, demand.contacts;
        contact_archive, ContactArchive: "contact archive", ContactRecord, 1, Retention::Durable, demand.contacts;
        resting_contacts, RestingContacts: "resting contacts", ContactRecord, 1, Retention::Durable, demand.resting;
        resting_live, RestingLive: "resting contact live", u32, 1, Retention::Durable, demand.resting;
        resting_next, RestingNext: "resting contact next", u32, 1, Retention::Durable, demand.resting;
        resting_free, RestingFree: "resting contact free", u32, 1, Retention::Seeded(NO_SLOT), 1;
        resting_index_major, RestingIndexMajor: "resting index major", u32, 1, Retention::Durable, demand.resting;
        resting_index_minor, RestingIndexMinor: "resting index minor", u32, 1, Retention::Durable, demand.resting;
        resting_index_slots, RestingIndexSlots: "resting index slots", u32, 1, Retention::Durable, demand.resting;
        solver_segments, SolverSegments: "solver segments", u32, 1, Retention::Scratch, SOLVER_BLOCK_KINDS;
        solver_block_counts, SolverBlockCounts: "solver block counts", u32, 1, Retention::Scratch, demand.bodies;
        solver_blocks, SolverBlocks: "solver block lanes", u32, 1, Retention::Scratch, demand.solver_words();
        solver_velocity_deltas, SolverVelocityDeltas: "solver velocity deltas", u32, 1, Retention::Scratch, demand.body_words();
        solver_position_deltas, SolverPositionDeltas: "solver position deltas", u32, 1, Retention::Scratch, demand.body_words();
        solver_resolution, SolverResolution: "solver resolution", [f32; 4], 1, Retention::Scratch, demand.bodies;
        solver_contributions, SolverContributions: "solver contributions", u32, 1, Retention::Scratch, demand.bodies;
        island_parents, IslandParents: "island parents", u32, 1, Retention::Scratch, demand.bodies;
        island_state, IslandState: "island state", u32, 1, Retention::Scratch, demand.bodies;
        events, Events: "contact events", ContactEventRecord, 1, Retention::Scratch, demand.events.saturating_mul(SEGMENT_COUNT);
        impacts, Impacts: "impact events", ImpactEventRecord, 1, Retention::Scratch, demand.impacts.saturating_mul(SEGMENT_COUNT);
        sort_scratch_major, SortScratchMajor: "sort scratch major", u32, 1, Retention::Scratch, demand.sort;
        sort_scratch_minor, SortScratchMinor: "sort scratch minor", u32, 1, Retention::Scratch, demand.sort;
        sort_scratch_payload, SortScratchPayload: "sort scratch payload", u32, 1, Retention::Scratch, demand.sort;
        sort_dummy_payload, SortDummyPayload: "sort dummy payload", u32, 1, Retention::Scratch, demand.sort;
        characters, Characters: "characters", CharacterRecord, 1, Retention::Durable, demand.characters;
        character_inputs, CharacterInputs: "character inputs", CharacterInputRecord, 1, Retention::Durable, demand.characters;
        character_states, CharacterStates: "character states", CharacterStateRecord, 1, Retention::Durable, demand.characters;
        character_sweeps, CharacterSweeps: "character sweeps", QueryRecord, 1, Retention::Durable, demand.character_sweeps();
        character_hits, CharacterHits: "character sweep hits", QueryHitRecord, 1, Retention::Durable, demand.character_sweeps();
        vehicles, Vehicles: "vehicles", VehicleRecord, 1, Retention::Durable, demand.vehicles;
        vehicle_wheels, VehicleWheels: "vehicle wheels", VehicleWheelRecord, 1, Retention::Durable, demand.wheels;
        vehicle_inputs, VehicleInputs: "vehicle inputs", VehicleInputRecord, 1, Retention::Durable, demand.vehicles;
        vehicle_states, VehicleStates: "vehicle states", VehicleStateRecord, 1, Retention::Durable, demand.vehicles;
        vehicle_sweeps, VehicleSweeps: "vehicle sweeps", QueryRecord, 1, Retention::Durable, demand.wheels;
        vehicle_hits, VehicleHits: "vehicle wheel hits", QueryHitRecord, 1, Retention::Durable, demand.wheels;
    }
}

impl RigidDemand {
    pub fn character_sweeps(&self) -> u32 {
        self.characters.saturating_mul(CHARACTER_SWEEPS)
    }

    pub fn blocks(&self) -> u32 {
        self.contacts.saturating_add(self.constraints)
    }

    pub fn solver_words(&self) -> u32 {
        self.blocks().saturating_mul(BLOCK_LANES)
    }

    pub fn body_words(&self) -> u32 {
        self.bodies.saturating_mul(DELTA_WORDS)
    }

    pub fn compact_blocks(&self) -> u32 {
        self.pairs.div_ceil(COMPACT_BLOCK)
    }

    pub fn sort_slots(pairs: u32, constraints: u32) -> u32 {
        pairs.saturating_add(constraints)
    }
}

pub fn event_capacity<R: dynamis_gpu::ResourceSource>(resources: &R) -> u32 {
    resources.slots(RigidStream::Events.into()) / SEGMENT_COUNT
}

pub fn impact_capacity<R: dynamis_gpu::ResourceSource>(resources: &R) -> u32 {
    resources.slots(RigidStream::Impacts.into()) / SEGMENT_COUNT
}
