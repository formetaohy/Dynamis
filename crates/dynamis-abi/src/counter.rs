pub mod device {
    use crate::wgsl::declare_constants;

    declare_constants! {
        pub const COUNTER_ENTRIES: usize = 0;
        pub const COUNTER_PAIRS: usize = 1;
        pub const COUNTER_GRID_LEVELS: usize = 2;
        pub const COUNTER_CONTACTS: usize = 3;
        pub const COUNTER_ARCHIVED: usize = 4;
        pub const COUNTER_JOINTS: usize = 5;
        pub const COUNTER_EVENTS: usize = 6;
        pub const COUNTER_SPILLOVER_PAIRS: usize = 7;
        pub const COUNTER_SPILLOVER_EVENTS: usize = 8;
        pub const COUNTER_SPILLOVER_ENTRIES: usize = 9;
        pub const COUNTER_RESTING: usize = 10;
        pub const COUNTER_SPILLOVER_RESTING: usize = 11;
        pub const COUNTER_SLEPT: usize = 12;
        pub const COUNTER_WOKE: usize = 13;
        pub const COUNTER_ACTIVE: usize = 14;
        pub const COUNTER_WOKE_DEFERRED: usize = 15;
        pub const COUNTER_RESTING_GATHER: usize = 16;
        pub const COUNTER_RESTING_INDEX: usize = 17;
        pub const COUNTER_RESTING_PENDING: usize = 18;
        pub const COUNTER_BLOCKS: usize = 19;
        pub const COUNTER_COARSE_ACTIVE: usize = 20;
        pub const COUNTER_GRID_SCALE: usize = 21;
        pub const COUNTER_GRID_EXTENT: usize = 22;
        pub const COUNTER_BREAKS: usize = 23;
        pub const COUNTER_PARTICLE_REACH: usize = 24;
        pub const COUNTER_SPILLOVER_NEIGHBOURS: usize = 25;
        pub const COUNTER_LIVE: usize = 26;
        pub const COUNTER_SPILLOVER_LIVE: usize = 27;
        pub const COUNTER_SOFT_ACTIVE: usize = 28;
        pub const COUNTER_SOFT_SLEPT: usize = 29;
        pub const COUNTER_SOFT_WOKE: usize = 30;
        pub const COUNTER_DEVICE_COUNT: usize = 31;
    }
}

pub mod host {
    use super::device::COUNTER_DEVICE_COUNT;

    pub const COUNT: usize = 7;

    pub const COUNTER_BODIES: usize = COUNTER_DEVICE_COUNT;
    pub const COUNTER_COLLIDERS: usize = COUNTER_DEVICE_COUNT + 1;
    pub const COUNTER_CONSTRAINTS: usize = COUNTER_DEVICE_COUNT + 2;
    pub const COUNTER_BODY_EDITS: usize = COUNTER_DEVICE_COUNT + 3;
    pub const COUNTER_BODY_MOVES: usize = COUNTER_DEVICE_COUNT + 4;
    pub const COUNTER_CONSTRAINT_COMMANDS: usize = COUNTER_DEVICE_COUNT + 5;
    pub const COUNTER_CONSTRAINT_MOVES: usize = COUNTER_DEVICE_COUNT + 6;
}

pub use device::COUNTER_DEVICE_COUNT;

pub const COUNTER_COUNT: usize = COUNTER_DEVICE_COUNT + host::COUNT;

pub const COUNTER_STRIDE: u64 = 256;

pub type Counters = [u32; COUNTER_COUNT];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeclaredCounters {
    pub bodies: u32,
    pub colliders: u32,
    pub constraints: u32,
    pub body_edits: u32,
    pub body_moves: u32,
    pub constraint_commands: u32,
    pub constraint_moves: u32,
}

impl DeclaredCounters {
    pub fn write_into(&self, counters: &mut Counters) {
        counters[host::COUNTER_BODIES] = self.bodies;
        counters[host::COUNTER_COLLIDERS] = self.colliders;
        counters[host::COUNTER_CONSTRAINTS] = self.constraints;
        counters[host::COUNTER_BODY_EDITS] = self.body_edits;
        counters[host::COUNTER_BODY_MOVES] = self.body_moves;
        counters[host::COUNTER_CONSTRAINT_COMMANDS] = self.constraint_commands;
        counters[host::COUNTER_CONSTRAINT_MOVES] = self.constraint_moves;
    }
}
