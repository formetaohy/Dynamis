pub const COUNTER_ENTRIES: usize = 0;

pub const COUNTER_PAIRS: usize = 1;

pub const COUNTER_LARGE: usize = 2;

pub const COUNTER_CONTACTS: usize = 3;

pub const COUNTER_PREV_CONTACTS: usize = 4;

pub const COUNTER_JOINTS: usize = 5;

pub const COUNTER_CONSTRAINTS: usize = 6;

pub const COUNTER_EVENTS: usize = 7;

pub const COUNTER_BODY_EDITS: usize = 8;

pub const COUNTER_CONSTRAINT_COMMANDS: usize = 9;

pub const COUNTER_SPILLOVER_PAIRS: usize = 10;

pub const COUNTER_SPILLOVER_EVENTS: usize = 11;

pub const COUNTER_BODIES: usize = 12;

pub const COUNTER_SPILLOVER_ENTRIES: usize = 13;

pub const COUNTER_RESTING: usize = 14;

pub const COUNTER_SPILLOVER_RESTING: usize = 15;

pub const COUNTER_SLEPT: usize = 16;

pub const COUNTER_WOKE: usize = 17;

pub const COUNTER_ACTIVE: usize = 18;

pub const COUNTER_BODY_MOVES: usize = 19;

pub const COUNTER_CONSTRAINT_MOVES: usize = 20;

pub const COUNTER_COUNT: usize = 21;

pub const COUNTER_STRIDE: u64 = 256;

pub type Counters = [u32; COUNTER_COUNT];
