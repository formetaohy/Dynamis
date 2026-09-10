//! The single device-authority counter vector every stage reads and writes.
//!
//! Slots written by the device measure what a step actually produced; slots written by
//! the host declare what a step is about to consume. One vector means one reset, one
//! readback, and one place to look when a stream needs more room.

/// Live bodies, in slot order.
pub const COUNTER_BODIES: usize = 12;
/// Live constraints.
pub const COUNTER_CONSTRAINTS: usize = 6;
/// Pending body commands.
pub const COUNTER_BODY_COMMANDS: usize = 8;
/// Pending constraint commands.
pub const COUNTER_CONSTRAINT_COMMANDS: usize = 9;

/// Grid entries emitted by the broadphase.
pub const COUNTER_ENTRIES: usize = 0;
/// Collider pairs emitted by the broadphase.
pub const COUNTER_PAIRS: usize = 1;
/// Colliders too large for the grid to index.
pub const COUNTER_LARGE: usize = 2;
/// Contact manifolds produced by the narrowphase.
pub const COUNTER_CONTACTS: usize = 3;
/// Contact manifolds produced by the previous step.
pub const COUNTER_PREV_CONTACTS: usize = 4;
/// Body pairs joined by a collision-disabling constraint.
pub const COUNTER_JOINTS: usize = 5;
/// Contact events produced by this step.
pub const COUNTER_EVENTS: usize = 7;
/// Pairs dropped because the pair stream was full.
pub const COUNTER_SPILLOVER_PAIRS: usize = 10;
/// Events dropped because the event stream was full.
pub const COUNTER_SPILLOVER_EVENTS: usize = 11;
/// Grid cells dropped because the entry stream was full.
pub const COUNTER_SPILLOVER_ENTRIES: usize = 13;

/// The width of the counter vector.
pub const COUNTER_COUNT: usize = 14;

/// Bytes between counter slots. A slot is bound on its own, so the stride satisfies
/// the storage binding offset alignment every backend guarantees.
pub const COUNTER_STRIDE: u64 = 256;

/// The counter vector, mirrored on both sides of the API.
pub type Counters = [u32; COUNTER_COUNT];
