mod capacity;
mod order;
mod resource;
mod schedule;
mod stage;
mod streams;

pub use capacity::{MIN_SLOTS, STREAM_FLOOR, StreamWatch, grown, product, settled};
pub use order::{DomainPasses, PassOrder};
pub use resource::{ResourceId, Resources, SlotRef};
pub use schedule::Schedule;
pub use stage::{
    Dispatch, MAX_DISPATCH_WORKGROUPS, Program, Stage, WORKGROUP_SIZE, entry_rows, entry_stream,
    workgroups_of,
};
pub use streams::{EVENT_SLOTS, PACK, STREAM, UNIFORM};
