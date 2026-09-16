mod buffer;
mod context;
mod library;
mod pipeline;
mod readback;
mod recorder;
mod resource;
mod runtime;
mod segment;
mod stream;
mod submission;
#[cfg(feature = "profile")]
mod timing;

pub use buffer::{GpuBuffer, GpuSlot, StorageId};
pub use context::{DeviceLost, GpuContext, GpuRequest, GpuUnavailable, LimitsPolicy};
pub use library::{WarmupBudget, WarmupProgress};
pub use pipeline::{BindingKind, BindingSpec, ComputePipeline, ComputeProgram, PipelineHandle};
pub use readback::{FACT_LAG, Publication, Readback, SEGMENT_COUNT, read_regions};
pub use recorder::ComputeRecorder;
pub use resource::{ResourceId, ResourceSource, SlotRef};
pub use runtime::GpuRuntime;
pub use segment::SegmentRing;
pub use stream::{PACK, Retention, STREAM, Stream, StreamDesc, StreamElement, TypedSlot, UNIFORM};
pub use submission::SubmissionEncoder;
#[cfg(feature = "profile")]
pub use timing::{GpuPassTiming, GpuTimer};
pub use wgpu::{
    Adapter, AdapterInfo, Backend, Backends, Device, DeviceLostReason, DeviceType,
    ExperimentalFeatures, Features, Limits, PowerPreference, Queue,
};
