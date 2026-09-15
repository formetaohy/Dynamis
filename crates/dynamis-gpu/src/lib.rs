mod bindings;
mod buffer;
mod context;
mod library;
mod pipeline;
mod readback;
mod recorder;
mod resource;
mod runtime;
mod stream;
mod submission;
#[cfg(feature = "profile")]
mod timing;

pub use bindings::{ShaderBinding, assert_binding_element, parse_bindings};
pub use buffer::{GpuBuffer, GpuSlot, StorageId};
pub use context::{DeviceLost, GpuContext, GpuRequest, GpuUnavailable, LimitsPolicy};
pub use library::{WarmupBudget, WarmupProgress};
pub use pipeline::{BindingKind, BindingSpec, ComputePipeline, ComputeProgram, PipelineHandle};
pub use readback::{EVENT_SLOTS, FACT_LAG, Publication, Readback, read_regions};
pub use recorder::ComputeRecorder;
pub use resource::{ResourceId, Resources, SlotRef};
pub use runtime::GpuRuntime;
pub use stream::{Contents, PACK, STREAM, Stream, StreamDesc, StreamElement, TypedSlot, UNIFORM};
pub use submission::SubmissionEncoder;
#[cfg(feature = "profile")]
pub use timing::{GpuPassTiming, GpuTimer};
pub use wgpu::{
    Adapter, AdapterInfo, Backend, Backends, Device, DeviceLostReason, DeviceType,
    ExperimentalFeatures, Features, Limits, PowerPreference, Queue,
};
