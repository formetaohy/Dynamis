mod bindings;
mod buffer;
mod context;
mod library;
mod pipeline;
mod readback;
mod recorder;
mod runtime;
mod stream;
mod submission;
#[cfg(feature = "profile")]
mod timing;

pub use bindings::{ShaderBinding, parse_bindings};
pub use buffer::{GpuBuffer, GpuSlot};
pub use context::{DeviceLost, GpuContext, GpuRequest, GpuUnavailable, LimitsPolicy};
pub use library::{WarmupBudget, WarmupProgress};
pub use pipeline::{BindingKind, BindingSpec, ComputePipeline, ComputeProgram, PipelineHandle};
pub use readback::{Readback, read_regions};
pub use recorder::ComputeRecorder;
pub use runtime::GpuRuntime;
pub use stream::{Contents, Stream};
pub use submission::SubmissionEncoder;
#[cfg(feature = "profile")]
pub use timing::{GpuPassTiming, GpuTimer};
pub use wgpu::{
    AdapterInfo, Backend, Backends, DeviceLostReason, DeviceType, Features, Limits, PowerPreference,
};
