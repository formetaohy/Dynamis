//! Device acquisition and the compute-only GPU primitives the engine runs on.

mod buffer;
mod compute;
mod context;
mod timing;

pub use buffer::{GpuBuffer, GpuReadback};
pub use compute::{BindingKind, BindingSpec, ComputePipeline, ComputeRecorder};
pub use context::{DeviceLost, GpuContext, GpuRequest, GpuUnavailable, LimitsPolicy};
pub use timing::{GpuPassTiming, GpuTimer};
pub use wgpu::{
    AdapterInfo, Backend, Backends, DeviceLostReason, DeviceType, Features, Limits, PowerPreference,
};
