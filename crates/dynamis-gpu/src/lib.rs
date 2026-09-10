//! Device acquisition and the compute-only GPU primitives the engine runs on.

mod buffer;
mod compute;
mod context;
mod dispatch;
#[cfg(feature = "profile")]
mod timing;

pub use buffer::{GpuBuffer, GpuReadback, GpuSlot};
pub use compute::{BindingKind, BindingSpec, ComputePipeline, ComputeRecorder};
pub use context::{DeviceLost, GpuContext, GpuRequest, GpuUnavailable, LimitsPolicy};
pub use dispatch::{DISPATCH_ARGS_BYTES, DispatchTable};
#[cfg(feature = "profile")]
pub use timing::{GpuPassTiming, GpuTimer};
pub use wgpu::{
    AdapterInfo, Backend, Backends, DeviceLostReason, DeviceType, Features, Limits, PowerPreference,
};
