//! A fully GPU-driven, data-oriented rigid-body engine.
//!
//! ```no_run
//! use dynamis::{BodyDesc, GpuContext, PhysicsConfig, Simulation};
//!
//! # async fn run() {
//! let gpu = GpuContext::new().await;
//! let mut sim = Simulation::new(gpu, 1024, PhysicsConfig::default());
//! let ball = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 5.0, 0.0]));
//! for _ in 0..60 {
//!     sim.step(1.0 / 60.0);
//! }
//! sim.wait();
//! assert!(sim.read_state(ball).position[1] < 5.0);
//! # }
//! ```

pub use dynamis_gpu::{
    AdapterInfo, Backend, Backends, BindingKind, BindingSpec, ComputePipeline, ComputeRecorder,
    DeviceLost, DeviceType, Features, GpuBuffer, GpuContext, GpuReadback, GpuRequest,
    GpuUnavailable, Limits, LimitsPolicy, PowerPreference,
};
#[cfg(feature = "profile")]
pub use dynamis_gpu::{GpuPassTiming, GpuTimer};
pub use dynamis_kernel::{BucketChannels, BucketSort, RadixSort, SortChannels};
pub use dynamis_model::{
    BodyDesc, BodyHandle, BodyState, ColliderDesc, ConstraintBreak, ConstraintDesc,
    ConstraintHandle, ConstraintKind, ConstraintLimit, ConstraintMotor, ConstraintSpring,
    ConstraintSwing, ContactEvent, ContactEventKind, ContactEventMode, DofDesc,
    MAX_COLLIDERS_PER_BODY, MassProperties, MaterialCombine, PhysicsConfig, QueryFilter, Shape,
    ShapeSourceHandle,
};
pub use dynamis_simulate::{
    Character, CharacterDesc, DebugBuffer, HullDecomposeSettings, QueryHandle, QueryHit, Simulation,
};
