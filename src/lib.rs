//! A fully GPU-driven, data-oriented rigid-body engine.
//!
//! ```no_run
//! use dynamis::{BodyDesc, GpuContext, PhysicsConfig, Simulation, StreamBudget};
//!
//! # async fn run() {
//! let gpu = GpuContext::new().await;
//! let mut sim = Simulation::new(gpu, 1024, PhysicsConfig::default(), StreamBudget::default());
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
    DeviceLost, DeviceType, Features, GpuBuffer, GpuContext, GpuReadback, GpuRequest, GpuSlot,
    GpuUnavailable, Limits, LimitsPolicy, PowerPreference,
};
#[cfg(feature = "profile")]
pub use dynamis_gpu::{GpuPassTiming, GpuTimer};
pub use dynamis_kernel::{RadixSort, SortChannels};
pub use dynamis_model::{
    BodyDesc, BodyHandle, BodyState, ColliderDesc, ConstraintBreak, ConstraintDesc,
    ConstraintHandle, ConstraintKind, ConstraintLimit, ConstraintMotor, ConstraintSpring,
    ConstraintSwing, ContactEvent, ContactEventKind, ContactEventMode, DofDesc,
    MAX_COLLIDERS_PER_BODY, MassProperties, MaterialCombine, PhysicsConfig, QueryFilter, Shape,
    ShapeSourceHandle,
};
pub use dynamis_simulate::{
    COUNTER_CONTACTS, COUNTER_COUNT, COUNTER_ENTRIES, COUNTER_EVENTS, COUNTER_PAIRS,
    COUNTER_PREV_CONTACTS, COUNTER_SPILLOVER_ENTRIES, COUNTER_SPILLOVER_EVENTS,
    COUNTER_SPILLOVER_PAIRS, Character, CharacterDesc, Counters, DebugBuffer,
    HullDecomposeSettings, QueryHandle, QueryHit, Simulation, StreamBudget,
};
