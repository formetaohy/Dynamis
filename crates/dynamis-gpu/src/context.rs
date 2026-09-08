use crate::{BindingKind, BindingSpec, ComputePipeline};
use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};
use std::sync::{Arc, Mutex};
use wgpu::{
    Adapter, AdapterInfo, Backend, Backends, Device, DeviceLostReason, DeviceType, Features,
    Instance, InstanceDescriptor, Limits, PowerPreference, Queue,
};

/// How much adapter capability the created device requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitsPolicy {
    /// Exactly [`GpuContext::MINIMUM_LIMITS`].
    Minimum,
    /// Everything the adapter reports.
    Adapter,
}

/// Device acquisition options.
pub struct GpuRequest {
    pub backends: Backends,
    pub power_preference: PowerPreference,
    /// Case-insensitive substring of [`AdapterInfo::name`].
    pub device_name: Option<String>,
    /// Features the device must offer.
    pub required_features: Features,
    /// Features used only when the adapter offers them.
    pub optional_features: Features,
    pub limits: LimitsPolicy,
}

impl Default for GpuRequest {
    fn default() -> Self {
        Self {
            backends: GpuRequest::NATIVE_BACKENDS,
            power_preference: PowerPreference::HighPerformance,
            device_name: None,
            required_features: Features::empty(),
            optional_features: GpuRequest::PROFILING_FEATURES,
            limits: LimitsPolicy::Adapter,
        }
    }
}

impl GpuRequest {
    /// Timestamp queries, requested from the device only when the `profile`
    /// feature is built in.
    #[cfg(feature = "profile")]
    pub const PROFILING_FEATURES: Features =
        Features::TIMESTAMP_QUERY.union(Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
    #[cfg(not(feature = "profile"))]
    pub const PROFILING_FEATURES: Features = Features::empty();

    /// The native APIs this engine runs on.
    pub const NATIVE_BACKENDS: Backends = Backends::DX12
        .union(Backends::METAL)
        .union(Backends::VULKAN);

    pub fn adapter_named(name: impl Into<String>) -> Self {
        Self {
            device_name: Some(name.into()),
            ..Self::default()
        }
    }

    /// Request [`GpuContext::MINIMUM_LIMITS`] rather than the hardware maximum.
    pub fn minimum_limits() -> Self {
        Self {
            limits: LimitsPolicy::Minimum,
            ..Self::default()
        }
    }
}

/// Why no device could be created.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuUnavailable {
    NoAdapter {
        backends: Backends,
    },
    DeviceNotFound {
        requested: String,
        available: Vec<String>,
    },
    MissingFeatures {
        missing: Features,
        available: Features,
    },
    DeviceRejected {
        message: String,
    },
    InsufficientLimits {
        adapter: String,
    },
}

impl Display for GpuUnavailable {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter { backends } => {
                write!(
                    out,
                    "no adapter exposed by the enabled backends {backends:?}"
                )
            }
            Self::DeviceNotFound {
                requested,
                available,
            } => write!(
                out,
                "no adapter name contains {requested:?}; available: {available:?}"
            ),
            Self::MissingFeatures { missing, available } => write!(
                out,
                "adapter lacks required features {missing:?} (offers {available:?})"
            ),
            Self::DeviceRejected { message } => write!(out, "device request rejected: {message}"),
            Self::InsufficientLimits { adapter } => write!(
                out,
                "adapter {adapter:?} cannot satisfy the storage-buffer limits dynamis requires"
            ),
        }
    }
}

impl std::error::Error for GpuUnavailable {}

/// Why a device stopped working.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceLost {
    pub reason: DeviceLostReason,
    pub message: String,
}

impl Display for DeviceLost {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        write!(out, "{:?}: {}", self.reason, self.message)
    }
}

#[derive(Default)]
struct Health {
    lost: Mutex<Option<DeviceLost>>,
}

pub struct GpuContext {
    adapter: Adapter,
    device: Device,
    queue: Queue,
    features: Features,
    limits: Limits,
    timestamp_period_ns: f32,
    health: Arc<Health>,
    pipelines: Arc<Mutex<HashMap<PipelineKey, Arc<ComputePipeline>>>>,
}

#[derive(PartialEq, Eq, Hash, Clone)]
struct PipelineKey {
    label: String,
    shader: String,
    entry: String,
    workgroup_size: u32,
    bindings: Vec<(u32, BindingKind)>,
}

impl PipelineKey {
    fn of(
        label: &str,
        shader: &str,
        entry: &str,
        groups: &[&[BindingSpec]],
        workgroup_size: u32,
    ) -> Self {
        let bindings = groups
            .iter()
            .flat_map(|group| group.iter())
            .map(|spec| (spec.binding, spec.kind))
            .collect();
        Self {
            label: label.to_owned(),
            shader: shader.to_owned(),
            entry: entry.to_owned(),
            workgroup_size,
            bindings,
        }
    }
}

impl GpuContext {
    /// The adapter capability this engine requires.
    pub const MINIMUM_LIMITS: Limits = Limits {
        max_storage_buffers_per_shader_stage: 16,
        max_storage_textures_per_shader_stage: 0,
        max_storage_buffer_binding_size: 16 * 1024 * 1024,
        max_buffer_size: 64 * 1024 * 1024,
        max_compute_workgroup_size_x: 64,
        max_compute_workgroup_size_y: 64,
        max_compute_workgroup_size_z: 64,
        max_compute_workgroups_per_dimension: 65_535,
        ..Limits::defaults()
    };

    /// Acquire a device for `request`.
    pub async fn open(request: &GpuRequest) -> Result<Self, GpuUnavailable> {
        let adapter = select_adapter(request).await?;
        let available = adapter.features();
        let missing = request.required_features.difference(available);
        if !missing.is_empty() {
            return Err(GpuUnavailable::MissingFeatures { missing, available });
        }
        let features = request.required_features | (request.optional_features & available);
        let limits = match request.limits {
            LimitsPolicy::Adapter => adapter.limits(),
            LimitsPolicy::Minimum => Self::MINIMUM_LIMITS,
        };
        if !Self::MINIMUM_LIMITS.check_limits(&adapter.limits()) {
            return Err(GpuUnavailable::InsufficientLimits {
                adapter: adapter.get_info().name,
            });
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("dynamis device"),
                required_features: features,
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(|error| GpuUnavailable::DeviceRejected {
                message: error.to_string(),
            })?;
        let health = Arc::new(Health::default());
        let callback = health.clone();
        device.set_device_lost_callback(move |reason, message| {
            *callback.lost.lock().unwrap() = Some(DeviceLost { reason, message });
        });
        let timestamp_period_ns = queue.get_timestamp_period();
        Ok(Self {
            adapter,
            device,
            queue,
            features,
            limits,
            timestamp_period_ns,
            health,
            pipelines: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Acquire a device for the default request.
    pub async fn new() -> Self {
        match Self::open(&GpuRequest::default()).await {
            Ok(context) => context,
            Err(error) => panic!("{error}"),
        }
    }

    /// Every adapter reachable through `backends`.
    pub async fn available_adapters(backends: Backends) -> Vec<AdapterInfo> {
        let instance = create_instance(backends);
        instance
            .enumerate_adapters(backends)
            .await
            .iter()
            .map(|adapter| adapter.get_info())
            .collect()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn adapter_info(&self) -> AdapterInfo {
        self.adapter.get_info()
    }

    pub fn features(&self) -> Features {
        self.features
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    pub fn supports(&self, features: Features) -> bool {
        self.features.contains(features)
    }

    /// Nanoseconds per timestamp tick.
    pub fn timestamp_period_ns(&self) -> f32 {
        self.timestamp_period_ns
    }

    /// Whether per-pass GPU timing is available.
    #[cfg(feature = "profile")]
    pub fn supports_pass_timing(&self) -> bool {
        self.supports(GpuRequest::PROFILING_FEATURES)
    }

    pub fn device_lost(&self) -> Option<DeviceLost> {
        self.health.lost.lock().unwrap().clone()
    }

    pub fn assert_alive(&self) {
        if let Some(lost) = self.device_lost() {
            panic!("gpu device lost: {lost}");
        }
    }

    pub fn compute_pipeline(
        &self,
        label: &str,
        shader: &str,
        entry: &str,
        groups: &[&[BindingSpec]],
        workgroup_size: u32,
    ) -> ComputePipeline {
        let key = PipelineKey::of(label, shader, entry, groups, workgroup_size);
        let mut cache = self.pipelines.lock().unwrap();
        if let Some(pipeline) = cache.get(&key) {
            return (**pipeline).clone();
        }
        let pipeline = Arc::new(ComputePipeline::new(
            &self.device,
            label,
            shader,
            entry,
            groups,
        ));
        cache.insert(key, pipeline.clone());
        (*pipeline).clone()
    }
}

impl Clone for GpuContext {
    fn clone(&self) -> Self {
        Self {
            adapter: self.adapter.clone(),
            device: self.device.clone(),
            queue: self.queue.clone(),
            features: self.features,
            limits: self.limits.clone(),
            timestamp_period_ns: self.timestamp_period_ns,
            health: self.health.clone(),
            pipelines: self.pipelines.clone(),
        }
    }
}

fn create_instance(backends: Backends) -> Instance {
    Instance::new(InstanceDescriptor {
        backends,
        ..InstanceDescriptor::new_without_display_handle()
    })
}

async fn select_adapter(request: &GpuRequest) -> Result<Adapter, GpuUnavailable> {
    let instance = create_instance(request.backends);
    let adapters = instance.enumerate_adapters(request.backends).await;
    if adapters.is_empty() {
        return Err(GpuUnavailable::NoAdapter {
            backends: request.backends,
        });
    }
    let candidates = match &request.device_name {
        Some(needle) => {
            let wanted = needle.to_lowercase();
            let matched = adapters
                .iter()
                .filter(|adapter| adapter.get_info().name.to_lowercase().contains(&wanted))
                .cloned()
                .collect::<Vec<_>>();
            if matched.is_empty() {
                return Err(GpuUnavailable::DeviceNotFound {
                    requested: needle.clone(),
                    available: adapters
                        .iter()
                        .map(|adapter| adapter.get_info().name)
                        .collect(),
                });
            }
            matched
        }
        None => adapters,
    };
    Ok(prefer(candidates, request.power_preference))
}

fn prefer(candidates: Vec<Adapter>, power: PowerPreference) -> Adapter {
    let mut candidates = candidates;
    let rank = |adapter: &Adapter| {
        let info = adapter.get_info();
        let device = match power {
            PowerPreference::HighPerformance => match info.device_type {
                DeviceType::DiscreteGpu => 0u8,
                DeviceType::VirtualGpu => 1,
                DeviceType::IntegratedGpu => 2,
                _ => 3,
            },
            PowerPreference::LowPower => match info.device_type {
                DeviceType::IntegratedGpu => 0u8,
                DeviceType::VirtualGpu => 1,
                DeviceType::DiscreteGpu => 2,
                _ => 3,
            },
            PowerPreference::None => 0u8,
        };
        (native_backend_rank(info.backend), device)
    };
    candidates.sort_by_key(|adapter| rank(adapter));
    candidates
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("adapter candidates must be non-empty"))
}

fn native_backend_rank(backend: Backend) -> u8 {
    #[cfg(target_os = "windows")]
    const PRIMARY: Backend = Backend::Dx12;
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "visionos"))]
    const PRIMARY: Backend = Backend::Metal;
    #[cfg(not(any(
        target_os = "windows",
        target_os = "macos",
        target_os = "ios",
        target_os = "visionos"
    )))]
    const PRIMARY: Backend = Backend::Vulkan;

    u8::from(backend != PRIMARY)
}
