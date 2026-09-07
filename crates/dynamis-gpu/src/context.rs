use wgpu::{
    Adapter, Device, Features, Instance, InstanceDescriptor, MemoryHints, PowerPreference, Queue,
};

pub struct GpuContext {
    adapter: Adapter,
    device: Device,
    queue: Queue,
}

impl Clone for GpuContext {
    fn clone(&self) -> Self {
        Self {
            adapter: self.adapter.clone(),
            device: self.device.clone(),
            queue: self.queue.clone(),
        }
    }
}

impl GpuContext {
    pub async fn new() -> Self {
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .expect("no compatible GPU adapter found");
        let mut required_features = Features::empty();
        for feature in [
            Features::TIMESTAMP_QUERY,
            Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
        ] {
            if adapter.features().contains(feature) {
                required_features |= feature;
            }
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("dynamis device"),
                required_features,
                required_limits: adapter.limits(),
                memory_hints: MemoryHints::default(),
                ..Default::default()
            })
            .await
            .expect("failed to create GPU device");
        Self {
            adapter,
            device,
            queue,
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    pub fn supports_timestamps(&self) -> bool {
        self.device
            .features()
            .contains(Features::TIMESTAMP_QUERY)
            && self
                .device
                .features()
                .contains(Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
    }
}
