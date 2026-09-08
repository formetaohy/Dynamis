use crate::{BindingKind, BindingSpec, ComputePipeline};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::{
    Adapter, Backends, Device, Instance, InstanceDescriptor, MemoryHints, PowerPreference, Queue,
};

const BACKEND_PRIORITY: [Backends; 2] = [Backends::DX12.union(Backends::METAL), Backends::VULKAN];

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

pub struct GpuContext {
    adapter: Adapter,
    device: Device,
    queue: Queue,
    pipelines: Arc<Mutex<HashMap<PipelineKey, Arc<ComputePipeline>>>>,
}

impl GpuContext {
    pub async fn new() -> Self {
        let adapter = request_adapter().await;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("dynamis device"),
                required_features: wgpu::Features::empty(),
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
            pipelines: Arc::new(Mutex::new(HashMap::new())),
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
            pipelines: self.pipelines.clone(),
        }
    }
}

async fn request_adapter() -> Adapter {
    for backends in BACKEND_PRIORITY {
        let instance = Instance::new(InstanceDescriptor {
            backends,
            ..InstanceDescriptor::new_without_display_handle()
        });
        let options = wgpu::RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            ..Default::default()
        };
        if let Ok(adapter) = instance.request_adapter(&options).await {
            return adapter;
        }
    }
    panic!("no compatible GPU adapter found for backends {BACKEND_PRIORITY:?}");
}
