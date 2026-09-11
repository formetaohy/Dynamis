use async_lock::OnceCell;
use wgpu::{Adapter, Backends, Instance, InstanceDescriptor};

static RUNTIME: OnceCell<GpuRuntime> = OnceCell::new();

pub struct GpuRuntime {
    instance: Instance,
    adapters: Vec<Adapter>,
}

impl GpuRuntime {
    pub async fn shared() -> &'static Self {
        RUNTIME
            .get_or_init(|| async {
                let instance = Instance::new(InstanceDescriptor {
                    backends: Backends::all(),
                    ..InstanceDescriptor::new_without_display_handle()
                });
                let adapters = instance.enumerate_adapters(Backends::all()).await;
                Self { instance, adapters }
            })
            .await
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub(crate) fn adapters(&self, backends: Backends) -> Vec<Adapter> {
        self.adapters
            .iter()
            .filter(|adapter| backends.contains(adapter.get_info().backend.into()))
            .cloned()
            .collect()
    }
}
