use async_lock::OnceCell;
use wgpu::{Adapter, Backends, Instance, InstanceDescriptor, InstanceFlags};

static RUNTIME: OnceCell<GpuRuntime> = OnceCell::new();

pub const INSTANCE_DIAGNOSTICS: InstanceFlags =
    InstanceFlags::VALIDATION_INDIRECT_CALL.union(if cfg!(debug_assertions) {
        InstanceFlags::VALIDATION
    } else {
        InstanceFlags::empty()
    });

const _: () = assert!(
    !INSTANCE_DIAGNOSTICS.contains(InstanceFlags::DEBUG),
    "the device program is declared once and must not carry backend debug codegen"
);

pub fn instance_flags() -> InstanceFlags {
    INSTANCE_DIAGNOSTICS.with_env()
}

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
                    flags: instance_flags(),
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
