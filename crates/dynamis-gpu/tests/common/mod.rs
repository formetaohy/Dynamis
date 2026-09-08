use dynamis_gpu::GpuContext;
use std::sync::OnceLock;

static CONTEXT: OnceLock<GpuContext> = OnceLock::new();

pub fn shared() -> &'static GpuContext {
    CONTEXT.get_or_init(|| pollster::block_on(GpuContext::new()))
}
