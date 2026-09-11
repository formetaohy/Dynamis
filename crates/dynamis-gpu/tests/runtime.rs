use dynamis_gpu::{Backends, BufferReadback, GpuBuffer, GpuContext, GpuRequest, GpuRuntime};
use std::sync::Barrier;
use wgpu::BufferUsages;

#[test]
fn concurrent_cold_start_and_backend_discovery_share_one_runtime() {
    const WORKERS: usize = 8;
    let start = Barrier::new(WORKERS);
    std::thread::scope(|scope| {
        let workers = (0..WORKERS)
            .map(|worker| {
                let start = &start;
                scope.spawn(move || {
                    start.wait();
                    let request = if worker % 2 == 0 {
                        GpuRequest::default()
                    } else {
                        GpuRequest::minimum_limits()
                    };
                    let context = pollster::block_on(GpuContext::open(&request)).unwrap();
                    let runtime = pollster::block_on(GpuRuntime::shared());
                    for backends in [
                        Backends::DX12,
                        Backends::VULKAN,
                        Backends::METAL,
                        Backends::empty(),
                    ] {
                        let adapters = pollster::block_on(GpuContext::available_adapters(backends));
                        assert!(
                            adapters
                                .iter()
                                .all(|adapter| backends.contains(adapter.backend.into()))
                        );
                    }
                    let buffer = GpuBuffer::new(
                        context.device(),
                        "runtime probe",
                        4,
                        BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    );
                    let expected = (worker as u32).to_le_bytes();
                    buffer.write(context.queue(), &expected);
                    let mut readback = BufferReadback::new(context.device(), "runtime probe", 4);
                    assert_eq!(
                        readback.read(context.queue(), buffer.buffer(), 0, 4),
                        expected
                    );
                    context.assert_alive();
                    runtime
                })
            })
            .collect::<Vec<_>>();
        let runtimes = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        assert!(
            runtimes
                .iter()
                .all(|runtime| std::ptr::eq(*runtime, runtimes[0]))
        );
    });
}
