use super::World;
use crate::dynamics::Pipeline;
use crate::dynamics::rigid::buffers::RigidBuffers;
use crate::dynamics::rigid::capacity::Capacity;
use dynamis_gpu::GpuContext;
#[cfg(feature = "profile")]
use dynamis_gpu::GpuPassTiming;

pub(crate) struct Backend {
    pub(crate) gpu: GpuContext,
    pub(crate) buffers: RigidBuffers,
    pub(crate) pipeline: Pipeline,
    pub(crate) capacity: Capacity,
    pub(crate) measured: dynamis_layout::Counters,
    pub(crate) measured_step: Option<u64>,
    pub(crate) commanded_step: Option<u64>,
    pub(crate) state_readback: Option<dynamis_gpu::BufferReadback>,
    #[cfg(feature = "profile")]
    pub(crate) pass_timings: Vec<GpuPassTiming>,
}

impl Backend {
    pub(crate) fn new(gpu: GpuContext) -> Self {
        let demand = Capacity::minimum();
        let buffers = RigidBuffers::new(gpu.device(), gpu.queue(), &demand);
        let pipeline = Pipeline::new(&gpu, &buffers);
        Self {
            gpu,
            buffers,
            pipeline,
            capacity: Capacity::new(),
            measured: [0; dynamis_layout::COUNTER_COUNT],
            measured_step: None,
            commanded_step: None,
            state_readback: None,
            #[cfg(feature = "profile")]
            pass_timings: Vec::new(),
        }
    }
}

impl World {
    pub(crate) fn apply_plan(&mut self) {
        let live = self.live();
        let demand =
            self.backend
                .capacity
                .demand(&self.backend.measured, &live, &self.backend.buffers);
        if self.backend.buffers.matches(&demand) {
            return;
        }
        if !self.backend.buffers.readback_matches(&demand) {
            self.sync_events();
            self.drain_readbacks();
        }
        let device = self.backend.gpu.device().clone();
        let queue = self.backend.gpu.queue().clone();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dynamis buffer plan"),
        });
        assert!(
            self.backend.buffers.reserve(&device, &mut encoder, &demand),
            "a buffer plan that changes capacity must reallocate"
        );
        queue.submit([encoder.finish()]);
    }
}
