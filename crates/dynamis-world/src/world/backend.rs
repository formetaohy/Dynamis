use super::World;
use crate::dynamics::Pipeline;
use crate::dynamics::buffers::WorldBuffers;
use crate::dynamics::capacity::Capacity;
use dynamis_gpu::GpuContext;
#[cfg(feature = "profile")]
use dynamis_gpu::GpuPassTiming;

pub(crate) struct Backend {
    pub(crate) gpu: GpuContext,
    pub(crate) buffers: WorldBuffers,
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
        let buffers = WorldBuffers::new(gpu.device(), gpu.queue(), &demand);
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
        self.sync_events();
        self.drain_readbacks();
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
        self.upload_host_state();
        self.shapes.pool.reset_upload_cursor();
        self.shapes.pool.upload_pending(
            &queue,
            &self.backend.buffers.shape_sources,
            &self.backend.buffers.shape_vertices,
            &self.backend.buffers.shape_triangles,
            &self.backend.buffers.shape_nodes,
        );
        self.backend.pipeline = Pipeline::new(&self.backend.gpu, &self.backend.buffers);
    }

    fn upload_host_state(&self) {
        let queue = self.backend.gpu.queue();
        let buffers = &self.backend.buffers;
        let mut descriptors = Vec::with_capacity(self.bodies.alive.len());
        for handle in &self.bodies.alive {
            descriptors.push(self.bodies.descriptors[handle.id as usize]);
        }
        buffers
            .body_descriptors
            .write(queue, bytemuck::cast_slice(&descriptors));
        buffers
            .colliders
            .write(queue, bytemuck::cast_slice(self.colliders.records()));
        let owners = self
            .colliders
            .owners()
            .iter()
            .map(|id| {
                if *id == dynamis_layout::NO_BODY {
                    return dynamis_layout::NO_BODY;
                }
                let row = self.bodies.index_of[*id as usize];
                if row == u32::MAX {
                    dynamis_layout::NO_BODY
                } else {
                    row
                }
            })
            .collect::<Vec<_>>();
        buffers
            .collider_owners
            .write(queue, bytemuck::cast_slice(&owners));
        buffers.constraint_descriptors.write(
            queue,
            bytemuck::cast_slice(&self.constraints.records[..self.constraints.alive.len()]),
        );
    }
}
