use super::{Shapes, Simulation};
use crate::buffers::WorldBuffers;
use crate::capacity::{Capacity, Reservation, ShapeReservation};
use crate::pipeline::Pipeline;
use dynamis_gpu::GpuContext;
#[cfg(feature = "profile")]
use dynamis_gpu::GpuPassTiming;
use dynamis_layout::{COUNTER_COUNT, COUNTER_PREV_CONTACTS, COUNTER_STRIDE, Counters};
use dynamis_model::MAX_COLLIDERS_PER_BODY;

pub(crate) struct Device {
    pub(crate) gpu: GpuContext,
    pub(crate) buffers: WorldBuffers,
    pub(crate) pipeline: Pipeline,
    pub(crate) reservation: Reservation,
    pub(crate) shape_reservation: ShapeReservation,
    pub(crate) capacity: Capacity,
    pub(crate) measured: Counters,
    pub(crate) sync_staging: Option<(wgpu::Buffer, u64)>,
    #[cfg(feature = "profile")]
    pub(crate) pass_timings: Vec<GpuPassTiming>,
}

impl Device {
    pub(crate) fn new(gpu: GpuContext, shapes: &Shapes) -> Self {
        let reservation = Reservation::initial();
        let shape_reservation =
            ShapeReservation::planned(&ShapeReservation::EMPTY, &shapes.pool.used());
        let buffers = WorldBuffers::new(gpu.device(), &reservation, &shape_reservation);
        let pipeline = Pipeline::new(&gpu, &buffers, &reservation);
        Self {
            gpu,
            buffers,
            pipeline,
            reservation,
            shape_reservation,
            capacity: Capacity::new(),
            measured: [0; COUNTER_COUNT],
            sync_staging: None,
            #[cfg(feature = "profile")]
            pass_timings: Vec::new(),
        }
    }
}

impl Simulation {
    /// A reallocation destroys the buffers holding in-flight reads, so every pending
    /// result is brought home first.
    pub(crate) fn apply_plan(&mut self) {
        let demand = self
            .device
            .capacity
            .observe(&self.device.measured, &self.device.reservation);
        let next = Reservation::planned(&self.device.reservation, &self.live(), demand);
        let shapes =
            ShapeReservation::planned(&self.device.shape_reservation, &self.shapes.pool.used());
        if next == self.device.reservation && shapes == self.device.shape_reservation {
            return;
        }
        self.device.reservation = next;
        self.device.shape_reservation = shapes;
        self.drain_readbacks();
        self.rebuild();
    }

    pub(crate) fn rebuild(&mut self) {
        let buffers = WorldBuffers::new(
            self.device.gpu.device(),
            &self.device.reservation,
            &self.device.shape_reservation,
        );

        self.transfer_device_state(&buffers);

        self.upload_host_state(&buffers);
        self.shapes.pool.reset_upload_cursor();
        self.shapes.pool.upload_pending(
            self.device.gpu.queue(),
            &buffers.shapes.sources,
            &buffers.shapes.vertices,
            &buffers.shapes.triangles,
            &buffers.shapes.nodes,
        );

        self.device.pipeline = Pipeline::new(&self.device.gpu, &buffers, &self.device.reservation);

        self.device.buffers = buffers;
    }

    /// Device-owned rows survive a reallocation untouched; only their storage moves.
    fn transfer_device_state(&self, next: &WorldBuffers) {
        let previous = &self.device.buffers;
        let bodies = previous.bodies.states.size().min(next.bodies.states.size());
        let aabbs = previous.bodies.aabbs.size().min(next.bodies.aabbs.size());
        let contacts = previous
            .contacts
            .manifolds
            .size()
            .min(next.contacts.manifolds.size());
        let island = previous
            .islands
            .parents
            .size()
            .min(next.islands.parents.size());
        let wake_flags = previous
            .islands
            .wake_flags
            .size()
            .min(next.islands.wake_flags.size());
        let constraints = previous
            .constraints
            .runtime
            .size()
            .min(next.constraints.runtime.size());
        let mut encoder =
            self.device
                .gpu
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dynamis reallocate"),
                });
        let mut copy =
            |source: &dynamis_gpu::GpuBuffer, target: &dynamis_gpu::GpuBuffer, bytes: u64| {
                encoder.copy_buffer_to_buffer(source.buffer(), 0, target.buffer(), 0, bytes);
            };
        copy(&previous.bodies.states, &next.bodies.states, bodies);
        copy(&previous.bodies.aabbs, &next.bodies.aabbs, aabbs);
        copy(
            &previous.contacts.previous,
            &next.contacts.previous,
            contacts,
        );
        copy(
            &previous.islands.wake_flags,
            &next.islands.wake_flags,
            wake_flags,
        );
        copy(&previous.islands.parents, &next.islands.parents, island);
        copy(&previous.islands.state, &next.islands.state, island);
        copy(
            &previous.constraints.runtime,
            &next.constraints.runtime,
            constraints,
        );
        let (offset, width) = (COUNTER_PREV_CONTACTS as u64 * COUNTER_STRIDE, 4);
        encoder.copy_buffer_to_buffer(
            previous.counters.buffer(),
            offset,
            next.counters.buffer(),
            offset,
            width,
        );
        self.device.gpu.queue().submit([encoder.finish()]);
    }

    /// Host-owned rows are re-emitted in slot order from the host authority.
    fn upload_host_state(&self, next: &WorldBuffers) {
        let queue = self.device.gpu.queue();
        let mut descriptors = Vec::with_capacity(self.bodies.alive.len());
        let mut colliders = Vec::with_capacity(self.bodies.alive.len() * MAX_COLLIDERS_PER_BODY);
        for handle in &self.bodies.alive {
            descriptors.push(self.bodies.descriptors[handle.id as usize]);
            colliders.extend_from_slice(&self.collider_block_of(handle.id as usize));
        }
        next.bodies
            .descriptors
            .write(queue, bytemuck::cast_slice(&descriptors));
        next.bodies
            .colliders
            .write(queue, bytemuck::cast_slice(&colliders));
        next.constraints.descriptors.write(
            queue,
            bytemuck::cast_slice(&self.constraints.records[..self.constraints.alive.len()]),
        );
    }
}
