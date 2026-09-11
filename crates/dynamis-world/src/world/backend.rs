use super::{Shapes, World};
use crate::dynamics::Pipeline;
use crate::dynamics::buffers::WorldBuffers;
use crate::dynamics::capacity::{Capacity, Reservation, ShapeReservation};
use dynamis_gpu::GpuContext;
#[cfg(feature = "profile")]
use dynamis_gpu::GpuPassTiming;
use dynamis_layout::{
    COUNTER_ARCHIVED, COUNTER_COUNT, COUNTER_RESTING, COUNTER_RESTING_INDEX,
    COUNTER_RESTING_PENDING, COUNTER_STRIDE, Counters,
};
use dynamis_model::MAX_COLLIDERS_PER_BODY;

pub(crate) struct Backend {
    pub(crate) gpu: GpuContext,
    pub(crate) buffers: WorldBuffers,
    pub(crate) pipeline: Pipeline,
    pub(crate) reservation: Reservation,
    pub(crate) shape_reservation: ShapeReservation,
    pub(crate) capacity: Capacity,
    pub(crate) measured: Counters,
    pub(crate) measured_step: Option<u64>,
    pub(crate) commanded_step: Option<u64>,
    pub(crate) state_readback: Option<dynamis_gpu::BufferReadback>,
    #[cfg(feature = "profile")]
    pub(crate) pass_timings: Vec<GpuPassTiming>,
}

impl Backend {
    pub(crate) fn new(gpu: GpuContext, shapes: &Shapes) -> Self {
        let reservation = Reservation::initial();
        let shape_reservation =
            ShapeReservation::planned(&ShapeReservation::EMPTY, &shapes.pool.used());
        let buffers =
            WorldBuffers::new(gpu.device(), gpu.queue(), &reservation, &shape_reservation);
        let pipeline = Pipeline::new(&gpu, &buffers, &reservation);
        Self {
            gpu,
            buffers,
            pipeline,
            reservation,
            shape_reservation,
            capacity: Capacity::new(),
            measured: [0; COUNTER_COUNT],
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
        let demand = self
            .backend
            .capacity
            .observe(&self.backend.measured, &self.backend.reservation);
        let next = Reservation::planned(&self.backend.reservation, &self.live(), demand);
        let shapes =
            ShapeReservation::planned(&self.backend.shape_reservation, &self.shapes.pool.used());
        if next == self.backend.reservation && shapes == self.backend.shape_reservation {
            return;
        }
        self.backend.reservation = next;
        self.backend.shape_reservation = shapes;
        self.drain_readbacks();
        self.rebuild();
    }

    pub(crate) fn rebuild(&mut self) {
        let buffers = WorldBuffers::new(
            self.backend.gpu.device(),
            self.backend.gpu.queue(),
            &self.backend.reservation,
            &self.backend.shape_reservation,
        );

        self.transfer_device_state(&buffers);

        self.upload_host_state(&buffers);
        self.shapes.pool.reset_upload_cursor();
        self.shapes.pool.upload_pending(
            self.backend.gpu.queue(),
            &buffers.shapes.sources,
            &buffers.shapes.vertices,
            &buffers.shapes.triangles,
            &buffers.shapes.nodes,
        );

        self.backend.pipeline =
            Pipeline::new(&self.backend.gpu, &buffers, &self.backend.reservation);

        self.backend.buffers = buffers;
    }

    fn transfer_device_state(&self, next: &WorldBuffers) {
        let previous = &self.backend.buffers;
        let bodies = previous.bodies.states.size().min(next.bodies.states.size());
        let aabbs = previous.bodies.aabbs.size().min(next.bodies.aabbs.size());
        let rows = previous
            .bodies
            .row_of_body
            .size()
            .min(next.bodies.row_of_body.size());
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
        let resting = previous
            .contacts
            .resting
            .size()
            .min(next.contacts.resting.size());
        let resting_index = previous
            .contacts
            .resting_index
            .major
            .size()
            .min(next.contacts.resting_index.major.size());
        let resting_live = previous
            .contacts
            .resting_live
            .size()
            .min(next.contacts.resting_live.size());
        let resting_next = previous
            .contacts
            .resting_next
            .size()
            .min(next.contacts.resting_next.size());
        let resting_free = previous
            .contacts
            .resting_free
            .size()
            .min(next.contacts.resting_free.size());
        let mut encoder =
            self.backend
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
        copy(&previous.bodies.row_of_body, &next.bodies.row_of_body, rows);
        copy(&previous.bodies.aabbs, &next.bodies.aabbs, aabbs);
        copy(&previous.contacts.archive, &next.contacts.archive, contacts);
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
        copy(&previous.contacts.resting, &next.contacts.resting, resting);
        copy(
            &previous.contacts.resting_index.major,
            &next.contacts.resting_index.major,
            resting_index,
        );
        copy(
            &previous.contacts.resting_index.minor,
            &next.contacts.resting_index.minor,
            resting_index,
        );
        copy(
            &previous.contacts.resting_index.payload,
            &next.contacts.resting_index.payload,
            resting_index,
        );
        copy(
            &previous.contacts.resting_live,
            &next.contacts.resting_live,
            resting_live,
        );
        copy(
            &previous.contacts.resting_next,
            &next.contacts.resting_next,
            resting_next,
        );
        copy(
            &previous.contacts.resting_free,
            &next.contacts.resting_free,
            resting_free,
        );
        for slot in [
            COUNTER_ARCHIVED,
            COUNTER_RESTING,
            COUNTER_RESTING_INDEX,
            COUNTER_RESTING_PENDING,
        ] {
            let offset = slot as u64 * COUNTER_STRIDE;
            encoder.copy_buffer_to_buffer(
                previous.counters.buffer(),
                offset,
                next.counters.buffer(),
                offset,
                4,
            );
        }
        self.backend.gpu.queue().submit([encoder.finish()]);
    }

    fn upload_host_state(&self, next: &WorldBuffers) {
        let queue = self.backend.gpu.queue();
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
