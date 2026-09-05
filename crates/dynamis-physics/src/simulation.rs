use crate::body::{BodyDesc, BodyHandle, BodyState};
use crate::buffers::StageBuffers;
use crate::config::PhysicsConfig;
use crate::query::{QueryHandle, QueryHit, QueryPool};
use crate::records::{
    BodyCommandRecord, DispatchArgs, PATCH_ANGULAR_VELOCITY, PATCH_FRICTION, PATCH_INVERSE_MASS,
    PATCH_ORIENTATION, PATCH_POSITION, PATCH_RADIUS, PATCH_RESTITUTION, PATCH_VELOCITY,
    QueryRecord, QueryResultRecord, RigidBodyRecord, SimParamsRecord,
};
use crate::stages::{Stages, build_stages, encode_physics};
use bytemuck::Zeroable;
use dynamis_gpu::GpuContext;

pub struct Simulation {
    gpu: GpuContext,
    config: PhysicsConfig,
    capacity: usize,
    step_index: u64,
    alive: Vec<BodyHandle>,
    index_of: Vec<u32>,
    generations: Vec<u32>,
    free_ids: Vec<u32>,
    commands: Vec<BodyCommandRecord>,
    queries: Vec<QueryRecord>,
    query_pool: QueryPool,
    buffers: StageBuffers,
    stages: Stages,
    states: Vec<Option<BodyState>>,
}

impl Simulation {
    pub fn new(gpu: GpuContext, capacity: usize, config: PhysicsConfig) -> Self {
        assert!(capacity > 0, "simulation capacity must be positive");
        let pair_capacity = capacity * (capacity.saturating_sub(1)) / 2;
        let query_capacity = capacity * 2;
        let buffers = StageBuffers::new(gpu.device(), capacity, pair_capacity, query_capacity);
        let stages = build_stages(gpu.device(), &buffers);
        Self {
            gpu,
            config,
            capacity,
            step_index: 0,
            alive: Vec::new(),
            index_of: vec![u32::MAX; capacity],
            generations: vec![1; capacity],
            free_ids: (0..capacity as u32).rev().collect(),
            commands: Vec::new(),
            queries: Vec::new(),
            query_pool: QueryPool::new(query_capacity),
            buffers,
            stages,
            states: vec![None; capacity],
        }
    }

    pub fn config(&self) -> &PhysicsConfig {
        &self.config
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn bodies(&self) -> &[BodyHandle] {
        &self.alive
    }

    pub fn count(&self) -> usize {
        self.alive.len()
    }

    pub fn bodies_buffer(&self) -> &wgpu::Buffer {
        self.buffers.bodies.buffer()
    }

    pub fn spawn(&mut self, desc: BodyDesc) -> BodyHandle {
        let id = self
            .free_ids
            .pop()
            .expect("simulation body capacity exhausted");
        self.generations[id as usize] += 1;
        let handle = BodyHandle {
            id,
            generation: self.generations[id as usize],
        };
        let slot = self.alive.len() as u32;
        self.index_of[id as usize] = slot;
        self.alive.push(handle);
        self.commands.push(BodyCommandRecord::add(
            slot,
            RigidBodyRecord::build(&desc, handle.id, handle.generation),
        ));
        self.states[id as usize] = Some(BodyState {
            position: desc.position,
            orientation: desc.orientation,
            velocity: desc.velocity,
            angular_velocity: desc.angular_velocity,
            inverse_mass: if desc.mass > 0.0 {
                1.0 / desc.mass
            } else {
                0.0
            },
            step: self.step_index,
        });
        handle
    }

    pub fn remove(&mut self, handle: BodyHandle) {
        self.validate(handle);
        let id = handle.id as usize;
        let slot = self.index_of[id] as usize;
        let tail = self.alive.len() - 1;
        let moved = self.alive[tail];
        self.alive.swap_remove(slot);
        self.index_of[moved.id as usize] = slot as u32;
        self.index_of[id] = u32::MAX;
        self.free_ids.push(handle.id);
        self.states[id] = None;
        self.commands
            .push(BodyCommandRecord::remove(slot as u32, tail as u32));
    }

    pub fn set_position(&mut self, handle: BodyHandle, position: [f32; 3]) {
        let mut record = RigidBodyRecord::zeroed();
        record.position = position;
        self.schedule_patch(handle, PATCH_POSITION, record);
    }

    pub fn set_orientation(&mut self, handle: BodyHandle, orientation: [f32; 4]) {
        assert!(
            (orientation[0] * orientation[0]
                + orientation[1] * orientation[1]
                + orientation[2] * orientation[2]
                + orientation[3] * orientation[3]
                - 1.0)
                .abs()
                < 1e-4,
            "orientation must be a unit quaternion"
        );
        let mut record = RigidBodyRecord::zeroed();
        record.orientation = orientation;
        self.schedule_patch(handle, PATCH_ORIENTATION, record);
    }

    pub fn set_velocity(&mut self, handle: BodyHandle, velocity: [f32; 3]) {
        let mut record = RigidBodyRecord::zeroed();
        record.velocity = velocity;
        self.schedule_patch(handle, PATCH_VELOCITY, record);
    }

    pub fn set_angular_velocity(&mut self, handle: BodyHandle, angular_velocity: [f32; 3]) {
        let mut record = RigidBodyRecord::zeroed();
        record.angular_velocity = angular_velocity;
        self.schedule_patch(handle, PATCH_ANGULAR_VELOCITY, record);
    }

    pub fn set_mass(&mut self, handle: BodyHandle, mass: f32) {
        assert!(mass >= 0.0, "mass must be non-negative");
        let mut record = RigidBodyRecord::zeroed();
        record.inverse_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        self.schedule_patch(handle, PATCH_INVERSE_MASS, record);
    }

    pub fn set_radius(&mut self, handle: BodyHandle, radius: f32) {
        assert!(radius > 0.0, "collider radius must be strictly positive");
        let mut record = RigidBodyRecord::zeroed();
        record.radius = radius;
        self.schedule_patch(handle, PATCH_RADIUS, record);
    }

    pub fn set_restitution(&mut self, handle: BodyHandle, restitution: f32) {
        let mut record = RigidBodyRecord::zeroed();
        record.restitution = restitution;
        self.schedule_patch(handle, PATCH_RESTITUTION, record);
    }

    pub fn set_friction(&mut self, handle: BodyHandle, friction: f32) {
        assert!(friction >= 0.0, "friction must be non-negative");
        let mut record = RigidBodyRecord::zeroed();
        record.friction = friction;
        self.schedule_patch(handle, PATCH_FRICTION, record);
    }

    pub fn apply_force(&mut self, handle: BodyHandle, force: [f32; 3]) {
        let slot = self.command_slot(handle);
        self.commands.push(BodyCommandRecord::force(slot, force));
    }

    pub fn apply_torque(&mut self, handle: BodyHandle, torque: [f32; 3]) {
        let slot = self.command_slot(handle);
        self.commands.push(BodyCommandRecord::torque(slot, torque));
    }

    pub fn apply_impulse(&mut self, handle: BodyHandle, impulse: [f32; 3]) {
        let slot = self.command_slot(handle);
        self.commands
            .push(BodyCommandRecord::impulse(slot, impulse));
    }

    pub fn apply_impulse_at_point(
        &mut self,
        handle: BodyHandle,
        impulse: [f32; 3],
        point: [f32; 3],
    ) {
        let slot = self.command_slot(handle);
        self.commands
            .push(BodyCommandRecord::impulse_at_point(slot, impulse, point));
    }

    pub fn raycast(&mut self, origin: [f32; 3], direction: [f32; 3], max_t: f32) -> QueryHandle {
        assert!(max_t > 0.0, "raycast distance must be positive");
        assert!(direction != [0.0; 3], "raycast direction must be non-zero");
        let slot = self.query_pool.allocate();
        let generation = self.query_pool.generation(slot);
        self.queries
            .push(QueryRecord::ray(origin, direction, max_t, slot as u32));
        QueryHandle {
            slot: slot as u32,
            generation,
        }
    }

    pub fn sphere_query(&mut self, center: [f32; 3], radius: f32) -> QueryHandle {
        assert!(radius > 0.0, "sphere query radius must be positive");
        let slot = self.query_pool.allocate();
        let generation = self.query_pool.generation(slot);
        self.queries
            .push(QueryRecord::sphere(center, radius, slot as u32));
        QueryHandle {
            slot: slot as u32,
            generation,
        }
    }

    pub fn query_hit(&self, handle: QueryHandle) -> Option<QueryHit> {
        let slot = handle.slot as usize;
        assert!(
            slot < self.query_pool.capacity(),
            "query handle {handle:?} is out of range"
        );
        assert_eq!(
            self.query_pool.generation(slot),
            handle.generation,
            "query handle {handle:?} is stale"
        );
        self.query_pool.hit(slot)
    }

    fn command_slot(&self, handle: BodyHandle) -> u32 {
        self.validate(handle);
        self.index_of[handle.id as usize]
    }

    fn schedule_patch(&mut self, handle: BodyHandle, mask: u32, record: RigidBodyRecord) {
        let slot = self.command_slot(handle);
        self.commands
            .push(BodyCommandRecord::patch(slot, mask, record));
    }

    fn validate(&self, handle: BodyHandle) {
        let id = handle.id as usize;
        if id >= self.capacity {
            panic!("body handle {handle:?} is out of range");
        }
        if self.generations[id] != handle.generation {
            panic!("body handle {handle:?} is stale");
        }
        if self.index_of[id] == u32::MAX {
            panic!("body handle {handle:?} is not alive");
        }
    }

    pub fn step(&mut self, dt: f32) {
        assert!(dt > 0.0, "timestep must be strictly positive");
        let step = self.step_index;
        self.collect_readbacks();
        let queue = self.gpu.queue();
        self.buffers
            .commands
            .write(queue, bytemuck::cast_slice(&self.commands));
        self.buffers.command_count.write(
            queue,
            bytemuck::cast_slice(&[DispatchArgs::sized(self.commands.len() as u32)]),
        );
        let params = SimParamsRecord::new(&self.config, dt, self.alive.len() as u32);
        self.buffers
            .params
            .write(queue, bytemuck::cast_slice(&[params]));
        self.buffers.reset_counters(queue);
        if !self.queries.is_empty() {
            self.buffers
                .queries
                .write(queue, bytemuck::cast_slice(&self.queries));
            let slots = self.queries.iter().map(|query| query.slot).collect();
            self.query_pool.mark_batch(step, slots);
        }

        let mut encoder =
            self.gpu
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dynamis step encoder"),
                });
        encode_physics(
            &self.stages,
            &self.buffers,
            &mut encoder,
            self.alive.len() as u32,
            self.config.solve_iterations,
            self.queries.len() as u32,
        );
        let stale_bodies = self.buffers.bodies_readback.enqueue(
            self.gpu.device(),
            &mut encoder,
            self.buffers.bodies.buffer(),
            step,
        );
        let stale_queries = if self.queries.is_empty() {
            None
        } else {
            self.buffers.queries_readback.enqueue(
                self.gpu.device(),
                &mut encoder,
                self.buffers.query_results.buffer(),
                step,
            )
        };
        self.gpu.queue().submit([encoder.finish()]);
        self.buffers.bodies_readback.arm();
        self.buffers.queries_readback.arm();
        if let Some((stale_step, bytes)) = stale_bodies {
            self.consume_bodies(stale_step, &bytes);
        }
        if let Some((stale_step, bytes)) = stale_queries {
            self.consume_queries(stale_step, &bytes);
        }
        self.commands.clear();
        self.queries.clear();
        self.step_index += 1;
    }

    pub fn poll(&mut self) {
        self.collect_readbacks();
    }

    pub fn wait(&mut self) {
        self.gpu
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device lost while awaiting readback");
        self.collect_readbacks();
    }

    pub fn read_state(&self, handle: BodyHandle) -> BodyState {
        self.validate(handle);
        self.states[handle.id as usize].expect("body state is unavailable")
    }

    fn collect_readbacks(&mut self) {
        for (step, bytes) in self.buffers.bodies_readback.poll(self.gpu.device()) {
            self.consume_bodies(step, &bytes);
        }
        for (step, bytes) in self.buffers.queries_readback.poll(self.gpu.device()) {
            self.consume_queries(step, &bytes);
        }
    }

    fn consume_queries(&mut self, step: u64, bytes: &[u8]) {
        let records: &[QueryResultRecord] = bytemuck::cast_slice(bytes);
        self.query_pool.consume(step, records);
    }

    fn consume_bodies(&mut self, step: u64, bytes: &[u8]) {
        let records: &[RigidBodyRecord] = bytemuck::cast_slice(bytes);
        for record in records {
            let id = record.body_id as usize;
            assert!(
                id < self.capacity,
                "GPU readback returned an out-of-range body id"
            );
            if record.generation != self.generations[id] {
                continue;
            }
            if self.index_of[id] == u32::MAX {
                continue;
            }
            self.states[id] = Some(BodyState {
                position: record.position,
                orientation: record.orientation,
                velocity: record.velocity,
                angular_velocity: record.angular_velocity,
                inverse_mass: record.inverse_mass,
                step,
            });
        }
    }
}
