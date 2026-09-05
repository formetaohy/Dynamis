use crate::body::{BodyDesc, BodyHandle, BodyState};
use crate::config::PhysicsConfig;
use crate::records::{
    BodyCommandRecord, DispatchCount, PATCH_ANGULAR_VELOCITY, PATCH_FRICTION, PATCH_INVERSE_MASS,
    PATCH_ORIENTATION, PATCH_POSITION, PATCH_RADIUS, PATCH_RESTITUTION, PATCH_VELOCITY,
    RigidBodyRecord, SimParamsRecord,
};
use crate::stages::{StageBuffers, Stages, build_stages, record_physics};
use bytemuck::Zeroable;
use dynamis_gpu::GpuContext;

pub struct Simulation {
    gpu: GpuContext,
    config: PhysicsConfig,
    capacity: usize,
    frame: u64,
    feedback_frame: Option<u64>,
    alive: Vec<BodyHandle>,
    index_of: Vec<u32>,
    generations: Vec<u32>,
    free_ids: Vec<u32>,
    commands: Vec<BodyCommandRecord>,
    data: StageBuffers,
    stages: Stages,
    states: Vec<BodyState>,
}

impl Simulation {
    pub fn new(gpu: GpuContext, capacity: usize, config: PhysicsConfig) -> Self {
        assert!(capacity > 0, "simulation capacity must be positive");
        let pair_capacity = capacity * (capacity.saturating_sub(1)) / 2;
        let data = StageBuffers::new(gpu.device(), capacity, pair_capacity);
        let stages = build_stages(gpu.device(), &data);
        Self {
            gpu,
            config,
            capacity,
            frame: 0,
            feedback_frame: None,
            alive: Vec::new(),
            index_of: vec![u32::MAX; capacity],
            generations: vec![1; capacity],
            free_ids: (0..capacity as u32).rev().collect(),
            commands: Vec::new(),
            data,
            stages,
            states: vec![BodyState::default(); capacity],
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
        self.commands
            .push(BodyCommandRecord::add(slot, RigidBodyRecord::build(&desc)));
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
        let queue = self.gpu.queue();
        self.data
            .commands
            .write(queue, bytemuck::cast_slice(&self.commands));
        self.data.command_count.write(
            queue,
            bytemuck::cast_slice(&[DispatchCount::sized(self.commands.len() as u32)]),
        );
        let params = SimParamsRecord::new(&self.config, dt, self.alive.len() as u32);
        self.data
            .params
            .write(queue, bytemuck::cast_slice(&[params]));
        self.data.reset_counters(queue);

        let mut encoder =
            self.gpu
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dynamis step encoder"),
                });
        record_physics(
            &self.stages,
            &self.data,
            &mut encoder,
            self.alive.len() as u32,
            self.config.solve_iterations,
        );
        self.data
            .readback
            .enqueue(&mut encoder, self.data.bodies.buffer());
        self.gpu.queue().submit([encoder.finish()]);
        self.commands.clear();
        self.frame += 1;
    }

    fn refresh_feedback(&mut self) {
        if self.feedback_frame == Some(self.frame) {
            return;
        }
        let bytes = self.data.readback.read(self.gpu.device());
        let records: &[RigidBodyRecord] = bytemuck::cast_slice(&bytes);
        for (state, record) in self.states.iter_mut().zip(records) {
            state.position = record.position;
            state.orientation = record.orientation;
            state.velocity = record.velocity;
            state.angular_velocity = record.angular_velocity;
            state.inverse_mass = record.inverse_mass;
        }
        self.feedback_frame = Some(self.frame);
    }

    pub fn read_state(&mut self, handle: BodyHandle) -> BodyState {
        self.validate(handle);
        self.refresh_feedback();
        self.states[self.index_of[handle.id as usize] as usize]
    }

    pub fn states(&self) -> &[BodyState] {
        &self.states
    }
}
