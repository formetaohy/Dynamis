use crate::components::{Mass, Restitution, SphereCollider, Transform, Velocity};
use crate::records::{RigidBodyRecord, SimParamsRecord};
use crate::stages::{StageBuffers, Stages, build_stages, record_physics};
use dynamis_ecs::{Entity, World};
use dynamis_gpu::GpuContext;

pub struct PhysicsConfig {
    pub gravity: [f32; 3],
    pub damping: f32,
    pub solve_iterations: u32,
    pub relaxation: f32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            gravity: [0.0, -9.81, 0.0],
            damping: 0.05,
            solve_iterations: 12,
            relaxation: 0.8,
        }
    }
}

pub struct Simulation {
    gpu: GpuContext,
    config: PhysicsConfig,
    capacity: usize,
    bodies: Vec<Entity>,
    data: StageBuffers,
    stages: Stages,
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
            bodies: Vec::new(),
            data,
            stages,
        }
    }

    pub fn config(&self) -> &PhysicsConfig {
        &self.config
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn bodies(&self) -> &[Entity] {
        &self.bodies
    }

    pub fn add(&mut self, entity: Entity) {
        assert!(
            self.bodies.len() < self.capacity,
            "attached bodies exceed simulation capacity"
        );
        self.bodies.push(entity);
    }

    pub fn remove(&mut self, entity: Entity) {
        self.bodies.retain(|candidate| *candidate != entity);
    }

    fn collect_records(&self, world: &World) -> Vec<RigidBodyRecord> {
        self.bodies
            .iter()
            .map(|entity| {
                let transform = world
                    .get::<Transform>(*entity)
                    .unwrap_or_else(|| panic!("attached body {entity:?} lacks Transform"));
                let velocity = world
                    .get::<Velocity>(*entity)
                    .unwrap_or_else(|| panic!("attached body {entity:?} lacks Velocity"));
                let mass = world
                    .get::<Mass>(*entity)
                    .unwrap_or_else(|| panic!("attached body {entity:?} lacks Mass"));
                let restitution = world
                    .get::<Restitution>(*entity)
                    .copied()
                    .unwrap_or_default();
                let collider = world
                    .get::<SphereCollider>(*entity)
                    .unwrap_or_else(|| panic!("attached body {entity:?} lacks SphereCollider"));
                RigidBodyRecord::build(transform, velocity, mass, restitution, collider)
            })
            .collect()
    }

    pub fn step(&mut self, world: &World, dt: f32) {
        assert!(dt > 0.0, "timestep must be strictly positive");
        let records = self.collect_records(world);
        let body_count = records.len() as u32;
        self.data
            .bodies
            .write(self.gpu.queue(), bytemuck::cast_slice(&records));
        let params = SimParamsRecord::new(
            self.config.gravity,
            dt,
            self.config.damping,
            body_count,
            self.config.relaxation,
        );
        self.data
            .params
            .write(self.gpu.queue(), bytemuck::cast_slice(&[params]));
        self.data.reset_counters(self.gpu.queue());

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
            body_count,
            self.config.solve_iterations,
        );
        self.gpu.queue().submit([encoder.finish()]);
    }

    pub fn sync_back(&mut self, world: &mut World) {
        let bytes = self
            .data
            .bodies
            .read_sync(self.gpu.device(), self.gpu.queue());
        let records: &[RigidBodyRecord] = bytemuck::cast_slice(&bytes);
        for (entity, record) in self.bodies.iter().zip(records) {
            if !world.alive(*entity) {
                continue;
            }
            let transform = world
                .get_mut::<Transform>(*entity)
                .unwrap_or_else(|| panic!("attached body {entity:?} lost Transform"));
            transform.position = record.position;
            let velocity = world
                .get_mut::<Velocity>(*entity)
                .unwrap_or_else(|| panic!("attached body {entity:?} lost Velocity"));
            velocity.linear = record.velocity;
        }
    }
}
