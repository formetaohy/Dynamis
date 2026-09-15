mod arena;
mod backend;
mod body;
mod character;
mod clock;
mod colliders;
mod commands;
mod constraint;
mod event;
mod facts;
mod ids;
mod impact;
mod journal;
mod observation;
mod query;
mod query_pool;
mod readback;
mod rows;
mod run;
mod scene;
mod shape;
mod shape_pool;
mod snapshot;
mod soft;
mod step;
mod upload;
mod vehicle;

use backend::Backend;
use body::Bodies;
use character::Characters;
use clock::Clock;
use colliders::ColliderPool;
use commands::BodyCommand;
use constraint::Constraints;
use dynamis_gpu::{GpuBuffer, GpuContext, WarmupBudget, WarmupProgress};
use dynamis_model::{BodyHandle, PhysicsConfig};
use event::Events;
use impact::Impacts;
use observation::Observations;
use query::Queries;
use shape::Shapes;
use soft::SoftBodies;
use vehicle::Vehicles;

pub use backend::StreamCapacity;
pub use dynamis_model::SceneTarget;
pub use dynamis_rigid::RigidShape;
pub use dynamis_soft::SoftCapacity;
pub use dynamis_state::{ShapeCapacity, StateCapacity};
pub use observation::Observation;
pub use query_pool::{QueryHandle, QueryHit, QueryState};
pub use readback::{ConstraintForce, ContactManifold, ContactPoint};
pub use scene::SceneCapacity;
pub use snapshot::Snapshot;

pub struct World {
    config: PhysicsConfig,

    clock: Clock,
    backend: Backend,
    bodies: Bodies,
    colliders: ColliderPool,
    constraints: Constraints,
    shapes: Shapes,
    queries: Queries,
    characters: Characters,
    events: Events,
    impacts: Impacts,
    observed: Observations,
    soft: SoftBodies,
    vehicles: Vehicles,
}

impl World {
    pub fn new(gpu: GpuContext, config: PhysicsConfig) -> Self {
        config.assert_valid();
        let shapes = Shapes::new();
        let backend = Backend::new(gpu);
        Self {
            config,
            clock: Clock::new(),
            backend,
            bodies: Bodies::new(),
            colliders: ColliderPool::new(),
            constraints: Constraints::new(),
            shapes,
            queries: Queries::new(),
            characters: Characters::new(),
            events: Events::new(),
            impacts: Impacts::new(),
            observed: Observations::new(),
            soft: SoftBodies::new(),
            vehicles: Vehicles::new(),
        }
    }

    pub fn config(&self) -> &PhysicsConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: PhysicsConfig) {
        config.assert_valid();
        self.config = config;
        self.wake_all();
    }

    pub fn set_gravity(&mut self, gravity: [f32; 3]) {
        self.config.gravity = gravity;
        self.wake_all();
    }

    fn wake_all(&mut self) {
        for row in 0..self.bodies.alive.len() as u32 {
            self.bodies.commands.push(BodyCommand::Wake { row });
        }
        self.soft.wake_all();
    }

    pub fn stream_capacity(&self) -> StreamCapacity {
        self.backend.streams.capacity()
    }

    pub fn bodies(&self) -> &[BodyHandle] {
        &self.bodies.alive
    }

    pub fn count(&self) -> usize {
        self.bodies.alive.len()
    }

    pub fn is_idle(&self) -> bool {
        !self.backend.published && !self.busy(&self.host_work())
    }

    pub fn state_buffer(&self) -> &GpuBuffer {
        self.backend.streams.state.body_states.gpu()
    }

    pub fn collider_buffer(&self) -> &GpuBuffer {
        self.backend.streams.state.colliders.gpu()
    }

    pub fn gpu(&self) -> &GpuContext {
        &self.backend.gpu
    }

    pub fn warmup(&self, budget: WarmupBudget) -> WarmupProgress {
        self.backend.gpu.warmup(budget)
    }

    pub fn is_warm(&self) -> bool {
        self.backend.gpu.is_warm()
    }

    pub fn pass_labels(&self) -> Vec<&'static str> {
        self.backend
            .passes
            .declared()
            .iter()
            .map(|pass| pass.label)
            .collect()
    }

    pub fn ran_passes(&self) -> Vec<&'static str> {
        self.backend.passes.ran()
    }
}
