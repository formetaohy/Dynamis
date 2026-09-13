mod arena;
mod backend;
mod body;
mod clock;
mod colliders;
mod commands;
mod constraint;
mod event;
mod ids;
mod query;
mod query_pool;
mod readback;
mod rows;
mod shape;
mod shape_pool;
mod soft;
mod step;
mod upload;
mod view;

use backend::Backend;
use body::Bodies;
use clock::Clock;
use colliders::ColliderPool;
use constraint::Constraints;
use dynamis_gpu::{GpuBuffer, GpuContext, WarmupBudget, WarmupProgress};
use dynamis_model::{BodyHandle, PhysicsConfig};
use dynamis_pass::EVENT_SLOTS;
use dynamis_state::Live;
use event::Events;
use query::Queries;
use shape::Shapes;
use soft::SoftBodies;
use view::View;

pub use backend::StreamCapacity;
pub use dynamis_soft::SoftCapacity;
pub use dynamis_state::ShapeCapacity;
pub use query_pool::{QueryHandle, QueryHit};
pub use readback::{ContactManifold, ContactPoint};

pub struct World {
    config: PhysicsConfig,

    clock: Clock,
    backend: Backend,
    bodies: Bodies,
    colliders: ColliderPool,
    constraints: Constraints,
    shapes: Shapes,
    queries: Queries,
    events: Events,
    view: View,
    soft: SoftBodies,
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
            events: Events::new(),
            view: View::new(),
            soft: SoftBodies::new(),
        }
    }

    pub fn config(&self) -> &PhysicsConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: PhysicsConfig) {
        config.assert_valid();
        self.config = config;
    }

    pub fn set_gravity(&mut self, gravity: [f32; 3]) {
        self.config.gravity = gravity;
    }

    pub fn stream_capacity(&self) -> StreamCapacity {
        self.backend.streams.stream_capacity()
    }

    pub fn bodies(&self) -> &[BodyHandle] {
        &self.bodies.alive
    }

    pub fn count(&self) -> usize {
        self.bodies.alive.len()
    }

    pub(crate) fn live(&self) -> Live {
        let (particles, elements, adjacency) = self.soft.used();
        Live {
            bodies: self.bodies.alive.len() as u32,
            colliders: self.colliders.live(),
            collider_pool: self.colliders.used(),
            body_ids: self.bodies.ids.len() as u32,
            constraints: self.constraints.alive.len() as u32,
            body_commands: self.bodies.commands.len() as u32,
            constraint_commands: self.constraints.commands.len() as u32,
            queries: self.queries.pending.len() as u32,
            shapes: self.shapes.pool.used(),
            particles,
            elements,
            adjacency,
        }
    }

    pub(crate) fn soft_active(&self) -> bool {
        self.soft.count() > 0
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

    pub(crate) fn event_slot_of(&self, step: u64) -> u32 {
        (step % EVENT_SLOTS as u64) as u32
    }
}
