mod arena;
mod backend;
mod body;
mod character;
mod clock;
mod collider;
mod command;
mod constraint;
mod derivation;
mod device;
mod event;
mod fact;
mod field;
mod host;
mod id;
mod impact;
mod journal;
mod observation;
mod pool;
mod query;
mod query_pool;
mod readback;
mod row;
mod run;
mod schedule;
mod shape;
mod shape_pool;
mod snapshot;
mod soft;
mod step;
mod upload;
mod vehicle;

use backend::Backend;
use body::BodyStore;
use character::CharacterStore;
use clock::Clock;
use collider::ColliderStore;
use constraint::ConstraintStore;
use derivation::SceneFacts;
use dynamis_gpu::{GpuBuffer, GpuContext, WarmupBudget, WarmupProgress};
use dynamis_model::domain;
use dynamis_model::{BodyHandle, PhysicsConfig};
use event::EventStore;
use field::FieldStore;
use impact::ImpactStore;
use observation::ObservationStore;
use query::QueryStore;
use shape::ShapeStore;
use soft::SoftBodyStore;
use vehicle::VehicleStore;

pub use backend::StreamCapacity;
pub use dynamis_model::SceneTarget;
pub use dynamis_rigid::RigidShape;
pub use dynamis_scene::SceneCapacity;
pub use dynamis_soft::SoftCapacity;
pub use dynamis_state::{ShapeCapacity, StateCapacity};
pub use observation::Observation;
pub use query_pool::{QueryHandle, QueryHit, QueryState};
pub use readback::{ConstraintForce, ContactManifold, ContactPoint, Refusal, SolveResidual};
pub use snapshot::Snapshot;

pub struct World {
    config: PhysicsConfig,
    wake_all: bool,
    facts: SceneFacts,

    clock: Clock,
    backend: Backend,
    bodies: BodyStore,
    colliders: ColliderStore,
    constraints: ConstraintStore,
    shapes: ShapeStore,
    queries: QueryStore,
    characters: CharacterStore,
    events: EventStore,
    impacts: ImpactStore,
    fields: FieldStore,
    observed: ObservationStore,
    soft: SoftBodyStore,
    vehicles: VehicleStore,
}

impl World {
    pub fn new(gpu: GpuContext, config: PhysicsConfig) -> Self {
        config.assert_valid();
        let shapes = ShapeStore::new();
        let backend = Backend::new(gpu);
        Self {
            config,
            wake_all: false,
            facts: SceneFacts::default(),
            clock: Clock::new(),
            backend,
            bodies: BodyStore::new(),
            colliders: ColliderStore::new(),
            constraints: ConstraintStore::new(),
            shapes,
            queries: QueryStore::new(),
            characters: CharacterStore::new(),
            events: EventStore::new(),
            impacts: ImpactStore::new(),
            fields: FieldStore::new(),
            observed: ObservationStore::new(),
            soft: SoftBodyStore::new(),
            vehicles: VehicleStore::new(),
        }
    }

    pub fn config(&self) -> &PhysicsConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: PhysicsConfig) {
        config.assert_valid();
        self.config = config;
        self.encode_bodies();
        self.wake_all();
    }

    pub fn set_gravity(&mut self, gravity: [f32; 3]) {
        domain::finite_vector(gravity, "gravity");
        self.config.gravity = gravity;
        self.wake_all();
    }

    fn wake_all(&mut self) {
        self.wake_all = true;
    }

    pub fn stream_capacity(&self) -> StreamCapacity {
        self.backend.streams.capacity()
    }

    pub fn bodies(&self) -> &[BodyHandle] {
        self.bodies.pool.alive()
    }

    pub fn count(&self) -> usize {
        self.bodies.pool.len() as usize
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
