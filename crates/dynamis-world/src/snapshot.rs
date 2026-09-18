use super::World;
use super::backend::archive::StreamArchive;
use super::body::BodyStore;
use super::character::CharacterStore;
use super::clock::Clock;
use super::collider::ColliderStore;
use super::constraint::ConstraintStore;
use super::device::Facts;
use super::query_pool::QueryPool;
use super::shape::ShapeStore;
use super::soft::SoftBodyStore;
use super::vehicle::VehicleStore;
use dynamis_abi::COUNTER_COUNT;
use dynamis_domain::Settling;
use dynamis_model::PhysicsConfig;

#[derive(Clone)]
struct Scene {
    bodies: BodyStore,
    colliders: ColliderStore,
    constraints: ConstraintStore,
    shapes: ShapeStore,
    soft: SoftBodyStore,
    characters: CharacterStore,
    vehicles: VehicleStore,
}

impl Scene {
    fn capture(world: &World) -> Self {
        Self {
            bodies: world.bodies.clone(),
            colliders: world.colliders.clone(),
            constraints: world.constraints.clone(),
            shapes: world.shapes.clone(),
            soft: world.soft.clone(),
            characters: world.characters.clone(),
            vehicles: world.vehicles.clone(),
        }
    }

    fn restore(&self, world: &mut World) {
        world.bodies = self.bodies.clone();
        world.colliders = self.colliders.clone();
        world.constraints = self.constraints.clone();
        world.shapes = self.shapes.clone();
        world.soft = self.soft.clone();
        world.characters = self.characters.clone();
        world.vehicles = self.vehicles.clone();
    }
}

pub struct Snapshot {
    config: PhysicsConfig,
    clock: Clock,
    scene: Scene,
    streams: StreamArchive,
}

impl World {
    pub fn snapshot(&mut self) -> Snapshot {
        self.sync(Facts::Retired);
        let regions = self.backend.streams.durable_regions();
        let bytes = self.read_regions("dynamis snapshot", &regions);
        Snapshot {
            config: self.config,
            clock: self.clock,
            scene: Scene::capture(self),
            streams: StreamArchive::capture(&self.backend.streams, bytes),
        }
    }

    pub fn restore(&mut self, snapshot: &Snapshot) {
        self.sync(Facts::Retired);
        self.config = snapshot.config;
        self.clock = snapshot.clock;
        self.wake_all = false;
        snapshot.scene.restore(self);
        self.abandon_observations();
        self.restart_measures();
        let census = self.census();
        let live = self.live(&census);
        self.apply_plan(&live);
        self.install_streams(&snapshot.streams);
        self.constraints.schedule.publish();
        self.backend
            .streams
            .write(self.backend.gpu.queue(), &snapshot.streams);
        self.backend.immovable.invalidate();
        self.backend.resting.invalidate();
    }

    fn abandon_observations(&mut self) {
        let observes_whole_body_set = self.observed.observes_whole_body_set();
        self.observed.reset();
        if observes_whole_body_set {
            self.observe_all_bodies();
        }
        self.events.contact.clear();
        self.impacts.impact.clear();
        self.queries.pending.clear();
        self.queries.pool = QueryPool::new();
        self.constraints.broken.clear();
    }

    fn restart_measures(&mut self) {
        self.backend.measured = [0; COUNTER_COUNT];
        self.backend.measured_step = None;
        self.backend.settling = Settling::IDLE;
        #[cfg(feature = "profile")]
        self.backend.pass_timings.clear();
    }

    fn install_streams(&mut self, archive: &StreamArchive) {
        let (device, queue) = (
            self.backend.gpu.device().clone(),
            self.backend.gpu.queue().clone(),
        );
        self.backend
            .streams
            .install(&device, &queue, |label| archive.slots(label));
    }
}
