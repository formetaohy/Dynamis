use crate::collision::CollisionFilter;
use crate::domain;
use crate::shape::Shape;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactEventMode {
    None,
    BeginEnd,
    Persist,
}

#[derive(Clone, Copy, Debug)]
pub struct ColliderDesc {
    pub shape: Shape,
    pub offset: [f32; 3],
    pub rotation: [f32; 4],
    pub friction: f32,
    pub restitution: f32,
    pub scale: [f32; 3],
    pub sensor: bool,
    pub filter: Option<CollisionFilter>,
    pub rolling_friction: f32,
    pub spin_friction: f32,
    pub contact_frequency: f32,
    pub contact_damping_ratio: f32,
    pub events: ContactEventMode,
    pub impact_force: Option<f32>,
}

impl ColliderDesc {
    pub fn assert_valid(&self) {
        self.shape.assert_valid();
        domain::finite_vector(self.offset, "a collider offset");
        domain::unit_quaternion(self.rotation, "a collider rotation");
        domain::non_negative(self.friction, "collider friction");
        domain::non_negative(self.restitution, "collider restitution");
        for axis in self.scale {
            domain::positive(axis, "a collider scale axis");
        }
        domain::non_negative(self.rolling_friction, "collider rolling friction");
        domain::non_negative(self.spin_friction, "collider spin friction");
        assert!(
            self.contact_frequency > 0.0,
            "a contact frequency must be strictly positive"
        );
        domain::non_negative(self.contact_damping_ratio, "a contact damping ratio");
        if let Some(force) = self.impact_force {
            domain::non_negative(force, "an impact force threshold");
        }
    }

    pub fn new(shape: Shape) -> Self {
        Self {
            shape,
            offset: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            friction: 0.5,
            restitution: 0.0,
            scale: [1.0; 3],
            sensor: false,
            filter: None,
            rolling_friction: 0.0,
            spin_friction: 0.0,
            contact_frequency: f32::INFINITY,
            contact_damping_ratio: 1.0,
            events: ContactEventMode::BeginEnd,
            impact_force: None,
        }
    }

    pub fn contact_frequency(mut self, contact_frequency: f32) -> Self {
        assert!(
            contact_frequency > 0.0,
            "a contact frequency must be strictly positive"
        );
        self.contact_frequency = contact_frequency;
        self
    }

    pub fn contact_damping_ratio(mut self, contact_damping_ratio: f32) -> Self {
        domain::non_negative(contact_damping_ratio, "a contact damping ratio");
        self.contact_damping_ratio = contact_damping_ratio;
        self
    }

    pub fn impact(mut self, force: f32) -> Self {
        domain::non_negative(force, "an impact force threshold");
        self.impact_force = Some(force);
        self
    }

    pub fn events(mut self, events: ContactEventMode) -> Self {
        self.events = events;
        self
    }

    pub fn offset(mut self, offset: [f32; 3]) -> Self {
        self.offset = offset;
        self
    }

    pub fn rotation(mut self, rotation: [f32; 4]) -> Self {
        domain::unit_quaternion(rotation, "a collider rotation");
        self.rotation = rotation;
        self
    }

    pub fn friction(mut self, friction: f32) -> Self {
        domain::non_negative(friction, "friction");
        self.friction = friction;
        self
    }

    pub fn restitution(mut self, restitution: f32) -> Self {
        domain::non_negative(restitution, "restitution");
        self.restitution = restitution;
        self
    }

    pub fn scale(mut self, scale: [f32; 3]) -> Self {
        for axis in scale {
            domain::positive(axis, "a collider scale axis");
        }
        self.scale = scale;
        self
    }

    pub fn sensor(mut self, sensor: bool) -> Self {
        self.sensor = sensor;
        self
    }

    pub fn filter(mut self, filter: CollisionFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn rolling_friction(mut self, rolling_friction: f32) -> Self {
        domain::non_negative(rolling_friction, "rolling friction");
        self.rolling_friction = rolling_friction;
        self
    }

    pub fn spin_friction(mut self, spin_friction: f32) -> Self {
        domain::non_negative(spin_friction, "spin friction");
        self.spin_friction = spin_friction;
        self
    }
}
