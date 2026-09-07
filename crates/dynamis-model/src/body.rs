use crate::collider::ColliderDesc;
use crate::shape::Shape;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyHandle {
    pub id: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyState {
    pub position: [f32; 3],
    pub orientation: [f32; 4],
    pub velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub inverse_mass: f32,
    pub sleeping: bool,
    pub step: u64,
}

pub const DEFAULT_COLLISION_GROUP: u32 = 0x0000_0001;
pub const DEFAULT_COLLISION_MASK: u32 = 0xFFFF_FFFF;
pub const BODY_DESC_COLLIDERS_MAX: usize = 4;

#[derive(Clone, Debug)]
pub struct BodyDesc {
    pub colliders: Vec<ColliderDesc>,
    pub position: [f32; 3],
    pub orientation: [f32; 4],
    pub velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub mass: f32,
    pub collision_group: u32,
    pub collision_mask: u32,
    pub kinematic: bool,
    pub ccd: bool,
}

impl BodyDesc {
    pub fn new(collider: ColliderDesc) -> Self {
        Self {
            colliders: vec![collider],
            position: [0.0; 3],
            orientation: [0.0, 0.0, 0.0, 1.0],
            velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            mass: 1.0,
            collision_group: DEFAULT_COLLISION_GROUP,
            collision_mask: DEFAULT_COLLISION_MASK,
            kinematic: false,
            ccd: false,
        }
    }

    pub fn collider(mut self, collider: ColliderDesc) -> Self {
        assert!(
            self.colliders.len() < BODY_DESC_COLLIDERS_MAX,
            "a body supports at most four colliders"
        );
        self.colliders.push(collider);
        self
    }

    pub fn sphere(radius: f32) -> Self {
        Self::new(ColliderDesc::new(Shape::sphere(radius)))
    }

    pub fn cuboid(half_extents: [f32; 3]) -> Self {
        Self::new(ColliderDesc::new(Shape::cuboid(half_extents)))
    }

    pub fn capsule(radius: f32, half_height: f32) -> Self {
        Self::new(ColliderDesc::new(Shape::capsule(radius, half_height)))
    }

    pub fn cylinder(radius: f32, half_height: f32) -> Self {
        Self::new(ColliderDesc::new(Shape::cylinder(radius, half_height)))
    }

    pub fn static_sphere(radius: f32) -> Self {
        Self {
            mass: 0.0,
            ..Self::sphere(radius)
        }
    }

    pub fn position(mut self, position: [f32; 3]) -> Self {
        self.position = position;
        self
    }

    pub fn restitution(mut self, restitution: f32) -> Self {
        self.colliders[0].restitution = restitution;
        self
    }

    pub fn friction(mut self, friction: f32) -> Self {
        assert!(friction >= 0.0, "friction must be non-negative");
        self.colliders[0].friction = friction;
        self
    }

    pub fn sensor(mut self, sensor: bool) -> Self {
        assert!(
            !self.colliders.is_empty(),
            "a body needs at least one collider"
        );
        self.colliders[0].sensor = sensor;
        self
    }

    pub fn orientation(mut self, orientation: [f32; 4]) -> Self {
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
        self.orientation = orientation;
        self
    }

    pub fn velocity(mut self, velocity: [f32; 3]) -> Self {
        self.velocity = velocity;
        self
    }

    pub fn angular_velocity(mut self, angular_velocity: [f32; 3]) -> Self {
        self.angular_velocity = angular_velocity;
        self
    }

    pub fn mass(mut self, mass: f32) -> Self {
        assert!(mass >= 0.0, "mass must be non-negative");
        self.mass = mass;
        self
    }

    pub fn collision_group(mut self, group: u32) -> Self {
        self.collision_group = group;
        self
    }

    pub fn collision_mask(mut self, mask: u32) -> Self {
        self.collision_mask = mask;
        self
    }

    pub fn kinematic(mut self, kinematic: bool) -> Self {
        self.kinematic = kinematic;
        self
    }

    pub fn ccd(mut self, ccd: bool) -> Self {
        self.ccd = ccd;
        self
    }
}
