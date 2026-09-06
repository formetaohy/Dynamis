use crate::shape::ShapeDesc;

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

#[derive(Clone, Copy)]
pub struct BodyDesc {
    pub shape: ShapeDesc,
    pub position: [f32; 3],
    pub orientation: [f32; 4],
    pub velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub mass: f32,
    pub restitution: f32,
    pub friction: f32,
    pub collision_group: u32,
    pub collision_mask: u32,
    pub kinematic: bool,
}

impl BodyDesc {
    pub fn sphere(radius: f32) -> Self {
        assert!(radius > 0.0, "collider radius must be strictly positive");
        Self::new(ShapeDesc::sphere(radius))
    }

    pub fn cuboid(half_extents: [f32; 3]) -> Self {
        assert!(
            half_extents.iter().all(|extent| *extent > 0.0),
            "box half extents must be strictly positive"
        );
        Self::new(ShapeDesc::cuboid(half_extents))
    }

    pub fn capsule(radius: f32, half_height: f32) -> Self {
        assert!(radius > 0.0, "capsule radius must be strictly positive");
        assert!(
            half_height >= 0.0,
            "capsule half height must be non-negative"
        );
        Self::new(ShapeDesc::capsule(radius, half_height))
    }

    pub fn new(shape: ShapeDesc) -> Self {
        Self {
            shape,
            position: [0.0; 3],
            orientation: [0.0, 0.0, 0.0, 1.0],
            velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            mass: 1.0,
            restitution: 0.0,
            friction: 0.5,
            collision_group: DEFAULT_COLLISION_GROUP,
            collision_mask: DEFAULT_COLLISION_MASK,
            kinematic: false,
        }
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

    pub fn restitution(mut self, restitution: f32) -> Self {
        self.restitution = restitution;
        self
    }

    pub fn friction(mut self, friction: f32) -> Self {
        assert!(friction >= 0.0, "friction must be non-negative");
        self.friction = friction;
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
}
