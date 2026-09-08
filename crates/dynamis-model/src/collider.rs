use crate::shape::Shape;

#[derive(Clone, Copy, Debug)]
pub struct ColliderDesc {
    pub shape: Shape,
    pub offset: [f32; 3],
    pub rotation: [f32; 4],
    pub friction: f32,
    pub restitution: f32,
    pub scale: [f32; 3],
    pub sensor: bool,
    pub collision_group: Option<u32>,
    pub collision_mask: Option<u32>,
    pub rolling_friction: f32,
    pub spin_friction: f32,
}

impl ColliderDesc {
    pub fn new(shape: Shape) -> Self {
        Self {
            shape,
            offset: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            friction: 0.5,
            restitution: 0.0,
            scale: [1.0; 3],
            sensor: false,
            collision_group: None,
            collision_mask: None,
            rolling_friction: 0.0,
            spin_friction: 0.0,
        }
    }

    pub fn offset(mut self, offset: [f32; 3]) -> Self {
        self.offset = offset;
        self
    }

    pub fn rotation(mut self, rotation: [f32; 4]) -> Self {
        assert!(
            (rotation[0] * rotation[0]
                + rotation[1] * rotation[1]
                + rotation[2] * rotation[2]
                + rotation[3] * rotation[3]
                - 1.0)
                .abs()
                < 1e-4,
            "rotation must be a unit quaternion"
        );
        self.rotation = rotation;
        self
    }

    pub fn friction(mut self, friction: f32) -> Self {
        assert!(friction >= 0.0, "friction must be non-negative");
        self.friction = friction;
        self
    }

    pub fn restitution(mut self, restitution: f32) -> Self {
        self.restitution = restitution;
        self
    }

    pub fn scale(mut self, scale: [f32; 3]) -> Self {
        assert!(
            scale.iter().all(|value| *value > 0.0),
            "collider scale must be strictly positive"
        );
        self.scale = scale;
        self
    }

    pub fn sensor(mut self, sensor: bool) -> Self {
        self.sensor = sensor;
        self
    }

    pub fn collision_group(mut self, group: u32) -> Self {
        self.collision_group = Some(group);
        self
    }

    pub fn collision_mask(mut self, mask: u32) -> Self {
        self.collision_mask = Some(mask);
        self
    }

    pub fn rolling_friction(mut self, rolling_friction: f32) -> Self {
        assert!(
            rolling_friction >= 0.0,
            "rolling friction must be non-negative"
        );
        self.rolling_friction = rolling_friction;
        self
    }

    pub fn spin_friction(mut self, spin_friction: f32) -> Self {
        assert!(spin_friction >= 0.0, "spin friction must be non-negative");
        self.spin_friction = spin_friction;
        self
    }
}
