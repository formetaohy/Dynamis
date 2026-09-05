#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BodyHandle {
    pub(crate) id: u32,
    pub(crate) generation: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BodyState {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub inverse_mass: f32,
}

#[derive(Clone, Copy)]
pub struct BodyDesc {
    pub(crate) position: [f32; 3],
    pub(crate) velocity: [f32; 3],
    pub(crate) mass: f32,
    pub(crate) restitution: f32,
    pub(crate) radius: f32,
}

impl BodyDesc {
    pub fn sphere(radius: f32) -> Self {
        assert!(radius > 0.0, "collider radius must be strictly positive");
        Self {
            position: [0.0; 3],
            velocity: [0.0; 3],
            mass: 1.0,
            restitution: 0.0,
            radius,
        }
    }

    pub fn static_sphere(radius: f32) -> Self {
        assert!(radius > 0.0, "collider radius must be strictly positive");
        Self {
            mass: 0.0,
            ..Self::sphere(radius)
        }
    }

    pub fn position(mut self, position: [f32; 3]) -> Self {
        self.position = position;
        self
    }

    pub fn velocity(mut self, velocity: [f32; 3]) -> Self {
        self.velocity = velocity;
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
}
