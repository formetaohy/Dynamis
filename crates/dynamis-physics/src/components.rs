#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
}

impl Transform {
    pub fn at(position: [f32; 3]) -> Self {
        Self {
            position,
            rotation: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::at([0.0, 0.0, 0.0])
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Velocity {
    pub linear: [f32; 3],
}

impl Velocity {
    pub fn at(linear: [f32; 3]) -> Self {
        Self { linear }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mass {
    pub inverse: f32,
}

impl Mass {
    pub fn new(mass: f32) -> Self {
        assert!(mass > 0.0, "dynamis bodies require strictly positive mass");
        Self {
            inverse: 1.0 / mass,
        }
    }

    pub fn static_body() -> Self {
        Self { inverse: 0.0 }
    }

    pub fn is_static(&self) -> bool {
        self.inverse == 0.0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Restitution(pub f32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereCollider {
    pub radius: f32,
}

impl SphereCollider {
    pub fn new(radius: f32) -> Self {
        assert!(radius > 0.0, "collider radius must be strictly positive");
        Self { radius }
    }
}
