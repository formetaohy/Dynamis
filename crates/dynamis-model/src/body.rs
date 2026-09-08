use crate::collider::ColliderDesc;
use crate::shape::{Shape, ShapeSourceHandle};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyHandle {
    pub id: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyState {
    pub position: [f32; 3],
    pub prev_position: [f32; 3],
    pub orientation: [f32; 4],
    pub velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub inverse_mass: f32,
    pub com: [f32; 3],
    pub sleeping: bool,
    pub step: u64,
}

pub const DEFAULT_COLLISION_GROUP: u32 = 0x0000_0001;
pub const DEFAULT_COLLISION_MASK: u32 = 0xFFFF_FFFF;
pub const MAX_COLLIDERS_PER_BODY: usize = 16;

#[derive(Clone, Debug)]
pub struct BodyDesc {
    pub colliders: Vec<ColliderDesc>,
    pub position: [f32; 3],
    pub orientation: [f32; 4],
    pub velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub mass: f32,
    pub density: Option<f32>,
    pub com: Option<[f32; 3]>,
    pub inertia: Option<[f32; 6]>,
    pub collision_group: u32,
    pub collision_mask: u32,
    pub linear_damping: Option<f32>,
    pub angular_damping: Option<f32>,
    pub gravity_scale: f32,
    pub sleep_velocity: Option<f32>,
    pub sleep_angular_velocity: Option<f32>,
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
            density: None,
            com: None,
            inertia: None,
            collision_group: DEFAULT_COLLISION_GROUP,
            collision_mask: DEFAULT_COLLISION_MASK,
            linear_damping: None,
            angular_damping: None,
            gravity_scale: 1.0,
            sleep_velocity: None,
            sleep_angular_velocity: None,
            kinematic: false,
            ccd: false,
        }
    }

    pub fn collider(mut self, collider: ColliderDesc) -> Self {
        assert!(
            self.colliders.len() < MAX_COLLIDERS_PER_BODY,
            "a body supports at most {MAX_COLLIDERS_PER_BODY} colliders"
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

    pub fn compound(handles: &[ShapeSourceHandle]) -> Self {
        assert!(
            !handles.is_empty() && handles.len() <= MAX_COLLIDERS_PER_BODY,
            "a compound body requires between 1 and {MAX_COLLIDERS_PER_BODY} hulls"
        );
        let mut members = handles.iter().copied();
        let first = members.next().expect("compound body hulls are non-empty");
        let mut body = Self::new(ColliderDesc::new(Shape::hull(first)));
        for handle in members {
            body = body.collider(ColliderDesc::new(Shape::hull(handle)));
        }
        body
    }

    pub fn inverse_mass(&self) -> f32 {
        if self.kinematic || self.mass <= 0.0 {
            0.0
        } else {
            1.0 / self.mass
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
        self.density = None;
        self
    }

    pub fn density(mut self, density: f32) -> Self {
        assert!(density >= 0.0, "density must be non-negative");
        self.density = Some(density);
        self
    }

    pub fn damping(mut self, damping: f32) -> Self {
        assert!(damping >= 0.0, "damping must be non-negative");
        self.linear_damping = Some(damping);
        self
    }

    pub fn angular_damping(mut self, angular_damping: f32) -> Self {
        assert!(
            angular_damping >= 0.0,
            "angular damping must be non-negative"
        );
        self.angular_damping = Some(angular_damping);
        self
    }

    pub fn gravity_scale(mut self, gravity_scale: f32) -> Self {
        self.gravity_scale = gravity_scale;
        self
    }

    pub fn sleep_thresholds(mut self, velocity: f32, angular_velocity: f32) -> Self {
        assert!(velocity >= 0.0, "sleep velocity must be non-negative");
        assert!(
            angular_velocity >= 0.0,
            "sleep angular velocity must be non-negative"
        );
        self.sleep_velocity = Some(velocity);
        self.sleep_angular_velocity = Some(angular_velocity);
        self
    }

    pub fn com(mut self, com: [f32; 3]) -> Self {
        self.com = Some(com);
        self
    }

    pub fn inertia(mut self, inertia: [f32; 6]) -> Self {
        assert!(
            inertia.iter().all(|value| value.is_finite()),
            "inertia tensor must be finite"
        );
        self.inertia = Some(inertia);
        self
    }

    pub fn mass_properties(
        &self,
        bounds: impl Fn(&Shape) -> Option<([f32; 3], [f32; 3])>,
    ) -> crate::mass::MassProperties {
        if let Some(inertia) = self.inertia {
            return crate::mass::mass_properties_of_intent(
                &self.colliders,
                self.mass,
                self.com,
                Some(inertia),
                bounds,
            );
        }
        match self.density {
            Some(density) => crate::mass::compute_mass_properties(
                &self.colliders,
                crate::mass::MassSource::Density(density),
                self.com,
                bounds,
            ),
            None => crate::mass::mass_properties_of_intent(
                &self.colliders,
                self.mass,
                self.com,
                None,
                bounds,
            ),
        }
    }

    pub fn effective_mass(&self, bounds: impl Fn(&Shape) -> Option<([f32; 3], [f32; 3])>) -> f32 {
        match self.density {
            Some(density) => density * crate::mass::solid_volume_of(&self.colliders, &bounds),
            None => self.mass,
        }
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
