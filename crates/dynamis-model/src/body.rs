use crate::collider::ColliderDesc;
use crate::collision::CollisionFilter;
use crate::domain;
use crate::shape::{Shape, ShapeSourceHandle, SolidGeometry};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyHandle {
    pub id: u32,
    pub generation: u32,
}

/// The kind of body a world holds. A kind declares what the physics owns: a dynamic body's motion
/// is solved, a kinematic body's motion is declared by the host and merely consumed, and a static
/// body neither simulates nor moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyKind {
    Static,
    Kinematic,
    Dynamic,
}

impl BodyKind {
    pub const ALL: [Self; 3] = [Self::Static, Self::Kinematic, Self::Dynamic];

    pub const fn code(self) -> u32 {
        match self {
            Self::Static => 0,
            Self::Kinematic => 1,
            Self::Dynamic => 2,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Static => "STATIC",
            Self::Kinematic => "KINEMATIC",
            Self::Dynamic => "DYNAMIC",
        }
    }

    /// Whether the physics evolves this body's motion: forces, gravity, damping, the velocity
    /// limits, contacts, joints, and the continuous collision response.
    pub const fn simulates(self) -> bool {
        matches!(self, Self::Dynamic)
    }

    /// Whether the physics advances this body's pose from the velocity it holds.
    pub const fn moves(self) -> bool {
        !matches!(self, Self::Static)
    }

    pub const fn of(inverse_mass: f32, kinematic: bool) -> Self {
        if kinematic {
            Self::Kinematic
        } else if inverse_mass > 0.0 {
            Self::Dynamic
        } else {
            Self::Static
        }
    }
}

const _: () = {
    let mut index = 0;
    while index < BodyKind::ALL.len() {
        assert!(
            BodyKind::ALL[index].code() == index as u32,
            "a body kind must be declared once",
        );
        index += 1;
    }
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyState {
    pub position: [f32; 3],
    pub prev_position: [f32; 3],
    pub prev_orientation: [f32; 4],
    pub orientation: [f32; 4],
    pub velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub inverse_mass: f32,
    pub com: [f32; 3],
    pub sleeping: bool,
    pub step: u64,
}

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
    pub filter: CollisionFilter,
    pub linear_damping: Option<f32>,
    pub angular_damping: Option<f32>,
    pub gravity_scale: f32,
    pub sleep_velocity: Option<f32>,
    pub sleep_angular_velocity: Option<f32>,
    pub kinematic: bool,
    pub ccd: bool,
}

impl BodyDesc {
    pub fn assert_valid(&self) {
        assert!(
            !self.colliders.is_empty(),
            "a body requires at least one collider"
        );
        for collider in &self.colliders {
            collider.assert_valid();
        }
        domain::finite_vector(self.position, "a body position");
        domain::unit_quaternion(self.orientation, "a body orientation");
        domain::finite_vector(self.velocity, "a body velocity");
        domain::finite_vector(self.angular_velocity, "a body angular velocity");
        domain::non_negative(self.mass, "body mass");
        if let Some(density) = self.density {
            domain::non_negative(density, "body density");
        }
        if let Some(com) = self.com {
            domain::finite_vector(com, "a body center of mass");
        }
        if let Some(inertia) = self.inertia {
            assert!(
                inertia.iter().all(|value| value.is_finite()),
                "an inertia tensor must be finite"
            );
        }
        if let Some(damping) = self.linear_damping {
            domain::non_negative(damping, "linear damping");
        }
        if let Some(damping) = self.angular_damping {
            domain::non_negative(damping, "angular damping");
        }
        domain::finite(self.gravity_scale, "a body gravity scale");
        assert!(
            !(self.kinematic && self.ccd),
            "a kinematic body declares its own motion, so it cannot also declare continuous collision",
        );
        if let Some(velocity) = self.sleep_velocity {
            domain::non_negative(velocity, "a sleep velocity");
        }
        if let Some(velocity) = self.sleep_angular_velocity {
            domain::non_negative(velocity, "a sleep angular velocity");
        }
    }

    pub const VACANT: Self = Self {
        colliders: Vec::new(),
        position: [0.0; 3],
        orientation: [0.0, 0.0, 0.0, 1.0],
        velocity: [0.0; 3],
        angular_velocity: [0.0; 3],
        mass: 0.0,
        density: None,
        com: None,
        inertia: None,
        filter: CollisionFilter::DEFAULT,
        linear_damping: None,
        angular_damping: None,
        gravity_scale: 0.0,
        sleep_velocity: None,
        sleep_angular_velocity: None,
        kinematic: false,
        ccd: false,
    };

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
            filter: CollisionFilter::DEFAULT,
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
        let first = handles
            .first()
            .expect("compound body requires at least one hull");
        let mut body = Self::new(ColliderDesc::new(Shape::hull(*first)));
        for handle in &handles[1..] {
            body = body.collider(ColliderDesc::new(Shape::hull(*handle)));
        }
        body
    }

    pub fn position(mut self, position: [f32; 3]) -> Self {
        domain::finite_vector(position, "a body position");
        self.position = position;
        self
    }

    pub fn restitution(mut self, restitution: f32) -> Self {
        self.colliders[0].restitution = domain::non_negative(restitution, "restitution");
        self
    }

    pub fn friction(mut self, friction: f32) -> Self {
        self.colliders[0].friction = domain::non_negative(friction, "friction");
        self
    }

    pub fn contact_frequency(mut self, contact_frequency: f32) -> Self {
        self.colliders[0] = self.colliders[0].contact_frequency(contact_frequency);
        self
    }

    pub fn contact_damping_ratio(mut self, contact_damping_ratio: f32) -> Self {
        self.colliders[0] = self.colliders[0].contact_damping_ratio(contact_damping_ratio);
        self
    }

    pub fn sensor(mut self, sensor: bool) -> Self {
        self.colliders[0].sensor = sensor;
        self
    }

    pub fn orientation(mut self, orientation: [f32; 4]) -> Self {
        domain::unit_quaternion(orientation, "a body orientation");
        self.orientation = orientation;
        self
    }

    pub fn velocity(mut self, velocity: [f32; 3]) -> Self {
        domain::finite_vector(velocity, "a body velocity");
        self.velocity = velocity;
        self
    }

    pub fn angular_velocity(mut self, angular_velocity: [f32; 3]) -> Self {
        domain::finite_vector(angular_velocity, "a body angular velocity");
        self.angular_velocity = angular_velocity;
        self
    }

    pub fn mass(mut self, mass: f32) -> Self {
        self.mass = domain::non_negative(mass, "mass");
        self.density = None;
        self
    }

    pub fn density(mut self, density: f32) -> Self {
        self.density = Some(domain::non_negative(density, "density"));
        self
    }

    pub fn damping(mut self, damping: f32) -> Self {
        self.linear_damping = Some(domain::non_negative(damping, "damping"));
        self
    }

    pub fn angular_damping(mut self, angular_damping: f32) -> Self {
        self.angular_damping = Some(domain::non_negative(angular_damping, "angular damping"));
        self
    }

    pub fn gravity_scale(mut self, gravity_scale: f32) -> Self {
        self.gravity_scale = domain::finite(gravity_scale, "a gravity scale");
        self
    }

    pub fn sleep_thresholds(mut self, velocity: f32, angular_velocity: f32) -> Self {
        self.sleep_velocity = Some(domain::non_negative(velocity, "a sleep velocity"));
        self.sleep_angular_velocity = Some(domain::non_negative(
            angular_velocity,
            "a sleep angular velocity",
        ));
        self
    }

    pub fn com(mut self, com: [f32; 3]) -> Self {
        self.com = Some(domain::finite_vector(com, "a center of mass"));
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
        geometry: impl Fn(&Shape) -> Option<SolidGeometry>,
    ) -> crate::mass::MassProperties {
        if let Some(inertia) = self.inertia {
            return crate::mass::mass_properties_of_intent(
                &self.colliders,
                self.mass,
                self.com,
                Some(inertia),
                geometry,
            );
        }
        match self.density {
            Some(density) => crate::mass::compute_mass_properties(
                &self.colliders,
                crate::mass::MassSource::Density(density),
                self.com,
                geometry,
            ),
            None => crate::mass::mass_properties_of_intent(
                &self.colliders,
                self.mass,
                self.com,
                None,
                geometry,
            ),
        }
    }

    pub fn effective_mass(&self, geometry: impl Fn(&Shape) -> Option<SolidGeometry>) -> f32 {
        match self.density {
            Some(density) => density * crate::mass::solid_volume_of(&self.colliders, &geometry),
            None => self.mass,
        }
    }

    pub fn kind(&self, geometry: impl Fn(&Shape) -> Option<SolidGeometry>) -> BodyKind {
        BodyKind::of(self.effective_mass(geometry), self.kinematic)
    }

    pub fn filter(mut self, filter: CollisionFilter) -> Self {
        self.filter = filter;
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
