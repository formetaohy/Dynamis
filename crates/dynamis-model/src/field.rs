use crate::collision::CollisionFilter;
use crate::domain;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldHandle {
    pub id: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldRegion {
    Global,
    Sphere {
        radius: f32,
    },
    Cuboid {
        half_extents: [f32; 3],
        orientation: [f32; 4],
    },
}

impl FieldRegion {
    pub fn assert_valid(&self) {
        match *self {
            Self::Global => {}
            Self::Sphere { radius } => {
                domain::positive(radius, "a field sphere radius");
            }
            Self::Cuboid {
                half_extents,
                orientation,
            } => {
                domain::finite_vector(half_extents, "field cuboid half extents");
                for extent in half_extents {
                    domain::positive(extent, "a field cuboid half extent");
                }
                domain::unit_quaternion(orientation, "a field cuboid orientation");
            }
        }
    }

    pub const fn sphere(radius: f32) -> Self {
        assert!(radius > 0.0, "a field sphere radius must be positive");
        Self::Sphere { radius }
    }

    pub fn cuboid(half_extents: [f32; 3]) -> Self {
        Self::Cuboid {
            half_extents,
            orientation: [0.0, 0.0, 0.0, 1.0],
        }
    }

    pub fn oriented_cuboid(half_extents: [f32; 3], orientation: [f32; 4]) -> Self {
        Self::Cuboid {
            half_extents,
            orientation,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldDesc {
    pub region: FieldRegion,
    pub position: [f32; 3],
    pub push: [f32; 3],
    pub pull: f32,
    pub swirl: f32,
    pub axis: [f32; 3],
    pub buoyancy: f32,
    pub medium: [f32; 3],
    pub linear_drag: f32,
    pub quadratic_drag: f32,
    pub angular_drag: f32,
    pub filter: CollisionFilter,
}

impl FieldDesc {
    pub const VACANT: Self = Self::of(FieldRegion::Global);

    pub const fn of(region: FieldRegion) -> Self {
        Self {
            region,
            position: [0.0; 3],
            push: [0.0; 3],
            pull: 0.0,
            swirl: 0.0,
            axis: [0.0, 1.0, 0.0],
            buoyancy: 0.0,
            medium: [0.0; 3],
            linear_drag: 0.0,
            quadratic_drag: 0.0,
            angular_drag: 0.0,
            filter: CollisionFilter::DEFAULT,
        }
    }

    pub fn uniform(region: FieldRegion, acceleration: [f32; 3]) -> Self {
        Self::of(region).push(acceleration)
    }

    pub fn radial(region: FieldRegion, position: [f32; 3], acceleration: f32) -> Self {
        Self::of(region).position(position).pull(acceleration)
    }

    pub fn vortex(
        region: FieldRegion,
        position: [f32; 3],
        axis: [f32; 3],
        acceleration: f32,
    ) -> Self {
        Self::of(region)
            .position(position)
            .swirl(axis, acceleration)
    }

    pub fn wind(region: FieldRegion, velocity: [f32; 3], drag: f32) -> Self {
        Self::of(region).medium(velocity).drag(drag, 0.0)
    }

    pub fn buoyant(region: FieldRegion, medium_density_ratio: f32, drag: f32) -> Self {
        Self::of(region)
            .buoyancy(medium_density_ratio)
            .drag(drag, 0.0)
    }

    pub fn assert_valid(&self) {
        self.region.assert_valid();
        domain::finite_vector(self.position, "a field position");
        domain::finite_vector(self.push, "a field push");
        domain::finite_vector(self.axis, "a field swirl axis");
        domain::finite_vector(self.medium, "a field medium velocity");
        domain::finite(self.pull, "a field pull");
        domain::finite(self.swirl, "a field swirl");
        domain::non_negative(self.buoyancy, "a field buoyancy");
        domain::non_negative(self.linear_drag, "a field linear drag");
        domain::non_negative(self.quadratic_drag, "a field quadratic drag");
        domain::non_negative(self.angular_drag, "a field angular drag");
        assert!(
            self.push != [0.0; 3]
                || self.pull != 0.0
                || self.swirl != 0.0
                || self.buoyancy > 0.0
                || self.linear_drag > 0.0
                || self.quadratic_drag > 0.0
                || self.angular_drag > 0.0,
            "a field must declare an effect",
        );
        if self.swirl != 0.0 {
            unit_axis(self.axis);
        }
    }

    pub fn position(mut self, position: [f32; 3]) -> Self {
        domain::finite_vector(position, "a field position");
        self.position = position;
        self
    }

    pub fn push(mut self, acceleration: [f32; 3]) -> Self {
        domain::finite_vector(acceleration, "a field push");
        self.push = acceleration;
        self
    }

    pub fn pull(mut self, acceleration: f32) -> Self {
        domain::finite(acceleration, "a field pull");
        self.pull = acceleration;
        self
    }

    pub fn buoyancy(mut self, ratio: f32) -> Self {
        domain::non_negative(ratio, "a field buoyancy");
        self.buoyancy = ratio;
        self
    }

    pub fn swirl(mut self, axis: [f32; 3], acceleration: f32) -> Self {
        domain::finite(acceleration, "a field swirl");
        unit_axis(axis);
        self.axis = axis;
        self.swirl = acceleration;
        self
    }

    pub fn medium(mut self, velocity: [f32; 3]) -> Self {
        domain::finite_vector(velocity, "a field medium velocity");
        self.medium = velocity;
        self
    }

    pub fn drag(mut self, linear: f32, quadratic: f32) -> Self {
        domain::non_negative(linear, "a field linear drag");
        domain::non_negative(quadratic, "a field quadratic drag");
        self.linear_drag = linear;
        self.quadratic_drag = quadratic;
        self
    }

    pub fn angular_drag(mut self, drag: f32) -> Self {
        domain::non_negative(drag, "a field angular drag");
        self.angular_drag = drag;
        self
    }

    pub fn filter(mut self, filter: CollisionFilter) -> Self {
        self.filter = filter;
        self
    }
}

fn unit_axis(axis: [f32; 3]) {
    domain::finite_vector(axis, "a field swirl axis");
    let square = axis.iter().map(|value| value * value).sum::<f32>();
    assert!(
        (square - 1.0).abs() < 1e-4,
        "a field swirl axis must be a unit vector"
    );
}
