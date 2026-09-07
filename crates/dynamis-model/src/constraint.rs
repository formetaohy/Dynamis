#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintKind {
    Ball,
    Distance,
    Revolute,
    Prismatic,
    Fixed,
}

#[derive(Clone, Copy, Debug)]
pub struct ConstraintLimit {
    pub min: f32,
    pub max: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ConstraintMotor {
    pub speed: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ConstraintSpring {
    pub frequency: f32,
    pub damping_ratio: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ConstraintDesc {
    pub kind: ConstraintKind,
    pub anchor_a: [f32; 3],
    pub anchor_b: [f32; 3],
    pub axis: [f32; 3],
    pub distance: f32,
    pub limit: Option<ConstraintLimit>,
    pub motor: Option<ConstraintMotor>,
    pub spring: Option<ConstraintSpring>,
    pub disable_collisions: bool,
}

impl ConstraintDesc {
    fn base(kind: ConstraintKind) -> Self {
        Self {
            kind,
            anchor_a: [0.0; 3],
            anchor_b: [0.0; 3],
            axis: [0.0, 1.0, 0.0],
            distance: 0.0,
            limit: None,
            motor: None,
            spring: None,
            disable_collisions: true,
        }
    }

    pub fn ball(anchor_a: [f32; 3], anchor_b: [f32; 3]) -> Self {
        Self::base(ConstraintKind::Ball).anchors(anchor_a, anchor_b)
    }

    pub fn distance(anchor_a: [f32; 3], anchor_b: [f32; 3], distance: f32) -> Self {
        assert!(distance >= 0.0, "constraint distance must be non-negative");
        Self::base(ConstraintKind::Distance)
            .anchors(anchor_a, anchor_b)
            .with_distance(distance)
    }

    pub fn revolute(anchor_a: [f32; 3], anchor_b: [f32; 3], axis: [f32; 3]) -> Self {
        assert!(axis != [0.0; 3], "revolute axis must be non-zero");
        Self::base(ConstraintKind::Revolute)
            .anchors(anchor_a, anchor_b)
            .with_axis(axis)
    }

    pub fn prismatic(anchor_a: [f32; 3], anchor_b: [f32; 3], axis: [f32; 3]) -> Self {
        assert!(axis != [0.0; 3], "prismatic axis must be non-zero");
        Self::base(ConstraintKind::Prismatic)
            .anchors(anchor_a, anchor_b)
            .with_axis(axis)
    }

    pub fn fixed(anchor_a: [f32; 3], anchor_b: [f32; 3]) -> Self {
        Self::base(ConstraintKind::Fixed).anchors(anchor_a, anchor_b)
    }

    pub fn anchors(mut self, anchor_a: [f32; 3], anchor_b: [f32; 3]) -> Self {
        self.anchor_a = anchor_a;
        self.anchor_b = anchor_b;
        self
    }

    pub fn with_axis(mut self, axis: [f32; 3]) -> Self {
        assert!(axis != [0.0; 3], "constraint axis must be non-zero");
        self.axis = axis;
        self
    }

    pub fn with_distance(mut self, distance: f32) -> Self {
        assert!(distance >= 0.0, "constraint distance must be non-negative");
        self.distance = distance;
        self
    }

    pub fn limit(mut self, min: f32, max: f32) -> Self {
        assert!(max >= min, "constraint limit max must not be below min");
        self.limit = Some(ConstraintLimit { min, max });
        self
    }

    pub fn motor(mut self, speed: f32) -> Self {
        self.motor = Some(ConstraintMotor { speed });
        self
    }

    pub fn spring(mut self, frequency: f32, damping_ratio: f32) -> Self {
        assert!(frequency >= 0.0, "spring frequency must be non-negative");
        assert!(
            damping_ratio >= 0.0,
            "spring damping ratio must be non-negative"
        );
        self.spring = Some(ConstraintSpring {
            frequency,
            damping_ratio,
        });
        self
    }

    pub fn disable_collisions(mut self, disable: bool) -> Self {
        self.disable_collisions = disable;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstraintHandle {
    pub id: u32,
    pub generation: u32,
}
