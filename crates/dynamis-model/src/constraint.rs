#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintKind {
    Ball,
    Distance,
    Revolute,
    Prismatic,
    Fixed,
    Gear,
    Pulley,
}

#[derive(Clone, Copy, Debug)]
pub struct ConstraintLimit {
    pub min: f32,
    pub max: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ConstraintMotor {
    pub target_velocity: f32,
    pub max_force: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ConstraintSpring {
    pub frequency: f32,
    pub damping_ratio: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ConstraintSwing {
    pub swing_a: f32,
    pub swing_b: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ConstraintBreak {
    pub force: f32,
    pub torque: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ConstraintDesc {
    pub kind: ConstraintKind,
    pub anchor_a: [f32; 3],
    pub anchor_b: [f32; 3],
    pub axis_a: [f32; 3],
    pub axis_b: [f32; 3],
    pub rest_length: f32,
    pub limit: Option<ConstraintLimit>,
    pub swing: Option<ConstraintSwing>,
    pub motor: Option<ConstraintMotor>,
    pub spring: Option<ConstraintSpring>,
    pub break_threshold: Option<ConstraintBreak>,
    pub gear_ratio: f32,
    pub pulley_fixed_a: [f32; 3],
    pub pulley_fixed_b: [f32; 3],
    pub disable_collisions: bool,
}

impl ConstraintDesc {
    fn base(kind: ConstraintKind) -> Self {
        Self {
            kind,
            anchor_a: [0.0; 3],
            anchor_b: [0.0; 3],
            axis_a: [0.0, 1.0, 0.0],
            axis_b: [0.0, 1.0, 0.0],
            rest_length: 0.0,
            limit: None,
            swing: None,
            motor: None,
            spring: None,
            break_threshold: None,
            gear_ratio: 1.0,
            pulley_fixed_a: [0.0; 3],
            pulley_fixed_b: [0.0; 3],
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
            .rest_length(distance)
    }

    pub fn revolute(anchor_a: [f32; 3], anchor_b: [f32; 3], axis: [f32; 3]) -> Self {
        assert!(axis != [0.0; 3], "revolute axis must be non-zero");
        Self::base(ConstraintKind::Revolute)
            .anchors(anchor_a, anchor_b)
            .axis(axis)
    }

    pub fn prismatic(anchor_a: [f32; 3], anchor_b: [f32; 3], axis: [f32; 3]) -> Self {
        assert!(axis != [0.0; 3], "prismatic axis must be non-zero");
        Self::base(ConstraintKind::Prismatic)
            .anchors(anchor_a, anchor_b)
            .axis(axis)
    }

    pub fn fixed(anchor_a: [f32; 3], anchor_b: [f32; 3]) -> Self {
        Self::base(ConstraintKind::Fixed).anchors(anchor_a, anchor_b)
    }

    pub fn gear(axis_a: [f32; 3], axis_b: [f32; 3], ratio: f32) -> Self {
        assert!(axis_a != [0.0; 3], "gear axis a must be non-zero");
        assert!(axis_b != [0.0; 3], "gear axis b must be non-zero");
        assert!(ratio != 0.0, "gear ratio must be non-zero");
        let mut desc = Self::base(ConstraintKind::Gear).axis(axis_a);
        desc.axis_b = axis_b;
        desc.gear_ratio = ratio;
        desc
    }

    pub fn pulley(
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
        fixed_a: [f32; 3],
        fixed_b: [f32; 3],
        length: f32,
    ) -> Self {
        assert!(length > 0.0, "pulley length must be positive");
        let mut desc = Self::base(ConstraintKind::Pulley)
            .anchors(anchor_a, anchor_b)
            .rest_length(length);
        desc.pulley_fixed_a = fixed_a;
        desc.pulley_fixed_b = fixed_b;
        desc
    }

    pub fn anchors(mut self, anchor_a: [f32; 3], anchor_b: [f32; 3]) -> Self {
        self.anchor_a = anchor_a;
        self.anchor_b = anchor_b;
        self
    }

    pub fn axis(mut self, axis: [f32; 3]) -> Self {
        assert!(axis != [0.0; 3], "constraint axis must be non-zero");
        self.axis_a = axis;
        self
    }

    pub fn axis_b(mut self, axis_b: [f32; 3]) -> Self {
        assert!(axis_b != [0.0; 3], "constraint axis b must be non-zero");
        self.axis_b = axis_b;
        self
    }

    pub fn rest_length(mut self, rest_length: f32) -> Self {
        assert!(
            rest_length >= 0.0,
            "constraint distance must be non-negative"
        );
        self.rest_length = rest_length;
        self
    }

    pub fn limit(mut self, min: f32, max: f32) -> Self {
        assert!(max >= min, "constraint limit max must not be below min");
        self.limit = Some(ConstraintLimit { min, max });
        self
    }

    pub fn swing(mut self, swing_a: f32, swing_b: f32) -> Self {
        assert!(swing_a >= 0.0, "swing limit must be non-negative");
        assert!(swing_b >= 0.0, "swing limit must be non-negative");
        self.swing = Some(ConstraintSwing { swing_a, swing_b });
        self
    }

    pub fn motor(mut self, target_velocity: f32) -> Self {
        self.motor = Some(ConstraintMotor {
            target_velocity,
            max_force: 0.0,
        });
        self
    }

    pub fn motor_force(mut self, max_force: f32) -> Self {
        assert!(max_force >= 0.0, "motor force must be non-negative");
        let motor = self.motor.get_or_insert(ConstraintMotor {
            target_velocity: 0.0,
            max_force: 0.0,
        });
        motor.max_force = max_force;
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

    pub fn break_threshold(mut self, force: f32, torque: f32) -> Self {
        assert!(force >= 0.0, "break force must be non-negative");
        assert!(torque >= 0.0, "break torque must be non-negative");
        self.break_threshold = Some(ConstraintBreak { force, torque });
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
