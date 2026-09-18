#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintKind {
    Ball,
    Distance,
    Revolute,
    Prismatic,
    Fixed,
    Gear,
    Pulley,
    Cone,
    SixDof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointDof {
    Separation,
    Hinge,
    Slide,
    SwingA,
    SwingB,
    Twist,
    Cone,
    Rope,
    Linear(usize),
    Angular(usize),
}

impl ConstraintKind {
    pub const ALL: &'static [Self] = &[
        Self::Ball,
        Self::Distance,
        Self::Revolute,
        Self::Prismatic,
        Self::Fixed,
        Self::Gear,
        Self::Pulley,
        Self::Cone,
        Self::SixDof,
    ];

    pub const fn dofs(self) -> &'static [JointDof] {
        match self {
            Self::Distance => &[JointDof::Separation],
            Self::Revolute => &[JointDof::Hinge],
            Self::Prismatic => &[JointDof::Slide],
            Self::Ball => &[JointDof::SwingA, JointDof::SwingB, JointDof::Twist],
            Self::Cone => &[JointDof::Cone],
            Self::Gear => &[],
            Self::Pulley => &[JointDof::Rope],
            Self::SixDof => &[
                JointDof::Linear(0),
                JointDof::Linear(1),
                JointDof::Linear(2),
                JointDof::Angular(0),
                JointDof::Angular(1),
                JointDof::Angular(2),
            ],
            Self::Fixed => &[],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointState {
    kind: ConstraintKind,
    coordinates: [f32; 6],
    rates: [f32; 6],
    impulses: [f32; 6],
}

impl JointState {
    pub fn new(
        kind: ConstraintKind,
        coordinates: [f32; 6],
        rates: [f32; 6],
        impulses: [f32; 6],
    ) -> Self {
        Self {
            kind,
            coordinates,
            rates,
            impulses,
        }
    }

    pub const fn kind(&self) -> ConstraintKind {
        self.kind
    }

    pub const fn dofs(&self) -> &'static [JointDof] {
        self.kind.dofs()
    }

    pub fn coordinate(&self, dof: JointDof) -> f32 {
        self.coordinates[self.slot(dof)]
    }

    pub fn rate(&self, dof: JointDof) -> f32 {
        self.rates[self.slot(dof)]
    }

    pub fn impulse(&self, dof: JointDof) -> f32 {
        self.impulses[self.slot(dof)]
    }

    fn slot(&self, dof: JointDof) -> usize {
        self.dofs()
            .iter()
            .position(|candidate| *candidate == dof)
            .unwrap_or_else(|| panic!("joint dof {dof:?} is outside the {:?} layout", self.kind))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstraintLimit {
    pub min: f32,
    pub max: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstraintPositionTarget {
    coordinate: f32,
    stiffness: f32,
    damping: f32,
}

impl ConstraintPositionTarget {
    pub fn new(coordinate: f32, stiffness: f32, damping: f32) -> Self {
        assert!(
            coordinate.is_finite(),
            "a position target coordinate must be finite"
        );
        assert!(
            (0.0..=1.0).contains(&stiffness),
            "a position target stiffness must be within [0, 1]"
        );
        assert!(
            (0.0..=1.0).contains(&damping),
            "a position target damping must be within [0, 1]"
        );
        Self {
            coordinate,
            stiffness,
            damping,
        }
    }

    pub const fn coordinate(&self) -> f32 {
        self.coordinate
    }

    pub const fn stiffness(&self) -> f32 {
        self.stiffness
    }

    pub const fn damping(&self) -> f32 {
        self.damping
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstraintMotor {
    target_velocity: f32,
    position: Option<ConstraintPositionTarget>,
    max_force: f32,
}

impl ConstraintMotor {
    pub fn new(target_velocity: f32, max_force: f32) -> Self {
        assert!(
            target_velocity.is_finite(),
            "a motor target velocity must be finite"
        );
        assert!(max_force >= 0.0, "a motor force cap must be non-negative");
        Self {
            target_velocity,
            position: None,
            max_force,
        }
    }

    pub fn velocity(mut self, target_velocity: f32) -> Self {
        assert!(
            target_velocity.is_finite(),
            "a motor target velocity must be finite"
        );
        self.target_velocity = target_velocity;
        self
    }

    pub fn position(mut self, target: ConstraintPositionTarget) -> Self {
        self.position = Some(target);
        self
    }

    pub fn force(mut self, max_force: f32) -> Self {
        assert!(max_force >= 0.0, "a motor force cap must be non-negative");
        self.max_force = max_force;
        self
    }

    pub const fn target_velocity(&self) -> f32 {
        self.target_velocity
    }

    pub const fn position_target(&self) -> Option<ConstraintPositionTarget> {
        self.position
    }

    pub const fn max_force(&self) -> f32 {
        self.max_force
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstraintSpring {
    pub frequency: f32,
    pub damping_ratio: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstraintSwing {
    pub swing_a: f32,
    pub swing_b: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstraintBreak {
    pub force: f32,
    pub torque: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DofDesc {
    pub locked: bool,
    pub limit: Option<ConstraintLimit>,
    pub motor: Option<ConstraintMotor>,
}

impl DofDesc {
    pub fn free() -> Self {
        Self {
            locked: false,
            limit: None,
            motor: None,
        }
    }

    pub fn locked() -> Self {
        Self::free().lock()
    }

    pub fn limited(min: f32, max: f32) -> Self {
        Self::free().limit(min, max)
    }

    pub fn driven(motor: ConstraintMotor) -> Self {
        Self::free().motor(motor)
    }

    pub fn lock(mut self) -> Self {
        self.set_locked(true);
        self
    }

    pub fn limit(mut self, min: f32, max: f32) -> Self {
        self.set_limit(Some(ConstraintLimit { min, max }));
        self
    }

    pub fn motor(mut self, motor: ConstraintMotor) -> Self {
        self.set_motor(Some(motor));
        self
    }

    pub fn set_locked(&mut self, locked: bool) {
        self.locked = locked;
    }

    pub fn set_limit(&mut self, limit: Option<ConstraintLimit>) {
        if let Some(limit) = limit {
            assert!(
                limit.max >= limit.min,
                "dof limit max must not be below min"
            );
        }
        self.limit = limit;
    }

    pub fn set_motor(&mut self, motor: Option<ConstraintMotor>) {
        self.motor = motor;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConstraintData {
    Ball {
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
        axis: [f32; 3],
        twist: Option<ConstraintLimit>,
        swing: Option<ConstraintSwing>,
    },
    Distance {
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
        length: f32,
        spring: Option<ConstraintSpring>,
    },
    Revolute {
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
        axis: [f32; 3],
        limit: Option<ConstraintLimit>,
        motor: Option<ConstraintMotor>,
    },
    Prismatic {
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
        axis: [f32; 3],
        limit: Option<ConstraintLimit>,
        motor: Option<ConstraintMotor>,
    },
    Fixed {
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
    },
    Gear {
        axis_a: [f32; 3],
        axis_b: [f32; 3],
        ratio: f32,
    },
    Pulley {
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
        fixed_a: [f32; 3],
        fixed_b: [f32; 3],
        length: f32,
    },
    Cone {
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
        axis_a: [f32; 3],
        axis_b: [f32; 3],
        half_angle: f32,
    },
    SixDof {
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
        axis: [f32; 3],
        dofs: Box<[DofDesc; 6]>,
    },
}

impl ConstraintData {
    pub const fn kind(&self) -> ConstraintKind {
        match self {
            Self::Ball { .. } => ConstraintKind::Ball,
            Self::Distance { .. } => ConstraintKind::Distance,
            Self::Revolute { .. } => ConstraintKind::Revolute,
            Self::Prismatic { .. } => ConstraintKind::Prismatic,
            Self::Fixed { .. } => ConstraintKind::Fixed,
            Self::Gear { .. } => ConstraintKind::Gear,
            Self::Pulley { .. } => ConstraintKind::Pulley,
            Self::Cone { .. } => ConstraintKind::Cone,
            Self::SixDof { .. } => ConstraintKind::SixDof,
        }
    }

    pub const fn limit_of(&self) -> Option<ConstraintLimit> {
        match self {
            Self::Ball { twist, .. } => *twist,
            Self::Revolute { limit, .. } | Self::Prismatic { limit, .. } => *limit,
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstraintDesc {
    data: ConstraintData,
    break_threshold: Option<ConstraintBreak>,
    warm_start: bool,
    disable_collisions: bool,
}

impl ConstraintDesc {
    pub const VACANT: Self = Self {
        data: ConstraintData::Fixed {
            anchor_a: [0.0; 3],
            anchor_b: [0.0; 3],
        },
        break_threshold: None,
        warm_start: true,
        disable_collisions: true,
    };

    fn held(data: ConstraintData) -> Self {
        Self {
            data,
            ..Self::VACANT
        }
    }

    pub const fn kind(&self) -> ConstraintKind {
        self.data.kind()
    }

    pub const fn data(&self) -> &ConstraintData {
        &self.data
    }

    pub const fn limit_of(&self) -> Option<ConstraintLimit> {
        self.data.limit_of()
    }

    pub const fn motor_of(&self) -> Option<ConstraintMotor> {
        match self.data {
            ConstraintData::Revolute { motor, .. } | ConstraintData::Prismatic { motor, .. } => {
                motor
            }
            _ => None,
        }
    }

    pub const fn spring_of(&self) -> Option<ConstraintSpring> {
        match self.data {
            ConstraintData::Distance { spring, .. } => spring,
            _ => None,
        }
    }

    pub const fn swing_of(&self) -> Option<ConstraintSwing> {
        match self.data {
            ConstraintData::Ball { swing, .. } => swing,
            _ => None,
        }
    }

    pub fn dofs_of(&self) -> Option<&[DofDesc; 6]> {
        match &self.data {
            ConstraintData::SixDof { dofs, .. } => Some(dofs),
            _ => None,
        }
    }

    pub const fn break_threshold_of(&self) -> Option<ConstraintBreak> {
        self.break_threshold
    }

    pub const fn warm_start_of(&self) -> bool {
        self.warm_start
    }

    pub const fn disable_collisions_of(&self) -> bool {
        self.disable_collisions
    }

    pub fn ball(anchor_a: [f32; 3], anchor_b: [f32; 3]) -> Self {
        Self::held(ConstraintData::Ball {
            anchor_a,
            anchor_b,
            axis: [0.0, 1.0, 0.0],
            twist: None,
            swing: None,
        })
    }

    pub fn distance(anchor_a: [f32; 3], anchor_b: [f32; 3], distance: f32) -> Self {
        assert!(distance >= 0.0, "constraint distance must be non-negative");
        Self::held(ConstraintData::Distance {
            anchor_a,
            anchor_b,
            length: distance,
            spring: None,
        })
    }

    pub fn revolute(anchor_a: [f32; 3], anchor_b: [f32; 3], axis: [f32; 3]) -> Self {
        assert!(axis != [0.0; 3], "revolute axis must be non-zero");
        Self::held(ConstraintData::Revolute {
            anchor_a,
            anchor_b,
            axis,
            limit: None,
            motor: None,
        })
    }

    pub fn prismatic(anchor_a: [f32; 3], anchor_b: [f32; 3], axis: [f32; 3]) -> Self {
        assert!(axis != [0.0; 3], "prismatic axis must be non-zero");
        Self::held(ConstraintData::Prismatic {
            anchor_a,
            anchor_b,
            axis,
            limit: None,
            motor: None,
        })
    }

    pub fn fixed(anchor_a: [f32; 3], anchor_b: [f32; 3]) -> Self {
        Self::held(ConstraintData::Fixed { anchor_a, anchor_b })
    }

    pub fn gear(axis_a: [f32; 3], axis_b: [f32; 3], ratio: f32) -> Self {
        assert!(axis_a != [0.0; 3], "gear axis a must be non-zero");
        assert!(axis_b != [0.0; 3], "gear axis b must be non-zero");
        assert!(ratio != 0.0, "gear ratio must be non-zero");
        Self::held(ConstraintData::Gear {
            axis_a,
            axis_b,
            ratio,
        })
    }

    pub fn pulley(
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
        fixed_a: [f32; 3],
        fixed_b: [f32; 3],
        length: f32,
    ) -> Self {
        assert!(length > 0.0, "pulley length must be positive");
        Self::held(ConstraintData::Pulley {
            anchor_a,
            anchor_b,
            fixed_a,
            fixed_b,
            length,
        })
    }

    pub fn cone(
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
        axis_a: [f32; 3],
        axis_b: [f32; 3],
        half_angle: f32,
    ) -> Self {
        assert!(axis_a != [0.0; 3], "cone axis a must be non-zero");
        assert!(axis_b != [0.0; 3], "cone axis b must be non-zero");
        assert!(
            (0.0..=std::f32::consts::PI).contains(&half_angle),
            "cone half angle must be within [0, pi]"
        );
        Self::held(ConstraintData::Cone {
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
            half_angle,
        })
    }

    pub fn six_dof(anchor_a: [f32; 3], anchor_b: [f32; 3], axis: [f32; 3]) -> Self {
        assert!(axis != [0.0; 3], "six dof axis must be non-zero");
        Self::held(ConstraintData::SixDof {
            anchor_a,
            anchor_b,
            axis,
            dofs: Box::new([DofDesc::free(); 6]),
        })
    }

    pub fn set_axis(&mut self, axis: [f32; 3]) {
        assert!(axis != [0.0; 3], "constraint axis must be non-zero");
        match &mut self.data {
            ConstraintData::Ball { axis: held, .. }
            | ConstraintData::Revolute { axis: held, .. }
            | ConstraintData::Prismatic { axis: held, .. }
            | ConstraintData::SixDof { axis: held, .. } => *held = axis,
            other => panic!("a {:?} joint declares no axis", other.kind()),
        }
    }

    pub fn set_limit(&mut self, limit: Option<ConstraintLimit>) {
        if let Some(limit) = limit {
            assert!(
                limit.max >= limit.min,
                "constraint limit max must not be below min"
            );
        }
        match &mut self.data {
            ConstraintData::Ball { twist, .. } => *twist = limit,
            ConstraintData::Revolute { limit: held, .. }
            | ConstraintData::Prismatic { limit: held, .. } => *held = limit,
            other => panic!("a {:?} joint declares no limit", other.kind()),
        }
    }

    pub fn set_swing(&mut self, swing: Option<ConstraintSwing>) {
        if let Some(swing) = swing {
            assert!(swing.swing_a >= 0.0, "swing limit must be non-negative");
            assert!(swing.swing_b >= 0.0, "swing limit must be non-negative");
        }
        match &mut self.data {
            ConstraintData::Ball { swing: held, .. } => *held = swing,
            other => panic!("a {:?} joint declares no swing limit", other.kind()),
        }
    }

    pub fn set_motor_velocity(&mut self, target_velocity: f32, max_force: f32) {
        let motor = self.motor_mut();
        *motor = motor.velocity(target_velocity).force(max_force);
    }

    pub fn set_spring(&mut self, spring: Option<ConstraintSpring>) {
        if let Some(spring) = spring {
            assert!(
                spring.frequency >= 0.0,
                "spring frequency must be non-negative"
            );
            assert!(
                spring.damping_ratio >= 0.0,
                "spring damping ratio must be non-negative"
            );
        }
        match &mut self.data {
            ConstraintData::Distance { spring: held, .. } => *held = spring,
            other => panic!("a {:?} joint declares no spring", other.kind()),
        }
    }

    pub fn set_break_threshold(&mut self, threshold: Option<ConstraintBreak>) {
        if let Some(threshold) = threshold {
            assert!(threshold.force >= 0.0, "break force must be non-negative");
            assert!(threshold.torque >= 0.0, "break torque must be non-negative");
        }
        self.break_threshold = threshold;
    }

    pub fn set_warm_start(&mut self, warm_start: bool) {
        self.warm_start = warm_start;
    }

    pub fn set_disable_collisions(&mut self, disable: bool) {
        self.disable_collisions = disable;
    }

    pub fn dof_mut(&mut self, index: usize) -> &mut DofDesc {
        assert!(index < 6, "dof index must be within 0..6");
        match &mut self.data {
            ConstraintData::SixDof { dofs, .. } => &mut dofs[index],
            other => panic!("a {:?} joint declares no dof", other.kind()),
        }
    }

    fn motor_mut(&mut self) -> &mut ConstraintMotor {
        match &mut self.data {
            ConstraintData::Revolute { motor, .. } | ConstraintData::Prismatic { motor, .. } => {
                motor.get_or_insert(ConstraintMotor::new(0.0, 0.0))
            }
            other => panic!("a {:?} joint declares no motor", other.kind()),
        }
    }

    pub fn axis(mut self, axis: [f32; 3]) -> Self {
        self.set_axis(axis);
        self
    }

    pub fn limit(mut self, min: f32, max: f32) -> Self {
        assert!(max >= min, "constraint limit max must not be below min");
        self.set_limit(Some(ConstraintLimit { min, max }));
        self
    }

    pub fn swing(mut self, swing_a: f32, swing_b: f32) -> Self {
        assert!(swing_a >= 0.0, "swing limit must be non-negative");
        assert!(swing_b >= 0.0, "swing limit must be non-negative");
        self.set_swing(Some(ConstraintSwing { swing_a, swing_b }));
        self
    }

    pub fn motor(mut self, target_velocity: f32) -> Self {
        let motor = self.motor_mut();
        *motor = motor.velocity(target_velocity);
        self
    }

    pub fn motor_force(mut self, max_force: f32) -> Self {
        let motor = self.motor_mut();
        *motor = motor.force(max_force);
        self
    }

    pub fn set_servo(&mut self, target_position: f32, stiffness: f32, damping: f32) {
        let target = ConstraintPositionTarget::new(target_position, stiffness, damping);
        let motor = self.motor_mut();
        *motor = motor.position(target);
    }

    pub fn servo(mut self, target_position: f32, stiffness: f32, damping: f32) -> Self {
        self.set_servo(target_position, stiffness, damping);
        self
    }
    pub fn spring(mut self, frequency: f32, damping_ratio: f32) -> Self {
        assert!(frequency >= 0.0, "spring frequency must be non-negative");
        assert!(
            damping_ratio >= 0.0,
            "spring damping ratio must be non-negative"
        );
        self.set_spring(Some(ConstraintSpring {
            frequency,
            damping_ratio,
        }));
        self
    }

    pub fn dofs(mut self, dofs: [DofDesc; 6]) -> Self {
        match &mut self.data {
            ConstraintData::SixDof { dofs: held, .. } => **held = dofs,
            other => panic!("a {:?} joint declares no dof", other.kind()),
        }
        self
    }

    pub fn dof(mut self, index: usize, desc: DofDesc) -> Self {
        *self.dof_mut(index) = desc;
        self
    }

    pub fn break_threshold(mut self, force: f32, torque: f32) -> Self {
        assert!(force >= 0.0, "break force must be non-negative");
        assert!(torque >= 0.0, "break torque must be non-negative");
        self.set_break_threshold(Some(ConstraintBreak { force, torque }));
        self
    }

    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.set_warm_start(warm_start);
        self
    }

    pub fn disable_collisions(mut self, disable: bool) -> Self {
        self.set_disable_collisions(disable);
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstraintHandle {
    pub id: u32,
    pub generation: u32,
}
