use super::Simulation;
use dynamis_layout::{ConstraintCommandRecord, ConstraintDescriptorRecord};
use dynamis_model::{
    BodyHandle, ConstraintBreak, ConstraintDesc, ConstraintHandle, ConstraintKind, ConstraintLimit,
    ConstraintMotor, ConstraintSpring, ConstraintSwing, DofDesc,
};

pub(crate) struct Constraints {
    pub(crate) alive: Vec<ConstraintHandle>,
    pub(crate) index_of: Vec<u32>,
    pub(crate) generations: Vec<u32>,
    pub(crate) free_ids: Vec<u32>,
    pub(crate) records: Vec<ConstraintDescriptorRecord>,
    pub(crate) commands: Vec<ConstraintCommandRecord>,
    pub(crate) dirty: Vec<u32>,
    pub(crate) structural: bool,
    pub(crate) last_commands: u32,
    pub(crate) broken: Vec<ConstraintHandle>,
}

impl Constraints {
    pub(crate) fn new(slots: usize) -> Self {
        Self {
            alive: Vec::new(),
            index_of: vec![u32::MAX; slots],
            generations: vec![1; slots],
            free_ids: (0..slots as u32).rev().collect(),
            records: Vec::new(),
            commands: Vec::new(),
            dirty: Vec::new(),
            structural: false,
            last_commands: 0,
            broken: Vec::new(),
        }
    }

    pub(crate) fn resize(&mut self, slots: usize) {
        self.index_of.resize(slots, u32::MAX);
        self.generations.resize(slots, 1);
    }
}

impl Simulation {
    pub fn add_constraint(
        &mut self,
        first: BodyHandle,
        second: BodyHandle,
        desc: ConstraintDesc,
    ) -> ConstraintHandle {
        self.validate(first);
        self.validate(second);
        if first == second {
            panic!("constraint bodies must be distinct");
        }
        self.validate_constraint_desc(&desc);
        let mut desc = desc;
        if desc.kind == ConstraintKind::SixDof
            && desc.reference == [0.0, 0.0, 0.0, 1.0]
            && let (Some(state_a), Some(state_b)) = (
                self.state_snapshot(first.id as usize),
                self.state_snapshot(second.id as usize),
            )
        {
            desc.reference = relative_reference(state_a.orientation, state_b.orientation);
        }
        let id = self
            .constraints
            .free_ids
            .pop()
            .expect("simulation constraint capacity exhausted");
        self.constraints.generations[id as usize] += 1;
        let handle = ConstraintHandle {
            id,
            generation: self.constraints.generations[id as usize],
        };
        let slot = self.constraints.alive.len() as u32;
        self.constraints.index_of[id as usize] = slot;
        self.constraints.alive.push(handle);
        self.constraints.dirty.push(slot);
        let record = ConstraintDescriptorRecord::build(
            &desc,
            self.bodies.index_of[first.id as usize],
            self.bodies.index_of[second.id as usize],
        );
        self.constraints.records.push(record);
        self.constraints.commands.push(ConstraintCommandRecord::add(
            slot,
            id,
            self.constraints.generations[id as usize],
        ));
        handle
    }

    pub fn update_constraint(&mut self, handle: ConstraintHandle, desc: ConstraintDesc) {
        self.validate_constraint(handle);
        self.validate_constraint_desc(&desc);
        let slot = self.constraints.index_of[handle.id as usize] as usize;
        let existing = self.constraints.records[slot];
        let record = ConstraintDescriptorRecord::build(&desc, existing.a, existing.b);
        self.constraints.records[slot] = record;
        self.constraints.dirty.push(slot as u32);
    }

    pub fn set_motor(&mut self, handle: ConstraintHandle, target_velocity: f32, max_force: f32) {
        assert!(max_force >= 0.0, "motor force must be non-negative");
        self.patch_constraint(handle, |desc| {
            let motor = desc.motor.get_or_insert(ConstraintMotor {
                target_velocity: 0.0,
                max_force: 0.0,
                target_position: None,
                stiffness: 0.0,
                damping: 0.0,
            });
            motor.target_velocity = target_velocity;
            motor.max_force = max_force;
            motor.target_position = None;
            motor.stiffness = 0.0;
            motor.damping = 0.0;
        });
    }

    pub fn set_limit(&mut self, handle: ConstraintHandle, limit: Option<ConstraintLimit>) {
        self.patch_constraint(handle, |desc| {
            desc.limit = limit;
        });
    }

    pub fn set_spring(&mut self, handle: ConstraintHandle, spring: Option<ConstraintSpring>) {
        self.patch_constraint(handle, |desc| {
            desc.spring = spring;
        });
    }

    pub fn set_break_threshold(
        &mut self,
        handle: ConstraintHandle,
        threshold: Option<ConstraintBreak>,
    ) {
        self.patch_constraint(handle, |desc| {
            desc.break_threshold = threshold;
        });
    }

    pub fn set_warm_start(&mut self, handle: ConstraintHandle, warm_start: bool) {
        self.patch_constraint(handle, |desc| {
            desc.warm_start = warm_start;
        });
    }

    pub fn set_servo(
        &mut self,
        handle: ConstraintHandle,
        target_position: f32,
        stiffness: f32,
        damping: f32,
    ) {
        self.patch_constraint(handle, |desc| {
            let motor = desc.motor.get_or_insert(ConstraintMotor {
                target_velocity: 0.0,
                max_force: 0.0,
                target_position: None,
                stiffness: 0.0,
                damping: 0.0,
            });
            motor.target_position = Some(target_position);
            motor.stiffness = stiffness;
            motor.damping = damping;
        });
    }

    pub fn set_swing_limits(&mut self, handle: ConstraintHandle, swing: Option<ConstraintSwing>) {
        self.patch_constraint(handle, |desc| {
            desc.swing = swing;
        });
    }

    pub fn set_constraint_disable_collisions(&mut self, handle: ConstraintHandle, disable: bool) {
        self.patch_constraint(handle, |desc| {
            desc.disable_collisions = disable;
        });
    }

    fn patch_constraint(
        &mut self,
        handle: ConstraintHandle,
        change: impl FnOnce(&mut ConstraintDesc),
    ) {
        self.validate_constraint(handle);
        let slot = self.constraints.index_of[handle.id as usize] as usize;
        let mut desc = constraint_desc_from_record(&self.constraints.records[slot]);
        change(&mut desc);
        self.validate_constraint_desc(&desc);
        let existing = self.constraints.records[slot];
        let record = ConstraintDescriptorRecord::build(&desc, existing.a, existing.b);
        self.constraints.records[slot] = record;
        self.constraints.dirty.push(slot as u32);
    }

    pub fn remove_constraint(&mut self, handle: ConstraintHandle) {
        self.validate_constraint(handle);
        let id = handle.id as usize;
        let slot = self.constraints.index_of[id] as usize;
        let tail = self.constraints.alive.len() - 1;
        self.constraints.alive.swap_remove(slot);
        self.constraints.records.remove(slot);
        if slot < tail {
            let moved = self.constraints.alive[slot];
            self.constraints.index_of[moved.id as usize] = slot as u32;
            self.constraints.dirty.push(slot as u32);
            self.constraints
                .commands
                .push(ConstraintCommandRecord::swap(slot as u32, tail as u32));
        }
        self.constraints.index_of[id] = u32::MAX;
        self.constraints.free_ids.push(handle.id);
        self.constraints.dirty.retain(|dirty| *dirty != tail as u32);
    }

    pub fn constraints(&self) -> &[ConstraintHandle] {
        &self.constraints.alive
    }

    fn validate_constraint(&self, handle: ConstraintHandle) {
        let id = handle.id as usize;
        if id >= self.slots {
            panic!("constraint handle {handle:?} is out of range");
        }
        if self.constraints.generations[id] != handle.generation {
            panic!("constraint handle {handle:?} is stale");
        }
        if self.constraints.index_of[id] == u32::MAX {
            panic!("constraint handle {handle:?} is not alive");
        }
    }

    fn validate_constraint_desc(&self, desc: &ConstraintDesc) {
        match desc.kind {
            ConstraintKind::Ball
            | ConstraintKind::Distance
            | ConstraintKind::Pulley
            | ConstraintKind::Gear => {}
            ConstraintKind::Cone => {
                if desc.axis_a == [0.0; 3] || desc.axis_b == [0.0; 3] {
                    panic!("constraint axis must be non-zero");
                }
            }
            _ => {
                if desc.axis_a == [0.0; 3] {
                    panic!("constraint axis must be non-zero");
                }
            }
        }
    }

    pub(super) fn remap_constraint_slots(&mut self, first: u32, second: u32) {
        for index in 0..self.constraints.records.len() {
            let record = &mut self.constraints.records[index];
            let mut changed = false;
            if record.a == first && record.b == second {
                record.a = second;
                record.b = first;
                changed = true;
            } else if record.a == second && record.b == first {
                record.a = first;
                record.b = second;
                changed = true;
            } else {
                if record.a == first {
                    record.a = second;
                    changed = true;
                } else if record.a == second {
                    record.a = first;
                    changed = true;
                }
                if record.b == first {
                    record.b = second;
                    changed = true;
                } else if record.b == second {
                    record.b = first;
                    changed = true;
                }
            }
            if changed {
                self.constraints.dirty.push(index as u32);
            }
        }
    }

    pub(super) fn assert_no_constraints(&self, handle: BodyHandle) {
        for constraint in &self.constraints.alive {
            if constraint.id == handle.id {
                panic!(
                    "body handle {handle:?} is referenced by a live constraint; remove it first"
                );
            }
        }
    }
}

fn relative_reference(orientation_a: [f32; 4], orientation_b: [f32; 4]) -> [f32; 4] {
    let a = [
        -orientation_a[0],
        -orientation_a[1],
        -orientation_a[2],
        orientation_a[3],
    ];
    dynamis_math::quat_mul(a, orientation_b)
}

fn constraint_desc_from_record(record: &ConstraintDescriptorRecord) -> ConstraintDesc {
    let kind = match record.kind {
        dynamis_layout::CONSTRAINT_BALL => ConstraintKind::Ball,
        dynamis_layout::CONSTRAINT_DISTANCE => ConstraintKind::Distance,
        dynamis_layout::CONSTRAINT_REVOLUTE => ConstraintKind::Revolute,
        dynamis_layout::CONSTRAINT_PRISMATIC => ConstraintKind::Prismatic,
        dynamis_layout::CONSTRAINT_FIXED => ConstraintKind::Fixed,
        dynamis_layout::CONSTRAINT_GEAR => ConstraintKind::Gear,
        dynamis_layout::CONSTRAINT_CONE => ConstraintKind::Cone,
        dynamis_layout::CONSTRAINT_SIXDOF => ConstraintKind::SixDof,
        other => panic!("constraint record has an invalid kind {other}"),
    };
    let mut desc = ConstraintDesc::ball(record.anchor_a, record.anchor_b).rekind(kind);
    desc.axis_a = record.axis_a;
    desc.axis_b = record.axis_b;
    desc.reference = record.reference;
    desc.rest_length = record.distance;
    desc.cone_angle = record.cone_angle;
    desc.warm_start = record.flags & dynamis_layout::CONSTRAINT_WARM_START != 0;
    if record.flags & dynamis_layout::CONSTRAINT_HAS_LIMIT != 0 {
        desc.limit = Some(ConstraintLimit {
            min: record.limit_min,
            max: record.limit_max,
        });
    }
    if record.flags & dynamis_layout::CONSTRAINT_HAS_SWING != 0 {
        desc.swing = Some(ConstraintSwing {
            swing_a: record.swing_a,
            swing_b: record.swing_b,
        });
    }
    if record.flags & dynamis_layout::CONSTRAINT_HAS_MOTOR != 0 {
        desc.motor = Some(ConstraintMotor {
            target_velocity: record.motor_speed,
            max_force: record.motor_max_force,
            target_position: (record.motor_stiffness > 0.0).then_some(record.motor_target),
            stiffness: record.motor_stiffness,
            damping: record.motor_damping,
        });
    }
    if record.flags & dynamis_layout::CONSTRAINT_IS_SPRING != 0 {
        desc.spring = Some(ConstraintSpring {
            frequency: record.spring_frequency,
            damping_ratio: record.spring_damping_ratio,
        });
    }
    if record.flags & dynamis_layout::CONSTRAINT_HAS_BREAK != 0 {
        desc.break_threshold = Some(ConstraintBreak {
            force: record.break_force,
            torque: record.break_torque,
        });
    }
    desc.gear_ratio = record.gear_ratio;
    desc.pulley_fixed_a = record.pulley_fixed_a;
    desc.pulley_fixed_b = record.pulley_fixed_b;
    desc.disable_collisions = record.flags & dynamis_layout::CONSTRAINT_DISABLE_COLLISIONS != 0;
    if desc.kind == ConstraintKind::SixDof || (record.kind == dynamis_layout::CONSTRAINT_SIXDOF) {
        let mode_of = |index: u32| -> DofDesc {
            let mode = dynamis_layout::dof_mode(record.flags, index);
            match mode {
                dynamis_layout::DOF_LOCKED => DofDesc::locked(),
                dynamis_layout::DOF_LIMITED => {
                    let (min, max) = if index < 3 {
                        (
                            record.linear_limit_min[index as usize],
                            record.linear_limit_max[index as usize],
                        )
                    } else {
                        (
                            record.angular_limit_min[index as usize - 3],
                            record.angular_limit_max[index as usize - 3],
                        )
                    };
                    DofDesc::limited(min, max)
                }
                dynamis_layout::DOF_DRIVEN => {
                    let (target, stiffness, damping, force) = if index < 3 {
                        (
                            record.linear_motor_target[index as usize],
                            record.linear_motor_stiffness[index as usize],
                            record.linear_motor_damping[index as usize],
                            record.linear_motor_force[index as usize],
                        )
                    } else {
                        (
                            record.angular_motor_target[index as usize - 3],
                            record.angular_motor_stiffness[index as usize - 3],
                            record.angular_motor_damping[index as usize - 3],
                            record.angular_motor_force[index as usize - 3],
                        )
                    };
                    DofDesc::driven(ConstraintMotor {
                        target_velocity: if stiffness <= 0.0 { target } else { 0.0 },
                        max_force: force,
                        target_position: (stiffness > 0.0).then_some(target),
                        stiffness,
                        damping,
                    })
                }
                _ => DofDesc::free(),
            }
        };
        desc.dofs = Some(std::array::from_fn(|index| mode_of(index as u32)));
    }
    desc
}
