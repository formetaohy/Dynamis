use super::World;
use super::commands::ConstraintCommand;
use super::ids::IdSpace;
use dynamis_layout::ConstraintDescriptorRecord;
use dynamis_model::{
    BodyHandle, ConstraintBreak, ConstraintDesc, ConstraintHandle, ConstraintKind, ConstraintLimit,
    ConstraintMotor, ConstraintSpring, ConstraintSwing,
};

pub(crate) struct Constraints {
    pub(crate) alive: Vec<ConstraintHandle>,
    pub(crate) ids: IdSpace,
    pub(crate) index_of: Vec<u32>,
    pub(crate) records: Vec<ConstraintDescriptorRecord>,
    pub(crate) commands: Vec<ConstraintCommand>,
    pub(crate) dirty: Vec<u32>,
    pub(crate) last_moves: u32,
    pub(crate) last_commands: u32,
    pub(crate) broken: Vec<ConstraintHandle>,
}

impl Constraints {
    pub(crate) const fn new() -> Self {
        Self {
            alive: Vec::new(),
            ids: IdSpace::new(),
            index_of: Vec::new(),
            records: Vec::new(),
            commands: Vec::new(),
            dirty: Vec::new(),
            last_moves: 0,
            last_commands: 0,
            broken: Vec::new(),
        }
    }

    fn grow_to(&mut self, id: u32) {
        let rows = id as usize + 1;
        if rows > self.index_of.len() {
            self.index_of.resize(rows, u32::MAX);
        }
    }

    fn attach(&mut self, handle: ConstraintHandle, record: ConstraintDescriptorRecord) -> u32 {
        let slot = self.alive.len() as u32;
        self.index_of[handle.id as usize] = slot;
        self.alive.push(handle);
        self.records.push(record);
        slot
    }

    fn detach(&mut self, slot: usize) -> bool {
        let tail = self.alive.len() - 1;
        self.alive.swap_remove(slot);
        self.records.swap_remove(slot);
        if slot == tail {
            return false;
        }
        let moved = self.alive[slot];
        self.index_of[moved.id as usize] = slot as u32;
        true
    }
}

impl World {
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
        if constrains_joint_frame(desc.kind)
            && desc.reference == [0.0, 0.0, 0.0, 1.0]
            && let (Some(state_a), Some(state_b)) = (
                self.state_snapshot(first.id as usize),
                self.state_snapshot(second.id as usize),
            )
        {
            desc.reference = relative_reference(state_a.orientation, state_b.orientation);
        }
        let (id, generation) = self.constraints.ids.acquire();
        self.constraints.grow_to(id);
        let handle = ConstraintHandle { id, generation };
        let record = ConstraintDescriptorRecord::build(
            &desc,
            self.bodies.index_of[first.id as usize],
            self.bodies.index_of[second.id as usize],
        );
        let slot = self.constraints.attach(handle, record);
        self.constraints.dirty.push(slot);
        self.constraints.commands.push(ConstraintCommand::Add {
            slot,
            id,
            generation,
        });
        handle
    }

    pub fn update_constraint(&mut self, handle: ConstraintHandle, desc: ConstraintDesc) {
        self.validate_constraint(handle);
        self.validate_constraint_desc(&desc);
        let slot = self.constraints.index_of[handle.id as usize] as usize;
        let existing = self.constraints.records[slot];
        let mut desc = desc;
        if constrains_joint_frame(desc.kind) && desc.reference == [0.0, 0.0, 0.0, 1.0] {
            desc.reference = existing.reference;
        }
        let record = ConstraintDescriptorRecord::build(&desc, existing.a, existing.b);
        self.constraints.records[slot] = record;
        self.constraints.dirty.push(slot as u32);
    }

    pub fn set_motor(&mut self, handle: ConstraintHandle, target_velocity: f32, max_force: f32) {
        assert!(max_force >= 0.0, "motor force must be non-negative");
        self.patch_record(handle, |record| {
            record.motor_speed = target_velocity;
            record.motor_max_force = max_force;
            record.motor_target = 0.0;
            record.motor_stiffness = 0.0;
            record.motor_damping = 0.0;
            record.flags |= dynamis_layout::CONSTRAINT_HAS_MOTOR;
        });
    }

    pub fn set_limit(&mut self, handle: ConstraintHandle, limit: Option<ConstraintLimit>) {
        if let Some(limit) = limit {
            assert!(
                limit.max >= limit.min,
                "constraint limit max must not be below min"
            );
        }
        self.patch_record(handle, |record| match limit {
            Some(limit) => {
                record.limit_min = limit.min;
                record.limit_max = limit.max;
                record.flags |= dynamis_layout::CONSTRAINT_HAS_LIMIT;
            }
            None => record.flags &= !dynamis_layout::CONSTRAINT_HAS_LIMIT,
        });
    }

    pub fn set_spring(&mut self, handle: ConstraintHandle, spring: Option<ConstraintSpring>) {
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
        self.patch_record(handle, |record| match spring {
            Some(spring) => {
                record.spring_frequency = spring.frequency;
                record.spring_damping_ratio = spring.damping_ratio;
                record.flags |= dynamis_layout::CONSTRAINT_IS_SPRING;
            }
            None => record.flags &= !dynamis_layout::CONSTRAINT_IS_SPRING,
        });
    }

    pub fn set_break_threshold(
        &mut self,
        handle: ConstraintHandle,
        threshold: Option<ConstraintBreak>,
    ) {
        if let Some(threshold) = threshold {
            assert!(threshold.force >= 0.0, "break force must be non-negative");
            assert!(threshold.torque >= 0.0, "break torque must be non-negative");
        }
        self.patch_record(handle, |record| match threshold {
            Some(threshold) => {
                record.break_force = threshold.force;
                record.break_torque = threshold.torque;
                record.flags |= dynamis_layout::CONSTRAINT_HAS_BREAK;
            }
            None => record.flags &= !dynamis_layout::CONSTRAINT_HAS_BREAK,
        });
    }

    pub fn set_warm_start(&mut self, handle: ConstraintHandle, warm_start: bool) {
        self.patch_record(handle, |record| {
            record.flags = (record.flags & !dynamis_layout::CONSTRAINT_WARM_START)
                | if warm_start {
                    dynamis_layout::CONSTRAINT_WARM_START
                } else {
                    0
                };
        });
    }

    pub fn set_servo(
        &mut self,
        handle: ConstraintHandle,
        target_position: f32,
        stiffness: f32,
        damping: f32,
    ) {
        assert!(
            (0.0..=1.0).contains(&stiffness),
            "servo stiffness must be within [0, 1]"
        );
        assert!(
            (0.0..=1.0).contains(&damping),
            "servo damping must be within [0, 1]"
        );
        self.patch_record(handle, |record| {
            record.motor_target = target_position;
            record.motor_stiffness = stiffness;
            record.motor_damping = damping;
            record.flags |= dynamis_layout::CONSTRAINT_HAS_MOTOR;
        });
    }

    pub fn set_swing_limits(&mut self, handle: ConstraintHandle, swing: Option<ConstraintSwing>) {
        if let Some(swing) = swing {
            assert!(swing.swing_a >= 0.0, "swing limit must be non-negative");
            assert!(swing.swing_b >= 0.0, "swing limit must be non-negative");
        }
        self.patch_record(handle, |record| match swing {
            Some(swing) => {
                record.swing_a = swing.swing_a;
                record.swing_b = swing.swing_b;
                record.flags |= dynamis_layout::CONSTRAINT_HAS_SWING;
            }
            None => record.flags &= !dynamis_layout::CONSTRAINT_HAS_SWING,
        });
    }

    pub fn set_constraint_disable_collisions(&mut self, handle: ConstraintHandle, disable: bool) {
        self.patch_record(handle, |record| {
            record.flags = (record.flags & !dynamis_layout::CONSTRAINT_DISABLE_COLLISIONS)
                | if disable {
                    dynamis_layout::CONSTRAINT_DISABLE_COLLISIONS
                } else {
                    0
                };
        });
    }

    pub fn set_dof_locked(&mut self, handle: ConstraintHandle, index: usize, locked: bool) {
        self.assert_dof_index(index);
        self.patch_record(handle, |record| {
            record.flags = dynamis_layout::set_dof_locked(record.flags, index as u32, locked);
        });
    }

    pub fn set_dof_limit(
        &mut self,
        handle: ConstraintHandle,
        index: usize,
        limit: Option<ConstraintLimit>,
    ) {
        self.assert_dof_index(index);
        if let Some(limit) = limit {
            assert!(
                limit.max >= limit.min,
                "dof limit max must not be below min"
            );
        }
        self.patch_record(handle, |record| {
            let (min, max) = dof_limit_pair(record, index);
            match limit {
                Some(limit) => {
                    *min = limit.min;
                    *max = limit.max;
                    record.flags =
                        dynamis_layout::set_dof_limited(record.flags, index as u32, true);
                }
                None => {
                    *min = 0.0;
                    *max = 0.0;
                    record.flags =
                        dynamis_layout::set_dof_limited(record.flags, index as u32, false);
                }
            }
        });
    }

    pub fn set_dof_motor(
        &mut self,
        handle: ConstraintHandle,
        index: usize,
        motor: Option<ConstraintMotor>,
    ) {
        self.assert_dof_index(index);
        self.patch_record(handle, |record| {
            let (target, stiffness, damping, force) = dof_motor_slots(record, index);
            match motor {
                Some(motor) => {
                    *target = motor.target_position.unwrap_or(motor.target_velocity);
                    *stiffness = motor.stiffness;
                    *damping = motor.damping;
                    *force = motor.max_force;
                    record.flags = dynamis_layout::set_dof_driven(record.flags, index as u32, true);
                }
                None => {
                    *target = 0.0;
                    *stiffness = 0.0;
                    *damping = 0.0;
                    *force = 0.0;
                    record.flags =
                        dynamis_layout::set_dof_driven(record.flags, index as u32, false);
                }
            }
        });
    }

    fn patch_record(
        &mut self,
        handle: ConstraintHandle,
        change: impl FnOnce(&mut ConstraintDescriptorRecord),
    ) {
        self.validate_constraint(handle);
        let slot = self.constraints.index_of[handle.id as usize] as usize;
        change(&mut self.constraints.records[slot]);
        self.constraints.dirty.push(slot as u32);
    }

    fn assert_dof_index(&self, index: usize) {
        assert!(
            index < dynamis_layout::DOF_COUNT as usize,
            "dof index must be below {}",
            dynamis_layout::DOF_COUNT
        );
    }

    pub fn remove_constraint(&mut self, handle: ConstraintHandle) {
        self.validate_constraint(handle);
        let id = handle.id as usize;
        let slot = self.constraints.index_of[id] as usize;
        let tail = self.constraints.alive.len() - 1;
        if self.constraints.detach(slot) {
            self.constraints.dirty.push(slot as u32);
            self.constraints.commands.push(ConstraintCommand::Swap {
                slot: slot as u32,
                tail: tail as u32,
            });
        }
        self.constraints.index_of[id] = u32::MAX;
        self.constraints.ids.release(handle.id);
        self.constraints.dirty.retain(|dirty| *dirty != tail as u32);
    }

    pub fn constraints(&self) -> &[ConstraintHandle] {
        &self.constraints.alive
    }

    fn validate_constraint(&self, handle: ConstraintHandle) {
        let id = handle.id as usize;
        if id >= self.constraints.ids.len() {
            panic!("constraint handle {handle:?} is out of range");
        }
        if self.constraints.ids.generation(handle.id) != handle.generation {
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

fn constrains_joint_frame(kind: ConstraintKind) -> bool {
    matches!(
        kind,
        ConstraintKind::Fixed
            | ConstraintKind::Revolute
            | ConstraintKind::Prismatic
            | ConstraintKind::SixDof
    )
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

fn dof_limit_pair(record: &mut ConstraintDescriptorRecord, index: usize) -> (&mut f32, &mut f32) {
    if index < 3 {
        (
            &mut record.linear_limit_min[index],
            &mut record.linear_limit_max[index],
        )
    } else {
        let axis = index - 3;
        (
            &mut record.angular_limit_min[axis],
            &mut record.angular_limit_max[axis],
        )
    }
}

type DofMotorSlots<'a> = (&'a mut f32, &'a mut f32, &'a mut f32, &'a mut f32);

fn dof_motor_slots(record: &mut ConstraintDescriptorRecord, index: usize) -> DofMotorSlots<'_> {
    let axis = index % 3;
    if index < 3 {
        (
            &mut record.linear_motor_target[axis],
            &mut record.linear_motor_stiffness[axis],
            &mut record.linear_motor_damping[axis],
            &mut record.linear_motor_force[axis],
        )
    } else {
        (
            &mut record.angular_motor_target[axis],
            &mut record.angular_motor_stiffness[axis],
            &mut record.angular_motor_damping[axis],
            &mut record.angular_motor_force[axis],
        )
    }
}
