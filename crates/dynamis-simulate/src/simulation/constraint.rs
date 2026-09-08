use super::Simulation;
use dynamis_layout::{ConstraintCommandRecord, ConstraintRecord};
use dynamis_model::{
    BodyHandle, ConstraintBreak, ConstraintDesc, ConstraintHandle, ConstraintKind, ConstraintLimit,
    ConstraintMotor, ConstraintSpring, ConstraintSwing,
};

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
        let id = self
            .constraint_free_ids
            .pop()
            .expect("simulation constraint capacity exhausted");
        self.constraint_generations[id as usize] += 1;
        let handle = ConstraintHandle {
            id,
            generation: self.constraint_generations[id as usize],
        };
        let slot = self.constraint_alive.len() as u32;
        self.constraint_index_of[id as usize] = slot;
        self.constraint_alive.push(handle);
        let record = ConstraintRecord::build(
            &desc,
            self.index_of[first.id as usize],
            self.index_of[second.id as usize],
        );
        self.constraint_records.push(record);
        self.constraint_commands
            .push(ConstraintCommandRecord::add(slot, record));
        handle
    }

    pub fn update_constraint(&mut self, handle: ConstraintHandle, desc: ConstraintDesc) {
        self.validate_constraint(handle);
        self.validate_constraint_desc(&desc);
        let slot = self.constraint_index_of[handle.id as usize] as usize;
        let existing = self.constraint_records[slot];
        let record = ConstraintRecord::build(&desc, existing.a, existing.b);
        self.constraint_records[slot] = record;
        self.constraint_commands
            .push(ConstraintCommandRecord::patch(slot as u32, record));
    }

    pub fn set_motor(&mut self, handle: ConstraintHandle, target_velocity: f32, max_force: f32) {
        assert!(max_force >= 0.0, "motor force must be non-negative");
        self.patch_constraint(handle, |desc| {
            desc.motor = Some(ConstraintMotor {
                target_velocity,
                max_force,
            });
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

    pub fn swing_limits(&mut self, handle: ConstraintHandle, swing: Option<ConstraintSwing>) {
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
        let slot = self.constraint_index_of[handle.id as usize] as usize;
        let mut desc = constraint_desc_from_record(&self.constraint_records[slot]);
        change(&mut desc);
        self.validate_constraint_desc(&desc);
        let existing = self.constraint_records[slot];
        let record = ConstraintRecord::build(&desc, existing.a, existing.b);
        self.constraint_records[slot] = record;
        self.constraint_commands
            .push(ConstraintCommandRecord::patch(slot as u32, record));
    }

    pub fn remove_constraint(&mut self, handle: ConstraintHandle) {
        self.validate_constraint(handle);
        let id = handle.id as usize;
        let slot = self.constraint_index_of[id] as usize;
        let tail = self.constraint_alive.len() - 1;
        let moved = self.constraint_alive[tail];
        self.constraint_alive.swap_remove(slot);
        self.constraint_index_of[moved.id as usize] = slot as u32;
        self.constraint_index_of[id] = u32::MAX;
        self.constraint_free_ids.push(handle.id);
        self.constraint_records.remove(slot);
        self.constraint_commands
            .push(ConstraintCommandRecord::remove(slot as u32));
    }

    pub fn constraints(&self) -> &[ConstraintHandle] {
        &self.constraint_alive
    }

    fn validate_constraint(&self, handle: ConstraintHandle) {
        let id = handle.id as usize;
        if id >= self.constraint_capacity {
            panic!("constraint handle {handle:?} is out of range");
        }
        if self.constraint_generations[id] != handle.generation {
            panic!("constraint handle {handle:?} is stale");
        }
        if self.constraint_index_of[id] == u32::MAX {
            panic!("constraint handle {handle:?} is not alive");
        }
    }

    fn validate_constraint_desc(&self, desc: &ConstraintDesc) {
        match desc.kind {
            ConstraintKind::Ball
            | ConstraintKind::Distance
            | ConstraintKind::Pulley
            | ConstraintKind::Gear => {}
            _ => {
                if desc.axis_a == [0.0; 3] {
                    panic!("constraint axis must be non-zero");
                }
            }
        }
    }

    pub(super) fn assert_no_constraints(&self, handle: BodyHandle) {
        for constraint in &self.constraint_alive {
            if constraint.id == handle.id {
                panic!(
                    "body handle {handle:?} is referenced by a live constraint; remove it first"
                );
            }
        }
    }
}

fn constraint_desc_from_record(record: &ConstraintRecord) -> ConstraintDesc {
    let kind = match record.kind {
        dynamis_layout::CONSTRAINT_BALL => ConstraintKind::Ball,
        dynamis_layout::CONSTRAINT_DISTANCE => ConstraintKind::Distance,
        dynamis_layout::CONSTRAINT_REVOLUTE => ConstraintKind::Revolute,
        dynamis_layout::CONSTRAINT_PRISMATIC => ConstraintKind::Prismatic,
        dynamis_layout::CONSTRAINT_FIXED => ConstraintKind::Fixed,
        dynamis_layout::CONSTRAINT_GEAR => ConstraintKind::Gear,
        _ => ConstraintKind::Pulley,
    };
    let mut desc = ConstraintDesc::ball(record.anchor_a, record.anchor_b).rekind(kind);
    desc.axis_a = record.axis_a;
    desc.axis_b = record.axis_b;
    desc.rest_length = record.distance;
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
    desc
}
