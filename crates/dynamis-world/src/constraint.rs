use super::World;
use super::command::ConstraintCommand;
use super::device::Facts;
use super::pool::{Pool, Retired};
use dynamis_abi::{BrokenConstraintRecord, ConstraintDescriptorRecord};
use dynamis_model::{
    BodyHandle, ConstraintBreak, ConstraintDesc, ConstraintHandle, ConstraintLimit,
    ConstraintMotor, ConstraintSpring, ConstraintSwing, DofDesc,
};

#[derive(Clone)]
pub(crate) struct JointDesc {
    pub(crate) first: u32,
    pub(crate) second: u32,
    pub(crate) desc: ConstraintDesc,
}

impl JointDesc {
    pub(crate) const VACANT: Self = Self {
        first: dynamis_abi::NO_BODY,
        second: dynamis_abi::NO_BODY,
        desc: ConstraintDesc::VACANT,
    };

    fn record(self) -> ConstraintDescriptorRecord {
        ConstraintDescriptorRecord::build(&self.desc, self.first, self.second)
    }
}

#[derive(Clone)]
pub(crate) struct ConstraintStore {
    pub(crate) pool: Pool<ConstraintHandle>,
    pub(crate) joints: Vec<JointDesc>,
    pub(crate) attached: Vec<Vec<u32>>,
    pub(crate) commands: Vec<ConstraintCommand>,
    pub(crate) last_moves: u32,
    pub(crate) last_commands: u32,
    pub(crate) broken: Vec<ConstraintHandle>,
}

impl ConstraintStore {
    pub(crate) const fn new() -> Self {
        Self {
            pool: Pool::compact("constraint"),
            joints: Vec::new(),
            attached: Vec::new(),
            commands: Vec::new(),
            last_moves: 0,
            last_commands: 0,
            broken: Vec::new(),
        }
    }

    pub(crate) fn attached_to(&self, body_id: u32) -> &[u32] {
        self.attached
            .get(body_id as usize)
            .map_or(&[], Vec::as_slice)
    }

    fn attach_to(&mut self, body_id: u32, constraint_id: u32) {
        let index = body_id as usize;
        if self.attached.len() <= index {
            self.attached.resize(index + 1, Vec::new());
        }
        self.attached[index].push(constraint_id);
    }

    fn detach_from(&mut self, body_id: u32, constraint_id: u32) {
        let attached = &mut self.attached[body_id as usize];
        let slot = attached
            .iter()
            .position(|id| *id == constraint_id)
            .expect("an attached constraint must exist");
        attached.swap_remove(slot);
    }

    fn attach(&mut self, handle: ConstraintHandle, joint: JointDesc) -> u32 {
        let slot = self.pool.insert(handle);
        if self.joints.len() <= handle.id as usize {
            self.joints
                .resize(handle.id as usize + 1, JointDesc::VACANT);
        }
        self.joints[handle.id as usize] = joint;
        slot
    }

    pub(crate) fn is_alive(&self, handle: ConstraintHandle) -> bool {
        self.pool.contains(handle)
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
        let handle = self.constraints.pool.acquire();
        let first_id = first.id;
        let second_id = second.id;
        let slot = self.constraints.attach(
            handle,
            JointDesc {
                first: first_id,
                second: second_id,
                desc,
            },
        );
        self.constraints.attach_to(first_id, handle.id);
        self.constraints.attach_to(second_id, handle.id);
        self.constraints.commands.push(ConstraintCommand::Add {
            slot,
            id: handle.id,
            generation: handle.generation,
        });
        handle
    }

    pub fn update_constraint(&mut self, handle: ConstraintHandle, desc: ConstraintDesc) {
        self.validate_constraint(handle);
        self.edit_constraint(handle, |joint| joint.desc = desc);
    }

    pub fn constraint_desc(&self, handle: ConstraintHandle) -> ConstraintDesc {
        self.joint(handle).desc
    }

    pub fn set_motor(&mut self, handle: ConstraintHandle, target_velocity: f32, max_force: f32) {
        self.edit_constraint(handle, |joint| {
            joint.desc.set_motor(Some(ConstraintMotor {
                target_velocity,
                max_force,
                target_position: None,
                stiffness: 0.0,
                damping: 0.0,
            }));
        });
    }

    pub fn set_limit(&mut self, handle: ConstraintHandle, limit: Option<ConstraintLimit>) {
        self.edit_constraint(handle, |joint| joint.desc.set_limit(limit));
    }

    pub fn set_spring(&mut self, handle: ConstraintHandle, spring: Option<ConstraintSpring>) {
        self.edit_constraint(handle, |joint| joint.desc.set_spring(spring));
    }

    pub fn set_break_threshold(
        &mut self,
        handle: ConstraintHandle,
        threshold: Option<ConstraintBreak>,
    ) {
        self.edit_constraint(handle, |joint| joint.desc.set_break_threshold(threshold));
    }

    pub fn set_warm_start(&mut self, handle: ConstraintHandle, warm_start: bool) {
        self.edit_constraint(handle, |joint| joint.desc.set_warm_start(warm_start));
    }

    pub fn set_servo(
        &mut self,
        handle: ConstraintHandle,
        target_position: f32,
        stiffness: f32,
        damping: f32,
    ) {
        self.edit_constraint(handle, |joint| {
            joint.desc.set_servo(target_position, stiffness, damping);
        });
    }

    pub fn set_swing_limits(&mut self, handle: ConstraintHandle, swing: Option<ConstraintSwing>) {
        self.edit_constraint(handle, |joint| joint.desc.set_swing(swing));
    }

    pub fn set_constraint_disable_collisions(&mut self, handle: ConstraintHandle, disable: bool) {
        self.edit_constraint(handle, |joint| joint.desc.set_disable_collisions(disable));
    }

    pub fn set_dof_locked(&mut self, handle: ConstraintHandle, index: usize, locked: bool) {
        self.edit_dof(handle, index, |dof| dof.set_locked(locked));
    }

    pub fn set_dof_limit(
        &mut self,
        handle: ConstraintHandle,
        index: usize,
        limit: Option<ConstraintLimit>,
    ) {
        self.edit_dof(handle, index, |dof| dof.set_limit(limit));
    }

    pub fn set_dof_motor(
        &mut self,
        handle: ConstraintHandle,
        index: usize,
        motor: Option<ConstraintMotor>,
    ) {
        self.edit_dof(handle, index, |dof| dof.set_motor(motor));
    }

    fn edit_constraint(&mut self, handle: ConstraintHandle, change: impl FnOnce(&mut JointDesc)) {
        self.validate_constraint(handle);
        change(&mut self.constraints.joints[handle.id as usize]);
        self.constraints.pool.mark(handle);
    }

    fn edit_dof(
        &mut self,
        handle: ConstraintHandle,
        index: usize,
        change: impl FnOnce(&mut DofDesc),
    ) {
        self.edit_constraint(handle, |joint| change(joint.desc.dof_mut(index)));
    }

    pub fn remove_constraint(&mut self, handle: ConstraintHandle) {
        let joint = self.joint(handle);
        self.constraints.detach_from(joint.first, handle.id);
        self.constraints.detach_from(joint.second, handle.id);
        let Retired { row, moved } = self.constraints.pool.retire(handle);
        self.constraints.joints[handle.id as usize] = JointDesc::VACANT;
        if moved.is_some() {
            self.constraints.commands.push(ConstraintCommand::Swap {
                slot: row,
                tail: self.constraints.pool.len(),
            });
        }
        self.observed.joints.stop_watching(handle.id);
    }

    pub(crate) fn consume_breaks(&mut self, count: u32, bytes: &[u8]) {
        let records = dynamis_abi::decode::<BrokenConstraintRecord>(bytes);
        assert!(
            records.len() <= count as usize,
            "a break publication must not carry more reports than it declared"
        );
        for record in records {
            self.accept_constraint_break(record.constraint_id, record.generation);
        }
    }

    pub fn collect_constraint_breaks(&mut self) -> Vec<ConstraintHandle> {
        self.sync(Facts::Arrived);
        std::mem::take(&mut self.constraints.broken)
    }

    pub fn drain_constraint_breaks(&mut self) -> Vec<ConstraintHandle> {
        self.sync(Facts::Retired);
        std::mem::take(&mut self.constraints.broken)
    }

    pub fn constraints(&self) -> &[ConstraintHandle] {
        self.constraints.pool.alive()
    }

    pub fn body_constraints(&self, handle: BodyHandle) -> Vec<ConstraintHandle> {
        self.validate(handle);
        self.constraints
            .attached_to(handle.id)
            .iter()
            .map(|id| ConstraintHandle {
                id: *id,
                generation: self.constraints.pool.generation(*id),
            })
            .collect()
    }

    pub fn constraint_bodies(&self, handle: ConstraintHandle) -> (BodyHandle, BodyHandle) {
        let joint = self.joint(handle);
        (
            BodyHandle {
                id: joint.first,
                generation: self.bodies.pool.generation(joint.first),
            },
            BodyHandle {
                id: joint.second,
                generation: self.bodies.pool.generation(joint.second),
            },
        )
    }

    pub(crate) fn joint(&self, handle: ConstraintHandle) -> JointDesc {
        self.validate_constraint(handle);
        self.constraints.joints[handle.id as usize].clone()
    }

    pub(crate) fn record_of(&self, handle: ConstraintHandle) -> ConstraintDescriptorRecord {
        self.joint(handle).record()
    }

    pub(crate) fn validate_constraint(&self, handle: ConstraintHandle) {
        self.constraints.pool.validate(handle);
    }
}
