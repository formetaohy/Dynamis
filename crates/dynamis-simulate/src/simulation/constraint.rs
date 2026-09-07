use super::Simulation;
use dynamis_layout::{ConstraintCommandRecord, ConstraintRecord};
use dynamis_model::{BodyHandle, ConstraintDesc, ConstraintHandle};

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

    fn validate_constraint_desc(&self, desc: &ConstraintDesc) {
        match desc.kind {
            dynamis_model::ConstraintKind::Ball
            | dynamis_model::ConstraintKind::Distance
            | dynamis_model::ConstraintKind::Pulley => {}
            dynamis_model::ConstraintKind::Gear => {}
            _ => {
                if desc.axis_a == [0.0; 3] {
                    panic!("constraint axis must be non-zero");
                }
            }
        }
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
