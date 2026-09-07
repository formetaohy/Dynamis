use super::Simulation;
use bytemuck::Zeroable;
use dynamis_layout::{
    BODY_CCD, BODY_KINEMATIC, BodyCommandRecord, ColliderRecord, PATCH_ANGULAR_VELOCITY, PATCH_CCD,
    PATCH_COLLIDER, PATCH_FRICTION, PATCH_GROUP, PATCH_KINEMATIC, PATCH_MASK, PATCH_MASS,
    PATCH_ORIENTATION, PATCH_POSITION, PATCH_RESTITUTION, PATCH_VELOCITY, RigidBodyRecord,
};
use dynamis_model::{
    BodyDesc, BodyHandle, BodyState, ColliderDesc, MAX_COLLIDERS_PER_BODY, MassProperties, Shape,
};

impl Simulation {
    pub fn spawn(&mut self, desc: BodyDesc) -> BodyHandle {
        let id = self
            .free_ids
            .pop()
            .expect("simulation body capacity exhausted");
        self.generations[id as usize] += 1;
        let handle = BodyHandle {
            id,
            generation: self.generations[id as usize],
        };
        let slot = self.alive.len() as u32;
        self.index_of[id as usize] = slot;
        self.alive.push(handle);
        self.validate_world_geometry(&desc);
        self.masses.resize(self.alive.len(), 1.0);
        self.masses[id as usize] = desc.mass;
        self.com_overrides.resize(self.alive.len(), None);
        self.com_overrides[id as usize] = desc.com;
        self.inertia_overrides.resize(self.alive.len(), None);
        self.inertia_overrides[id as usize] = desc.inertia;
        self.collider_descs.resize(self.alive.len(), Vec::new());
        self.collider_descs[id as usize] = desc.colliders.clone();
        let mass = self.mass_properties_of(id as usize);
        let body = RigidBodyRecord::build(&desc, handle.id, handle.generation, mass);
        let colliders = self.collider_block(&desc);
        self.record_state(id as usize, &desc, &body);
        self.commands
            .push(BodyCommandRecord::add(slot, body, colliders));
        handle
    }

    fn collider_block(&self, desc: &BodyDesc) -> [ColliderRecord; 4] {
        let mut colliders = [ColliderRecord::zeroed(); 4];
        for (index, collider) in desc.colliders.iter().enumerate() {
            colliders[index] = self.collider_record(collider);
        }
        colliders
    }

    fn record_state(&mut self, id: usize, desc: &BodyDesc, body: &RigidBodyRecord) {
        if self.states.len() <= id {
            self.states.resize(id + 1, None);
        }
        self.states[id] = Some(BodyState {
            position: desc.position,
            orientation: desc.orientation,
            velocity: desc.velocity,
            angular_velocity: desc.angular_velocity,
            inverse_mass: body.inverse_mass,
            com: body.com,
            sleeping: false,
            step: self.step_index,
        });
    }

    fn validate_world_geometry(&self, desc: &BodyDesc) {
        for collider in &desc.colliders {
            if collider.shape.is_world_geometry() && (desc.mass > 0.0 && !desc.kinematic) {
                panic!("world geometry colliders must be static or kinematic");
            }
        }
    }

    pub fn remove(&mut self, handle: BodyHandle) {
        self.validate(handle);
        self.assert_no_constraints(handle);
        let id = handle.id as usize;
        let slot = self.index_of[id] as usize;
        let tail = self.alive.len() - 1;
        let moved = self.alive[tail];
        self.alive.swap_remove(slot);
        self.index_of[moved.id as usize] = slot as u32;
        self.index_of[id] = u32::MAX;
        self.free_ids.push(handle.id);
        self.collider_descs[id] = Vec::new();
        self.states[id] = None;
        self.commands
            .push(BodyCommandRecord::remove(slot as u32, tail as u32));
    }

    pub fn set_position(&mut self, handle: BodyHandle, position: [f32; 3]) {
        self.validate(handle);
        if let Some(state) = self.states[handle.id as usize].as_mut() {
            state.position = position;
        }
        let mut record = RigidBodyRecord::zeroed();
        record.position = position;
        self.schedule_patch(handle, PATCH_POSITION, record);
    }

    pub fn set_orientation(&mut self, handle: BodyHandle, orientation: [f32; 4]) {
        self.assert_unit(orientation);
        self.validate(handle);
        if let Some(state) = self.states[handle.id as usize].as_mut() {
            state.orientation = orientation;
        }
        let mut record = RigidBodyRecord::zeroed();
        record.orientation = orientation;
        self.schedule_patch(handle, PATCH_ORIENTATION, record);
    }

    pub fn set_velocity(&mut self, handle: BodyHandle, velocity: [f32; 3]) {
        self.validate(handle);
        if let Some(state) = self.states[handle.id as usize].as_mut() {
            state.velocity = velocity;
        }
        let mut record = RigidBodyRecord::zeroed();
        record.velocity = velocity;
        self.schedule_patch(handle, PATCH_VELOCITY, record);
    }

    pub fn set_angular_velocity(&mut self, handle: BodyHandle, angular_velocity: [f32; 3]) {
        self.validate(handle);
        if let Some(state) = self.states[handle.id as usize].as_mut() {
            state.angular_velocity = angular_velocity;
        }
        let mut record = RigidBodyRecord::zeroed();
        record.angular_velocity = angular_velocity;
        self.schedule_patch(handle, PATCH_ANGULAR_VELOCITY, record);
    }

    pub fn set_mass(&mut self, handle: BodyHandle, mass: f32) {
        assert!(mass >= 0.0, "mass must be non-negative");
        self.validate(handle);
        self.masses[handle.id as usize] = mass;
        let inverse_mass = self.inverse_mass_of(handle);
        let mass_properties = self.mass_properties_of(handle.id as usize);
        let mut record = RigidBodyRecord::zeroed();
        record.inverse_mass = inverse_mass;
        record.com = mass_properties.com;
        record.inverse_inertia_body = mass_properties.inverse_inertia;
        if let Some(state) = self.states[handle.id as usize].as_mut() {
            state.inverse_mass = inverse_mass;
            state.com = mass_properties.com;
        }
        self.schedule_patch(handle, PATCH_MASS, record);
    }

    pub fn set_com(&mut self, handle: BodyHandle, com: [f32; 3]) {
        self.validate(handle);
        self.com_overrides[handle.id as usize] = Some(com);
        let mass_properties = self.mass_properties_of(handle.id as usize);
        let mut record = RigidBodyRecord::zeroed();
        record.inverse_mass = self.inverse_mass_of(handle);
        record.com = mass_properties.com;
        record.inverse_inertia_body = mass_properties.inverse_inertia;
        if let Some(state) = self.states[handle.id as usize].as_mut() {
            state.com = mass_properties.com;
        }
        self.schedule_patch(handle, PATCH_MASS, record);
    }

    pub fn set_inertia(&mut self, handle: BodyHandle, inertia: [f32; 6]) {
        self.validate(handle);
        self.inertia_overrides[handle.id as usize] = Some(inertia);
        let mass_properties = self.mass_properties_of(handle.id as usize);
        let mut record = RigidBodyRecord::zeroed();
        record.inverse_mass = self.inverse_mass_of(handle);
        record.com = mass_properties.com;
        record.inverse_inertia_body = mass_properties.inverse_inertia;
        if let Some(state) = self.states[handle.id as usize].as_mut() {
            state.com = mass_properties.com;
        }
        self.schedule_patch(handle, PATCH_MASS, record);
    }

    fn inverse_mass_of(&self, handle: BodyHandle) -> f32 {
        let mass = self.masses[handle.id as usize];
        if mass > 0.0 { 1.0 / mass } else { 0.0 }
    }

    pub fn set_collider(&mut self, handle: BodyHandle, index: usize, collider: ColliderDesc) {
        self.validate(handle);
        assert!(
            index < MAX_COLLIDERS_PER_BODY,
            "collider index out of range"
        );
        self.collider_descs[handle.id as usize][index] = collider;
        let mass_properties = self.mass_properties_of(handle.id as usize);
        let mut record = RigidBodyRecord::zeroed();
        record.inverse_mass = self.states[handle.id as usize]
            .map(|state| state.inverse_mass)
            .unwrap_or(0.0);
        record.com = mass_properties.com;
        record.inverse_inertia_body = mass_properties.inverse_inertia;
        let record_collider = self.collider_record(&collider);
        self.schedule_patch_collider(
            handle,
            PATCH_COLLIDER,
            record,
            record_collider,
            index as u32,
        );
    }

    pub fn set_shape(&mut self, handle: BodyHandle, shape: Shape) {
        self.validate(handle);
        let collider = &self.collider_descs[handle.id as usize];
        let existing = collider[0];
        self.set_collider(
            handle,
            0,
            ColliderDesc {
                shape,
                offset: existing.offset,
                rotation: existing.rotation,
                friction: existing.friction,
                restitution: existing.restitution,
                sensor: existing.sensor,
            },
        );
    }

    pub fn set_restitution(&mut self, handle: BodyHandle, restitution: f32) {
        self.validate(handle);
        let mut record = RigidBodyRecord::zeroed();
        record.restitution = restitution;
        self.schedule_patch(handle, PATCH_RESTITUTION, record);
    }

    pub fn set_friction(&mut self, handle: BodyHandle, friction: f32) {
        assert!(friction >= 0.0, "friction must be non-negative");
        self.validate(handle);
        let mut record = RigidBodyRecord::zeroed();
        record.friction = friction;
        self.schedule_patch(handle, PATCH_FRICTION, record);
    }

    pub fn set_collision_group(&mut self, handle: BodyHandle, group: u32) {
        self.validate(handle);
        let mut record = RigidBodyRecord::zeroed();
        record.collision_group = group;
        self.schedule_patch(handle, PATCH_GROUP, record);
    }

    pub fn set_collision_mask(&mut self, handle: BodyHandle, mask: u32) {
        self.validate(handle);
        let mut record = RigidBodyRecord::zeroed();
        record.collision_mask = mask;
        self.schedule_patch(handle, PATCH_MASK, record);
    }

    pub fn set_kinematic(&mut self, handle: BodyHandle, kinematic: bool) {
        self.validate(handle);
        let mass = self.masses[handle.id as usize];
        let inverse_mass = if kinematic || mass <= 0.0 {
            0.0
        } else {
            1.0 / mass
        };
        let mass_properties = if kinematic {
            MassProperties::zeroed()
        } else {
            self.mass_properties_of(handle.id as usize)
        };
        let mut record = RigidBodyRecord::zeroed();
        record.flags = if kinematic { BODY_KINEMATIC } else { 0 };
        record.inverse_mass = inverse_mass;
        record.com = mass_properties.com;
        record.inverse_inertia_body = mass_properties.inverse_inertia;
        self.schedule_patch(handle, PATCH_KINEMATIC, record);
    }

    pub fn set_ccd(&mut self, handle: BodyHandle, ccd: bool) {
        self.validate(handle);
        let mut record = RigidBodyRecord::zeroed();
        record.flags = if ccd { BODY_CCD } else { 0 };
        self.schedule_patch(handle, PATCH_CCD, record);
    }

    pub fn apply_force(&mut self, handle: BodyHandle, force: [f32; 3]) {
        let slot = self.command_slot(handle);
        self.commands.push(BodyCommandRecord::force(slot, force));
    }

    pub fn apply_force_at_point(&mut self, handle: BodyHandle, force: [f32; 3], point: [f32; 3]) {
        let slot = self.command_slot(handle);
        self.commands
            .push(BodyCommandRecord::force_at_point(slot, force, point));
    }

    pub fn apply_torque(&mut self, handle: BodyHandle, torque: [f32; 3]) {
        let slot = self.command_slot(handle);
        self.commands.push(BodyCommandRecord::torque(slot, torque));
    }

    pub fn apply_impulse(&mut self, handle: BodyHandle, impulse: [f32; 3]) {
        let slot = self.command_slot(handle);
        self.commands
            .push(BodyCommandRecord::impulse(slot, impulse));
    }

    pub fn apply_impulse_at_point(
        &mut self,
        handle: BodyHandle,
        impulse: [f32; 3],
        point: [f32; 3],
    ) {
        let slot = self.command_slot(handle);
        self.commands
            .push(BodyCommandRecord::impulse_at_point(slot, impulse, point));
    }

    pub fn apply_angular_impulse(&mut self, handle: BodyHandle, impulse: [f32; 3]) {
        let slot = self.command_slot(handle);
        self.commands
            .push(BodyCommandRecord::angular_impulse(slot, impulse));
    }

    pub fn wake(&mut self, handle: BodyHandle) {
        let slot = self.command_slot(handle);
        self.commands.push(BodyCommandRecord::wake(slot));
    }

    pub fn sleep(&mut self, handle: BodyHandle) {
        let slot = self.command_slot(handle);
        self.commands.push(BodyCommandRecord::sleep(slot));
    }

    pub fn read_state(&self, handle: BodyHandle) -> BodyState {
        self.validate(handle);
        self.states[handle.id as usize].expect("body state is unavailable")
    }

    fn command_slot(&self, handle: BodyHandle) -> u32 {
        self.validate(handle);
        self.index_of[handle.id as usize]
    }

    fn schedule_patch(&mut self, handle: BodyHandle, mask: u32, record: RigidBodyRecord) {
        let slot = self.command_slot(handle);
        self.commands.push(BodyCommandRecord::patch(
            slot,
            mask,
            record,
            [ColliderRecord::zeroed(); 4],
            0,
        ));
    }

    fn schedule_patch_collider(
        &mut self,
        handle: BodyHandle,
        mask: u32,
        record: RigidBodyRecord,
        collider: ColliderRecord,
        index: u32,
    ) {
        let slot = self.command_slot(handle);
        self.commands.push(BodyCommandRecord::patch(
            slot,
            mask,
            record,
            [
                collider,
                ColliderRecord::zeroed(),
                ColliderRecord::zeroed(),
                ColliderRecord::zeroed(),
            ],
            index,
        ));
    }

    pub(super) fn validate(&self, handle: BodyHandle) {
        let id = handle.id as usize;
        if id >= self.capacity {
            panic!("body handle {handle:?} is out of range");
        }
        if self.generations[id] != handle.generation {
            panic!("body handle {handle:?} is stale");
        }
        if self.index_of[id] == u32::MAX {
            panic!("body handle {handle:?} is not alive");
        }
    }

    pub(super) fn assert_unit(&self, orientation: [f32; 4]) {
        assert!(
            (orientation[0] * orientation[0]
                + orientation[1] * orientation[1]
                + orientation[2] * orientation[2]
                + orientation[3] * orientation[3]
                - 1.0)
                .abs()
                < 1e-4,
            "orientation must be a unit quaternion"
        );
    }
}
