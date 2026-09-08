use super::Simulation;
use crate::static_aabb;
use bytemuck::Zeroable;
use dynamis_layout::{
    AabbRecord, BODY_CCD, BODY_KINEMATIC, BodyCommandRecord, ColliderRecord,
    PATCH_ANGULAR_VELOCITY, PATCH_CCD, PATCH_COLLIDER, PATCH_DYNAMICS, PATCH_GROUP,
    PATCH_KINEMATIC, PATCH_MASK, PATCH_MASS, PATCH_ORIENTATION, PATCH_POSITION, PATCH_VELOCITY,
    RigidBodyRecord,
};
use dynamis_model::{
    BodyDesc, BodyHandle, BodyState, ColliderDesc, MAX_COLLIDERS_PER_BODY, MassProperties, Shape,
};

impl Simulation {
    fn is_dynamic_desc(desc: &BodyDesc) -> bool {
        !(desc.mass <= 0.0 && !desc.kinematic)
    }

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
        self.validate_world_geometry(&desc);
        let effective_mass = desc.effective_mass(|shape| self.shape_bounds(shape));
        let mut spawn_desc = desc.clone();
        spawn_desc.mass = effective_mass;
        self.masses.resize(self.alive.len() + 1, 1.0);
        self.masses[id as usize] = effective_mass;
        self.com_overrides.resize(self.alive.len() + 1, None);
        self.com_overrides[id as usize] = desc.com;
        self.inertia_overrides.resize(self.alive.len() + 1, None);
        self.inertia_overrides[id as usize] = desc.inertia;
        self.dynamics
            .resize(self.alive.len() + 1, super::BodyDynamics::defaults());
        self.dynamics[id as usize] = super::BodyDynamics::from_desc(&desc);
        self.kinematic.resize(self.alive.len() + 1, false);
        self.kinematic[id as usize] = desc.kinematic;
        self.collider_descs.resize(self.alive.len() + 1, Vec::new());
        self.collider_descs[id as usize] = desc.colliders.clone();
        let mass = self.mass_properties_of(id as usize);
        let body = RigidBodyRecord::build(
            &spawn_desc,
            handle.id,
            handle.generation,
            mass,
            &self.config,
        );
        let colliders = self.collider_block(&desc);
        self.record_state(id as usize, &desc, &body);
        let slot = self.alive.len() as u32;
        self.index_of[id as usize] = slot;
        self.alive.push(handle);
        self.commands
            .push(BodyCommandRecord::add(slot, body, colliders));
        if Self::is_dynamic_desc(&desc) {
            if (self.dynamic_count as u32) < slot {
                self.swap_slots(self.dynamic_count as u32, slot);
            }
            self.dynamic_count += 1;
        } else {
            self.sync_slot_aabbs(slot);
        }
        handle
    }

    fn collider_block(&self, desc: &BodyDesc) -> [ColliderRecord; 4] {
        let mut colliders = [ColliderRecord::zeroed(); 4];
        for (index, collider) in desc.colliders.iter().enumerate() {
            colliders[index] = self.collider_record(collider);
        }
        colliders
    }

    pub(crate) fn collider_block_of(&self, id: usize) -> [ColliderRecord; 4] {
        let mut colliders = [ColliderRecord::zeroed(); 4];
        for (index, collider) in self.collider_descs[id].iter().enumerate() {
            colliders[index] = self.collider_record(collider);
        }
        colliders
    }

    fn refresh_static_aabbs(&self, handle: BodyHandle) {
        self.sync_slot_aabbs(self.index_of[handle.id as usize])
    }

    fn sync_slot_aabbs(&self, slot: u32) {
        let handle = self.alive[slot as usize];
        let id = handle.id as usize;
        let aabbs = if self.is_static_id(id) {
            let desc = self.build_desc(id);
            let body = RigidBodyRecord::build(
                &desc,
                handle.id,
                handle.generation,
                self.mass_properties_of(id),
                &self.config,
            );
            let body = self.override_state(&body, id);
            static_aabb::static_aabbs(&body, &self.collider_block_of(id), &self.shape_pool)
        } else {
            [AabbRecord::empty(); 4]
        };
        let offset = (slot as usize) * MAX_COLLIDERS_PER_BODY * std::mem::size_of::<AabbRecord>();
        self.buffers.aabbs.write_at(
            self.gpu.queue(),
            offset as u64,
            bytemuck::cast_slice(&aabbs),
        );
    }

    fn override_state(&self, body: &RigidBodyRecord, id: usize) -> RigidBodyRecord {
        let mut body = *body;
        if let Some(state) = self.state_snapshot(id) {
            body.position = state.position;
            body.prev_position = state.previous_position;
            body.orientation = state.orientation;
        }
        body
    }

    fn is_static_id(&self, id: usize) -> bool {
        self.masses[id] <= 0.0 && !self.is_kinematic_id(id)
    }

    fn is_kinematic_id(&self, id: usize) -> bool {
        self.kinematic.get(id).copied().unwrap_or(false)
    }

    fn swap_slots(&mut self, first: u32, second: u32) {
        self.alive.swap(first as usize, second as usize);
        self.index_of[self.alive[first as usize].id as usize] = first;
        self.index_of[self.alive[second as usize].id as usize] = second;
        self.remap_constraint_slots(first, second);
        self.commands.push(BodyCommandRecord::swap(first, second));
        self.sync_slot_aabbs(first);
        self.sync_slot_aabbs(second);
    }

    pub fn remove(&mut self, handle: BodyHandle) {
        self.validate(handle);
        self.assert_no_constraints(handle);
        let id = handle.id as usize;
        let slot = self.index_of[id];
        if (slot as usize) < self.dynamic_count {
            let tail_dynamic = (self.dynamic_count - 1) as u32;
            if slot != tail_dynamic {
                self.swap_slots(slot, tail_dynamic);
            }
            self.dynamic_count -= 1;
            let last = (self.alive.len() - 1) as u32;
            if tail_dynamic != last {
                self.swap_slots(tail_dynamic, last);
            }
        } else {
            let last = (self.alive.len() - 1) as u32;
            if slot != last {
                self.swap_slots(slot, last);
            }
        }
        self.discard_tail();
    }

    fn discard_tail(&mut self) {
        let last = (self.alive.len() - 1) as u32;
        let handle = self.alive[last as usize];
        self.index_of[handle.id as usize] = u32::MAX;
        self.free_ids.push(handle.id);
        self.alive.pop();
        self.collider_descs[handle.id as usize] = Vec::new();
        self.states[handle.id as usize] = None;
        self.kinematic[handle.id as usize] = false;
        self.commands.push(BodyCommandRecord::remove(last, last));
    }

    fn record_state(&mut self, id: usize, desc: &BodyDesc, body: &RigidBodyRecord) {
        if self.states.len() <= id {
            self.states.resize(id + 1, None);
        }
        self.states[id] = Some(BodyState {
            position: desc.position,
            previous_position: desc.position,
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

    pub fn set_position(&mut self, handle: BodyHandle, position: [f32; 3]) {
        self.validate(handle);
        if let Some(state) = self.states[handle.id as usize].as_mut() {
            state.position = position;
        }
        let mut record = RigidBodyRecord::zeroed();
        record.position = position;
        self.schedule_patch(handle, PATCH_POSITION, record);
        self.refresh_static_aabbs(handle);
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
        self.refresh_static_aabbs(handle);
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
        self.migrate_partition(handle);
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

    pub fn set_linear_damping(&mut self, handle: BodyHandle, damping: f32) {
        assert!(damping >= 0.0, "damping must be non-negative");
        self.validate(handle);
        self.dynamics[handle.id as usize].linear_damping = Some(damping);
        let record = self.dynamics_record(handle);
        self.schedule_patch(handle, PATCH_DYNAMICS, record);
    }

    pub fn set_angular_damping(&mut self, handle: BodyHandle, damping: f32) {
        assert!(damping >= 0.0, "angular damping must be non-negative");
        self.validate(handle);
        self.dynamics[handle.id as usize].angular_damping = Some(damping);
        let record = self.dynamics_record(handle);
        self.schedule_patch(handle, PATCH_DYNAMICS, record);
    }

    pub fn set_gravity_scale(&mut self, handle: BodyHandle, gravity_scale: f32) {
        self.validate(handle);
        self.dynamics[handle.id as usize].gravity_scale = gravity_scale;
        let record = self.dynamics_record(handle);
        self.schedule_patch(handle, PATCH_DYNAMICS, record);
    }

    pub fn set_sleep_thresholds(
        &mut self,
        handle: BodyHandle,
        velocity: f32,
        angular_velocity: f32,
    ) {
        assert!(velocity >= 0.0, "sleep velocity must be non-negative");
        assert!(
            angular_velocity >= 0.0,
            "sleep angular velocity must be non-negative"
        );
        self.validate(handle);
        let dynamics = &mut self.dynamics[handle.id as usize];
        dynamics.sleep_velocity = Some(velocity);
        dynamics.sleep_angular_velocity = Some(angular_velocity);
        let record = self.dynamics_record(handle);
        self.schedule_patch(handle, PATCH_DYNAMICS, record);
    }

    pub fn set_density(&mut self, handle: BodyHandle, density: f32) {
        assert!(density >= 0.0, "density must be non-negative");
        self.validate(handle);
        let desc = &self.build_desc(handle.id as usize);
        let mass = density
            * dynamis_model::solid_volume_of(&desc.colliders, &|shape| self.shape_bounds(shape));
        self.masses[handle.id as usize] = mass;
        let mass_properties = self.mass_properties_of(handle.id as usize);
        let mut record = RigidBodyRecord::zeroed();
        record.inverse_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        record.com = mass_properties.com;
        record.inverse_inertia_body = mass_properties.inverse_inertia;
        if let Some(state) = self.states[handle.id as usize].as_mut() {
            state.inverse_mass = record.inverse_mass;
            state.com = mass_properties.com;
        }
        self.schedule_patch(handle, PATCH_MASS, record);
        self.migrate_partition(handle);
    }

    fn dynamics_record(&self, handle: BodyHandle) -> RigidBodyRecord {
        let dynamics = self.dynamics[handle.id as usize];
        let mut record = RigidBodyRecord::zeroed();
        record.linear_damping = dynamics.linear_damping.unwrap_or(self.config.damping);
        record.angular_damping = dynamics
            .angular_damping
            .unwrap_or(self.config.angular_damping);
        record.gravity_scale = dynamics.gravity_scale;
        record.sleep_velocity_override = dynamics.sleep_velocity.unwrap_or(-1.0);
        record.sleep_angular_velocity_override = dynamics.sleep_angular_velocity.unwrap_or(-1.0);
        record
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
        let (record, record_collider) = self.collider_patch(handle, index);
        self.schedule_patch_collider(
            handle,
            PATCH_COLLIDER,
            record,
            record_collider,
            index as u32,
        );
        self.refresh_static_aabbs(handle);
    }

    fn collider_patch(
        &self,
        handle: BodyHandle,
        index: usize,
    ) -> (RigidBodyRecord, ColliderRecord) {
        let mass_properties = self.mass_properties_of(handle.id as usize);
        let mut record = RigidBodyRecord::zeroed();
        record.inverse_mass = self.states[handle.id as usize]
            .map(|state| state.inverse_mass)
            .unwrap_or(0.0);
        record.com = mass_properties.com;
        record.inverse_inertia_body = mass_properties.inverse_inertia;
        let record_collider = self.collider_record(&self.collider_descs[handle.id as usize][index]);
        (record, record_collider)
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
                scale: existing.scale,
                sensor: existing.sensor,
                collision_group: existing.collision_group,
                collision_mask: existing.collision_mask,
                rolling_friction: existing.rolling_friction,
                spin_friction: existing.spin_friction,
            },
        );
    }

    pub fn set_restitution(&mut self, handle: BodyHandle, restitution: f32) {
        self.validate(handle);
        self.collider_descs[handle.id as usize][0].restitution = restitution;
        let (record, record_collider) = self.collider_patch(handle, 0);
        self.schedule_patch_collider(handle, PATCH_COLLIDER, record, record_collider, 0);
    }

    pub fn set_friction(&mut self, handle: BodyHandle, friction: f32) {
        assert!(friction >= 0.0, "friction must be non-negative");
        self.validate(handle);
        self.collider_descs[handle.id as usize][0].friction = friction;
        let (record, record_collider) = self.collider_patch(handle, 0);
        self.schedule_patch_collider(handle, PATCH_COLLIDER, record, record_collider, 0);
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
        self.kinematic[handle.id as usize] = kinematic;
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
        self.migrate_partition(handle);
    }

    fn migrate_partition(&mut self, handle: BodyHandle) {
        let id = handle.id as usize;
        let static_now = self.is_static_id(id);
        let slot = self.index_of[id];
        let in_dynamic = (slot as usize) < self.dynamic_count;
        if static_now == !in_dynamic {
            if static_now {
                self.refresh_static_aabbs(handle);
            }
            return;
        }
        if static_now {
            let tail = (self.dynamic_count - 1) as u32;
            if slot != tail {
                self.swap_slots(slot, tail);
            }
            self.dynamic_count -= 1;
            let mut record = RigidBodyRecord::zeroed();
            record.velocity = [0.0; 3];
            record.angular_velocity = [0.0; 3];
            self.commands.push(BodyCommandRecord::patch(
                tail,
                PATCH_VELOCITY | PATCH_ANGULAR_VELOCITY,
                record,
                [ColliderRecord::zeroed(); 4],
                0,
            ));
            if let Some(state) = self.states[id].as_mut() {
                state.velocity = [0.0; 3];
                state.angular_velocity = [0.0; 3];
            }
            self.sync_slot_aabbs(tail);
        } else {
            let boundary = self.dynamic_count as u32;
            if slot != boundary {
                self.swap_slots(slot, boundary);
            }
            self.dynamic_count += 1;
        }
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
