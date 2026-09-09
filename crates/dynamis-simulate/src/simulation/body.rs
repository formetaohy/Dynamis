use super::Simulation;
use crate::static_aabb;
use bytemuck::Zeroable;
use dynamis_layout::{
    AabbRecord, BODY_CCD, BODY_KINEMATIC, BodyCommandRecord, BodyDescriptorRecord, BodyStateRecord,
    ColliderRecord, OVERRIDE_SLEEP_ANGULAR, OVERRIDE_SLEEP_LINEAR, PATCH_ANGULAR_VELOCITY,
    PATCH_ORIENTATION, PATCH_POSITION, PATCH_VELOCITY,
};
use dynamis_model::{
    BodyDesc, BodyHandle, BodyState, ColliderDesc, ContactEventMode, MAX_COLLIDERS_PER_BODY, Shape,
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
        let id = id as usize;
        self.validate_world_geometry(&desc);
        let mass = desc.effective_mass(|shape| self.shape_bounds(shape));
        let mut spawn_desc = desc.clone();
        spawn_desc.mass = mass;
        self.masses[id] = mass;
        self.com_overrides[id] = desc.com;
        self.inertia_overrides[id] = desc.inertia;
        self.kinematic[id] = desc.kinematic;
        self.collider_descs[id] = desc.colliders.clone();
        for collider in &desc.colliders {
            self.retain_shape_ref(&collider.shape);
        }
        let mass_properties = self.mass_properties_of(id);
        let descriptor = BodyDescriptorRecord::build(&spawn_desc, mass_properties, &self.config);
        self.descriptors[id] = descriptor;
        self.record_state(id, &spawn_desc);
        let state = BodyStateRecord::initial(&spawn_desc, id as u32, handle.generation);
        let slot = self.alive.len() as u32;
        self.index_of[id] = slot;
        self.alive.push(handle);
        self.dirty_bodies.push(slot);
        let _ = descriptor;
        self.commands.push(BodyCommandRecord::add(slot, state));
        if mass > 0.0 || desc.kinematic {
            if (self.dynamic_count as u32) < slot {
                self.swap_slots(self.dynamic_count as u32, slot);
            }
            self.dynamic_count += 1;
        }
        handle
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
        let id = handle.id as usize;
        self.index_of[id] = u32::MAX;
        self.free_ids.push(handle.id);
        self.alive.pop();
        self.dirty_bodies.retain(|dirty| *dirty != last);
        let removed = std::mem::take(&mut self.collider_descs[id]);
        for collider in &removed {
            self.release_shape_ref(&collider.shape);
        }
        self.states[id] = None;
        self.kinematic[id] = false;
        self.descriptors[id] = BodyDescriptorRecord::zeroed();
        self.commands.push(BodyCommandRecord::remove(last, last));
    }

    fn swap_slots(&mut self, first: u32, second: u32) {
        self.alive.swap(first as usize, second as usize);
        self.index_of[self.alive[first as usize].id as usize] = first;
        self.index_of[self.alive[second as usize].id as usize] = second;
        self.remap_constraint_slots(first, second);
        self.commands.push(BodyCommandRecord::swap(first, second));
        self.dirty_bodies.push(first);
        self.dirty_bodies.push(second);
    }

    fn migrate_partition(&mut self, handle: BodyHandle) {
        let id = handle.id as usize;
        let static_now = self.is_static_id(id);
        let slot = self.index_of[id];
        let in_dynamic = (slot as usize) < self.dynamic_count;
        if static_now == !in_dynamic {
            return;
        }
        if static_now {
            let tail = (self.dynamic_count - 1) as u32;
            if slot != tail {
                self.swap_slots(slot, tail);
            }
            self.dynamic_count -= 1;
            self.commands.push(BodyCommandRecord::patch(
                tail,
                PATCH_VELOCITY,
                BodyStateRecord::zeroed(),
            ));
            if let Some(state) = self.states[id].as_mut() {
                state.velocity = [0.0; 3];
                state.angular_velocity = [0.0; 3];
            }
            self.dirty_bodies.push(tail);
        } else {
            let boundary = self.dynamic_count as u32;
            if slot != boundary {
                self.swap_slots(slot, boundary);
            }
            self.dynamic_count += 1;
        }
        self.dirty_bodies.push(slot);
    }

    pub fn read_state(&self, handle: BodyHandle) -> BodyState {
        self.validate(handle);
        assert!(
            self.states_synchronized,
            "body states require synchronize_states() after stepping"
        );
        self.states[handle.id as usize].expect("body state is unavailable")
    }

    fn record_state(&mut self, id: usize, desc: &BodyDesc) {
        self.states[id] = Some(BodyState {
            position: desc.position,
            prev_position: desc.position,
            orientation: desc.orientation,
            velocity: desc.velocity,
            angular_velocity: desc.angular_velocity,
            inverse_mass: self.descriptors[id].inverse_mass,
            com: self.descriptors[id].com,
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

    fn is_static_id(&self, id: usize) -> bool {
        self.masses[id] <= 0.0 && !self.kinematic[id]
    }

    fn observed_pose(&self, id: usize) -> ([f32; 3], [f32; 4]) {
        match self.states[id] {
            Some(state) => (state.position, state.orientation),
            None => panic!("static body has no observed pose"),
        }
    }

    /// The collider boxes of a slot as the host authority sees them; dynamic slots
    /// leave their boxes for the device to compute.
    pub(super) fn aabb_block_of(&self, id: usize) -> [AabbRecord; MAX_COLLIDERS_PER_BODY] {
        if self.is_static_id(id) {
            let (position, orientation) = self.observed_pose(id);
            static_aabb::static_aabbs(
                position,
                orientation,
                &self.collider_block_of(id),
                &self.shape_pool,
            )
        } else {
            [AabbRecord::empty(); MAX_COLLIDERS_PER_BODY]
        }
    }

    /// Applies a host-side edit to one descriptor row and publishes it.
    fn patch_descriptor(
        &mut self,
        handle: BodyHandle,
        edit: impl FnOnce(&mut BodyDescriptorRecord),
    ) {
        self.validate(handle);
        let id = handle.id as usize;
        edit(&mut self.descriptors[id]);
        self.dirty_bodies.push(self.index_of[id]);
    }

    /// Republishes the mass-derived half of a descriptor after the colliders,
    /// mass or kinematic flag changed; the rest is already in the stored row.
    fn refresh_descriptor(&mut self, handle: BodyHandle) {
        let id = handle.id as usize;
        let mass = self.mass_properties_of(id);
        let kinematic = self.kinematic[id];
        let inverse_mass = if kinematic || self.masses[id] <= 0.0 {
            0.0
        } else {
            1.0 / self.masses[id]
        };
        let descriptor = &mut self.descriptors[id];
        descriptor.com = mass.com;
        descriptor.inverse_inertia = mass.inverse_inertia;
        descriptor.inverse_mass = inverse_mass;
        descriptor.flags =
            (descriptor.flags & !BODY_KINEMATIC) | if kinematic { BODY_KINEMATIC } else { 0 };
        let row = *descriptor;
        self.dirty_bodies.push(self.index_of[id]);
        if let Some(state) = self.states[id].as_mut() {
            state.inverse_mass = row.inverse_mass;
            state.com = row.com;
        }
    }

    fn apply_mass(&mut self, handle: BodyHandle, mass: f32) {
        self.masses[handle.id as usize] = mass;
        self.refresh_descriptor(handle);
    }

    fn command_slot(&self, handle: BodyHandle) -> u32 {
        self.validate(handle);
        self.index_of[handle.id as usize]
    }

    fn schedule_patch(&mut self, handle: BodyHandle, mask: u32, state: BodyStateRecord) {
        let slot = self.command_slot(handle);
        self.commands
            .push(BodyCommandRecord::patch(slot, mask, state));
    }

    pub fn set_position(&mut self, handle: BodyHandle, position: [f32; 3]) {
        self.validate(handle);
        if let Some(state) = self.states[handle.id as usize].as_mut() {
            state.position = position;
            state.prev_position = position;
        }
        let mut payload = BodyStateRecord::zeroed();
        payload.position = position;
        self.schedule_patch(handle, PATCH_POSITION, payload);
        self.dirty_bodies.push(self.index_of[handle.id as usize]);
    }

    pub fn set_orientation(&mut self, handle: BodyHandle, orientation: [f32; 4]) {
        self.assert_unit(orientation);
        self.validate(handle);
        if let Some(state) = self.states[handle.id as usize].as_mut() {
            state.orientation = orientation;
        }
        let mut payload = BodyStateRecord::zeroed();
        payload.orientation = orientation;
        self.schedule_patch(handle, PATCH_ORIENTATION, payload);
        self.dirty_bodies.push(self.index_of[handle.id as usize]);
    }

    pub fn set_velocity(&mut self, handle: BodyHandle, velocity: [f32; 3]) {
        self.validate(handle);
        if let Some(state) = self.states[handle.id as usize].as_mut() {
            state.velocity = velocity;
        }
        let mut payload = BodyStateRecord::zeroed();
        payload.velocity = velocity;
        self.schedule_patch(handle, PATCH_VELOCITY, payload);
    }

    pub fn set_angular_velocity(&mut self, handle: BodyHandle, angular_velocity: [f32; 3]) {
        self.validate(handle);
        if let Some(state) = self.states[handle.id as usize].as_mut() {
            state.angular_velocity = angular_velocity;
        }
        let mut payload = BodyStateRecord::zeroed();
        payload.angular_velocity = angular_velocity;
        self.schedule_patch(handle, PATCH_ANGULAR_VELOCITY, payload);
    }

    pub fn set_mass(&mut self, handle: BodyHandle, mass: f32) {
        assert!(mass >= 0.0, "mass must be non-negative");
        self.validate(handle);
        self.apply_mass(handle, mass);
        self.migrate_partition(handle);
    }

    pub fn set_density(&mut self, handle: BodyHandle, density: f32) {
        assert!(density >= 0.0, "density must be non-negative");
        self.validate(handle);
        let volume =
            dynamis_model::solid_volume_of(&self.collider_descs[handle.id as usize], &|shape| {
                self.shape_bounds(shape)
            });
        self.apply_mass(handle, density * volume);
        self.migrate_partition(handle);
    }

    pub fn set_com(&mut self, handle: BodyHandle, com: [f32; 3]) {
        self.validate(handle);
        self.com_overrides[handle.id as usize] = Some(com);
        self.refresh_descriptor(handle);
    }

    pub fn set_inertia(&mut self, handle: BodyHandle, inertia: [f32; 6]) {
        assert!(
            inertia.iter().all(|value| value.is_finite()),
            "inertia tensor must be finite"
        );
        self.validate(handle);
        self.inertia_overrides[handle.id as usize] = Some(inertia);
        self.refresh_descriptor(handle);
    }

    pub fn set_linear_damping(&mut self, handle: BodyHandle, damping: f32) {
        assert!(damping >= 0.0, "damping must be non-negative");
        self.patch_descriptor(handle, |row| row.linear_damping = damping);
    }

    pub fn set_angular_damping(&mut self, handle: BodyHandle, damping: f32) {
        assert!(damping >= 0.0, "angular damping must be non-negative");
        self.patch_descriptor(handle, |row| row.angular_damping = damping);
    }

    pub fn set_gravity_scale(&mut self, handle: BodyHandle, gravity_scale: f32) {
        self.patch_descriptor(handle, |row| row.gravity_scale = gravity_scale);
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
        self.patch_descriptor(handle, |row| {
            row.sleep_velocity = velocity;
            row.sleep_angular_velocity = angular_velocity;
            row.flags |= OVERRIDE_SLEEP_LINEAR | OVERRIDE_SLEEP_ANGULAR;
        });
    }

    pub fn set_collision_group(&mut self, handle: BodyHandle, group: u32) {
        self.patch_descriptor(handle, |row| row.collision_group = group);
    }

    pub fn set_collision_mask(&mut self, handle: BodyHandle, mask: u32) {
        self.patch_descriptor(handle, |row| row.collision_mask = mask);
    }

    pub fn set_ccd(&mut self, handle: BodyHandle, ccd: bool) {
        self.patch_descriptor(handle, |row| {
            row.flags = (row.flags & !BODY_CCD) | if ccd { BODY_CCD } else { 0 };
        });
    }

    pub fn set_kinematic(&mut self, handle: BodyHandle, kinematic: bool) {
        self.validate(handle);
        self.kinematic[handle.id as usize] = kinematic;
        self.refresh_descriptor(handle);
        self.migrate_partition(handle);
    }

    pub fn set_collider(&mut self, handle: BodyHandle, index: usize, collider: ColliderDesc) {
        self.validate(handle);
        let id = handle.id as usize;
        assert!(
            index < self.collider_descs[id].len(),
            "collider index out of range"
        );
        let existing = std::mem::replace(&mut self.collider_descs[id][index], collider);
        self.release_shape_ref(&existing.shape);
        self.retain_shape_ref(&collider.shape);
        self.refresh_descriptor(handle);
    }

    pub fn add_collider(&mut self, handle: BodyHandle, collider: ColliderDesc) {
        self.validate(handle);
        let id = handle.id as usize;
        let count = self.collider_descs[id].len();
        assert!(
            count < MAX_COLLIDERS_PER_BODY,
            "collider capacity per body reached"
        );
        self.retain_shape_ref(&collider.shape);
        self.collider_descs[id].push(collider);
        self.refresh_descriptor(handle);
    }

    pub fn remove_collider(&mut self, handle: BodyHandle, index: usize) {
        self.validate(handle);
        let id = handle.id as usize;
        assert!(
            index < self.collider_descs[id].len(),
            "collider index out of range"
        );
        assert!(
            self.collider_descs[id].len() > 1,
            "a body requires at least one collider"
        );
        let removed = self.collider_descs[id].swap_remove(index);
        self.release_shape_ref(&removed.shape);
        self.refresh_descriptor(handle);
    }

    pub fn set_shape(&mut self, handle: BodyHandle, shape: Shape) {
        self.validate(handle);
        let existing = self.collider_descs[handle.id as usize][0];
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
                events: existing.events,
            },
        );
    }

    pub fn set_restitution(&mut self, handle: BodyHandle, restitution: f32) {
        self.validate(handle);
        self.collider_descs[handle.id as usize][0].restitution = restitution;
        self.dirty_bodies.push(self.index_of[handle.id as usize]);
    }

    pub fn set_friction(&mut self, handle: BodyHandle, friction: f32) {
        assert!(friction >= 0.0, "friction must be non-negative");
        self.validate(handle);
        self.collider_descs[handle.id as usize][0].friction = friction;
        self.dirty_bodies.push(self.index_of[handle.id as usize]);
    }

    pub fn set_collider_events(
        &mut self,
        handle: BodyHandle,
        index: usize,
        events: ContactEventMode,
    ) {
        self.validate(handle);
        let id = handle.id as usize;
        assert!(
            index < self.collider_descs[id].len(),
            "collider index out of range"
        );
        self.collider_descs[id][index].events = events;
        self.dirty_bodies.push(self.index_of[id]);
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

    pub(crate) fn collider_block_of(&self, id: usize) -> [ColliderRecord; MAX_COLLIDERS_PER_BODY] {
        std::array::from_fn(|index| {
            self.collider_descs[id]
                .get(index)
                .map_or(ColliderRecord::zeroed(), |collider| {
                    self.collider_record(collider)
                })
        })
    }

    pub(super) fn validate(&self, handle: BodyHandle) {
        let id = handle.id as usize;
        if id >= self.slots {
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

    fn retain_shape_ref(&mut self, shape: &Shape) {
        if let Shape::Hull(handle) | Shape::Mesh(handle) | Shape::HeightField(handle) = shape {
            self.shape_pool.retain(*handle);
        }
    }

    fn release_shape_ref(&mut self, shape: &Shape) {
        if let Shape::Hull(handle) | Shape::Mesh(handle) | Shape::HeightField(handle) = shape {
            self.shape_pool.release(*handle);
        }
    }
}
