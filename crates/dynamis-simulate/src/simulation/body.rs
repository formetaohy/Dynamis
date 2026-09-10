use super::Simulation;
use super::commands::BodyCommand;
use crate::static_aabb;
use bytemuck::Zeroable;
use dynamis_layout::{
    AabbRecord, BODY_CCD, BODY_KINEMATIC, BodyDescriptorRecord, BodyStateRecord, ColliderRecord,
    OVERRIDE_SLEEP_ANGULAR, OVERRIDE_SLEEP_LINEAR, PATCH_ANGULAR_VELOCITY, PATCH_ORIENTATION,
    PATCH_POSITION, PATCH_VELOCITY,
};
use dynamis_model::{
    BodyDesc, BodyHandle, BodyState, ColliderDesc, ContactEventMode, MAX_COLLIDERS_PER_BODY,
    MassProperties, Shape,
};

pub(crate) struct Bodies {
    pub(crate) alive: Vec<BodyHandle>,
    pub(crate) index_of: Vec<u32>,
    pub(crate) generations: Vec<u32>,
    pub(crate) free_ids: Vec<u32>,
    pub(crate) collider_descs: Vec<Vec<ColliderDesc>>,
    pub(crate) masses: Vec<f32>,
    pub(crate) com_overrides: Vec<Option<[f32; 3]>>,
    pub(crate) inertia_overrides: Vec<Option<[f32; 6]>>,
    pub(crate) descriptors: Vec<BodyDescriptorRecord>,
    pub(crate) dynamic_count: usize,
    pub(crate) kinematic: Vec<bool>,
    pub(crate) states: Vec<Option<BodyState>>,
    pub(crate) states_ready: bool,
    pub(crate) device_count: u32,
    pub(crate) commands: Vec<BodyCommand>,
    pub(crate) dirty: Vec<u32>,
    pub(crate) last_moves: u32,
    pub(crate) last_edits: u32,
}

impl Bodies {
    pub(crate) fn new(slots: usize) -> Self {
        Self {
            alive: Vec::new(),
            index_of: vec![u32::MAX; slots],
            generations: vec![1; slots],
            free_ids: (0..slots as u32).rev().collect(),
            collider_descs: vec![Vec::new(); slots],
            masses: vec![1.0; slots],
            com_overrides: vec![None; slots],
            inertia_overrides: vec![None; slots],
            descriptors: vec![BodyDescriptorRecord::zeroed(); slots],
            dynamic_count: 0,
            kinematic: vec![false; slots],
            states: vec![None; slots],
            states_ready: true,
            device_count: 0,
            commands: Vec::new(),
            dirty: Vec::new(),
            last_moves: 0,
            last_edits: 0,
        }
    }

    pub(crate) fn resize(&mut self, slots: usize) {
        self.index_of.resize(slots, u32::MAX);
        self.generations.resize(slots, 1);
        self.collider_descs.resize(slots, Vec::new());
        self.masses.resize(slots, 1.0);
        self.com_overrides.resize(slots, None);
        self.inertia_overrides.resize(slots, None);
        self.descriptors
            .resize(slots, BodyDescriptorRecord::zeroed());
        self.states.resize(slots, None);
        self.kinematic.resize(slots, false);
    }
}

impl Simulation {
    pub fn spawn(&mut self, desc: BodyDesc) -> BodyHandle {
        let id = self
            .bodies
            .free_ids
            .pop()
            .expect("simulation body capacity exhausted");
        self.bodies.generations[id as usize] += 1;
        let handle = BodyHandle {
            id,
            generation: self.bodies.generations[id as usize],
        };
        let id = id as usize;
        self.validate_world_geometry(&desc);
        let mass = desc.effective_mass(|shape| self.shape_bounds(shape));
        let mut spawn_desc = desc.clone();
        spawn_desc.mass = mass;
        self.bodies.masses[id] = mass;
        self.bodies.com_overrides[id] = desc.com;
        self.bodies.inertia_overrides[id] = desc.inertia;
        self.bodies.kinematic[id] = desc.kinematic;
        self.bodies.collider_descs[id] = desc.colliders.clone();
        for collider in &desc.colliders {
            self.retain_shape_ref(&collider.shape);
        }
        let mass_properties = self.mass_properties_of(id);
        self.bodies.descriptors[id] =
            BodyDescriptorRecord::build(&spawn_desc, mass_properties, &self.config);
        self.record_state(id, &spawn_desc);
        let state = BodyStateRecord::initial(&spawn_desc, id as u32, handle.generation);
        let slot = self.bodies.alive.len() as u32;
        self.bodies.index_of[id] = slot;
        self.bodies.alive.push(handle);
        self.bodies.dirty.push(slot);
        self.bodies
            .commands
            .push(BodyCommand::Add { row: slot, state });
        if mass > 0.0 || desc.kinematic {
            if (self.bodies.dynamic_count as u32) < slot {
                self.swap_slots(self.bodies.dynamic_count as u32, slot);
            }
            self.bodies.dynamic_count += 1;
        }
        handle
    }

    pub fn remove(&mut self, handle: BodyHandle) {
        self.validate(handle);
        self.assert_no_constraints(handle);
        let id = handle.id as usize;
        let slot = self.bodies.index_of[id];
        if (slot as usize) < self.bodies.dynamic_count {
            let tail_dynamic = (self.bodies.dynamic_count - 1) as u32;
            if slot != tail_dynamic {
                self.swap_slots(slot, tail_dynamic);
            }
            self.bodies.dynamic_count -= 1;
            let last = (self.bodies.alive.len() - 1) as u32;
            if tail_dynamic != last {
                self.swap_slots(tail_dynamic, last);
            }
        } else {
            let last = (self.bodies.alive.len() - 1) as u32;
            if slot != last {
                self.swap_slots(slot, last);
            }
        }
        self.discard_tail();
    }

    fn discard_tail(&mut self) {
        let last = (self.bodies.alive.len() - 1) as u32;
        let handle = self.bodies.alive[last as usize];
        let id = handle.id as usize;
        self.bodies.index_of[id] = u32::MAX;
        self.bodies.free_ids.push(handle.id);
        self.bodies.alive.pop();
        self.bodies.dirty.retain(|dirty| *dirty != last);
        let removed = std::mem::take(&mut self.bodies.collider_descs[id]);
        for collider in &removed {
            self.release_shape_ref(&collider.shape);
        }
        self.bodies.states[id] = None;
        self.bodies.kinematic[id] = false;
        self.bodies.descriptors[id] = BodyDescriptorRecord::zeroed();
        self.bodies.commands.push(BodyCommand::Remove {
            hole: last,
            tail: last,
        });
    }

    fn swap_slots(&mut self, first: u32, second: u32) {
        self.bodies.alive.swap(first as usize, second as usize);
        self.bodies.index_of[self.bodies.alive[first as usize].id as usize] = first;
        self.bodies.index_of[self.bodies.alive[second as usize].id as usize] = second;
        self.remap_constraint_slots(first, second);
        self.bodies
            .commands
            .push(BodyCommand::Swap { first, second });
        self.bodies.dirty.push(first);
        self.bodies.dirty.push(second);
    }

    fn migrate_partition(&mut self, handle: BodyHandle) {
        let id = handle.id as usize;
        let static_now = self.is_static_id(id);
        let slot = self.bodies.index_of[id];
        let in_dynamic = (slot as usize) < self.bodies.dynamic_count;
        if static_now == !in_dynamic {
            return;
        }
        if static_now {
            let tail = (self.bodies.dynamic_count - 1) as u32;
            if slot != tail {
                self.swap_slots(slot, tail);
            }
            self.bodies.dynamic_count -= 1;
            self.bodies.commands.push(BodyCommand::Patch {
                row: tail,
                mask: PATCH_VELOCITY,
                state: BodyStateRecord::zeroed(),
            });
            if let Some(state) = self.bodies.states[id].as_mut() {
                state.velocity = [0.0; 3];
                state.angular_velocity = [0.0; 3];
            }
            self.bodies.dirty.push(tail);
        } else {
            let boundary = self.bodies.dynamic_count as u32;
            if slot != boundary {
                self.swap_slots(slot, boundary);
            }
            self.bodies.dynamic_count += 1;
        }
        self.bodies.dirty.push(slot);
    }

    pub fn read_state(&self, handle: BodyHandle) -> BodyState {
        self.validate(handle);
        assert!(
            self.bodies.states_ready,
            "body states require synchronize_states() after stepping"
        );
        self.bodies.states[handle.id as usize].expect("body state is unavailable")
    }

    fn record_state(&mut self, id: usize, desc: &BodyDesc) {
        self.bodies.states[id] = Some(BodyState {
            position: desc.position,
            prev_position: desc.position,
            orientation: desc.orientation,
            velocity: desc.velocity,
            angular_velocity: desc.angular_velocity,
            inverse_mass: self.bodies.descriptors[id].inverse_mass,
            com: self.bodies.descriptors[id].com,
            sleeping: false,
            step: self.clock.step,
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
        self.bodies.masses[id] <= 0.0 && !self.bodies.kinematic[id]
    }

    fn observed_pose(&self, id: usize) -> ([f32; 3], [f32; 4]) {
        match self.bodies.states[id] {
            Some(state) => (state.position, state.orientation),
            None => panic!("static body has no observed pose"),
        }
    }

    pub(super) fn aabb_block_of(&self, id: usize) -> [AabbRecord; MAX_COLLIDERS_PER_BODY] {
        if self.is_static_id(id) {
            let (position, orientation) = self.observed_pose(id);
            static_aabb::static_aabbs(
                position,
                orientation,
                &self.collider_block_of(id),
                &self.shapes.pool,
            )
        } else {
            [AabbRecord::empty(); MAX_COLLIDERS_PER_BODY]
        }
    }

    fn patch_descriptor(
        &mut self,
        handle: BodyHandle,
        edit: impl FnOnce(&mut BodyDescriptorRecord),
    ) {
        self.validate(handle);
        let id = handle.id as usize;
        edit(&mut self.bodies.descriptors[id]);
        self.bodies.dirty.push(self.bodies.index_of[id]);
    }

    fn refresh_descriptor(&mut self, handle: BodyHandle) {
        let id = handle.id as usize;
        let mass = self.mass_properties_of(id);
        let kinematic = self.bodies.kinematic[id];
        let inverse_mass = if kinematic || self.bodies.masses[id] <= 0.0 {
            0.0
        } else {
            1.0 / self.bodies.masses[id]
        };
        let descriptor = &mut self.bodies.descriptors[id];
        descriptor.com = mass.com;
        descriptor.inverse_inertia = mass.inverse_inertia;
        descriptor.inverse_mass = inverse_mass;
        descriptor.flags =
            (descriptor.flags & !BODY_KINEMATIC) | if kinematic { BODY_KINEMATIC } else { 0 };
        let row = *descriptor;
        self.bodies.dirty.push(self.bodies.index_of[id]);
        if let Some(state) = self.bodies.states[id].as_mut() {
            state.inverse_mass = row.inverse_mass;
            state.com = row.com;
        }
    }

    fn apply_mass(&mut self, handle: BodyHandle, mass: f32) {
        self.bodies.masses[handle.id as usize] = mass;
        self.refresh_descriptor(handle);
    }

    fn command_slot(&self, handle: BodyHandle) -> u32 {
        self.validate(handle);
        self.bodies.index_of[handle.id as usize]
    }

    fn schedule_patch(&mut self, handle: BodyHandle, mask: u32, state: BodyStateRecord) {
        let slot = self.command_slot(handle);
        self.bodies.commands.push(BodyCommand::Patch {
            row: slot,
            mask,
            state,
        });
    }

    pub fn set_position(&mut self, handle: BodyHandle, position: [f32; 3]) {
        self.validate(handle);
        if let Some(state) = self.bodies.states[handle.id as usize].as_mut() {
            state.position = position;
            state.prev_position = position;
        }
        let mut payload = BodyStateRecord::zeroed();
        payload.position = position;
        self.schedule_patch(handle, PATCH_POSITION, payload);
        self.bodies
            .dirty
            .push(self.bodies.index_of[handle.id as usize]);
    }

    pub fn set_orientation(&mut self, handle: BodyHandle, orientation: [f32; 4]) {
        self.assert_unit(orientation);
        self.validate(handle);
        if let Some(state) = self.bodies.states[handle.id as usize].as_mut() {
            state.orientation = orientation;
        }
        let mut payload = BodyStateRecord::zeroed();
        payload.orientation = orientation;
        self.schedule_patch(handle, PATCH_ORIENTATION, payload);
        self.bodies
            .dirty
            .push(self.bodies.index_of[handle.id as usize]);
    }

    pub fn set_velocity(&mut self, handle: BodyHandle, velocity: [f32; 3]) {
        self.validate(handle);
        if let Some(state) = self.bodies.states[handle.id as usize].as_mut() {
            state.velocity = velocity;
        }
        let mut payload = BodyStateRecord::zeroed();
        payload.velocity = velocity;
        self.schedule_patch(handle, PATCH_VELOCITY, payload);
    }

    pub fn set_angular_velocity(&mut self, handle: BodyHandle, angular_velocity: [f32; 3]) {
        self.validate(handle);
        if let Some(state) = self.bodies.states[handle.id as usize].as_mut() {
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
        let volume = dynamis_model::solid_volume_of(
            &self.bodies.collider_descs[handle.id as usize],
            &|shape| self.shape_bounds(shape),
        );
        self.apply_mass(handle, density * volume);
        self.migrate_partition(handle);
    }

    pub fn set_com(&mut self, handle: BodyHandle, com: [f32; 3]) {
        self.validate(handle);
        self.bodies.com_overrides[handle.id as usize] = Some(com);
        self.refresh_descriptor(handle);
    }

    pub fn set_inertia(&mut self, handle: BodyHandle, inertia: [f32; 6]) {
        assert!(
            inertia.iter().all(|value| value.is_finite()),
            "inertia tensor must be finite"
        );
        self.validate(handle);
        self.bodies.inertia_overrides[handle.id as usize] = Some(inertia);
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
        self.bodies.kinematic[handle.id as usize] = kinematic;
        self.refresh_descriptor(handle);
        self.migrate_partition(handle);
    }

    pub fn set_collider(&mut self, handle: BodyHandle, index: usize, collider: ColliderDesc) {
        self.validate(handle);
        let id = handle.id as usize;
        assert!(
            index < self.bodies.collider_descs[id].len(),
            "collider index out of range"
        );
        let existing = std::mem::replace(&mut self.bodies.collider_descs[id][index], collider);
        self.release_shape_ref(&existing.shape);
        self.retain_shape_ref(&collider.shape);
        self.refresh_descriptor(handle);
    }

    pub fn add_collider(&mut self, handle: BodyHandle, collider: ColliderDesc) {
        self.validate(handle);
        let id = handle.id as usize;
        let count = self.bodies.collider_descs[id].len();
        assert!(
            count < MAX_COLLIDERS_PER_BODY,
            "collider capacity per body reached"
        );
        self.retain_shape_ref(&collider.shape);
        self.bodies.collider_descs[id].push(collider);
        self.refresh_descriptor(handle);
    }

    pub fn remove_collider(&mut self, handle: BodyHandle, index: usize) {
        self.validate(handle);
        let id = handle.id as usize;
        assert!(
            index < self.bodies.collider_descs[id].len(),
            "collider index out of range"
        );
        assert!(
            self.bodies.collider_descs[id].len() > 1,
            "a body requires at least one collider"
        );
        let removed = self.bodies.collider_descs[id].swap_remove(index);
        self.release_shape_ref(&removed.shape);
        self.refresh_descriptor(handle);
    }

    pub fn set_shape(&mut self, handle: BodyHandle, shape: Shape) {
        self.validate(handle);
        let id = handle.id as usize;
        let replaced = std::mem::replace(&mut self.bodies.collider_descs[id][0].shape, shape);
        self.release_shape_ref(&replaced);
        self.retain_shape_ref(&shape);
        self.refresh_descriptor(handle);
    }

    pub fn set_restitution(&mut self, handle: BodyHandle, restitution: f32) {
        self.validate(handle);
        self.bodies.collider_descs[handle.id as usize][0].restitution = restitution;
        self.bodies
            .dirty
            .push(self.bodies.index_of[handle.id as usize]);
    }

    pub fn set_friction(&mut self, handle: BodyHandle, friction: f32) {
        assert!(friction >= 0.0, "friction must be non-negative");
        self.validate(handle);
        self.bodies.collider_descs[handle.id as usize][0].friction = friction;
        self.bodies
            .dirty
            .push(self.bodies.index_of[handle.id as usize]);
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
            index < self.bodies.collider_descs[id].len(),
            "collider index out of range"
        );
        self.bodies.collider_descs[id][index].events = events;
        self.bodies.dirty.push(self.bodies.index_of[id]);
    }

    pub fn apply_force(&mut self, handle: BodyHandle, force: [f32; 3]) {
        let slot = self.command_slot(handle);
        self.bodies
            .commands
            .push(BodyCommand::Force { row: slot, force });
    }

    pub fn apply_force_at_point(&mut self, handle: BodyHandle, force: [f32; 3], point: [f32; 3]) {
        let slot = self.command_slot(handle);
        self.bodies.commands.push(BodyCommand::ForceAtPoint {
            row: slot,
            force,
            point,
        });
    }

    pub fn apply_torque(&mut self, handle: BodyHandle, torque: [f32; 3]) {
        let slot = self.command_slot(handle);
        self.bodies
            .commands
            .push(BodyCommand::Torque { row: slot, torque });
    }

    pub fn apply_impulse(&mut self, handle: BodyHandle, impulse: [f32; 3]) {
        let slot = self.command_slot(handle);
        self.bodies
            .commands
            .push(BodyCommand::Impulse { row: slot, impulse });
    }

    pub fn apply_impulse_at_point(
        &mut self,
        handle: BodyHandle,
        impulse: [f32; 3],
        point: [f32; 3],
    ) {
        let slot = self.command_slot(handle);
        self.bodies.commands.push(BodyCommand::ImpulseAtPoint {
            row: slot,
            impulse,
            point,
        });
    }

    pub fn apply_angular_impulse(&mut self, handle: BodyHandle, impulse: [f32; 3]) {
        let slot = self.command_slot(handle);
        self.bodies
            .commands
            .push(BodyCommand::AngularImpulse { row: slot, impulse });
    }

    pub fn wake(&mut self, handle: BodyHandle) {
        let slot = self.command_slot(handle);
        self.bodies.commands.push(BodyCommand::Wake { row: slot });
    }

    pub fn sleep(&mut self, handle: BodyHandle) {
        let slot = self.command_slot(handle);
        self.bodies.commands.push(BodyCommand::Sleep { row: slot });
    }

    pub(crate) fn collider_block_of(&self, id: usize) -> [ColliderRecord; MAX_COLLIDERS_PER_BODY] {
        std::array::from_fn(|index| {
            self.bodies.collider_descs[id]
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
        if self.bodies.generations[id] != handle.generation {
            panic!("body handle {handle:?} is stale");
        }
        if self.bodies.index_of[id] == u32::MAX {
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
            self.shapes.pool.retain(*handle);
        }
    }

    fn release_shape_ref(&mut self, shape: &Shape) {
        if let Shape::Hull(handle) | Shape::Mesh(handle) | Shape::HeightField(handle) = shape {
            self.shapes.pool.release(*handle);
        }
    }
}

impl Simulation {
    pub(crate) fn state_snapshot(&self, id: usize) -> Option<BodyState> {
        assert!(
            self.bodies.states_ready,
            "body states require synchronize_states() after stepping"
        );
        self.bodies.states[id]
    }

    pub(crate) fn mass_properties_of(&self, id: usize) -> MassProperties {
        dynamis_model::mass_properties_of_intent(
            &self.bodies.collider_descs[id],
            self.bodies.masses[id],
            self.bodies.com_overrides[id],
            self.bodies.inertia_overrides[id],
            |shape| self.shape_bounds(shape),
        )
    }

    pub(crate) fn collider_record(&self, desc: &ColliderDesc) -> ColliderRecord {
        let source = match desc.shape {
            Shape::Hull(handle) | Shape::Mesh(handle) | Shape::HeightField(handle) => handle.id,
            _ => 0,
        };
        ColliderRecord::build(desc, source)
    }
}
