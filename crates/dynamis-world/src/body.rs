use super::World;
use super::command::BodyCommand;
use super::pool::Pool;
use bytemuck::Zeroable;
use dynamis_abi::{
    BODY_CCD, BODY_KINEMATIC, BodyDescriptorRecord, BodyStateRecord, ColliderRecord,
    PATCH_ANGULAR_VELOCITY, PATCH_ORIENTATION, PATCH_POSITION, PATCH_VELOCITY, ShapeRole,
    shape_source_handle,
};
use dynamis_model::{
    BodyDesc, BodyHandle, BodyState, ColliderDesc, CollisionFilter, ContactEventMode, Shape,
};

#[derive(Clone)]
pub(crate) struct BodyStore {
    pub(crate) pool: Pool<BodyHandle>,
    pub(crate) descs: Vec<BodyDesc>,
    pub(crate) records: Vec<BodyDescriptorRecord>,
    pub(crate) dynamic: u32,
    pub(crate) ccd: u32,
    pub(crate) commands: Vec<BodyCommand>,
    pub(crate) last_moves: u32,
    pub(crate) last_edits: u32,
}

impl BodyStore {
    pub(crate) const fn new() -> Self {
        Self {
            pool: Pool::compact("body"),
            descs: Vec::new(),
            records: Vec::new(),
            dynamic: 0,
            ccd: 0,
            commands: Vec::new(),
            last_moves: 0,
            last_edits: 0,
        }
    }

    pub(crate) fn handle_of(&self, id: u32) -> Option<BodyHandle> {
        self.pool.handle_of(id)
    }

    fn grow_to(&mut self, id: u32) {
        let rows = id as usize + 1;
        if rows <= self.descs.len() {
            return;
        }
        self.descs.resize(rows, BodyDesc::VACANT);
        self.records.resize(rows, BodyDescriptorRecord::zeroed());
    }
}

impl World {
    pub fn spawn(&mut self, desc: BodyDesc) -> BodyHandle {
        self.validate_world_geometry(&desc);
        let handle = self.bodies.pool.acquire();
        self.bodies.grow_to(handle.id);
        self.bodies.descs[handle.id as usize] = desc.clone();
        for collider in &desc.colliders {
            self.retain_shape_ref(&collider.shape);
        }
        let slot = self.bodies.pool.insert(handle);
        let state = BodyStateRecord::initial(&desc, handle.id, handle.generation);
        self.bodies
            .commands
            .push(BodyCommand::Add { row: slot, state });
        self.encode_body(handle);
        self.record_state(handle, &state);
        if self.observed.observes_whole_body_set() {
            self.observe_body(handle);
        }
        handle
    }

    pub fn remove(&mut self, handle: BodyHandle) {
        self.validate(handle);
        self.assert_unreferenced(handle);
        let slot = self.bodies.pool.row_of(handle);
        if slot < self.bodies.dynamic {
            let tail_dynamic = self.bodies.dynamic - 1;
            if slot != tail_dynamic {
                self.swap_slots(slot, tail_dynamic);
            }
            self.bodies.dynamic -= 1;
            let last = self.bodies.pool.len() - 1;
            if tail_dynamic != last {
                self.swap_slots(tail_dynamic, last);
            }
        } else {
            let last = self.bodies.pool.len() - 1;
            if slot != last {
                self.swap_slots(slot, last);
            }
        }
        self.discard_tail();
    }

    fn discard_tail(&mut self) {
        let last = self.bodies.pool.len() - 1;
        let handle = self.bodies.pool.handle_of_row(last);
        let id = handle.id as usize;
        let retired = self.bodies.pool.retire(handle);
        assert!(
            retired.moved.is_none(),
            "the tail row retires without a move"
        );
        let removed = std::mem::replace(&mut self.bodies.descs[id], BodyDesc::VACANT);
        for collider in &removed.colliders {
            self.release_shape_ref(&collider.shape);
        }
        self.colliders.release(id as u32);
        self.observed.bodies.stop_watching(handle.id);
        if self.bodies.records[id].flags & BODY_CCD != 0 {
            self.bodies.ccd -= 1;
        }
        self.bodies.records[id] = BodyDescriptorRecord::zeroed();
        self.bodies.commands.push(BodyCommand::Remove {
            hole: last,
            tail: last,
        });
    }

    fn swap_slots(&mut self, first: u32, second: u32) {
        self.bodies.pool.swap_rows(first, second);
        self.bodies
            .commands
            .push(BodyCommand::Swap { first, second });
    }

    fn encode_body(&mut self, handle: BodyHandle) {
        let id = handle.id as usize;
        let record = BodyDescriptorRecord::build(&self.bodies.descs[id], &self.config, |shape| {
            self.shape_solid(shape)
        });
        let previous = self.bodies.records[id];
        self.bodies.ccd = match (previous.flags & BODY_CCD != 0, record.flags & BODY_CCD != 0) {
            (false, true) => self.bodies.ccd + 1,
            (true, false) => self.bodies.ccd - 1,
            _ => self.bodies.ccd,
        };
        self.bodies.records[id] = record;
        self.bodies.pool.mark(handle);
        self.observed.bodies.patch(id as u32, |state| {
            state.inverse_mass = record.inverse_mass;
            state.com = record.com;
        });
        self.migrate_partition(handle);
        self.repool_colliders(handle.id, !self.is_static(id));
    }

    pub(crate) fn encode_bodies(&mut self) {
        let handles = self.bodies.pool.alive().to_vec();
        for handle in handles {
            self.encode_body(handle);
        }
    }

    fn edit_body(&mut self, handle: BodyHandle, change: impl FnOnce(&mut BodyDesc)) {
        self.validate(handle);
        change(&mut self.bodies.descs[handle.id as usize]);
        self.encode_body(handle);
    }

    fn migrate_partition(&mut self, handle: BodyHandle) {
        let id = handle.id as usize;
        let static_now = self.is_static(id);
        let slot = self.bodies.pool.row_of(handle);
        let in_dynamic = slot < self.bodies.dynamic;
        if static_now == !in_dynamic {
            return;
        }
        if static_now {
            let tail = self.bodies.dynamic - 1;
            if slot != tail {
                self.swap_slots(slot, tail);
            }
            self.bodies.dynamic -= 1;
            self.bodies.commands.push(BodyCommand::Patch {
                row: tail,
                mask: PATCH_VELOCITY,
                state: BodyStateRecord::zeroed(),
            });
            self.observed.bodies.patch(id as u32, |state| {
                state.velocity = [0.0; 3];
                state.angular_velocity = [0.0; 3];
            });
            self.bodies.pool.mark_row(tail);
        } else {
            let boundary = self.bodies.dynamic;
            if slot != boundary {
                self.swap_slots(slot, boundary);
            }
            self.bodies.dynamic += 1;
        }
        self.bodies.pool.mark_row(slot);
    }

    pub fn read_state(&self, handle: BodyHandle) -> BodyState {
        self.validate(handle);
        assert!(
            self.body_current(handle.id),
            "body {handle:?} carries no state for the last step; observe it with observe_bodies or observe_all_bodies before waiting"
        );
        *self
            .observed
            .bodies
            .get(handle.id)
            .expect("body state is unavailable")
    }

    fn record_state(&mut self, handle: BodyHandle, initial: &BodyStateRecord) {
        let step = self.clock.step;
        let record = self.bodies.records[handle.id as usize];
        self.observed
            .bodies
            .insert(handle.id, step, initial.state(&record, step));
    }

    fn validate_world_geometry(&self, desc: &BodyDesc) {
        let mass = desc.effective_mass(|shape| self.shape_solid(shape));
        if mass <= 0.0 || desc.kinematic {
            return;
        }
        for collider in &desc.colliders {
            assert!(
                !ShapeRole::of_shape(&collider.shape).world_geometry(),
                "world geometry colliders must be static or kinematic"
            );
        }
    }

    pub(crate) fn is_static(&self, id: usize) -> bool {
        let record = self.bodies.records[id];
        record.inverse_mass == 0.0 && record.flags & BODY_KINEMATIC == 0
    }
    fn assert_unreferenced(&self, handle: BodyHandle) {
        let id = handle.id;
        assert!(
            self.constraints.attached_to(id).is_empty(),
            "body handle {handle:?} is referenced by a live constraint"
        );
        assert!(
            self.soft.attachment_refs(id) == 0,
            "body handle {handle:?} anchors a soft attachment"
        );
        assert!(
            !self.vehicles.owns_body(id),
            "body handle {handle:?} is the chassis of a vehicle"
        );
        assert!(
            !self.characters.owns_body(id),
            "body handle {handle:?} is the body of a character"
        );
    }

    fn command_slot(&self, handle: BodyHandle) -> u32 {
        self.bodies.pool.row_of(handle)
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
        self.observed.bodies.patch(handle.id, |state| {
            state.position = position;
            state.prev_position = position;
        });
        let mut payload = BodyStateRecord::zeroed();
        payload.position = position;
        self.schedule_patch(handle, PATCH_POSITION, payload);
        self.bodies.pool.mark(handle);
    }

    pub fn set_orientation(&mut self, handle: BodyHandle, orientation: [f32; 4]) {
        self.assert_unit(orientation);
        self.validate(handle);
        self.observed.bodies.patch(handle.id, |state| {
            state.orientation = orientation;
        });
        let mut payload = BodyStateRecord::zeroed();
        payload.orientation = orientation;
        self.schedule_patch(handle, PATCH_ORIENTATION, payload);
        self.bodies.pool.mark(handle);
    }

    pub fn set_velocity(&mut self, handle: BodyHandle, velocity: [f32; 3]) {
        self.validate(handle);
        self.observed.bodies.patch(handle.id, |state| {
            state.velocity = velocity;
        });
        let mut payload = BodyStateRecord::zeroed();
        payload.velocity = velocity;
        self.schedule_patch(handle, PATCH_VELOCITY, payload);
        self.bodies.pool.mark(handle);
    }

    pub fn set_angular_velocity(&mut self, handle: BodyHandle, angular_velocity: [f32; 3]) {
        self.validate(handle);
        self.observed.bodies.patch(handle.id, |state| {
            state.angular_velocity = angular_velocity;
        });
        let mut payload = BodyStateRecord::zeroed();
        payload.angular_velocity = angular_velocity;
        self.schedule_patch(handle, PATCH_ANGULAR_VELOCITY, payload);
        self.bodies.pool.mark(handle);
    }

    pub fn set_mass(&mut self, handle: BodyHandle, mass: f32) {
        assert!(mass >= 0.0, "mass must be non-negative");
        self.edit_body(handle, |desc| {
            desc.mass = mass;
            desc.density = None;
        });
    }

    pub fn set_density(&mut self, handle: BodyHandle, density: f32) {
        assert!(density >= 0.0, "density must be non-negative");
        self.edit_body(handle, |desc| desc.density = Some(density));
    }

    pub fn set_com(&mut self, handle: BodyHandle, com: [f32; 3]) {
        self.edit_body(handle, |desc| desc.com = Some(com));
    }

    pub fn set_inertia(&mut self, handle: BodyHandle, inertia: [f32; 6]) {
        assert!(
            inertia.iter().all(|value| value.is_finite()),
            "inertia tensor must be finite"
        );
        self.edit_body(handle, |desc| desc.inertia = Some(inertia));
    }

    pub fn set_linear_damping(&mut self, handle: BodyHandle, damping: f32) {
        assert!(damping >= 0.0, "damping must be non-negative");
        self.edit_body(handle, |desc| desc.linear_damping = Some(damping));
    }

    pub fn set_angular_damping(&mut self, handle: BodyHandle, angular_damping: f32) {
        assert!(
            angular_damping >= 0.0,
            "angular damping must be non-negative"
        );
        self.edit_body(handle, |desc| desc.angular_damping = Some(angular_damping));
    }

    pub fn set_gravity_scale(&mut self, handle: BodyHandle, gravity_scale: f32) {
        self.edit_body(handle, |desc| desc.gravity_scale = gravity_scale);
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
        self.edit_body(handle, |desc| {
            desc.sleep_velocity = Some(velocity);
            desc.sleep_angular_velocity = Some(angular_velocity);
        });
    }

    pub fn set_collision_filter(&mut self, handle: BodyHandle, filter: CollisionFilter) {
        self.edit_body(handle, |desc| desc.filter = filter);
    }

    pub fn set_ccd(&mut self, handle: BodyHandle, ccd: bool) {
        self.edit_body(handle, |desc| desc.ccd = ccd);
    }

    pub(crate) fn ccd_active(&self) -> bool {
        debug_assert_eq!(
            self.bodies.ccd,
            self.bodies
                .records
                .iter()
                .filter(|row| row.flags & BODY_CCD != 0)
                .count() as u32,
            "the ccd body count must mirror the body records"
        );
        self.bodies.ccd > 0
    }

    pub fn set_kinematic(&mut self, handle: BodyHandle, kinematic: bool) {
        self.edit_body(handle, |desc| desc.kinematic = kinematic);
    }

    pub fn set_collider(&mut self, handle: BodyHandle, index: usize, collider: ColliderDesc) {
        self.validate(handle);
        let id = handle.id as usize;
        assert!(
            index < self.bodies.descs[id].colliders.len(),
            "collider index out of range"
        );
        let existing = std::mem::replace(&mut self.bodies.descs[id].colliders[index], collider);
        self.release_shape_ref(&existing.shape);
        self.retain_shape_ref(&collider.shape);
        self.encode_body(handle);
    }

    pub fn add_collider(&mut self, handle: BodyHandle, collider: ColliderDesc) {
        self.validate(handle);
        self.retain_shape_ref(&collider.shape);
        self.bodies.descs[handle.id as usize]
            .colliders
            .push(collider);
        self.encode_body(handle);
    }

    pub fn remove_collider(&mut self, handle: BodyHandle, index: usize) {
        self.validate(handle);
        let id = handle.id as usize;
        assert!(
            index < self.bodies.descs[id].colliders.len(),
            "collider index out of range"
        );
        assert!(
            self.bodies.descs[id].colliders.len() > 1,
            "a body requires at least one collider"
        );
        let removed = self.bodies.descs[id].colliders.swap_remove(index);
        self.release_shape_ref(&removed.shape);
        self.encode_body(handle);
    }

    pub fn set_shape(&mut self, handle: BodyHandle, shape: Shape) {
        self.validate(handle);
        let id = handle.id as usize;
        let replaced = std::mem::replace(&mut self.bodies.descs[id].colliders[0].shape, shape);
        self.release_shape_ref(&replaced);
        self.retain_shape_ref(&shape);
        self.encode_body(handle);
    }

    pub fn set_restitution(&mut self, handle: BodyHandle, restitution: f32) {
        self.edit_body(handle, |desc| desc.colliders[0].restitution = restitution);
    }

    pub fn set_friction(&mut self, handle: BodyHandle, friction: f32) {
        assert!(friction >= 0.0, "friction must be non-negative");
        self.edit_body(handle, |desc| desc.colliders[0].friction = friction);
    }

    pub fn set_contact_frequency(&mut self, handle: BodyHandle, contact_frequency: f32) {
        assert!(
            contact_frequency > 0.0,
            "a contact frequency must be strictly positive"
        );
        self.edit_body(handle, |desc| {
            desc.colliders[0].contact_frequency = contact_frequency
        });
    }

    pub fn set_contact_damping_ratio(&mut self, handle: BodyHandle, contact_damping_ratio: f32) {
        assert!(
            contact_damping_ratio >= 0.0,
            "a contact damping ratio must be non-negative"
        );
        self.edit_body(handle, |desc| {
            desc.colliders[0].contact_damping_ratio = contact_damping_ratio
        });
    }

    pub fn set_collider_events(
        &mut self,
        handle: BodyHandle,
        index: usize,
        events: ContactEventMode,
    ) {
        self.validate(handle);
        assert!(
            index < self.bodies.descs[handle.id as usize].colliders.len(),
            "collider index out of range"
        );
        self.edit_body(handle, |desc| desc.colliders[index].events = events);
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

    pub(crate) fn repool_colliders(&mut self, id: u32, movable: bool) {
        let records = self.collider_block_of(id as usize);
        self.colliders.assign(id, movable, &records);
    }

    pub(crate) fn collider_block_of(&self, id: usize) -> Vec<ColliderRecord> {
        self.bodies.descs[id]
            .colliders
            .iter()
            .enumerate()
            .map(|(slot, collider)| self.collider_record(collider, slot as u32))
            .collect()
    }

    pub(super) fn validate(&self, handle: BodyHandle) {
        self.bodies.pool.validate(handle);
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
        if let Some(handle) = shape_source_handle(shape) {
            self.shapes.pool.retain(handle);
        }
    }

    fn release_shape_ref(&mut self, shape: &Shape) {
        if let Some(handle) = shape_source_handle(shape) {
            self.shapes.pool.release(handle);
        }
    }
}

impl World {
    pub(crate) fn collider_record(&self, desc: &ColliderDesc, slot: u32) -> ColliderRecord {
        let source = shape_source_handle(&desc.shape).map_or(0, |handle| handle.id);
        ColliderRecord::build(desc, source, slot)
    }
}
