use super::World;
use super::collider::ColliderDelta;
use super::command::BodyCommand;
use super::pool::Pool;
use super::row::RowLayout;
use bytemuck::Zeroable;
use dynamis_abi::{
    BODY_CCD, BodyDescriptorRecord, BodyStateRecord, ColliderRecord, PATCH_ANGULAR_VELOCITY,
    PATCH_ORIENTATION, PATCH_POSITION, PATCH_VELOCITY, RowMoveRecord, assert_body_kind,
    shape_source_handle,
};
use dynamis_model::domain;
use dynamis_model::{
    BodyDesc, BodyHandle, BodyKind, BodyState, ColliderDesc, CollisionFilter, ContactEventMode,
    Shape, ShapeSourceHandle,
};
use std::collections::HashMap;

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
    pub(crate) rows: RowLayout<BodyHandle>,
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
            rows: RowLayout::new(),
        }
    }

    /// The moves the device owes to hold the rows at hand, and the state a row whose body the
    /// device has never held must be seeded with: the state the body's birth declared.
    pub(crate) fn row_moves(&self) -> (Vec<RowMoveRecord>, Vec<BodyStateRecord>) {
        if self.rows.settled() {
            return (Vec::new(), Vec::new());
        }
        let births = self.births();
        self.rows.difference(self.pool.alive(), |handle| {
            births
                .get(&handle.id)
                .copied()
                .filter(|state| state.generation == handle.generation)
                .unwrap_or_else(|| {
                    panic!(
                        "row of {handle:?} must be seeded with the state its birth declared, while no pending command carries it"
                    )
                })
        })
    }

    /// Records that the device holds the rows at hand, which the handover of the move stream does.
    pub(crate) fn settle_rows(&mut self) {
        let alive = self.pool.alive();
        self.rows.settle(alive);
    }

    pub(crate) fn rows_held(&self) -> u32 {
        self.rows.rows()
    }

    fn births(&self) -> HashMap<u32, BodyStateRecord> {
        self.commands
            .iter()
            .filter_map(|command| match *command {
                BodyCommand::Add { state, .. } => Some((state.body_id, state)),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn handle_of(&self, id: u32) -> Option<BodyHandle> {
        self.pool.handle_of(id)
    }

    pub(crate) fn kind(&self, id: u32) -> BodyKind {
        self.records[id as usize].kind()
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
        desc.assert_valid();
        assert_body_kind(&desc, self.body_kind_of(&desc));
        self.facts.anchors += 1;
        let handle = self.bodies.pool.acquire();
        self.bodies.grow_to(handle.id);
        self.bodies.descs[handle.id as usize] = desc.clone();
        for collider in &desc.colliders {
            self.retain_shape_ref(handle.id, &collider.shape);
        }
        let slot = self.bodies.pool.insert(handle);
        self.bodies.rows.touch(slot);
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
        self.facts.anchors += 1;
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
            self.release_shape_ref(id as u32, &collider.shape);
        }
        let movable = !self.is_static(id);
        let delta = self.colliders.release(id as u32);
        self.note_collider_delta(delta, movable);
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
        self.facts.layout += 1;
        self.bodies.rows.touch(first);
        self.bodies.rows.touch(second);
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
        if previous.kind().simulates() != record.kind().simulates() {
            self.facts.anchors += 1;
        }
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
        let mut candidate = self.bodies.descs[handle.id as usize].clone();
        change(&mut candidate);
        self.assert_body_intent(handle.id, &candidate);
        self.install_body(handle, candidate);
    }

    /// The kind a description declares in this world, where a hull answers the solid geometry its
    /// source carries and world geometry answers none.
    fn body_kind_of(&self, desc: &BodyDesc) -> BodyKind {
        desc.kind(|shape| self.shape_solid(shape))
    }

    /// The one authority a host entry passes before it installs a description: the description is
    /// a valid intent, the kind it declares may carry the geometry its colliders hold, and no actor
    /// that owns the motion of the body is contradicted.
    fn assert_body_intent(&self, id: u32, desc: &BodyDesc) {
        desc.assert_valid();
        let kind = self.body_kind_of(desc);
        assert_body_kind(desc, kind);
        self.assert_kind_owner(id, kind);
    }

    /// The one authority a body's declared kind passes: a character declares the kinematic motion of
    /// the body it drives, and a vehicle drives a chassis the solver must evolve, so no host edit
    /// may declare the opposite of either.
    fn assert_kind_owner(&self, id: u32, kind: BodyKind) {
        assert!(
            !(self.characters.owns_body(id) && kind != BodyKind::Kinematic),
            "a character declares the motion of the body it drives, which stays kinematic: {kind:?}",
        );
        assert!(
            !(self.vehicles.owns_body(id) && !kind.simulates()),
            "a vehicle drives its chassis through the solver, which stays simulated: {kind:?}",
        );
    }

    /// The one authority a declaration of a body's motion passes: a body the solver evolves answers
    /// the state the host loads it at, and a body the device never moves keeps the motion the host
    /// declares, so a declaration reaches both. A character declares the motion of the body it
    /// drives from the pose the character holds, so a host declaration of that motion would be
    /// overwritten by the character's next declaration rather than reach the simulation.
    fn assert_motion_owner(&self, handle: BodyHandle, what: &str) {
        assert!(
            !self.characters.owns_body(handle.id),
            "a character declares the motion of the body it drives, so the host cannot {what}: place the character instead",
        );
    }

    /// A character declares the capsule it sweeps, so the host may not re-declare the collider of
    /// the body the character drives.
    fn assert_collider_owner(&self, id: u32) {
        assert!(
            !self.characters.owns_body(id),
            "a character declares the collider of the body it drives",
        );
    }

    fn install_body(&mut self, handle: BodyHandle, desc: BodyDesc) {
        self.bodies.descs[handle.id as usize] = desc;
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

    pub(crate) fn is_static(&self, id: usize) -> bool {
        self.bodies.kind(id as u32) == BodyKind::Static
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
        domain::finite_vector(position, "a body position");
        self.assert_motion_owner(handle, "declare its position");
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
        domain::unit_quaternion(orientation, "a body orientation");
        self.validate(handle);
        self.assert_motion_owner(handle, "declare its orientation");
        self.observed.bodies.patch(handle.id, |state| {
            state.orientation = orientation;
            state.prev_orientation = orientation;
        });
        let mut payload = BodyStateRecord::zeroed();
        payload.orientation = orientation;
        self.schedule_patch(handle, PATCH_ORIENTATION, payload);
        self.bodies.pool.mark(handle);
    }

    pub fn set_velocity(&mut self, handle: BodyHandle, velocity: [f32; 3]) {
        self.validate(handle);
        domain::finite_vector(velocity, "a body velocity");
        self.assert_motion_owner(handle, "declare its velocity");
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
        domain::finite_vector(angular_velocity, "a body angular velocity");
        self.assert_motion_owner(handle, "declare its angular velocity");
        self.observed.bodies.patch(handle.id, |state| {
            state.angular_velocity = angular_velocity;
        });
        let mut payload = BodyStateRecord::zeroed();
        payload.angular_velocity = angular_velocity;
        self.schedule_patch(handle, PATCH_ANGULAR_VELOCITY, payload);
        self.bodies.pool.mark(handle);
    }

    pub fn set_mass(&mut self, handle: BodyHandle, mass: f32) {
        domain::non_negative(mass, "mass");
        self.edit_body(handle, |desc| {
            desc.mass = mass;
            desc.density = None;
        });
    }

    pub fn set_density(&mut self, handle: BodyHandle, density: f32) {
        domain::non_negative(density, "density");
        self.edit_body(handle, |desc| desc.density = Some(density));
    }

    pub fn set_com(&mut self, handle: BodyHandle, com: [f32; 3]) {
        domain::finite_vector(com, "a center of mass");
        self.edit_body(handle, |desc| desc.com = Some(com));
    }

    pub fn set_inertia(&mut self, handle: BodyHandle, inertia: [f32; 6]) {
        assert!(
            inertia.iter().all(|value| value.is_finite()),
            "an inertia tensor must be finite"
        );
        self.edit_body(handle, |desc| desc.inertia = Some(inertia));
    }

    pub fn set_linear_damping(&mut self, handle: BodyHandle, damping: f32) {
        domain::non_negative(damping, "damping");
        self.edit_body(handle, |desc| desc.linear_damping = Some(damping));
    }

    pub fn set_angular_damping(&mut self, handle: BodyHandle, angular_damping: f32) {
        domain::non_negative(angular_damping, "angular damping");
        self.edit_body(handle, |desc| desc.angular_damping = Some(angular_damping));
    }

    pub fn set_gravity_scale(&mut self, handle: BodyHandle, gravity_scale: f32) {
        domain::finite(gravity_scale, "a gravity scale");
        self.edit_body(handle, |desc| desc.gravity_scale = gravity_scale);
    }

    pub fn set_sleep_thresholds(
        &mut self,
        handle: BodyHandle,
        velocity: f32,
        angular_velocity: f32,
    ) {
        domain::non_negative(velocity, "a sleep velocity");
        domain::non_negative(angular_velocity, "a sleep angular velocity");
        self.edit_body(handle, |desc| {
            desc.sleep_velocity = Some(velocity);
            desc.sleep_angular_velocity = Some(angular_velocity);
        });
    }

    pub fn set_collision_filter(&mut self, handle: BodyHandle, filter: CollisionFilter) {
        self.edit_body(handle, |desc| desc.filter = filter);
    }

    pub fn set_ccd(&mut self, handle: BodyHandle, ccd: bool) {
        if ccd {
            self.assert_simulating(handle, "declare continuous collision");
        }
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
        self.validate(handle);
        assert!(
            !(kinematic && self.bodies.descs[handle.id as usize].ccd),
            "a body that declares continuous collision cannot be kinematic: {handle:?}",
        );
        self.edit_body(handle, |desc| desc.kinematic = kinematic);
    }

    pub fn set_collider(&mut self, handle: BodyHandle, index: usize, collider: ColliderDesc) {
        self.validate(handle);
        collider.assert_valid();
        let id = handle.id as usize;
        assert!(
            index < self.bodies.descs[id].colliders.len(),
            "collider index out of range"
        );
        self.assert_collider_owner(handle.id);
        let mut candidate = self.bodies.descs[id].clone();
        let existing = std::mem::replace(&mut candidate.colliders[index], collider);
        self.assert_body_intent(handle.id, &candidate);
        self.retain_shape_ref(handle.id, &collider.shape);
        self.install_body(handle, candidate);
        self.release_shape_ref(handle.id, &existing.shape);
    }

    pub fn add_collider(&mut self, handle: BodyHandle, collider: ColliderDesc) {
        self.validate(handle);
        collider.assert_valid();
        self.assert_collider_owner(handle.id);
        let mut candidate = self.bodies.descs[handle.id as usize].clone();
        candidate.colliders.push(collider);
        self.assert_body_intent(handle.id, &candidate);
        self.retain_shape_ref(handle.id, &collider.shape);
        self.install_body(handle, candidate);
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
        self.assert_collider_owner(handle.id);
        let mut candidate = self.bodies.descs[id].clone();
        let removed = candidate.colliders.swap_remove(index);
        self.assert_body_intent(handle.id, &candidate);
        self.install_body(handle, candidate);
        self.release_shape_ref(handle.id, &removed.shape);
    }

    pub fn set_shape(&mut self, handle: BodyHandle, shape: Shape) {
        self.validate(handle);
        shape.assert_valid();
        self.assert_collider_owner(handle.id);
        let id = handle.id as usize;
        let mut candidate = self.bodies.descs[id].clone();
        let replaced = std::mem::replace(&mut candidate.colliders[0].shape, shape);
        self.assert_body_intent(handle.id, &candidate);
        self.retain_shape_ref(handle.id, &shape);
        self.install_body(handle, candidate);
        self.release_shape_ref(handle.id, &replaced);
    }

    pub fn set_restitution(&mut self, handle: BodyHandle, restitution: f32) {
        domain::non_negative(restitution, "restitution");
        self.edit_body(handle, |desc| desc.colliders[0].restitution = restitution);
    }

    pub fn set_friction(&mut self, handle: BodyHandle, friction: f32) {
        domain::non_negative(friction, "friction");
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
        domain::non_negative(contact_damping_ratio, "a contact damping ratio");
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
        domain::finite_vector(force, "a body force");
        self.assert_simulating(handle, "answer a force");
        let slot = self.command_slot(handle);
        self.bodies
            .commands
            .push(BodyCommand::Force { row: slot, force });
    }

    pub fn apply_force_at_point(&mut self, handle: BodyHandle, force: [f32; 3], point: [f32; 3]) {
        domain::finite_vector(force, "a body force");
        domain::finite_vector(point, "a force application point");
        self.assert_simulating(handle, "answer a force");
        let slot = self.command_slot(handle);
        self.bodies.commands.push(BodyCommand::ForceAtPoint {
            row: slot,
            force,
            point,
        });
    }

    pub fn apply_torque(&mut self, handle: BodyHandle, torque: [f32; 3]) {
        domain::finite_vector(torque, "a body torque");
        self.assert_simulating(handle, "answer a torque");
        let slot = self.command_slot(handle);
        self.bodies
            .commands
            .push(BodyCommand::Torque { row: slot, torque });
    }

    pub fn apply_impulse(&mut self, handle: BodyHandle, impulse: [f32; 3]) {
        domain::finite_vector(impulse, "a body impulse");
        self.assert_simulating(handle, "answer an impulse");
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
        domain::finite_vector(impulse, "a body impulse");
        domain::finite_vector(point, "an impulse application point");
        self.assert_simulating(handle, "answer an impulse");
        let slot = self.command_slot(handle);
        self.bodies.commands.push(BodyCommand::ImpulseAtPoint {
            row: slot,
            impulse,
            point,
        });
    }

    pub fn apply_angular_impulse(&mut self, handle: BodyHandle, impulse: [f32; 3]) {
        domain::finite_vector(impulse, "a body angular impulse");
        self.assert_simulating(handle, "answer an angular impulse");
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
        self.assert_simulating(handle, "sleep");
        let slot = self.command_slot(handle);
        self.bodies.commands.push(BodyCommand::Sleep { row: slot });
    }

    fn assert_simulating(&self, handle: BodyHandle, what: &str) {
        self.validate(handle);
        let kind = self.bodies.kind(handle.id);
        assert!(
            kind.simulates(),
            "only a simulated body can {what}: {handle:?} is {kind:?}",
        );
    }

    pub(crate) fn repool_colliders(&mut self, id: u32, movable: bool) {
        let records = self.collider_block_of(id as usize);
        let delta = self.colliders.assign(id, movable, &records);
        self.note_collider_delta(delta, movable);
    }

    /// Notes the facts a collider block's replacement or retirement moved: the grid resolution is a
    /// maximum over every block's sizing, and each half of the grid holds the entries of the
    /// bodies that reach it. A block that is replaced with the same sizing and placement moves
    /// neither, whatever else its records carry.
    fn note_collider_delta(&mut self, delta: ColliderDelta, movable: bool) {
        if delta.sizing {
            self.facts.geometry += 1;
        }
        if delta.entries() {
            if delta.crossed || movable {
                self.facts.movable_colliders += 1;
            }
            if delta.crossed || !movable {
                self.facts.immovable_colliders += 1;
            }
        }
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

    fn retain_shape_ref(&mut self, body: u32, shape: &Shape) {
        if let Some(handle) = shape_source_handle(shape) {
            self.shapes.pool.retain(handle, body);
        }
    }

    fn release_shape_ref(&mut self, body: u32, shape: &Shape) {
        if let Some(handle) = shape_source_handle(shape) {
            self.shapes.pool.release(handle, body);
        }
    }

    /// Re-derives every body that reads a shape source: a body's record answers the mass its
    /// colliders derive from the solid geometry of their sources, so it must follow the source's
    /// geometry wherever the host moves it.
    pub(crate) fn encode_shape_readers(&mut self, source: ShapeSourceHandle) {
        for id in self.shapes.pool.readers_of(source) {
            if let Some(handle) = self.bodies.pool.handle_of(id) {
                self.encode_body(handle);
            }
        }
    }
}

impl World {
    pub(crate) fn collider_record(&self, desc: &ColliderDesc, slot: u32) -> ColliderRecord {
        let source = shape_source_handle(&desc.shape).map_or(0, |handle| handle.id);
        ColliderRecord::build(desc, source, slot)
    }
}
