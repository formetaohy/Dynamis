use super::World;
use super::arena::{Cleared, Mirror, Run};
use super::command::Consumption;
use super::device::{Facts, Region};
use super::journal::EditJournal;
use super::pool::Pool;
use dynamis_abi::{
    ELEMENT_PARTICLES, ELEMENT_ROLE_BITS, NO_SLOT, SoftAttachmentInit, SoftAttachmentRecord,
    SoftBodyEditRecord, SoftBodyRecord, SoftEditRecord, SoftElementInit, SoftElementRecord,
    SoftParticleInit, SoftParticleRecord,
};
use dynamis_model::math::{add, mul, quat_rotate};
use dynamis_model::{
    BodyHandle, ContactEventMode, SoftBodyDesc, SoftBodyHandle, SoftElement, SoftElementState,
    SoftParticleState,
};
use dynamis_soft::SoftStreams;

#[derive(Clone, Copy)]
pub(crate) struct SoftRuns {
    pub(crate) particles: Run,
    pub(crate) elements: Run,
    pub(crate) attachments: Run,
    pub(crate) adjacency: Run,
    pub(crate) strength: bool,
}

impl SoftRuns {
    const EMPTY: Self = Self {
        particles: Run::EMPTY,
        elements: Run::EMPTY,
        attachments: Run::EMPTY,
        adjacency: Run::EMPTY,
        strength: false,
    };
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum SoftCommand {
    InverseMass(f32),
    Radius(f32),
    Friction(f32),
    Position([f32; 3]),
    Velocity([f32; 3]),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum SoftBodyCommand {
    Accelerate([f32; 3]),
    Wake,
}

#[derive(Clone)]
pub(crate) struct SoftBodyStore {
    pool: Pool<SoftBodyHandle>,
    runs: Vec<SoftRuns>,
    masses: Vec<f32>,
    events: Vec<ContactEventMode>,
    records: Vec<SoftBodyRecord>,
    body_commands: EditJournal<u32, SoftBodyCommand>,
    commands: EditJournal<u32, SoftCommand>,
    strength: u32,
    persistent: u32,
    pub(crate) last_body_edits: u32,
    pub(crate) last_edits: u32,
    particles: Mirror<SoftParticleRecord>,
    elements: Mirror<SoftElementRecord>,
    attachments: Mirror<SoftAttachmentRecord>,
    attachment_refs: Vec<u32>,
    adjacency: Mirror<u32>,
    pub(crate) uploaded: bool,
}

impl Cleared for SoftParticleRecord {
    const CLEARED: Self = Self::cleared();
}

impl Cleared for SoftElementRecord {
    const CLEARED: Self = Self::cleared();
}

impl Cleared for SoftAttachmentRecord {
    const CLEARED: Self = Self::cleared();
}

impl Cleared for u32 {
    const CLEARED: Self = u32::MAX;
}

impl SoftBodyStore {
    pub(crate) fn new() -> Self {
        Self {
            pool: Pool::compact("soft body"),
            runs: Vec::new(),
            masses: Vec::new(),
            events: Vec::new(),
            records: Vec::new(),
            body_commands: EditJournal::new(),
            commands: EditJournal::new(),
            strength: 0,
            persistent: 0,
            last_body_edits: 0,
            last_edits: 0,
            particles: Mirror::new(),
            elements: Mirror::new(),
            attachments: Mirror::new(),
            attachment_refs: Vec::new(),
            adjacency: Mirror::new(),
            uploaded: false,
        }
    }

    pub(crate) fn count(&self) -> usize {
        self.pool.len() as usize
    }

    pub(crate) fn ids_len(&self) -> usize {
        self.pool.ids() as usize
    }

    pub(crate) fn carries_strength(&self) -> bool {
        self.strength > 0
    }

    pub(crate) fn carries_events(&self) -> bool {
        self.persistent > 0
    }

    pub(crate) fn bodies(&self) -> &[SoftBodyHandle] {
        self.pool.alive()
    }

    pub(crate) fn used(&self) -> (u32, u32, u32, u32) {
        (
            self.particles.used(),
            self.elements.used(),
            self.attachments.used(),
            self.adjacency.used(),
        )
    }

    pub(crate) fn attachment_refs(&self, id: u32) -> u32 {
        self.attachment_refs.get(id as usize).copied().unwrap_or(0)
    }

    pub(crate) fn run_of(&self, handle: SoftBodyHandle) -> Run {
        self.runs_of(handle).particles
    }

    pub(crate) fn local_particle_of(&self, body: SoftBodyHandle, slot: u32) -> u32 {
        if self.handle_of(body.id) != Some(body) {
            return slot;
        }
        let run = self.runs[body.id as usize].particles;
        slot.checked_sub(run.offset).unwrap_or_else(|| {
            panic!(
                "soft body {body:?} cannot hold the particle a query hit reported at slot {slot}"
            )
        })
    }

    pub(crate) fn mass_of(&self, handle: SoftBodyHandle) -> f32 {
        self.validate(handle);
        self.masses[handle.id as usize]
    }

    pub(crate) fn particle_slot_of(&self, handle: SoftBodyHandle, particle: u32) -> u32 {
        let run = self.run_of(handle);
        assert!(
            particle < run.len,
            "soft particle {particle} is outside the {} particles of {handle:?}",
            run.len,
        );
        run.offset + particle
    }

    pub(crate) fn particle_state(
        &self,
        handle: SoftBodyHandle,
        particle: u32,
    ) -> SoftParticleState {
        let record = &self.particles.records()[self.particle_slot_of(handle, particle) as usize];
        SoftParticleState::new(record.inverse_mass(), record.radius(), record.friction())
    }

    pub(crate) fn pending_edits(&self) -> u32 {
        self.commands.len() as u32
    }

    pub(crate) fn pending_body_edits(&self) -> u32 {
        self.body_commands.len() as u32
    }

    pub(crate) fn pending_uploads(&self) -> bool {
        self.pool.pending() || !self.body_commands.is_empty() || !self.commands.is_empty()
    }

    pub(crate) fn compile_body_commands(
        &self,
        consumption: Consumption,
    ) -> Vec<SoftBodyEditRecord> {
        if consumption != Consumption::Step {
            return Vec::new();
        }
        self.body_commands
            .iter()
            .map(|(owner, commands)| {
                commands
                    .iter()
                    .fold(
                        SoftBodyEditRecord::merged(owner),
                        |edit, command| match *command {
                            SoftBodyCommand::Accelerate(acceleration) => {
                                edit.accelerate(acceleration)
                            }
                            SoftBodyCommand::Wake => edit.wake(),
                        },
                    )
            })
            .collect()
    }

    pub(crate) fn compile_commands(&self, consumption: Consumption) -> Vec<SoftEditRecord> {
        if consumption != Consumption::Step {
            return Vec::new();
        }
        self.commands
            .iter()
            .map(|(particle, commands)| {
                commands
                    .iter()
                    .fold(
                        SoftEditRecord::merged(particle),
                        |edit, command| match *command {
                            SoftCommand::InverseMass(inverse_mass) => {
                                edit.inverse_mass(inverse_mass)
                            }
                            SoftCommand::Radius(radius) => edit.radius(radius),
                            SoftCommand::Friction(friction) => edit.friction(friction),
                            SoftCommand::Position(position) => edit.position(position),
                            SoftCommand::Velocity(velocity) => edit.velocity(velocity),
                        },
                    )
            })
            .collect()
    }

    pub(crate) fn accelerate(&mut self, handle: SoftBodyHandle, acceleration: [f32; 3]) {
        self.validate(handle);
        assert!(
            acceleration.iter().all(|value| value.is_finite()),
            "a soft body acceleration must be finite"
        );
        self.body_commands
            .push(handle.id, SoftBodyCommand::Accelerate(acceleration));
        self.body_commands.push(handle.id, SoftBodyCommand::Wake);
    }

    fn edit(
        &mut self,
        handle: SoftBodyHandle,
        particle: u32,
        command: SoftCommand,
        substep_dt: f32,
    ) {
        let slot = self.particle_slot_of(handle, particle);
        let id = handle.id as usize;
        match command {
            SoftCommand::InverseMass(inverse_mass) => {
                let record = &mut self.particles.records_mut()[slot as usize];
                let previous = record.inverse_mass();
                record.prev_position[3] = inverse_mass;
                let mass = &mut self.masses[id];
                if previous > 0.0 {
                    *mass -= 1.0 / previous;
                }
                if inverse_mass > 0.0 {
                    *mass += 1.0 / inverse_mass;
                }
            }
            SoftCommand::Radius(radius) => {
                self.particles.records_mut()[slot as usize].position[3] = radius;
            }
            SoftCommand::Friction(friction) => {
                self.particles.records_mut()[slot as usize].velocity[3] = friction;
            }
            SoftCommand::Position(position) => {
                let record = &mut self.particles.records_mut()[slot as usize];
                record.position[..3].copy_from_slice(&position);
                record.prev_position[..3].copy_from_slice(&position);
            }
            SoftCommand::Velocity(velocity) => {
                let record = &mut self.particles.records_mut()[slot as usize];
                let position = [record.position[0], record.position[1], record.position[2]];
                record.velocity[..3].copy_from_slice(&velocity);
                record.prev_position[..3].copy_from_slice(&[
                    position[0] - velocity[0] * substep_dt,
                    position[1] - velocity[1] * substep_dt,
                    position[2] - velocity[2] * substep_dt,
                ]);
            }
        }
        self.body_commands.push(handle.id, SoftBodyCommand::Wake);
        self.commands.push(slot, command);
    }

    pub(crate) fn attach(
        &mut self,
        handle: SoftBodyHandle,
        particle: u32,
        body: BodyHandle,
        local: [f32; 3],
    ) {
        self.validate(handle);
        assert!(
            local.iter().all(|value| value.is_finite()),
            "a soft attachment anchor must be finite"
        );
        let slot = self.particle_slot_of(handle, particle);
        let mut records = self.attachments_of(handle);
        assert!(
            records.iter().all(|record| record.particle != slot),
            "soft particle {particle} of {handle:?} already carries an attachment"
        );
        self.hold_attachment(body);
        records.push(SoftAttachmentRecord::build(SoftAttachmentInit {
            particle: slot,
            body_id: body.id,
            generation: body.generation,
            local,
        }));
        self.rewrite_attachments(handle, records);
    }

    pub(crate) fn detach(&mut self, handle: SoftBodyHandle, particle: u32) {
        self.validate(handle);
        let slot = self.particle_slot_of(handle, particle);
        let mut records = self.attachments_of(handle);
        let at = records
            .iter()
            .position(|record| record.particle == slot)
            .unwrap_or_else(|| {
                panic!("soft particle {particle} of {handle:?} carries no attachment")
            });
        let removed = records.remove(at);
        self.release_attachment(removed.body_id);
        self.rewrite_attachments(handle, records);
    }

    fn attachments_of(&self, handle: SoftBodyHandle) -> Vec<SoftAttachmentRecord> {
        let run = self.runs[handle.id as usize].attachments;
        self.attachments.records()[run.span()].to_vec()
    }

    fn rewrite_attachments(&mut self, handle: SoftBodyHandle, records: Vec<SoftAttachmentRecord>) {
        let id = handle.id as usize;
        let previous = self.runs[id].attachments;
        if previous.len > 0 {
            self.attachments.retire(previous);
            self.runs[id].attachments = Run::EMPTY;
        }
        if !records.is_empty() {
            let run = self.attachments.take(records.len() as u32);
            self.attachments.records_mut()[run.span()].copy_from_slice(&records);
            self.runs[id].attachments = run;
        }
        self.body_commands.push(handle.id, SoftBodyCommand::Wake);
    }

    pub(crate) fn runs_of(&self, handle: SoftBodyHandle) -> SoftRuns {
        self.validate(handle);
        self.runs[handle.id as usize]
    }

    pub(crate) fn handle_of(&self, id: u32) -> Option<SoftBodyHandle> {
        self.pool.handle_of(id)
    }

    pub(crate) fn validate(&self, handle: SoftBodyHandle) {
        self.pool.validate(handle);
    }

    fn grow_to(&mut self, id: u32) {
        if self.runs.len() > id as usize {
            return;
        }
        self.runs.resize(id as usize + 1, SoftRuns::EMPTY);
        self.masses.resize(id as usize + 1, 0.0);
        self.events.resize(id as usize + 1, ContactEventMode::None);
        self.records
            .resize(id as usize + 1, SoftBodyRecord::cleared());
    }

    fn hold_attachment(&mut self, body: BodyHandle) {
        let id = body.id as usize;
        if self.attachment_refs.len() <= id {
            self.attachment_refs.resize(id + 1, 0);
        }
        self.attachment_refs[id] += 1;
    }

    fn release_attachment(&mut self, body_id: u32) {
        let refs = self
            .attachment_refs
            .get_mut(body_id as usize)
            .filter(|refs| **refs > 0)
            .unwrap_or_else(|| panic!("soft attachments must hold a live body id {body_id}"));
        *refs -= 1;
    }

    pub(crate) fn spawn(&mut self, desc: &SoftBodyDesc) -> SoftBodyHandle {
        let handle = self.pool.acquire();
        let id = handle.id;
        self.grow_to(id);
        self.events[id as usize] = desc.events;
        self.records[id as usize] = SoftBodyRecord::awake(desc.filter, desc.events);
        self.masses[id as usize] = desc
            .inverse_masses
            .iter()
            .filter(|inverse_mass| **inverse_mass > 0.0)
            .map(|inverse_mass| 1.0 / inverse_mass)
            .sum();
        let particles = self.particles.take(desc.particles.len() as u32);
        let elements = self.elements.take(desc.elements.len() as u32);
        let attachments = self.attachments.take(desc.attachments.len() as u32);
        let adjacency = self
            .adjacency
            .take(desc.elements.len() as u32 * ELEMENT_PARTICLES);
        let neighbours = assemble_adjacency(
            self.adjacency.records_mut(),
            desc.particles.len(),
            &desc.elements,
            adjacency.offset,
            elements.offset,
        );
        for (slot, local) in desc.particles.iter().enumerate() {
            let position = add(desc.position, quat_rotate(desc.orientation, *local));
            let (offset, count) = neighbours[slot];
            self.particles.records_mut()[particles.offset as usize + slot] =
                SoftParticleRecord::build(SoftParticleInit {
                    position,
                    prev_position: position,
                    velocity: desc.velocity,
                    radius: desc.radius,
                    inverse_mass: desc.inverse_masses[slot],
                    friction: desc.friction,
                    support: desc.fluid.map_or(0.0, |fluid| fluid.support()),
                    rest_spacing: desc.fluid.map_or(0.0, |fluid| fluid.spacing()),
                    neighbour_offset: offset,
                    neighbour_count: count,
                    owner: id,
                    generation: handle.generation,
                });
        }
        for (slot, element) in desc.elements.iter().enumerate() {
            self.elements.records_mut()[elements.offset as usize + slot] =
                SoftElementRecord::build(SoftElementInit {
                    kind: element.kind() as u32,
                    particles: global_particles(element, particles.offset),
                    rest: element.rest(),
                    compliance: element.compliance_of(),
                    yield_strain: element.yield_strain_of(),
                    break_strain: element.break_strain_of(),
                    plastic_flow: element.plastic_flow_of(),
                });
        }
        for (slot, attachment) in desc.attachments.iter().enumerate() {
            self.hold_attachment(attachment.body());
            self.attachments.records_mut()[attachments.offset as usize + slot] =
                SoftAttachmentRecord::build(SoftAttachmentInit {
                    particle: particles.offset + attachment.particle(),
                    body_id: attachment.body().id,
                    generation: attachment.body().generation,
                    local: attachment.local(),
                });
        }
        let strength = desc.carries_strength();
        if strength {
            self.strength += 1;
        }
        if !matches!(desc.events, ContactEventMode::None) {
            self.persistent += 1;
        }
        self.runs[id as usize] = SoftRuns {
            particles,
            elements,
            attachments,
            adjacency,
            strength,
        };
        self.pool.insert(handle);
        handle
    }

    pub(crate) fn remove(&mut self, handle: SoftBodyHandle) {
        self.validate(handle);
        let id = handle.id as usize;
        let runs = self.runs[id];
        assert!(
            self.commands
                .iter()
                .all(|(particle, _)| !runs.particles.span().contains(&(particle as usize))),
            "soft body {handle:?} must consume its particle edits before it is removed"
        );
        self.body_commands.remove(handle.id);
        if runs.strength {
            self.strength -= 1;
        }
        if !matches!(self.events[id], ContactEventMode::None) {
            self.persistent -= 1;
        }
        self.records[id] = SoftBodyRecord::cleared();
        for slot in runs.attachments.span() {
            self.release_attachment(self.attachments.records()[slot].body_id);
        }
        self.particles.retire(runs.particles);
        self.elements.retire(runs.elements);
        self.attachments.retire(runs.attachments);
        self.adjacency.retire(runs.adjacency);
        self.runs[id] = SoftRuns::EMPTY;
        self.events[id] = ContactEventMode::None;
        self.pool.retire(handle);
    }

    pub(crate) fn consume(&mut self) {
        self.body_commands.clear();
        self.commands.clear();
    }

    pub(crate) fn upload(&mut self, queue: &wgpu::Queue, streams: &SoftStreams) {
        if self.pool.pending() {
            self.uploaded = true;
            let stride = streams.bodies.stride();
            for id in self.pool.take_retired() {
                streams.bodies.write_at(
                    queue,
                    u64::from(id) * stride,
                    bytemuck::bytes_of(&SoftBodyRecord::cleared()),
                );
            }
            for id in self.pool.changed() {
                streams.bodies.write_at(
                    queue,
                    u64::from(id) * stride,
                    bytemuck::bytes_of(&self.records[id as usize]),
                );
            }
        }
        let particle_stride = streams.particles.stride();
        self.particles.flush(|offset, records| {
            streams.particles.write_at(
                queue,
                u64::from(offset) * particle_stride,
                bytemuck::cast_slice(records),
            );
        });
        let element_stride = streams.elements.stride();
        self.elements.flush(|offset, records| {
            streams.elements.write_at(
                queue,
                u64::from(offset) * element_stride,
                bytemuck::cast_slice(records),
            );
        });
        let attachment_stride = streams.attachments.stride();
        self.attachments.flush(|offset, records| {
            streams.attachments.write_at(
                queue,
                u64::from(offset) * attachment_stride,
                bytemuck::cast_slice(records),
            );
        });
        self.adjacency.flush(|offset, records| {
            streams
                .adjacency
                .write_at(queue, u64::from(offset) * 4, bytemuck::cast_slice(records));
        });
    }
}

fn global_particles(element: &SoftElement, base: u32) -> [u32; ELEMENT_PARTICLES as usize] {
    element.particles().map(|particle| {
        if particle == SoftElement::UNUSED {
            NO_SLOT
        } else {
            base + particle
        }
    })
}

fn assemble_adjacency(
    adjacency: &mut [u32],
    particles: usize,
    elements: &[SoftElement],
    adjacency_base: u32,
    element_base: u32,
) -> Vec<(u32, u32)> {
    let mut counts = vec![0u32; particles];
    for element in elements {
        for particle in element.participants() {
            counts[particle as usize] += 1;
        }
    }
    let mut ranges = Vec::with_capacity(particles);
    let mut cursor = Vec::with_capacity(particles);
    let mut offset = adjacency_base;
    for count in counts {
        ranges.push((offset, count));
        cursor.push(offset);
        offset += count;
    }
    for (slot, element) in elements.iter().enumerate() {
        for (role, particle) in element.participants().enumerate() {
            let entry = ((element_base + slot as u32) << ELEMENT_ROLE_BITS) | role as u32;
            adjacency[cursor[particle as usize] as usize] = entry;
            cursor[particle as usize] += 1;
        }
    }
    ranges
}

impl World {
    fn soft_substep_dt(&self) -> f32 {
        self.clock.sub_dt / self.config.soft_substeps as f32
    }

    pub fn add_soft_body(&mut self, desc: SoftBodyDesc) -> SoftBodyHandle {
        desc.assert_attachments();
        for attachment in &desc.attachments {
            self.validate(attachment.body());
        }
        self.soft.spawn(&desc)
    }

    pub fn remove_soft_body(&mut self, handle: SoftBodyHandle) {
        self.soft.remove(handle);
        self.observed.soft.stop_watching(handle.id);
    }

    pub fn apply_soft_force(&mut self, handle: SoftBodyHandle, force: [f32; 3]) {
        assert!(
            force.iter().all(|value| value.is_finite()),
            "a soft body force must be finite"
        );
        let mass = self.soft.mass_of(handle);
        assert!(
            mass > 0.0,
            "soft body {handle:?} carries no dynamic mass to accelerate"
        );
        self.soft.accelerate(handle, mul(force, 1.0 / mass));
    }

    pub fn apply_soft_acceleration(&mut self, handle: SoftBodyHandle, acceleration: [f32; 3]) {
        self.soft.accelerate(handle, acceleration);
    }

    pub fn set_soft_particle_inverse_mass(
        &mut self,
        handle: SoftBodyHandle,
        particle: u32,
        inverse_mass: f32,
    ) {
        assert!(
            inverse_mass >= 0.0,
            "a soft particle inverse mass must be non-negative"
        );
        let substep_dt = self.soft_substep_dt();
        self.soft.edit(
            handle,
            particle,
            SoftCommand::InverseMass(inverse_mass),
            substep_dt,
        );
    }

    pub fn set_soft_particle_radius(&mut self, handle: SoftBodyHandle, particle: u32, radius: f32) {
        assert!(radius >= 0.0, "a soft particle radius must be non-negative");
        let substep_dt = self.soft_substep_dt();
        self.soft
            .edit(handle, particle, SoftCommand::Radius(radius), substep_dt);
    }

    pub fn set_soft_particle_friction(
        &mut self,
        handle: SoftBodyHandle,
        particle: u32,
        friction: f32,
    ) {
        assert!(
            friction >= 0.0,
            "a soft particle friction must be non-negative"
        );
        let substep_dt = self.soft_substep_dt();
        self.soft.edit(
            handle,
            particle,
            SoftCommand::Friction(friction),
            substep_dt,
        );
    }

    pub fn set_soft_particle_position(
        &mut self,
        handle: SoftBodyHandle,
        particle: u32,
        position: [f32; 3],
    ) {
        assert!(
            position.iter().all(|value| value.is_finite()),
            "a soft particle position must be finite"
        );
        let substep_dt = self.soft_substep_dt();
        self.soft.edit(
            handle,
            particle,
            SoftCommand::Position(position),
            substep_dt,
        );
    }

    pub fn set_soft_particle_velocity(
        &mut self,
        handle: SoftBodyHandle,
        particle: u32,
        velocity: [f32; 3],
    ) {
        assert!(
            velocity.iter().all(|value| value.is_finite()),
            "a soft particle velocity must be finite"
        );
        let substep_dt = self.soft_substep_dt();
        self.soft.edit(
            handle,
            particle,
            SoftCommand::Velocity(velocity),
            substep_dt,
        );
    }

    pub fn attach_soft_particle(
        &mut self,
        handle: SoftBodyHandle,
        particle: u32,
        body: BodyHandle,
        local: [f32; 3],
    ) {
        self.validate(body);
        self.soft.attach(handle, particle, body, local);
    }

    pub fn detach_soft_particle(&mut self, handle: SoftBodyHandle, particle: u32) {
        self.soft.detach(handle, particle);
    }

    pub fn soft_particle_state(&self, handle: SoftBodyHandle, particle: u32) -> SoftParticleState {
        self.soft.particle_state(handle, particle)
    }

    pub fn soft_bodies(&self) -> &[SoftBodyHandle] {
        self.soft.bodies()
    }

    pub fn soft_body_count(&self) -> usize {
        self.soft.count()
    }

    pub fn inspect_soft_particles(&mut self, handle: SoftBodyHandle) -> Vec<[f32; 3]> {
        self.sync(Facts::Retired);
        let run = self.soft.run_of(handle);
        let region = Region::of(&self.backend.streams.soft.particles, run.offset, run.len);
        let records = self.read::<SoftParticleRecord>("soft body particles", region);
        assert!(
            records
                .iter()
                .all(|record| record.owner == handle.id && record.generation == handle.generation),
            "a soft body readback must return only its own particles"
        );
        records
            .iter()
            .map(|record| [record.position[0], record.position[1], record.position[2]])
            .collect()
    }

    pub fn inspect_soft_elements(&mut self, handle: SoftBodyHandle) -> Vec<SoftElementState> {
        self.sync(Facts::Retired);
        let runs = self.soft.runs_of(handle);
        if runs.elements.len == 0 {
            return Vec::new();
        }
        let region = Region::of(
            &self.backend.streams.soft.elements,
            runs.elements.offset,
            runs.elements.len,
        );
        let states = self
            .read::<SoftElementRecord>("soft body elements", region)
            .iter()
            .map(SoftElementRecord::state)
            .collect::<Vec<_>>();
        assert!(
            states
                .iter()
                .flat_map(|state| state.participants())
                .all(|particle| runs.particles.span().contains(&(particle as usize))),
            "a soft body readback must return only its own elements"
        );
        states
    }
}

pub(crate) struct CompiledSoftCommands {
    pub(crate) body_edits: Vec<SoftBodyEditRecord>,
    pub(crate) edits: Vec<SoftEditRecord>,
}

impl World {
    pub(crate) fn compile_soft_commands(&self, consumption: Consumption) -> CompiledSoftCommands {
        CompiledSoftCommands {
            body_edits: self.soft.compile_body_commands(consumption),
            edits: self.soft.compile_commands(consumption),
        }
    }

    pub(crate) fn upload_soft_commands(&mut self, compiled: &CompiledSoftCommands) {
        let queue = self.backend.gpu.queue().clone();
        if !compiled.body_edits.is_empty() {
            self.backend
                .streams
                .soft
                .body_edits
                .write(&queue, bytemuck::cast_slice(&compiled.body_edits));
        }
        if !compiled.edits.is_empty() {
            self.backend
                .streams
                .soft
                .edits
                .write(&queue, bytemuck::cast_slice(&compiled.edits));
        }
    }
}
