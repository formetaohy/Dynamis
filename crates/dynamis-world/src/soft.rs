use super::World;
use super::arena::{Cleared, Mirror, Run};
use super::commands::Consumption;
use super::ids::IdSpace;
use super::journal::EditJournal;
use dynamis_abi::{
    ELEMENT_PARTICLES, ELEMENT_ROLE_BITS, NO_SLOT, SoftAttachmentInit, SoftAttachmentRecord,
    SoftBodyEditRecord, SoftBodyRecord, SoftEditRecord, SoftElementInit, SoftElementRecord,
    SoftParticleInit, SoftParticleRecord,
};
use dynamis_model::math::{add, mul, quat_rotate};
use dynamis_model::{
    BodyHandle, SoftBodyDesc, SoftBodyHandle, SoftElement, SoftElementState, SoftParticleState,
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
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum SoftBodyCommand {
    Accelerate([f32; 3]),
    Wake,
}

#[derive(Clone)]
pub(crate) struct SoftBodies {
    alive: Vec<SoftBodyHandle>,
    ids: IdSpace,
    index_of: Vec<u32>,
    runs: Vec<SoftRuns>,
    masses: Vec<f32>,
    created: Vec<(u32, SoftBodyRecord)>,
    body_commands: EditJournal<u32, SoftBodyCommand>,
    commands: EditJournal<u32, SoftCommand>,
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

impl SoftBodies {
    pub(crate) fn new() -> Self {
        Self {
            alive: Vec::new(),
            ids: IdSpace::new(),
            index_of: Vec::new(),
            runs: Vec::new(),
            masses: Vec::new(),
            created: Vec::new(),
            body_commands: EditJournal::new(),
            commands: EditJournal::new(),
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
        self.alive.len()
    }

    pub(crate) fn ids_len(&self) -> usize {
        self.ids.len()
    }

    pub(crate) fn carries_strength(&self) -> bool {
        self.alive
            .iter()
            .any(|handle| self.runs[handle.id as usize].strength)
    }

    pub(crate) fn bodies(&self) -> &[SoftBodyHandle] {
        &self.alive
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
        !self.created.is_empty() || !self.body_commands.is_empty() || !self.commands.is_empty()
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

    fn edit(&mut self, handle: SoftBodyHandle, particle: u32, command: SoftCommand) {
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
        }
        self.body_commands.push(handle.id, SoftBodyCommand::Wake);
        self.commands.push(slot, command);
    }

    pub(crate) fn wake_all(&mut self) {
        for handle in &self.alive {
            self.body_commands.push(handle.id, SoftBodyCommand::Wake);
        }
    }

    pub(crate) fn runs_of(&self, handle: SoftBodyHandle) -> SoftRuns {
        self.validate(handle);
        self.runs[handle.id as usize]
    }

    pub(crate) fn is_alive(&self, handle: SoftBodyHandle) -> bool {
        (handle.id as usize) < self.ids.len()
            && self.ids.generation(handle.id) == handle.generation
            && self.index_of[handle.id as usize] != u32::MAX
    }

    pub(crate) fn validate(&self, handle: SoftBodyHandle) {
        let id = handle.id as usize;
        if id >= self.ids.len() {
            panic!("soft body handle {handle:?} is out of range");
        }
        if self.ids.generation(handle.id) != handle.generation {
            panic!("soft body handle {handle:?} is stale");
        }
        if self.index_of[id] == u32::MAX {
            panic!("soft body handle {handle:?} is not alive");
        }
    }

    fn grow_to(&mut self, id: u32) {
        if self.runs.len() > id as usize {
            return;
        }
        self.index_of.resize(id as usize + 1, u32::MAX);
        self.runs.resize(id as usize + 1, SoftRuns::EMPTY);
        self.masses.resize(id as usize + 1, 0.0);
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
        let (id, generation) = self.ids.acquire();
        self.grow_to(id);
        self.created.push((id, SoftBodyRecord::awake(desc.filter)));
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
                    generation,
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
        self.runs[id as usize] = SoftRuns {
            particles,
            elements,
            attachments,
            adjacency,
            strength: desc.carries_strength(),
        };
        let handle = SoftBodyHandle { id, generation };
        self.index_of[id as usize] = self.alive.len() as u32;
        self.alive.push(handle);
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
        self.created.push((handle.id, SoftBodyRecord::cleared()));
        for slot in runs.attachments.span() {
            self.release_attachment(self.attachments.records()[slot].body_id);
        }
        self.particles.retire(runs.particles);
        self.elements.retire(runs.elements);
        self.attachments.retire(runs.attachments);
        self.adjacency.retire(runs.adjacency);
        self.runs[id] = SoftRuns::EMPTY;
        let slot = self.index_of[id] as usize;
        self.alive.swap_remove(slot);
        if slot < self.alive.len() {
            self.index_of[self.alive[slot].id as usize] = slot as u32;
        }
        self.index_of[id] = u32::MAX;
        self.ids.release(handle.id);
    }

    pub(crate) fn consume(&mut self) {
        self.body_commands.clear();
        self.commands.clear();
    }

    pub(crate) fn upload(&mut self, queue: &wgpu::Queue, streams: &SoftStreams) {
        if !self.created.is_empty() {
            self.uploaded = true;
            let stride = streams.bodies.stride();
            for (id, record) in self.created.drain(..) {
                streams
                    .bodies
                    .write_at(queue, u64::from(id) * stride, bytemuck::bytes_of(&record));
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
    pub fn add_soft_body(&mut self, desc: SoftBodyDesc) -> SoftBodyHandle {
        desc.assert_attachments();
        for attachment in &desc.attachments {
            self.validate(attachment.body());
        }
        self.soft.spawn(&desc)
    }

    pub fn remove_soft_body(&mut self, handle: SoftBodyHandle) {
        self.soft.remove(handle);
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
        self.soft
            .edit(handle, particle, SoftCommand::InverseMass(inverse_mass));
    }

    pub fn set_soft_particle_radius(&mut self, handle: SoftBodyHandle, particle: u32, radius: f32) {
        assert!(radius >= 0.0, "a soft particle radius must be non-negative");
        self.soft
            .edit(handle, particle, SoftCommand::Radius(radius));
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
        self.soft
            .edit(handle, particle, SoftCommand::Friction(friction));
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
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
        let live = self.live();
        self.apply_plan(&live);
        self.flush_rows();
        let run = self.soft.run_of(handle);
        let stride = self.backend.streams.soft.particles.stride();
        let bytes = run.len as u64 * stride;
        let buffer = self.backend.streams.soft.particles.buffer().clone();
        let raw = self.read_regions(
            "soft body particles",
            &[(&buffer, run.offset as u64 * stride, bytes)],
        );
        let records = dynamis_abi::decode::<SoftParticleRecord>(&raw);
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
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
        let live = self.live();
        self.apply_plan(&live);
        self.flush_rows();
        let runs = self.soft.runs_of(handle);
        if runs.elements.len == 0 {
            return Vec::new();
        }
        let stride = self.backend.streams.soft.elements.stride();
        let bytes = runs.elements.len as u64 * stride;
        let buffer = self.backend.streams.soft.elements.buffer().clone();
        let raw = self.read_regions(
            "soft body elements",
            &[(&buffer, runs.elements.offset as u64 * stride, bytes)],
        );
        let records = dynamis_abi::decode::<SoftElementRecord>(&raw);
        let states = records
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
