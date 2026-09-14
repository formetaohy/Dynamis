use super::World;
use super::arena::{Arena, Run};
use super::ids::IdSpace;
use dynamis_abi::{
    ELEMENT_PARTICLES, ELEMENT_ROLE_BITS, NO_SLOT, SoftAttachmentInit, SoftAttachmentRecord,
    SoftBodyRecord, SoftElementInit, SoftElementRecord, SoftParticleInit, SoftParticleRecord,
};
use dynamis_math::{add, quat_rotate};
use dynamis_model::{BodyHandle, SoftBodyDesc, SoftBodyHandle, SoftElement, SoftElementState};
use dynamis_soft::SoftStreams;

#[derive(Clone, Copy)]
pub(crate) struct SoftRuns {
    particles: Run,
    elements: Run,
    attachments: Run,
    adjacency: Run,
    strength: bool,
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

#[derive(Clone)]
pub(crate) struct SoftBodies {
    alive: Vec<SoftBodyHandle>,
    ids: IdSpace,
    index_of: Vec<u32>,
    runs: Vec<SoftRuns>,
    states: Vec<SoftBodyRecord>,
    dirty_states: Vec<u32>,
    particles: Vec<SoftParticleRecord>,
    elements: Vec<SoftElementRecord>,
    attachments: Vec<SoftAttachmentRecord>,
    attachment_refs: Vec<u32>,
    adjacency: Vec<u32>,
    particle_arena: Arena,
    element_arena: Arena,
    attachment_arena: Arena,
    adjacency_arena: Arena,
    pending_particles: Vec<Run>,
    pending_elements: Vec<Run>,
    pending_attachments: Vec<Run>,
    pending_adjacency: Vec<Run>,
    pub(crate) uploaded: bool,
}

impl SoftBodies {
    pub(crate) const fn new() -> Self {
        Self {
            alive: Vec::new(),
            ids: IdSpace::new(),
            index_of: Vec::new(),
            runs: Vec::new(),
            states: Vec::new(),
            dirty_states: Vec::new(),
            particles: Vec::new(),
            elements: Vec::new(),
            attachments: Vec::new(),
            attachment_refs: Vec::new(),
            adjacency: Vec::new(),
            particle_arena: Arena::new(),
            element_arena: Arena::new(),
            attachment_arena: Arena::new(),
            adjacency_arena: Arena::new(),
            pending_particles: Vec::new(),
            pending_elements: Vec::new(),
            pending_attachments: Vec::new(),
            pending_adjacency: Vec::new(),
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
            self.particles.len() as u32,
            self.elements.len() as u32,
            self.attachments.len() as u32,
            self.adjacency.len() as u32,
        )
    }

    pub(crate) fn attachment_refs(&self, id: u32) -> u32 {
        self.attachment_refs.get(id as usize).copied().unwrap_or(0)
    }

    pub(crate) fn run_of(&self, handle: SoftBodyHandle) -> Run {
        self.runs_of(handle).particles
    }

    pub(crate) fn wake_all(&mut self) {
        let states = &mut self.states;
        let dirty = &mut self.dirty_states;
        for handle in &self.alive {
            states[handle.id as usize] = SoftBodyRecord::awake();
            dirty.push(handle.id);
        }
    }

    pub(crate) fn runs_of(&self, handle: SoftBodyHandle) -> SoftRuns {
        self.validate(handle);
        self.runs[handle.id as usize]
    }

    fn validate(&self, handle: SoftBodyHandle) {
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
    }

    fn take_element_run(&mut self, elements: usize) -> Run {
        if elements == 0 {
            return Run::EMPTY;
        }
        self.element_arena.take(elements as u32)
    }

    fn take_adjacency_run(&mut self, entries: usize) -> Run {
        if entries == 0 {
            return Run::EMPTY;
        }
        self.adjacency_arena.take(entries as u32)
    }

    fn take_attachment_run(&mut self, attachments: usize) -> Run {
        if attachments == 0 {
            return Run::EMPTY;
        }
        self.attachment_arena.take(attachments as u32)
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
        self.states
            .resize(self.ids.len(), SoftBodyRecord::cleared());
        self.states[id as usize] = SoftBodyRecord::awake();
        self.dirty_states.push(id);
        let particles = self.particle_arena.take(desc.particles.len() as u32);
        let elements = self.take_element_run(desc.elements.len());
        let attachments = self.take_attachment_run(desc.attachments.len());
        let adjacency = self.take_adjacency_run(desc.elements.len() * ELEMENT_PARTICLES as usize);
        self.particles.resize(
            self.particle_arena.used() as usize,
            SoftParticleRecord::cleared(),
        );
        self.elements.resize(
            self.element_arena.used() as usize,
            SoftElementRecord::cleared(),
        );
        self.attachments.resize(
            self.attachment_arena.used() as usize,
            SoftAttachmentRecord::cleared(),
        );
        self.adjacency
            .resize(self.adjacency_arena.used() as usize, u32::MAX);
        let neighbours = assemble_adjacency(
            &mut self.adjacency,
            desc.particles.len(),
            &desc.elements,
            adjacency.offset,
            elements.offset,
        );
        for (slot, local) in desc.particles.iter().enumerate() {
            let position = add(desc.position, quat_rotate(desc.orientation, *local));
            let (offset, count) = neighbours[slot];
            self.particles[particles.offset as usize + slot] =
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
            self.elements[elements.offset as usize + slot] =
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
            self.attachments[attachments.offset as usize + slot] =
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
        self.pending_particles.push(particles);
        self.pending_elements.push(elements);
        self.pending_attachments.push(attachments);
        self.pending_adjacency.push(adjacency);
        handle
    }

    pub(crate) fn remove(&mut self, handle: SoftBodyHandle) {
        self.validate(handle);
        let id = handle.id as usize;
        let runs = self.runs[id];
        self.states[id] = SoftBodyRecord::cleared();
        self.dirty_states.push(handle.id);
        for index in runs.particles.span() {
            self.particles[index] = SoftParticleRecord::cleared();
        }
        for index in runs.elements.span() {
            self.elements[index] = SoftElementRecord::cleared();
        }
        for index in runs.attachments.span() {
            let released = self.attachments[index];
            self.release_attachment(released.body_id);
            self.attachments[index] = SoftAttachmentRecord::cleared();
        }
        self.particle_arena.release(runs.particles);
        if runs.elements.len > 0 {
            self.element_arena.release(runs.elements);
        }
        if runs.attachments.len > 0 {
            self.attachment_arena.release(runs.attachments);
        }
        if runs.adjacency.len > 0 {
            self.adjacency_arena.release(runs.adjacency);
        }
        self.runs[id] = SoftRuns::EMPTY;
        let slot = self.index_of[id] as usize;
        self.alive.swap_remove(slot);
        if slot < self.alive.len() {
            self.index_of[self.alive[slot].id as usize] = slot as u32;
        }
        self.index_of[id] = u32::MAX;
        self.ids.release(handle.id);
        self.pending_particles.push(runs.particles);
        if runs.elements.len > 0 {
            self.pending_elements.push(runs.elements);
        }
        if runs.attachments.len > 0 {
            self.pending_attachments.push(runs.attachments);
        }
        if runs.adjacency.len > 0 {
            self.pending_adjacency.push(runs.adjacency);
        }
    }

    pub(crate) fn upload(&mut self, queue: &wgpu::Queue, streams: &SoftStreams) {
        if !self.dirty_states.is_empty() {
            self.uploaded = true;
            let stride = streams.bodies.stride();
            for id in self.dirty_states.drain(..) {
                streams.bodies.write_at(
                    queue,
                    u64::from(id) * stride,
                    bytemuck::bytes_of(&self.states[id as usize]),
                );
            }
        }
        let particle_stride = streams.particles.stride();
        for run in self.pending_particles.drain(..) {
            if run.len == 0 {
                continue;
            }
            streams.particles.write_at(
                queue,
                run.offset as u64 * particle_stride,
                bytemuck::cast_slice(&self.particles[run.span()]),
            );
        }
        let element_stride = streams.elements.stride();
        for run in self.pending_elements.drain(..) {
            if run.len == 0 {
                continue;
            }
            streams.elements.write_at(
                queue,
                run.offset as u64 * element_stride,
                bytemuck::cast_slice(&self.elements[run.span()]),
            );
        }
        let attachment_stride = streams.attachments.stride();
        for run in self.pending_attachments.drain(..) {
            if run.len == 0 {
                continue;
            }
            streams.attachments.write_at(
                queue,
                run.offset as u64 * attachment_stride,
                bytemuck::cast_slice(&self.attachments[run.span()]),
            );
        }
        for run in self.pending_adjacency.drain(..) {
            if run.len == 0 {
                continue;
            }
            streams.adjacency.write_at(
                queue,
                run.offset as u64 * 4,
                bytemuck::cast_slice(&self.adjacency[run.span()]),
            );
        }
    }

    pub(crate) fn observe(&mut self, run: Run, records: &[SoftParticleRecord]) {
        for (slot, record) in records.iter().enumerate() {
            self.particles[run.offset as usize + slot] = *record;
        }
    }

    pub(crate) fn observe_elements(&mut self, run: Run, records: &[SoftElementRecord]) {
        for (slot, record) in records.iter().enumerate() {
            self.elements[run.offset as usize + slot] = *record;
        }
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

    pub fn soft_bodies(&self) -> &[SoftBodyHandle] {
        self.soft.bodies()
    }

    pub fn soft_body_count(&self) -> usize {
        self.soft.count()
    }

    pub fn soft_body_positions(&mut self, handle: SoftBodyHandle) -> Vec<[f32; 3]> {
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
        let positions = records
            .iter()
            .map(|record| [record.position[0], record.position[1], record.position[2]])
            .collect();
        self.soft.observe(run, &records);
        positions
    }

    pub fn soft_body_elements(&mut self, handle: SoftBodyHandle) -> Vec<SoftElementState> {
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
        self.soft.observe_elements(runs.elements, &records);
        states
    }
}
