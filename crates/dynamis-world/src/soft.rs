use super::World;
use super::arena::{Cleared, Mirror, Run};
use super::ids::IdSpace;
use dynamis_abi::{
    ELEMENT_PARTICLES, ELEMENT_ROLE_BITS, NO_SLOT, SoftAttachmentInit, SoftAttachmentRecord,
    SoftBodyRecord, SoftElementInit, SoftElementRecord, SoftParticleInit, SoftParticleRecord,
};
use dynamis_model::math::{add, quat_rotate};
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
    pub(crate) const fn new() -> Self {
        Self {
            alive: Vec::new(),
            ids: IdSpace::new(),
            index_of: Vec::new(),
            runs: Vec::new(),
            states: Vec::new(),
            dirty_states: Vec::new(),
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
        self.states[id] = SoftBodyRecord::cleared();
        self.dirty_states.push(handle.id);
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

    pub(crate) fn observe(&mut self, run: Run, records: &[SoftParticleRecord]) {
        for (slot, record) in records.iter().enumerate() {
            self.particles.records_mut()[run.offset as usize + slot] = *record;
        }
    }

    pub(crate) fn observe_elements(&mut self, run: Run, records: &[SoftElementRecord]) {
        for (slot, record) in records.iter().enumerate() {
            self.elements.records_mut()[run.offset as usize + slot] = *record;
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
