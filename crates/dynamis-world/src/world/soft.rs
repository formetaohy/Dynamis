use super::World;
use super::arena::{Arena, Run};
use super::ids::IdSpace;
use crate::dynamics::soft::SoftStreams;
use dynamis_layout::{SoftLinkRecord, SoftParticleInit, SoftParticleRecord};
use dynamis_math::{add, quat_rotate};
use dynamis_model::{SoftBodyDesc, SoftBodyHandle};

#[derive(Clone, Copy)]
struct SoftRuns {
    particles: Run,
    links: Run,
    adjacency: Run,
}

impl SoftRuns {
    const EMPTY: Self = Self {
        particles: Run::EMPTY,
        links: Run::EMPTY,
        adjacency: Run::EMPTY,
    };
}

pub(crate) struct SoftBodies {
    alive: Vec<SoftBodyHandle>,
    ids: IdSpace,
    index_of: Vec<u32>,
    runs: Vec<SoftRuns>,
    particles: Vec<SoftParticleRecord>,
    links: Vec<SoftLinkRecord>,
    adjacency: Vec<u32>,
    particle_arena: Arena,
    link_arena: Arena,
    adjacency_arena: Arena,
    pending_particles: Vec<Run>,
    pending_links: Vec<Run>,
    pending_adjacency: Vec<Run>,
}

impl SoftBodies {
    pub(crate) const fn new() -> Self {
        Self {
            alive: Vec::new(),
            ids: IdSpace::new(),
            index_of: Vec::new(),
            runs: Vec::new(),
            particles: Vec::new(),
            links: Vec::new(),
            adjacency: Vec::new(),
            particle_arena: Arena::new(),
            link_arena: Arena::new(),
            adjacency_arena: Arena::new(),
            pending_particles: Vec::new(),
            pending_links: Vec::new(),
            pending_adjacency: Vec::new(),
        }
    }

    pub(crate) fn count(&self) -> usize {
        self.alive.len()
    }

    pub(crate) fn bodies(&self) -> &[SoftBodyHandle] {
        &self.alive
    }

    pub(crate) fn used(&self) -> (u32, u32, u32) {
        (
            self.particles.len() as u32,
            self.links.len() as u32,
            self.adjacency.len() as u32,
        )
    }

    pub(crate) fn run_of(&self, handle: SoftBodyHandle) -> Run {
        self.validate(handle);
        self.runs[handle.id as usize].particles
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

    fn take_link_run(&mut self, links: usize) -> Run {
        if links == 0 {
            return Run::EMPTY;
        }
        self.link_arena.take(links as u32)
    }

    fn take_adjacency_run(&mut self, entries: usize) -> Run {
        if entries == 0 {
            return Run::EMPTY;
        }
        self.adjacency_arena.take(entries as u32)
    }

    pub(crate) fn spawn(&mut self, desc: &SoftBodyDesc) -> SoftBodyHandle {
        let (id, generation) = self.ids.acquire();
        self.grow_to(id);
        let particles = self.particle_arena.take(desc.particles.len() as u32);
        let links = self.take_link_run(desc.links.len());
        let adjacency = self.take_adjacency_run(desc.links.len() * 2);
        self.particles.resize(
            self.particle_arena.used() as usize,
            SoftParticleRecord::cleared(),
        );
        self.links
            .resize(self.link_arena.used() as usize, SoftLinkRecord::cleared());
        self.adjacency
            .resize(self.adjacency_arena.used() as usize, u32::MAX);
        let neighbours = assemble_adjacency(
            &mut self.adjacency,
            desc.particles.len(),
            &desc.links,
            adjacency.offset,
            links.offset,
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
                    neighbour_offset: offset,
                    neighbour_count: count,
                    owner: id,
                    generation,
                });
        }
        for (slot, link) in desc.links.iter().enumerate() {
            self.links[links.offset as usize + slot] = SoftLinkRecord::build(
                particles.offset + link[0],
                particles.offset + link[1],
                particle_distance(
                    &desc.particles[link[0] as usize],
                    &desc.particles[link[1] as usize],
                ),
            );
        }
        self.runs[id as usize] = SoftRuns {
            particles,
            links,
            adjacency,
        };
        let handle = SoftBodyHandle { id, generation };
        self.index_of[id as usize] = self.alive.len() as u32;
        self.alive.push(handle);
        self.pending_particles.push(particles);
        self.pending_links.push(links);
        self.pending_adjacency.push(adjacency);
        handle
    }

    pub(crate) fn remove(&mut self, handle: SoftBodyHandle) {
        self.validate(handle);
        let id = handle.id as usize;
        let runs = self.runs[id];
        for index in runs.particles.span() {
            self.particles[index] = SoftParticleRecord::cleared();
        }
        for index in runs.links.span() {
            self.links[index] = SoftLinkRecord::cleared();
        }
        self.particle_arena.release(runs.particles);
        if runs.links.len > 0 {
            self.link_arena.release(runs.links);
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
        if runs.links.len > 0 {
            self.pending_links.push(runs.links);
        }
        if runs.adjacency.len > 0 {
            self.pending_adjacency.push(runs.adjacency);
        }
    }

    pub(crate) fn upload(&mut self, queue: &wgpu::Queue, streams: &SoftStreams) {
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
        let link_stride = streams.links.stride();
        for run in self.pending_links.drain(..) {
            if run.len == 0 {
                continue;
            }
            streams.links.write_at(
                queue,
                run.offset as u64 * link_stride,
                bytemuck::cast_slice(&self.links[run.span()]),
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
}

fn assemble_adjacency(
    adjacency: &mut [u32],
    particles: usize,
    links: &[[u32; 2]],
    adjacency_base: u32,
    link_base: u32,
) -> Vec<(u32, u32)> {
    let mut counts = vec![0u32; particles];
    for link in links {
        counts[link[0] as usize] += 1;
        counts[link[1] as usize] += 1;
    }
    let mut ranges = Vec::with_capacity(particles);
    let mut cursor = Vec::with_capacity(particles);
    let mut offset = adjacency_base;
    for count in counts {
        ranges.push((offset, count));
        cursor.push(offset);
        offset += count;
    }
    for (slot, link) in links.iter().enumerate() {
        let index = link_base + slot as u32;
        adjacency[cursor[link[0] as usize] as usize] = index;
        cursor[link[0] as usize] += 1;
        adjacency[cursor[link[1] as usize] as usize] = index;
        cursor[link[1] as usize] += 1;
    }
    ranges
}

fn particle_distance(first: &[f32; 3], second: &[f32; 3]) -> f32 {
    let dx = second[0] - first[0];
    let dy = second[1] - first[1];
    let dz = second[2] - first[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

impl World {
    pub fn add_soft_body(&mut self, desc: SoftBodyDesc) -> SoftBodyHandle {
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
        self.apply_plan();
        self.flush_rows();
        let run = self.soft.run_of(handle);
        let stride = self.backend.streams.soft.particles.stride();
        let bytes = run.len as u64 * stride;
        let buffer = self.backend.streams.soft.particles.buffer().clone();
        let raw = self.read_regions(
            "soft body particles",
            &[(&buffer, run.offset as u64 * stride, bytes)],
        );
        let records = dynamis_layout::decode::<SoftParticleRecord>(&raw);
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
}
