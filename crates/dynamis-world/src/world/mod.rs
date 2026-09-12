mod backend;
mod body;
mod clock;
mod colliders;
mod commands;
mod constraint;
mod event;
mod ids;
mod query;
pub(crate) mod query_pool;
mod readback;
mod rows;
mod shape;
pub(crate) mod shape_pool;
pub(crate) mod static_aabb;
mod step;

use crate::dynamics::capacity::{Live, StreamCapacity};
use backend::Backend;
use body::Bodies;
use clock::Clock;
use colliders::ColliderPool;
use constraint::Constraints;
use dynamis_gpu::{GpuBuffer, GpuContext, WarmupBudget, WarmupProgress};
use dynamis_layout::ColliderRecord;
use dynamis_layout::{
    COUNTER_BODIES, COUNTER_BODY_EDITS, COUNTER_BODY_MOVES, COUNTER_CONSTRAINT_COMMANDS,
    COUNTER_CONSTRAINT_MOVES, COUNTER_CONSTRAINTS,
};
use dynamis_model::{BodyHandle, PhysicsConfig};
use event::Events;
use query::Queries;
use shape::Shapes;

pub use readback::{ContactManifold, ContactPoint};

pub struct World {
    config: PhysicsConfig,

    clock: Clock,
    backend: Backend,
    bodies: Bodies,
    colliders: ColliderPool,
    constraints: Constraints,
    shapes: Shapes,
    queries: Queries,
    events: Events,
}

fn flush_pool_range(
    buffers: &crate::dynamics::buffers::WorldBuffers,
    queue: &wgpu::Queue,
    head: Option<(u32, u32)>,
    records: &[ColliderRecord],
    aabbs: &[dynamis_layout::AabbRecord],
    owners: &[u32],
) {
    let Some((start, end)) = head else {
        return;
    };
    debug_assert_eq!((end - start) as usize, records.len());
    if records.is_empty() {
        return;
    }
    buffers.bodies.colliders.write_at(
        queue,
        start as u64 * std::mem::size_of::<ColliderRecord>() as u64,
        bytemuck::cast_slice(records),
    );
    buffers.bodies.aabbs.write_at(
        queue,
        start as u64 * std::mem::size_of::<dynamis_layout::AabbRecord>() as u64,
        bytemuck::cast_slice(aabbs),
    );
    buffers
        .bodies
        .collider_owners
        .write_at(queue, start as u64 * 4, bytemuck::cast_slice(owners));
}

fn contiguous_runs(slots: &[u32]) -> Vec<&[u32]> {
    let mut runs = Vec::new();
    let mut start = 0usize;
    for index in 1..=slots.len() {
        let breaks = index == slots.len() || slots[index] != slots[index - 1] + 1;
        if breaks {
            if index > start {
                runs.push(&slots[start..index]);
            }
            start = index;
        }
    }
    runs
}

fn assert_config(config: &PhysicsConfig) {
    assert!(
        config.contact_margin >= 0.0,
        "contact margin must be non-negative"
    );
    assert!(
        config.position_iterations > 0,
        "position iterations must be strictly positive"
    );
    assert!(
        (0.0..=1.0).contains(&config.relaxation),
        "position relaxation must be within (0, 1]"
    );
    assert!(
        config.broadphase_cell_size > 0.0,
        "broadphase cell size must be strictly positive"
    );
}

impl World {
    pub fn new(gpu: GpuContext, config: PhysicsConfig) -> Self {
        assert_config(&config);
        let shapes = Shapes::new();
        let backend = Backend::new(gpu, &shapes);
        Self {
            config,
            clock: Clock::new(),
            backend,
            bodies: Bodies::new(),
            colliders: ColliderPool::new(),
            constraints: Constraints::new(),
            shapes,
            queries: Queries::new(),
            events: Events::new(),
        }
    }

    pub fn config(&self) -> &PhysicsConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: PhysicsConfig) {
        assert_config(&config);
        self.config = config;
    }

    pub fn set_gravity(&mut self, gravity: [f32; 3]) {
        self.config.gravity = gravity;
    }

    pub fn stream_capacity(&self) -> StreamCapacity {
        self.backend.reservation.streams()
    }

    pub fn bodies(&self) -> &[BodyHandle] {
        &self.bodies.alive
    }

    pub fn count(&self) -> usize {
        self.bodies.alive.len()
    }

    pub(crate) fn live(&self) -> Live {
        Live {
            bodies: self.bodies.alive.len() as u32,
            colliders: self.colliders.live(),
            collider_pool: self.colliders.used(),
            body_ids: self.bodies.ids.len() as u32,
            constraints: self.constraints.alive.len() as u32,
            body_commands: self.bodies.commands.len() as u32,
            constraint_commands: self.constraints.commands.len() as u32,
            queries: self.queries.pending.len() as u32,
        }
    }

    pub(crate) fn flush_rows(&mut self) {
        let queue = self.backend.gpu.queue().clone();
        if self.shapes.dirty {
            self.shapes.pool.upload_pending(
                &queue,
                &self.backend.buffers.shapes.sources,
                &self.backend.buffers.shapes.vertices,
                &self.backend.buffers.shapes.triangles,
                &self.backend.buffers.shapes.nodes,
            );
            self.shapes.dirty = false;
        }
        self.bodies.dirty.sort_unstable();
        self.bodies.dirty.dedup();
        let dirty = std::mem::take(&mut self.bodies.dirty);
        for run in contiguous_runs(&dirty) {
            let descriptors = run
                .iter()
                .map(|slot| self.bodies.descriptors[self.bodies.alive[*slot as usize].id as usize])
                .collect::<Vec<_>>();
            self.backend.buffers.bodies.descriptors.write_at(
                &queue,
                run[0] as u64 * self.backend.buffers.descriptor_row(),
                bytemuck::cast_slice(&descriptors),
            );
        }
        self.upload_colliders(&queue, &dirty);
        self.constraints.dirty.sort_unstable();
        self.constraints.dirty.dedup();
        for run in contiguous_runs(&std::mem::take(&mut self.constraints.dirty)) {
            let first = run[0];
            let records = run
                .iter()
                .map(|slot| self.constraints.records[*slot as usize])
                .collect::<Vec<_>>();
            self.backend.buffers.constraints.descriptors.write_at(
                &queue,
                first as u64 * self.backend.buffers.constraint_row(),
                bytemuck::cast_slice(&records),
            );
        }
    }

    fn upload_colliders(&mut self, queue: &wgpu::Queue, dirty: &[u32]) {
        for cleared in self.colliders.take_cleared() {
            let range = cleared.offset as usize..(cleared.offset + cleared.len) as usize;
            let owners = vec![dynamis_layout::NO_BODY; cleared.len as usize];
            flush_pool_range(
                &self.backend.buffers,
                queue,
                Some((cleared.offset, cleared.offset + cleared.len)),
                &self.colliders.records()[range.clone()],
                &self.colliders.aabbs()[range],
                &owners,
            );
        }
        let mut placed = Vec::with_capacity(dirty.len());
        for slot in dirty {
            let id = self.bodies.alive[*slot as usize].id;
            let run = self
                .colliders
                .run_of(id)
                .unwrap_or_else(|| panic!("row {slot} of body {id} has no collider run"));
            placed.push((run, *slot));
        }
        placed.sort_unstable_by_key(|(run, _)| run.offset);
        let mut records = Vec::new();
        let mut aabbs = Vec::new();
        let mut owners = Vec::new();
        let mut head: Option<(u32, u32)> = None;
        for (run, row) in placed {
            let contiguous = head.is_some_and(|(_, end)| end == run.offset);
            if !contiguous {
                flush_pool_range(
                    &self.backend.buffers,
                    queue,
                    head,
                    &records,
                    &aabbs,
                    &owners,
                );
                head = Some((run.offset, run.offset));
                records.clear();
                aabbs.clear();
                owners.clear();
            }
            let (_, end) = head.expect("a pool range is open");
            head = Some((head.expect("a pool range is open").0, end + run.len));
            let range = run.offset as usize..(run.offset + run.len) as usize;
            records.extend_from_slice(&self.colliders.records()[range.clone()]);
            aabbs.extend_from_slice(&self.colliders.aabbs()[range]);
            owners.extend(std::iter::repeat_n(row, run.len as usize));
        }
        flush_pool_range(
            &self.backend.buffers,
            queue,
            head,
            &records,
            &aabbs,
            &owners,
        );
    }

    pub(crate) fn write_declared_counters(&self) {
        let queue = self.backend.gpu.queue();
        let declared = [
            (COUNTER_BODIES, self.bodies.alive.len() as u32),
            (COUNTER_CONSTRAINTS, self.constraints.alive.len() as u32),
            (COUNTER_BODY_EDITS, self.bodies.last_edits),
            (COUNTER_BODY_MOVES, self.bodies.last_moves),
            (COUNTER_CONSTRAINT_COMMANDS, self.constraints.last_commands),
            (COUNTER_CONSTRAINT_MOVES, self.constraints.last_moves),
        ];
        for (slot, value) in declared {
            self.backend.buffers.counters.write_at(
                queue,
                slot as u64 * dynamis_layout::COUNTER_STRIDE,
                bytemuck::cast_slice(&[value]),
            );
        }
    }

    pub fn state_buffer(&self) -> &GpuBuffer {
        &self.backend.buffers.bodies.states
    }

    pub fn collider_buffer(&self) -> &GpuBuffer {
        &self.backend.buffers.bodies.colliders
    }

    pub fn gpu(&self) -> &GpuContext {
        &self.backend.gpu
    }

    pub fn warmup(&self, budget: WarmupBudget) -> WarmupProgress {
        self.backend.gpu.warmup(budget)
    }

    pub fn is_warm(&self) -> bool {
        self.backend.gpu.is_warm()
    }

    pub(crate) fn event_slot_of(&self, step: u64) -> u32 {
        (step % crate::dynamics::buffers::EVENT_SLOTS as u64) as u32
    }
}
