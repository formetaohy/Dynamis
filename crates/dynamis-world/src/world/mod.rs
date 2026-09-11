mod backend;
mod body;
mod clock;
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
use constraint::Constraints;
use dynamis_gpu::{GpuBuffer, GpuContext, WarmupBudget, WarmupProgress};
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
    constraints: Constraints,
    shapes: Shapes,
    queries: Queries,
    events: Events,
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
        (0.0..=1.0).contains(&config.relaxation),
        "position relaxation must be within (0, 1]"
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
            constraints: self.constraints.alive.len() as u32,
            body_commands: self.bodies.commands.len() as u32,
            constraint_commands: self.constraints.commands.len() as u32,
            queries: self.queries.pending.len() as u32,
        }
    }

    pub(crate) fn flush_rows(&mut self) {
        let queue = self.backend.gpu.queue();
        if self.shapes.dirty {
            self.shapes.pool.upload_pending(
                queue,
                &self.backend.buffers.shapes.sources,
                &self.backend.buffers.shapes.vertices,
                &self.backend.buffers.shapes.triangles,
                &self.backend.buffers.shapes.nodes,
            );
            self.shapes.dirty = false;
        }
        self.bodies.dirty.sort_unstable();
        self.bodies.dirty.dedup();
        for run in contiguous_runs(&std::mem::take(&mut self.bodies.dirty)) {
            let mut colliders = Vec::new();
            let mut descriptors = Vec::new();
            let mut aabbs = Vec::new();
            for slot in run {
                let id = self.bodies.alive[*slot as usize].id as usize;
                colliders.extend_from_slice(&self.collider_block_of(id));
                descriptors.push(self.bodies.descriptors[id]);
                aabbs.extend_from_slice(&self.aabb_block_of(id));
            }
            let first = run[0];

            self.backend.buffers.bodies.colliders.write_at(
                queue,
                first as u64 * self.backend.buffers.collider_row(),
                bytemuck::cast_slice(&colliders),
            );
            self.backend.buffers.bodies.descriptors.write_at(
                queue,
                first as u64 * self.backend.buffers.descriptor_row(),
                bytemuck::cast_slice(&descriptors),
            );
            self.backend.buffers.bodies.aabbs.write_at(
                queue,
                first as u64 * self.backend.buffers.aabb_row(),
                bytemuck::cast_slice(&aabbs),
            );
        }
        self.constraints.dirty.sort_unstable();
        self.constraints.dirty.dedup();
        for run in contiguous_runs(&std::mem::take(&mut self.constraints.dirty)) {
            let first = run[0];
            let records = run
                .iter()
                .map(|slot| self.constraints.records[*slot as usize])
                .collect::<Vec<_>>();
            self.backend.buffers.constraints.descriptors.write_at(
                queue,
                first as u64 * self.backend.buffers.constraint_row(),
                bytemuck::cast_slice(&records),
            );
        }
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
