mod body;
mod clock;
mod commands;
mod constraint;
mod device;
mod event;
mod query;
mod readback;
mod shape;
mod step;

use crate::capacity::{Live, StreamCapacity};
use body::Bodies;
use clock::Clock;
use constraint::Constraints;
use device::Device;
use dynamis_gpu::{GpuBuffer, GpuContext};
use dynamis_layout::{
    COUNTER_BODIES, COUNTER_BODY_COMMANDS, COUNTER_CONSTRAINT_COMMANDS, COUNTER_CONSTRAINTS,
};
use dynamis_model::{BodyHandle, PhysicsConfig};
use event::Events;
use query::Queries;
use shape::Shapes;

pub use readback::{ContactManifold, ContactPoint};

pub struct Simulation {
    config: PhysicsConfig,
    /// The host-side handle space; bounds ids, never device work.
    slots: usize,
    clock: Clock,
    device: Device,
    bodies: Bodies,
    constraints: Constraints,
    shapes: Shapes,
    queries: Queries,
    events: Events,
}

/// Splits a sorted, de-duplicated slot list into maximal contiguous runs, so a
/// stripe of dirty slots uploads as one write instead of one call per row.
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
    assert!(
        (0.0..=1.0).contains(&config.tempering),
        "solver tempering must be within (0, 1]"
    );
}

impl Simulation {
    /// A world able to address `slots` body and constraint ids. Stream lanes are
    /// sized from what the device measures, so the first plan serves a minimum and
    /// every step boundary widens or narrows them to the observed demand.
    pub fn new(gpu: GpuContext, slots: usize, config: PhysicsConfig) -> Self {
        assert!(slots > 0, "simulation slot count must be positive");
        assert!(
            u32::try_from(slots).is_ok(),
            "simulation slot count exceeds the handle space"
        );
        assert_config(&config);
        let shapes = Shapes::new();
        let device = Device::new(gpu, &shapes);
        Self {
            config,
            slots,
            clock: Clock::new(),
            device,
            bodies: Bodies::new(slots),
            constraints: Constraints::new(slots),
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

    /// The number of body ids this world can address.
    pub fn capacity(&self) -> usize {
        self.slots
    }

    pub fn stream_capacity(&self) -> StreamCapacity {
        self.device.reservation.streams()
    }

    pub fn bodies(&self) -> &[BodyHandle] {
        &self.bodies.alive
    }

    pub fn count(&self) -> usize {
        self.bodies.alive.len()
    }

    /// Raises the id space. Device storage follows live bodies, so this alone never
    /// reallocates anything on the card.
    pub fn grow(&mut self, slots: usize) {
        assert!(
            u32::try_from(slots).is_ok(),
            "simulation slot count exceeds the handle space"
        );
        assert!(
            slots >= self.bodies.alive.len(),
            "grow slot count must not drop below the live body count"
        );
        if slots <= self.slots {
            return;
        }
        let base = self.slots;
        self.slots = slots;
        self.bodies
            .free_ids
            .extend((base..slots).rev().map(|id| id as u32));
        self.constraints
            .free_ids
            .extend((base..slots).rev().map(|id| id as u32));
        self.bodies.resize(slots);
        self.constraints.resize(slots);
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

    /// Publishes every host-side row edit the world accumulated since the last step.
    ///
    /// Host mutations never touch the device directly, so this is the only place that
    /// writes rows and the only place that may need to reallocate first.
    pub(crate) fn flush_rows(&mut self) {
        let queue = self.device.gpu.queue();
        if self.shapes.dirty {
            self.shapes.pool.upload_pending(
                queue,
                &self.device.buffers.shapes.sources,
                &self.device.buffers.shapes.vertices,
                &self.device.buffers.shapes.triangles,
                &self.device.buffers.shapes.nodes,
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

            self.device.buffers.bodies.colliders.write_at(
                queue,
                first as u64 * self.device.buffers.collider_row(),
                bytemuck::cast_slice(&colliders),
            );
            self.device.buffers.bodies.descriptors.write_at(
                queue,
                first as u64 * self.device.buffers.descriptor_row(),
                bytemuck::cast_slice(&descriptors),
            );
            self.device.buffers.bodies.aabbs.write_at(
                queue,
                first as u64 * self.device.buffers.aabb_row(),
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
            self.device.buffers.constraints.descriptors.write_at(
                queue,
                first as u64 * self.device.buffers.constraint_row(),
                bytemuck::cast_slice(&records),
            );
        }
    }

    pub(crate) fn write_declared_counters(&self) {
        let queue = self.device.gpu.queue();
        let declared = [
            (COUNTER_BODIES, self.bodies.alive.len() as u32),
            (COUNTER_CONSTRAINTS, self.constraints.alive.len() as u32),
            (COUNTER_BODY_COMMANDS, self.bodies.last_commands),
            (COUNTER_CONSTRAINT_COMMANDS, self.constraints.last_commands),
        ];
        for (slot, value) in declared {
            self.device.buffers.counters.write_at(
                queue,
                slot as u64 * dynamis_layout::COUNTER_STRIDE,
                bytemuck::cast_slice(&[value]),
            );
        }
    }

    /// The device-owned kinematic state of every live body slot.
    pub fn state_buffer(&self) -> &GpuBuffer {
        &self.device.buffers.bodies.states
    }

    /// The device-owned collider rows of every live body slot.
    pub fn collider_buffer(&self) -> &GpuBuffer {
        &self.device.buffers.bodies.colliders
    }

    pub fn gpu(&self) -> &GpuContext {
        &self.device.gpu
    }

    /// Which event ring segment the step being encoded writes into.
    pub(crate) fn event_slot_of(&self, step: u64) -> u32 {
        (step % crate::buffers::EVENT_SLOTS as u64) as u32
    }
}
