use super::Simulation;
use crate::pipeline::FrameParams;
use crate::simulation::commands::{CompiledBodyEdits, CompiledConstraintEdits, MoveLanes, RowMove};
use dynamis_gpu::{GpuBuffer, GpuReadback};
use dynamis_layout::{COUNTER_ACTIVE, QueryResultRecord, SimParamsRecord};
use std::mem::size_of;

impl Simulation {
    pub fn set_time_scale(&mut self, time_scale: f32) {
        assert!(time_scale > 0.0, "time scale must be strictly positive");
        self.clock.time_scale = time_scale;
    }

    pub fn update(&mut self, real_dt: f32, sub_dt: f32, max_substeps: u32) {
        assert!(real_dt >= 0.0, "real dt must be non-negative");
        assert!(sub_dt > 0.0, "sub dt must be strictly positive");
        assert!(max_substeps > 0, "max substeps must be positive");
        self.clock.sub_dt = sub_dt;
        self.clock.accumulator += real_dt * self.clock.time_scale;
        let mut steps = 0;
        while self.clock.accumulator >= sub_dt && steps < max_substeps {
            self.step(sub_dt);
            self.clock.accumulator -= sub_dt;
            steps += 1;
        }
        if steps == max_substeps {
            self.clock.accumulator %= sub_dt;
        }
    }

    pub fn interpolation_alpha(&self) -> f32 {
        (self.clock.accumulator / self.clock.sub_dt).clamp(0.0, 1.0)
    }

    pub fn step(&mut self, dt: f32) {
        assert!(dt > 0.0, "timestep must be strictly positive");
        self.device.gpu.assert_alive();
        let step = self.clock.step;
        self.collect_readbacks();
        self.apply_plan();
        self.flush_rows();
        self.apply_pending_commands();
        self.write_step_records(dt);
        let query_count = self.queries.pending.len() as u32;
        let batch = self.submit_queries(step, query_count);
        let quiet = self.device.measured_step.is_some_and(|measured| {
            self.device.measured[COUNTER_ACTIVE] == 0
                && self
                    .device
                    .commanded_step
                    .is_none_or(|commanded| commanded <= measured)
        });
        let idle = quiet && self.constraints.alive.is_empty();
        let frame = FrameParams {
            dynamic_count: self.bodies.dynamic_count as u32,
            body_count: self.bodies.alive.len() as u32,
            solve_iterations: self.config.solve_iterations,
            position_iterations: self.config.position_iterations,
            island_rounds: self.island_rounds(),
            query_count,
            constraint_count: self.constraints.alive.len() as u32,
            edit_run_count: self.bodies.last_edits,
            body_structural: self.bodies.structural,
            constraint_structural: self.constraints.structural,
        };
        self.encode_step(&frame, batch, step, idle);
        self.bodies.device_count = frame.body_count;
        self.queries.pending.clear();
        self.clock.step += 1;
        self.bodies.states_ready = false;
    }

    pub(crate) fn apply_pending_commands(&mut self) {
        if self.bodies.commands.is_empty() && self.constraints.commands.is_empty() {
            self.bodies.last_edits = 0;
            self.constraints.last_commands = 0;
            self.bodies.structural = false;
            self.constraints.structural = false;
            return;
        }
        self.device.commanded_step = Some(self.clock.step);
        let body_edits = self.compile_body_edits();
        let constraint_edits = self.compile_constraint_edits();
        self.upload_body_edits(&body_edits);
        self.upload_constraint_edits(&constraint_edits);
        self.bodies.last_edits = body_edits.runs.len() as u32;
        self.constraints.last_commands = self.constraints.commands.len() as u32;
        self.bodies.commands.clear();
        self.constraints.commands.clear();
    }

    fn write_step_records(&self, dt: f32) {
        let queue = self.device.gpu.queue();
        self.write_declared_counters();
        let params = SimParamsRecord::new(
            &self.config,
            dt,
            self.bodies.dynamic_count as u32,
            self.bodies.alive.len() as u32,
            self.constraints.alive.len() as u32,
            self.bodies.last_edits,
            self.event_slot_of(self.clock.step),
        );
        self.device
            .buffers
            .params
            .write(queue, bytemuck::cast_slice(&[params]));
    }

    fn upload_body_edits(&mut self, compiled: &CompiledBodyEdits) {
        let queue = self.device.gpu.queue();
        let lanes = MoveLanes::of(&compiled.moves);
        self.device
            .buffers
            .bodies
            .row_src
            .write(queue, bytemuck::cast_slice(&lanes.src));
        self.device
            .buffers
            .bodies
            .row_fresh
            .write(queue, bytemuck::cast_slice(&lanes.fresh));
        self.device
            .buffers
            .bodies
            .fresh_states
            .write(queue, bytemuck::cast_slice(&compiled.fresh));
        self.device
            .buffers
            .bodies
            .edits
            .write(queue, bytemuck::cast_slice(&compiled.edits));
        self.device
            .buffers
            .bodies
            .edit_runs
            .write(queue, bytemuck::cast_slice(&compiled.runs));
        self.bodies.structural = compiled.structurally_dirty;
    }

    fn upload_constraint_edits(&mut self, compiled: &CompiledConstraintEdits) {
        let queue = self.device.gpu.queue();
        let lanes = MoveLanes::of(&compiled.moves);
        self.device
            .buffers
            .constraints
            .row_src
            .write(queue, bytemuck::cast_slice(&lanes.src));
        self.device
            .buffers
            .constraints
            .row_fresh
            .write(queue, bytemuck::cast_slice(&lanes.fresh));
        self.device
            .buffers
            .constraints
            .fresh
            .write(queue, bytemuck::cast_slice(&compiled.fresh));
        self.constraints.structural = compiled.moves.iter().any(RowMove::is_structural);
    }

    fn submit_queries(&mut self, step: u64, query_count: u32) -> Option<u64> {
        if query_count == 0 {
            return None;
        }
        self.device.buffers.queries.records.write(
            self.device.gpu.queue(),
            bytemuck::cast_slice(&self.queries.pending),
        );
        let batch = self.queries.next_batch;
        self.queries.pool.submit(batch, step, query_count as usize);
        self.queries.next_batch += 1;
        Some(batch)
    }

    fn encode_step(&mut self, frame: &FrameParams, batch: Option<u64>, step: u64, idle: bool) {
        let device = self.device.gpu.device().clone();
        let queue = self.device.gpu.queue().clone();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dynamis step encoder"),
        });

        self.copy_events(&mut encoder, &device);
        self.device
            .pipeline
            .encode(&mut encoder, &self.device.buffers, frame, idle);
        #[cfg(feature = "profile")]
        let timings = self.device.pipeline.capture_timings(&mut encoder, step);
        let pack_bytes = self.pack_step(&mut encoder);
        let pack = self.device.buffers.readback.step.enqueue(
            &device,
            &mut encoder,
            self.device.buffers.readback.pack.buffer(),
            0,
            pack_bytes,
            step,
        );
        let queries = match batch {
            Some(batch) => enqueue_readback(
                &device,
                &mut encoder,
                &mut self.device.buffers.readback.queries,
                &self.device.buffers.queries.results,
                self.queries.pending.len() as u64 * size_of::<QueryResultRecord>() as u64,
                batch,
            ),
            None => None,
        };
        queue.submit([encoder.finish()]);
        self.device.buffers.readback.step.arm();
        self.device.buffers.readback.events.arm();
        self.device.buffers.readback.queries.arm();
        #[cfg(feature = "profile")]
        self.device.pipeline.arm_timings();
        #[cfg(feature = "profile")]
        if let Some((_step, timings)) = timings {
            self.device.pass_timings = timings;
        }
        if let Some((step, bytes)) = pack {
            self.consume_pack(step, &bytes);
        }
        if let Some((batch, bytes)) = queries {
            self.queries.pool.collect(batch, &bytes);
        }
    }

    #[cfg(feature = "profile")]
    pub fn gpu_pass_timings(&self) -> &[dynamis_gpu::GpuPassTiming] {
        &self.device.pass_timings
    }

    #[cfg(feature = "profile")]
    pub fn gpu_timing_supported(&self) -> bool {
        self.device.gpu.supports_pass_timing()
    }
}

fn enqueue_readback(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    slot: &mut GpuReadback,
    source: &GpuBuffer,
    bytes: u64,
    sequence: u64,
) -> Option<(u64, Vec<u8>)> {
    if bytes == 0 {
        return None;
    }
    slot.enqueue(device, encoder, source.buffer(), 0, bytes, sequence)
}
