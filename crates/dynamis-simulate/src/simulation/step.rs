use super::Simulation;
use crate::pipeline::FrameParams;
use crate::simulation::commands::{
    CompiledBodyCommands, CompiledConstraintCommands, encode_constraint_move, encode_move,
};
use dynamis_gpu::{GpuBuffer, GpuReadback};
use dynamis_layout::{QueryResultRecord, SimParamsRecord};
use std::mem::size_of;

impl Simulation {
    pub fn set_time_scale(&mut self, time_scale: f32) {
        assert!(time_scale > 0.0, "time scale must be strictly positive");
        self.time_scale = time_scale;
    }

    pub fn update(&mut self, real_dt: f32, sub_dt: f32, max_substeps: u32) {
        assert!(real_dt >= 0.0, "real dt must be non-negative");
        assert!(sub_dt > 0.0, "sub dt must be strictly positive");
        assert!(max_substeps > 0, "max substeps must be positive");
        self.sub_dt = sub_dt;
        self.accumulator += real_dt * self.time_scale;
        let mut steps = 0;
        while self.accumulator >= sub_dt && steps < max_substeps {
            self.step(sub_dt);
            self.accumulator -= sub_dt;
            steps += 1;
        }
        if steps == max_substeps {
            self.accumulator %= sub_dt;
        }
    }

    pub fn interpolation_alpha(&self) -> f32 {
        (self.accumulator / self.sub_dt).clamp(0.0, 1.0)
    }

    pub fn step(&mut self, dt: f32) {
        assert!(dt > 0.0, "timestep must be strictly positive");
        self.gpu.assert_alive();
        let step = self.step_index;
        self.collect_readbacks();
        self.apply_plan();
        self.flush_rows();
        self.apply_pending_commands();
        self.write_step_records(dt);
        let query_count = self.queries.len() as u32;
        let batch = self.submit_queries(step, query_count);
        let frame = FrameParams {
            dynamic_count: self.dynamic_count as u32,
            body_count: self.alive.len() as u32,
            solve_iterations: self.config.solve_iterations,
            position_iterations: self.config.position_iterations,
            island_rounds: self.island_rounds(),
            query_count,
            constraint_count: self.constraint_alive.len() as u32,
            body_structural: self.body_structural,
            constraint_structural: self.constraint_structural,
            has_body_edits: self.has_body_edits,
        };
        self.encode_step(&frame, batch, step);
        self.device_body_count = frame.body_count;
        self.queries.clear();
        self.step_index += 1;
        self.states_synchronized = false;
    }

    /// Compiles every pending command, uploads the parallel stream, and consumes
    /// the sequence: a command applies exactly once whether the next step or a
    /// query flush is what runs first.
    pub(crate) fn apply_pending_commands(&mut self) {
        let body_compiled = self.compile_commands();
        let constraint_compiled = self.compile_constraint_commands();
        self.upload_body_compile(&body_compiled);
        self.upload_constraint_compile(&constraint_compiled);
        self.last_body_commands = self.commands.len() as u32;
        self.last_constraint_commands = self.constraint_commands.len() as u32;
        self.commands.clear();
        self.constraint_commands.clear();
    }

    fn write_step_records(&mut self, dt: f32) {
        let queue = self.gpu.queue();
        self.write_declared_counters(queue);
        let params = SimParamsRecord::new(
            &self.config,
            dt,
            self.dynamic_count as u32,
            self.alive.len() as u32,
            self.constraint_alive.len() as u32,
            self.event_slot_of(self.step_index),
        );
        self.buffers
            .params
            .write(queue, bytemuck::cast_slice(&[params]));
    }

    fn upload_body_compile(&mut self, compiled: &CompiledBodyCommands) {
        let queue = self.gpu.queue();
        let bodies = compiled.moves.len();
        let mut src = Vec::with_capacity(bodies);
        let mut fresh_lane = Vec::with_capacity(bodies);
        for (slot, move_) in compiled.moves.iter().enumerate() {
            let (source, fresh) = encode_move(move_, slot as u32);
            src.push(source);
            fresh_lane.push(fresh);
        }
        self.buffers
            .row_src
            .write(queue, bytemuck::cast_slice(&src));
        self.buffers
            .row_fresh
            .write(queue, bytemuck::cast_slice(&fresh_lane));
        self.buffers
            .fresh_states
            .write(queue, bytemuck::cast_slice(&compiled.fresh));
        self.buffers
            .commands
            .write(queue, bytemuck::cast_slice(&compiled.edits));
        self.buffers
            .command_first
            .write(queue, bytemuck::cast_slice(&compiled.edit_first));
        self.body_structural = compiled.structurally_dirty;
        self.has_body_edits = !compiled.edits.is_empty();
    }

    fn upload_constraint_compile(&mut self, compiled: &CompiledConstraintCommands) {
        let queue = self.gpu.queue();
        let mut src = Vec::with_capacity(compiled.moves.len());
        let mut fresh_lane = Vec::with_capacity(compiled.moves.len());
        for (slot, move_) in compiled.moves.iter().enumerate() {
            let (source, fresh) = encode_constraint_move(move_, slot as u32);
            src.push(source);
            fresh_lane.push(fresh);
        }
        self.buffers
            .constraint_row_src
            .write(queue, bytemuck::cast_slice(&src));
        self.buffers
            .constraint_row_fresh
            .write(queue, bytemuck::cast_slice(&fresh_lane));
        self.buffers
            .constraint_fresh
            .write(queue, bytemuck::cast_slice(&compiled.fresh));
        self.constraint_structural = compiled.moves.iter().any(|move_| move_.is_structural());
    }

    fn submit_queries(&mut self, step: u64, query_count: u32) -> Option<u64> {
        if query_count == 0 {
            return None;
        }
        self.buffers
            .queries
            .write(self.gpu.queue(), bytemuck::cast_slice(&self.queries));
        let batch = self.next_batch;
        self.query_pool.submit(batch, step, query_count as usize);
        self.next_batch += 1;
        Some(batch)
    }

    fn encode_step(&mut self, frame: &FrameParams, batch: Option<u64>, step: u64) {
        let device = self.gpu.device().clone();
        let queue = self.gpu.queue().clone();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dynamis step encoder"),
        });
        // Event segments are copied home at the head of the very next encoder, so a
        // segment is read before the step that would overwrite it is encoded.
        self.copy_events(&mut encoder, &device);
        self.pipeline.encode(&mut encoder, &self.buffers, frame);
        #[cfg(feature = "profile")]
        let timings = self.pipeline.capture_timings(&mut encoder, step);
        let pack_bytes = self.pack_step(&mut encoder);
        let pack = self.buffers.readback.enqueue(
            &device,
            &mut encoder,
            self.buffers.readback_pack.buffer(),
            0,
            pack_bytes,
            step,
        );
        let queries = match batch {
            Some(batch) => readback(
                &device,
                &mut encoder,
                &mut self.buffers.queries_readback,
                &self.buffers.query_results,
                self.queries.len() as u64 * size_of::<QueryResultRecord>() as u64,
                batch,
            ),
            None => None,
        };
        queue.submit([encoder.finish()]);
        self.buffers.readback.arm();
        self.buffers.events_readback.arm();
        self.buffers.queries_readback.arm();
        #[cfg(feature = "profile")]
        self.pipeline.arm_timings();
        #[cfg(feature = "profile")]
        if let Some((_step, timings)) = timings {
            self.pass_timings = timings;
        }
        if let Some((step, bytes)) = pack {
            self.consume_pack(step, &bytes);
        }
        if let Some((batch, bytes)) = queries {
            self.query_pool.collect(batch, &bytes);
        }
    }

    /// Per-pass GPU durations of the most recently drained step.
    #[cfg(feature = "profile")]
    pub fn gpu_pass_timings(&self) -> &[dynamis_gpu::GpuPassTiming] {
        &self.pass_timings
    }

    /// Whether this device can report pass timings.
    #[cfg(feature = "profile")]
    pub fn gpu_timing_supported(&self) -> bool {
        self.gpu.supports_pass_timing()
    }
}

/// Queues a readback of a device buffer; an empty buffer has nothing to bring back.
fn readback(
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
