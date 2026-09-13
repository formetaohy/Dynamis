use super::World;
use crate::backend::StepFrames;
use crate::commands::{CompiledBodyCommands, CompiledConstraintCommands};
use dynamis_abi::QueryResultRecord;
use dynamis_abi::StepParamsRecord;
use dynamis_rigid::RigidShape;
use std::mem::size_of;

impl World {
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
        self.backend.gpu.assert_alive();
        self.clock.sub_dt = dt;
        let step = self.clock.step;
        self.collect_readbacks();
        let live = self.live();
        self.apply_plan(&live);
        self.flush_rows();
        self.apply_pending_commands();
        let work = self.host_work();
        let query_count = work.state.queries;
        self.backend.working = work.pending();
        let params = self.step_params(dt);
        let frames = self.frames(&live, &work, params);
        self.shapes.uploaded = false;
        self.soft.uploaded = false;
        self.write_step_records(params);
        self.declare_step(step);
        let batch = self.submit_queries(step, query_count);
        self.encode_step(&frames, batch, step);
        self.bodies.device_count = params.body_count;
        self.queries.pending.clear();
        self.clock.step += 1;
    }

    pub(crate) fn apply_pending_commands(&mut self) {
        if self.bodies.commands.is_empty() && self.constraints.commands.is_empty() {
            self.bodies.last_edits = 0;
            self.bodies.last_moves = 0;
            self.constraints.last_commands = 0;
            self.constraints.last_moves = 0;
            return;
        }
        let body_commands = self.compile_body_commands();
        let constraint_commands = self.compile_constraint_commands();
        self.upload_body_commands(&body_commands);
        self.upload_constraint_commands(&constraint_commands);
        self.bodies.last_edits = body_commands.runs.len() as u32;
        self.constraints.last_commands = self.constraints.commands.len() as u32;
        self.bodies.commands.clear();
        self.constraints.commands.clear();
    }

    pub fn rigid_shape(&self) -> RigidShape {
        RigidShape::of(&self.frame_counts())
    }

    fn write_step_records(&self, params: StepParamsRecord) {
        self.backend
            .streams
            .state
            .params
            .write(self.backend.gpu.queue(), bytemuck::cast_slice(&[params]));
    }

    fn upload_body_commands(&mut self, compiled: &CompiledBodyCommands) {
        let queue = self.backend.gpu.queue();
        self.backend
            .streams
            .state
            .body_row_moves
            .write(queue, bytemuck::cast_slice(&compiled.moves));
        self.backend
            .streams
            .state
            .body_fresh_rows
            .write(queue, bytemuck::cast_slice(&compiled.fresh));
        self.backend
            .streams
            .state
            .body_edits
            .write(queue, bytemuck::cast_slice(&compiled.edits));
        self.backend
            .streams
            .state
            .body_edit_runs
            .write(queue, bytemuck::cast_slice(&compiled.runs));
        self.bodies.last_moves = compiled.moves.len() as u32;
    }

    fn upload_constraint_commands(&mut self, compiled: &CompiledConstraintCommands) {
        let queue = self.backend.gpu.queue();
        self.backend
            .streams
            .state
            .constraint_row_moves
            .write(queue, bytemuck::cast_slice(&compiled.moves));
        self.backend
            .streams
            .state
            .constraint_fresh_rows
            .write(queue, bytemuck::cast_slice(&compiled.fresh));
        self.constraints.last_moves = compiled.moves.len() as u32;
    }

    fn submit_queries(&mut self, step: u64, query_count: u32) -> Option<u64> {
        if query_count == 0 {
            return None;
        }
        self.backend.streams.state.query_records.write(
            self.backend.gpu.queue(),
            bytemuck::cast_slice(&self.queries.pending),
        );
        let batch = self.queries.next_batch;
        self.queries.pool.submit(batch, step, query_count as usize);
        self.queries.next_batch += 1;
        Some(batch)
    }

    fn encode_step(&mut self, frames: &StepFrames, batch: Option<u64>, step: u64) {
        let device = self.backend.gpu.device().clone();
        let mut encoder = dynamis_gpu::SubmissionEncoder::new(&device, "dynamis step");

        self.copy_events(&mut encoder);
        self.copy_breaks(&mut encoder);
        self.backend
            .passes
            .record(&mut encoder, &self.backend.streams, frames);
        #[cfg(feature = "profile")]
        let timings = self.backend.passes.capture_timings(&mut encoder);
        let pack_bytes = self.pack_step(&mut encoder);
        self.write_states(&mut encoder, step);
        let pack = self.backend.readback.step.enqueue(
            &mut encoder,
            self.backend.readback.pack.buffer(),
            0,
            pack_bytes,
            step,
        );
        let queries = match batch {
            Some(batch) => self.backend.readback.queries.enqueue(
                &mut encoder,
                self.backend.streams.state.query_results.buffer(),
                0,
                self.queries.pending.len() as u64 * size_of::<QueryResultRecord>() as u64,
                batch,
            ),
            None => None,
        };
        self.submit(encoder);
        #[cfg(feature = "profile")]
        if let Some(timings) = timings {
            self.backend.pass_timings = timings;
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
        &self.backend.pass_timings
    }

    #[cfg(feature = "profile")]
    pub fn gpu_timing_supported(&self) -> bool {
        self.backend.gpu.supports_pass_timing()
    }
}
