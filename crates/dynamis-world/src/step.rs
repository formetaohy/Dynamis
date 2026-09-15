use super::World;
use crate::commands::{CompiledBodyCommands, CompiledConstraintCommands, Consumption};
use dynamis_abi::StepParamsRecord;
use dynamis_rigid::RigidShape;

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
        self.execute(dynamis_pass::Run::Step);
        self.clock.step += 1;
    }

    pub(crate) fn apply_pending_commands(&mut self, consumption: Consumption) {
        if self.bodies.commands.is_empty()
            && self.constraints.commands.is_empty()
            && !self.soft.pending_uploads()
            && !self.characters.pending()
            && !self.vehicles.pending()
        {
            self.clear_work();
            return;
        }
        let body_commands = self.compile_body_commands(consumption);
        let constraint_commands = self.compile_constraint_commands();
        let soft_commands = self.compile_soft_commands(consumption);
        self.bodies.last_moves = body_commands.moves.len() as u32;
        self.bodies.last_edits = body_commands.runs.len() as u32;
        self.constraints.last_moves = constraint_commands.moves.len() as u32;
        self.constraints.last_commands = self.constraints.commands.len() as u32;
        self.soft.last_body_edits = soft_commands.body_edits.len() as u32;
        self.soft.last_edits = soft_commands.edits.len() as u32;
        self.upload_body_commands(&body_commands);
        self.upload_constraint_commands(&constraint_commands);
        self.upload_soft_commands(&soft_commands);
        if consumption == Consumption::Step {
            self.bodies.commands.clear();
            self.constraints.commands.clear();
            self.soft.consume();
            self.characters.consume();
            self.vehicles.consume();
        }
    }

    fn clear_work(&mut self) {
        self.bodies.last_edits = 0;
        self.bodies.last_moves = 0;
        self.constraints.last_commands = 0;
        self.constraints.last_moves = 0;
        self.soft.last_body_edits = 0;
        self.soft.last_edits = 0;
        self.characters.clear_work();
        self.vehicles.clear_work();
    }

    pub fn rigid_shape(&self) -> RigidShape {
        RigidShape::of(&self.frame_counts())
    }

    pub(crate) fn write_step_records(&mut self, params: StepParamsRecord) {
        let queue = self.backend.gpu.queue();
        if self.backend.written_params != Some(params) {
            self.backend
                .streams
                .state
                .params
                .write(queue, bytemuck::cast_slice(&[params]));
            self.backend.written_params = Some(params);
        }
        let rows: dynamis_abi::RowStreamsRecord = self.row_streams().into();
        self.backend
            .streams
            .state
            .row_streams
            .write(queue, bytemuck::cast_slice(&[rows]));
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
