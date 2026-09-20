use super::World;
use crate::command::{CompiledBodyCommands, Consumption};
use dynamis_abi::{BodyEditRecord, StepParamsRecord};
use dynamis_model::domain;
use dynamis_rigid::RigidShape;

impl World {
    pub fn set_time_scale(&mut self, time_scale: f32) {
        domain::positive(time_scale, "a time scale");
        self.clock.time_scale = time_scale;
    }

    pub fn update(&mut self, real_dt: f32, sub_dt: f32, max_substeps: u32) {
        domain::non_negative(real_dt, "a frame duration");
        domain::positive(sub_dt, "a substep duration");
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
        domain::positive(dt, "a timestep");
        self.clock.sub_dt = dt;
        self.execute(dynamis_pass::Run::Step);
        self.clock.step += 1;
    }

    pub(crate) fn apply_pending_commands(&mut self, consumption: Consumption) {
        if self.bodies.commands.is_empty()
            && self.constraints.declarations == 0
            && !self.soft.pending_uploads()
            && !self.characters.pending()
            && !self.vehicles.pending()
        {
            self.clear_work();
            return;
        }
        let body_commands = self.compile_body_commands(consumption);
        let constraint_rows = self.compile_constraint_rows();
        self.note_command_edits(&body_commands);
        let soft_commands = self.compile_soft_commands(consumption);
        self.bodies.last_moves = body_commands.moves.len() as u32;
        self.bodies.last_edits = body_commands.runs.len() as u32;
        self.bodies.last_layout = body_commands.layout;
        self.constraints.last_moves = constraint_rows.moves.len() as u32;
        self.constraints.last_declarations = self.constraints.declarations;
        self.soft.last_body_edits = soft_commands.body_edits.len() as u32;
        self.soft.last_edits = soft_commands.edits.len() as u32;
        self.backend.staged.body_commands = Some(body_commands);
        self.backend.staged.constraint_rows = Some(constraint_rows);
        self.backend.staged.soft_commands = Some(soft_commands);
        if consumption == Consumption::Step {
            self.bodies.commands.clear();
            self.soft.consume();
            self.characters.consume();
            self.vehicles.consume();
        }
    }

    fn clear_work(&mut self) {
        self.bodies.last_edits = 0;
        self.bodies.last_moves = 0;
        self.bodies.last_layout = 0;
        self.constraints.last_declarations = 0;
        self.constraints.last_moves = 0;
        self.soft.last_body_edits = 0;
        self.soft.last_edits = 0;
        self.characters.clear_work();
        self.vehicles.clear_work();
    }

    pub fn rigid_shape(&self) -> RigidShape {
        RigidShape::of(&self.census())
    }

    pub(crate) fn stage_step_records(&mut self, params: StepParamsRecord) {
        self.backend.staged.step_records = Some((params, self.row_streams().into()));
    }

    pub(crate) fn flush_step_parameters(&mut self) {
        let Some((params, rows)) = self.backend.staged.step_records.take() else {
            return;
        };
        let queue = self.backend.gpu.queue();
        if self.backend.written_params != Some(params) {
            self.backend
                .streams
                .state
                .params
                .write(queue, bytemuck::cast_slice(&[params]));
            self.backend.written_params = Some(params);
        }
        self.backend
            .streams
            .state
            .row_streams
            .write(queue, bytemuck::cast_slice(&[rows]));
    }

    pub(crate) fn flush_body_commands(&mut self) {
        let compiled = std::mem::take(&mut self.backend.staged.body_commands);
        let Some(compiled) = compiled else {
            return;
        };
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
        self.bodies.settle_rows();
    }

    pub(crate) fn flush_constraint_rows(&mut self) {
        let compiled = std::mem::take(&mut self.backend.staged.constraint_rows);
        let Some(compiled) = compiled else {
            return;
        };
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
        self.constraints.settle_rows();
        self.constraints.declarations = 0;
    }

    /// Notes the facts a compiled command stream declares: that a pose was declared, and, when the
    /// edited body is immovable, that the entries the immovable half of the grid was emitted from
    /// have moved. Every other input of a derivation is owned by the store that changes it.
    fn note_command_edits(&mut self, compiled: &CompiledBodyCommands) {
        let mut posed = false;
        for run in &compiled.runs {
            let edits = &compiled.edits[run.first as usize..(run.first + run.len) as usize];
            let declares_pose = edits.iter().any(BodyEditRecord::declares_pose);
            posed |= declares_pose;
            let id = self.bodies.pool.handle_of_row(run.row).id as usize;
            if declares_pose && self.is_static(id) {
                self.facts.immovable_edits += 1;
            }
        }
        for moved in &compiled.moves {
            let id = self.bodies.pool.handle_of_row(moved.row).id as usize;
            if self.is_static(id) {
                self.facts.immovable_edits += 1;
            }
        }
        if posed {
            self.facts.poses += 1;
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
