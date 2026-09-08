use super::Simulation;
use crate::pipeline::FrameParams;
#[cfg(feature = "profile")]
use dynamis_gpu::GpuPassTiming;
use dynamis_layout::{Counter, QueryResultHeader, SimParamsRecord};

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
        let queue = self.gpu.queue().clone();
        let device = self.gpu.device().clone();
        self.buffers
            .commands
            .write(&queue, bytemuck::cast_slice(&self.commands));
        self.buffers.command_count.write(
            &queue,
            bytemuck::cast_slice(&[Counter::sized(self.commands.len() as u32)]),
        );
        self.buffers
            .constraint_commands
            .write(&queue, bytemuck::cast_slice(&self.constraint_commands));
        self.buffers.constraint_command_count.write(
            &queue,
            bytemuck::cast_slice(&[Counter::sized(self.constraint_commands.len() as u32)]),
        );
        self.buffers.constraint_count_state.write(
            &queue,
            bytemuck::cast_slice(&[Counter::sized(self.constraint_alive.len() as u32)]),
        );
        let params = SimParamsRecord::new(
            &self.config,
            dt,
            self.dynamic_count as u32,
            self.alive.len() as u32,
            self.constraint_alive.len() as u32,
        );
        self.buffers
            .params
            .write(&queue, bytemuck::cast_slice(&[params]));
        self.buffers.reset_counters(&queue);
        let header_bytes = self.query_pool.capacity() * std::mem::size_of::<QueryResultHeader>();
        self.buffers
            .query_headers
            .write(&queue, &vec![0u8; header_bytes]);
        if !self.queries.is_empty() {
            self.buffers
                .queries
                .write(&queue, bytemuck::cast_slice(&self.queries));
            let slots = self.queries.iter().map(|query| query.slot).collect();
            self.query_pool.mark_batch(step, slots);
        }

        let frame = FrameParams {
            dynamic_count: self.dynamic_count as u32,
            body_count: self.alive.len() as u32,
            solve_iterations: self.config.solve_iterations,
            position_iterations: self.config.position_iterations,
            island_rounds: self.island_rounds(),
            query_count: self.queries.len() as u32,
            constraint_count: self.constraint_alive.len() as u32,
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dynamis step encoder"),
        });
        self.pipeline.encode(&mut encoder, &self.buffers, &frame);
        #[cfg(feature = "profile")]
        let stale_timings = self.pipeline.capture_timings(&mut encoder, step);

        let stale_bodies = self.buffers.body_states_readback.enqueue(
            &device,
            &mut encoder,
            self.buffers.body_states.buffer(),
            step,
        );
        let stale_constraints = self.buffers.constraint_runtime_readback.enqueue(
            &device,
            &mut encoder,
            self.buffers.constraint_runtime.buffer(),
            step,
        );
        let stale_queries = self.submit_queries_pack(&device, &mut encoder, step);
        let stale_events = self.submit_events_pack(&device, &mut encoder, step);
        queue.submit([encoder.finish()]);
        self.buffers.body_states_readback.arm();
        self.buffers.queries_readback.arm();
        self.buffers.events_readback.arm();
        self.buffers.constraint_runtime_readback.arm();
        #[cfg(feature = "profile")]
        self.pipeline.arm_timings();
        #[cfg(feature = "profile")]
        if let Some((_stale_step, timings)) = stale_timings {
            self.pass_timings = timings;
        }
        if let Some((stale_step, bytes)) = stale_bodies {
            self.consume_bodies(stale_step, &bytes);
        }
        if let Some((stale_step, bytes)) = stale_constraints {
            self.consume_constraints(stale_step, &bytes);
        }
        if let Some((stale_step, bytes)) = stale_queries {
            self.consume_queries(stale_step, &bytes);
        }
        if let Some((_stale_step, bytes)) = stale_events {
            self.consume_events(&bytes);
        }
        self.commands.clear();
        self.constraint_commands.clear();
        self.queries.clear();
        self.step_index += 1;
    }

    pub(crate) fn submit_queries_pack(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        step: u64,
    ) -> Option<(u64, Vec<u8>)> {
        if self.queries.is_empty() {
            return None;
        }
        let header_bytes = self.query_pool.capacity() * std::mem::size_of::<QueryResultHeader>();
        encoder.copy_buffer_to_buffer(
            self.buffers.query_headers.buffer(),
            0,
            self.buffers.query_pack.buffer(),
            0,
            header_bytes as u64,
        );
        encoder.copy_buffer_to_buffer(
            self.buffers.query_hits.buffer(),
            0,
            self.buffers.query_pack.buffer(),
            header_bytes as u64,
            self.buffers.query_hits.size(),
        );
        self.buffers.queries_readback.enqueue(
            device,
            encoder,
            self.buffers.query_pack.buffer(),
            step,
        )
    }

    pub(crate) fn submit_events_pack(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        step: u64,
    ) -> Option<(u64, Vec<u8>)> {
        encoder.copy_buffer_to_buffer(
            self.buffers.event_count.buffer(),
            0,
            self.buffers.events_pack.buffer(),
            0,
            12,
        );
        encoder.copy_buffer_to_buffer(
            self.buffers.events.buffer(),
            0,
            self.buffers.events_pack.buffer(),
            12,
            self.buffers.events.size(),
        );
        self.buffers.events_readback.enqueue(
            device,
            encoder,
            self.buffers.events_pack.buffer(),
            step,
        )
    }

    fn island_rounds(&self) -> u32 {
        (self.capacity as u32).ilog2() + 1
    }

    /// Per-pass GPU durations of the most recently drained step.
    #[cfg(feature = "profile")]
    pub fn gpu_pass_timings(&self) -> &[GpuPassTiming] {
        &self.pass_timings
    }

    /// Whether this device can report pass timings.
    #[cfg(feature = "profile")]
    pub fn gpu_timing_supported(&self) -> bool {
        self.gpu.supports_pass_timing()
    }
}
