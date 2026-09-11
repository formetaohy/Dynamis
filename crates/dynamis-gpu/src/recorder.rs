use crate::ComputePipeline;
use wgpu::{
    BindGroup, CommandEncoder, ComputePass, ComputePassDescriptor, ComputePassTimestampWrites,
};

pub struct ComputeRecorder<'a> {
    pass: ComputePass<'a>,
    per_row: u32,
}

impl<'a> ComputeRecorder<'a> {
    pub fn begin(encoder: &'a mut CommandEncoder, label: &'a str, per_row: u32) -> Self {
        Self::begin_timed(encoder, label, None, per_row)
    }

    pub fn begin_timed(
        encoder: &'a mut CommandEncoder,
        label: &'a str,
        timing: Option<ComputePassTimestampWrites<'a>>,
        per_row: u32,
    ) -> Self {
        let pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: timing,
        });
        Self { pass, per_row }
    }

    pub fn record(&mut self, pipeline: &ComputePipeline, bind_groups: &[&BindGroup], count: u32) {
        if count == 0 {
            return;
        }
        self.pass.set_pipeline(pipeline.wgpu());
        for (group, bind_group) in bind_groups.iter().enumerate() {
            self.pass.set_bind_group(group as u32, *bind_group, &[]);
        }
        self.pass
            .dispatch_workgroups(count.min(self.per_row), count.div_ceil(self.per_row), 1);
    }
}
