use crate::DispatchTable;
use wgpu::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, CommandEncoder, ComputePassDescriptor,
    ComputePassTimestampWrites, ComputePipeline as WgpuComputePipeline, ComputePipelineDescriptor,
    Device, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

/// Records dispatches for one compute pass, splitting a workgroup count that exceeds
/// the device's per-dimension limit across a second dimension.
pub struct ComputeRecorder<'a> {
    pass: wgpu::ComputePass<'a>,
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
        self.pass.set_pipeline(pipeline.pipeline());
        for (group, bind_group) in bind_groups.iter().enumerate() {
            self.pass.set_bind_group(group as u32, *bind_group, &[]);
        }
        self.pass
            .dispatch_workgroups(count.min(self.per_row), count.div_ceil(self.per_row), 1);
    }

    /// Records a dispatch whose workgroup counts live in the table: the device decides
    /// how much work this stage has, and the pass reads it as a direct argument.
    pub fn record_indirect(
        &mut self,
        pipeline: &ComputePipeline,
        bind_groups: &[&BindGroup],
        table: &DispatchTable,
        slot: u32,
    ) {
        self.pass.set_pipeline(pipeline.pipeline());
        for (group, bind_group) in bind_groups.iter().enumerate() {
            self.pass.set_bind_group(group as u32, *bind_group, &[]);
        }
        self.pass
            .dispatch_workgroups_indirect(table.buffer().buffer(), table.offset(slot));
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingKind {
    Uniform,
    ReadOnlyStorage,
    ReadWriteStorage,
}

pub struct BindingSpec {
    pub binding: u32,
    pub kind: BindingKind,
}

#[derive(Clone)]
pub struct ComputePipeline {
    pipeline: WgpuComputePipeline,
    bind_group_layouts: Vec<BindGroupLayout>,
}

impl ComputePipeline {
    pub fn new(
        device: &Device,
        label: &str,
        shader: &str,
        entry: &str,
        groups: &[&[BindingSpec]],
    ) -> Self {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(shader.into()),
        });
        let bind_group_layouts = groups
            .iter()
            .map(|bindings| {
                let entries: Vec<BindGroupLayoutEntry> = bindings
                    .iter()
                    .map(|spec| BindGroupLayoutEntry {
                        binding: spec.binding,
                        visibility: ShaderStages::COMPUTE,
                        ty: match spec.kind {
                            BindingKind::Uniform => BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            BindingKind::ReadOnlyStorage => BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            BindingKind::ReadWriteStorage => BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                        },
                        count: None,
                    })
                    .collect();
                device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some(label),
                    entries: &entries,
                })
            })
            .collect::<Vec<_>>();
        let layouts = bind_group_layouts.iter().map(Some).collect::<Vec<_>>();
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            pipeline,
            bind_group_layouts,
        }
    }

    pub fn pipeline(&self) -> &WgpuComputePipeline {
        &self.pipeline
    }

    pub fn create_bind_group(
        &self,
        device: &Device,
        group: usize,
        entries: &[BindGroupEntry<'_>],
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layouts[group],
            entries,
        })
    }
}
