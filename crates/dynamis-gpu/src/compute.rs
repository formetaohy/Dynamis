use crate::buffer::GpuBuffer;
use wgpu::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, CommandEncoder, ComputePassDescriptor,
    ComputePipeline as WgpuComputePipeline, ComputePipelineDescriptor, Device,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

pub struct ComputeRecorder<'a> {
    pass: wgpu::ComputePass<'a>,
}

impl<'a> ComputeRecorder<'a> {
    pub fn begin(encoder: &'a mut CommandEncoder, label: &str) -> Self {
        let pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        Self { pass }
    }

    pub fn record(&mut self, pipeline: &ComputePipeline, bind_groups: &[&BindGroup], count: u32) {
        self.pass.set_pipeline(pipeline.pipeline());
        for (group, bind_group) in bind_groups.iter().enumerate() {
            self.pass.set_bind_group(group as u32, *bind_group, &[]);
        }
        self.pass.dispatch_workgroups(count, 1, 1);
    }

    pub fn record_indirect(
        &mut self,
        pipeline: &ComputePipeline,
        bind_groups: &[&BindGroup],
        args: &GpuBuffer,
        offset: u64,
    ) {
        self.pass.set_pipeline(pipeline.pipeline());
        for (group, bind_group) in bind_groups.iter().enumerate() {
            self.pass.set_bind_group(group as u32, *bind_group, &[]);
        }
        self.pass
            .dispatch_workgroups_indirect(args.as_indirect_args(), offset);
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
    workgroup_size: u32,
}

impl ComputePipeline {
    pub fn new(
        device: &Device,
        label: &str,
        shader: &str,
        entry: &str,
        groups: &[&[BindingSpec]],
        workgroup_size: u32,
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
            workgroup_size,
        }
    }

    pub fn bind_group_layout(&self, group: usize) -> &BindGroupLayout {
        &self.bind_group_layouts[group]
    }

    pub fn pipeline(&self) -> &WgpuComputePipeline {
        &self.pipeline
    }

    pub fn workgroup_size(&self) -> u32 {
        self.workgroup_size
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

    pub fn workgroup_count(&self, elements: u32) -> u32 {
        elements.div_ceil(self.workgroup_size)
    }
}
