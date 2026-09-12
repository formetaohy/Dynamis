use std::sync::{Arc, OnceLock};
use wgpu::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, ComputePipeline as WgpuComputePipeline,
    ComputePipelineDescriptor, Device, PipelineLayout, PipelineLayoutDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BindingKind {
    Uniform,
    ReadOnlyStorage,
    ReadWriteStorage,
}

impl BindingKind {
    fn layout_entry(self, binding: u32) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: match self {
                Self::Uniform => BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                Self::ReadOnlyStorage => BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                Self::ReadWriteStorage => BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            },
            count: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindingSpec {
    pub binding: u32,
    pub kind: BindingKind,
}

#[derive(PartialEq, Eq, Hash)]
pub struct ComputeProgram {
    label: String,
    shader: Arc<str>,
    entry: String,
    groups: Vec<Vec<BindingSpec>>,
}

impl ComputeProgram {
    pub fn new(
        label: &str,
        shader: impl Into<Arc<str>>,
        entry: &str,
        groups: &[&[BindingSpec]],
    ) -> Self {
        Self {
            label: label.to_owned(),
            shader: shader.into(),
            entry: entry.to_owned(),
            groups: groups.iter().map(|group| group.to_vec()).collect(),
        }
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

pub struct ComputeLayout {
    pipeline: PipelineLayout,
    groups: Vec<BindGroupLayout>,
}

impl ComputeLayout {
    pub(crate) fn new(device: &Device, program: &ComputeProgram) -> Self {
        let groups = program
            .groups
            .iter()
            .map(|bindings| {
                let entries = bindings
                    .iter()
                    .map(|spec| spec.kind.layout_entry(spec.binding))
                    .collect::<Vec<_>>();
                device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some(program.label()),
                    entries: &entries,
                })
            })
            .collect::<Vec<_>>();
        let layouts = groups.iter().map(Some).collect::<Vec<_>>();
        let pipeline = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(program.label()),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        });
        Self { pipeline, groups }
    }

    pub fn create_bind_group(
        &self,
        device: &Device,
        group: usize,
        entries: &[BindGroupEntry<'_>],
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.groups[group],
            entries,
        })
    }
}

pub struct ComputePipeline {
    pipeline: WgpuComputePipeline,
}

impl ComputePipeline {
    fn compile(device: &Device, program: &ComputeProgram, layout: &ComputeLayout) -> Self {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(program.label()),
            source: ShaderSource::Wgsl(program.shader.as_ref().into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(program.label()),
            layout: Some(&layout.pipeline),
            module: &module,
            entry_point: Some(&program.entry),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline }
    }

    pub(crate) fn wgpu(&self) -> &WgpuComputePipeline {
        &self.pipeline
    }
}

struct PipelineSlot {
    program: Arc<ComputeProgram>,
    layout: ComputeLayout,
    compiled: OnceLock<ComputePipeline>,
}

#[derive(Clone)]
pub struct PipelineHandle {
    slot: Arc<PipelineSlot>,
}

impl PipelineHandle {
    pub(crate) fn new(program: Arc<ComputeProgram>, layout: ComputeLayout) -> Self {
        Self {
            slot: Arc::new(PipelineSlot {
                program,
                layout,
                compiled: OnceLock::new(),
            }),
        }
    }

    pub fn label(&self) -> &str {
        self.slot.program.label()
    }

    pub fn is_warmed(&self) -> bool {
        self.slot.compiled.get().is_some()
    }

    pub fn pipeline(&self) -> &ComputePipeline {
        self.slot.compiled.get().unwrap_or_else(|| {
            panic!(
                "pipeline {:?} must be warmed before it is recorded",
                self.slot.program.label()
            )
        })
    }

    pub fn create_bind_group(
        &self,
        device: &Device,
        group: usize,
        entries: &[BindGroupEntry<'_>],
    ) -> BindGroup {
        self.slot.layout.create_bind_group(device, group, entries)
    }

    pub(crate) fn compile(&self, device: &Device) {
        let pipeline = ComputePipeline::compile(device, &self.slot.program, &self.slot.layout);
        assert!(
            self.slot.compiled.set(pipeline).is_ok(),
            "pipeline {:?} compiled twice",
            self.slot.program.label()
        );
    }
}
