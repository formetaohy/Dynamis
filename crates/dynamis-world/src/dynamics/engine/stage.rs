use dynamis_gpu::{
    BindingKind, BindingSpec, ComputeProgram, ComputeRecorder, GpuContext, GpuSlot, PipelineHandle,
};
use std::sync::Arc;
use wgpu::{BindGroup, BindGroupEntry};

pub(crate) const WORKGROUP_SIZE: u32 = 64;

pub(crate) const MAX_DISPATCH_WORKGROUPS: u32 = 4096;

pub(crate) fn workgroups_of(elements: u32) -> u32 {
    elements.div_ceil(WORKGROUP_SIZE)
}

pub(crate) fn entry_rows(field: &str) -> String {
    format!(
        "
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {{
    let index = global_index(gid);
    if (index >= params.{field}) {{
        return;
    }}
    work(index);
}}
"
    )
}

pub(crate) fn entry_stream(name: &str, kernel: &str) -> String {
    assert!(
        kernel != "main" && kernel != "warm",
        "the streaming kernel {kernel:?} collides with the generated entry point"
    );
    format!(
        "
@compute @workgroup_size(WORKGROUP_SIZE)
fn {name}(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {{
    let live = extent();
    let stride = grid_stride(groups);
    for (var index = global_index(gid); index < live; index = index + stride) {{
        {kernel}(index);
    }}
}}
"
    )
}

#[derive(Clone, Copy)]
pub(crate) enum Dispatch {
    Rows,
    Stream { workgroups: u32 },
    Workgroups,
}

impl Dispatch {
    pub(crate) fn rows() -> Self {
        Self::Rows
    }

    pub(crate) fn stream(slots: u32) -> Self {
        Self::Stream {
            workgroups: workgroups_of(slots).min(MAX_DISPATCH_WORKGROUPS),
        }
    }

    pub(crate) fn workgroups() -> Self {
        Self::Workgroups
    }
}

pub(crate) struct Program {
    pub(crate) source: Arc<str>,
    pub(crate) dispatch: Dispatch,
    pub(crate) warm: bool,
}

pub(crate) fn whole<'a>(binding: impl Into<GpuSlot<'a>>) -> GpuSlot<'a> {
    binding.into()
}

struct Warm {
    pipeline: PipelineHandle,
    workgroups: u32,
}

pub(crate) struct Stage {
    pipeline: PipelineHandle,
    dispatch: Dispatch,
    warm: Option<Warm>,
    bind_group: BindGroup,
    shapes_group: Option<BindGroup>,
}

struct StageBindings<'a> {
    slots: &'a [(&'a str, GpuSlot<'a>)],
    shapes: &'a [(&'a str, GpuSlot<'a>)],
}

struct Bindings<'a> {
    groups: Vec<Vec<BindingSpec>>,
    storage: Vec<BindGroupEntry<'a>>,
    shapes: Vec<BindGroupEntry<'a>>,
}

impl<'a> Bindings<'a> {
    fn of(label: &str, source: &str, declared: StageBindings<'a>) -> Self {
        let StageBindings { slots, shapes } = declared;
        let declarations = dynamis_gpu::parse_bindings(source);
        let storage = storage_entries(&declarations, slots);
        assert!(
            storage.len() == specs_of(&declarations, 0).len(),
            "stage {label:?} declares {} bindings but provides {}",
            specs_of(&declarations, 0).len(),
            storage.len()
        );
        let shape_entries = shape_entries(&declarations, shapes);
        let mut groups = vec![specs_of(&declarations, 0)];
        if !shapes.is_empty() {
            groups.push(specs_of(&declarations, 1));
        }
        Self {
            groups,
            storage,
            shapes: shape_entries,
        }
    }
}

fn storage_entries<'a>(
    declarations: &[dynamis_gpu::ShaderBinding],
    slots: &[(&str, GpuSlot<'a>)],
) -> Vec<BindGroupEntry<'a>> {
    let mut storage = Vec::with_capacity(slots.len());
    for (name, slot) in slots {
        let declaration = declarations
            .iter()
            .find(|entry| entry.group == 0 && entry.name == *name)
            .unwrap_or_else(|| panic!("the shader never declares the binding {name:?}"));
        assert!(
            !storage
                .iter()
                .any(|entry: &BindGroupEntry| entry.binding == declaration.binding),
            "the shader binding {name:?} is bound twice"
        );
        storage.push(BindGroupEntry {
            binding: declaration.binding,
            resource: slot.as_binding(),
        });
    }
    storage
}

fn shape_entries<'a>(
    declarations: &[dynamis_gpu::ShaderBinding],
    shapes: &[(&str, GpuSlot<'a>)],
) -> Vec<BindGroupEntry<'a>> {
    if shapes.is_empty() {
        return Vec::new();
    }
    let declared = ordered(declarations, 1);
    assert!(
        declared.len() == shapes.len(),
        "the shader declares {} shape bindings but the stage provides {}",
        declared.len(),
        shapes.len()
    );
    declared
        .iter()
        .map(|declaration| {
            let (_, slot) = shapes
                .iter()
                .find(|(name, _)| *name == declaration.name)
                .unwrap_or_else(|| {
                    panic!(
                        "the shader never declares the binding {:?}",
                        declaration.name
                    )
                });
            BindGroupEntry {
                binding: declaration.binding,
                resource: slot.as_binding(),
            }
        })
        .collect()
}

fn ordered(
    declarations: &[dynamis_gpu::ShaderBinding],
    group: u32,
) -> Vec<&dynamis_gpu::ShaderBinding> {
    let mut declared = declarations
        .iter()
        .filter(|entry| entry.group == group)
        .collect::<Vec<_>>();
    declared.sort_by_key(|entry| entry.binding);
    for (position, entry) in declared.iter().enumerate() {
        assert!(
            entry.binding == position as u32,
            "the shader binds group {group} index {} where {position} is required",
            entry.binding
        );
    }
    declared
}

fn specs_of(declarations: &[dynamis_gpu::ShaderBinding], group: u32) -> Vec<BindingSpec> {
    ordered(declarations, group)
        .into_iter()
        .map(|entry| {
            assert!(
                group == 0 || entry.kind == BindingKind::ReadOnlyStorage,
                "group 1 bindings must be read only"
            );
            BindingSpec {
                binding: entry.binding,
                kind: entry.kind,
            }
        })
        .collect()
}

impl Stage {
    pub(crate) fn build(
        context: &GpuContext,
        label: &str,
        program: Program,
        slots: &[(&str, GpuSlot)],
        shapes: &[(&str, GpuSlot)],
    ) -> Self {
        let Program {
            source,
            dispatch,
            warm,
        } = program;
        let resolved = Bindings::of(label, &source, StageBindings { slots, shapes });
        let groups = resolved
            .groups
            .iter()
            .map(|group| group.as_slice())
            .collect::<Vec<_>>();
        let pipeline = context.declare(ComputeProgram::new(label, source.clone(), "main", &groups));
        let warm = warm.then(|| Warm {
            pipeline: context.declare(ComputeProgram::new(
                &format!("{label} warm"),
                source,
                "warm",
                &groups,
            )),
            workgroups: match dispatch {
                Dispatch::Stream { workgroups } => workgroups,
                _ => panic!("a warm entry point must stream"),
            },
        });
        let bind_group = pipeline.create_bind_group(context.device(), 0, &resolved.storage);
        let shapes_group = (!resolved.shapes.is_empty())
            .then(|| pipeline.create_bind_group(context.device(), 1, &resolved.shapes));
        Self {
            pipeline,
            dispatch,
            warm,
            bind_group,
            shapes_group,
        }
    }

    pub(crate) fn record_rows(&self, recorder: &mut ComputeRecorder, rows: u32) {
        assert!(
            matches!(self.dispatch, Dispatch::Rows),
            "an element stage derives its dispatch from its own row count"
        );
        self.record_handle(recorder, &self.pipeline, workgroups_of(rows));
    }

    pub(crate) fn record_stream(&self, recorder: &mut ComputeRecorder) {
        let Dispatch::Stream { workgroups } = self.dispatch else {
            panic!("a streaming stage captures its dispatch from its stream capacity");
        };
        self.record_handle(recorder, &self.pipeline, workgroups);
    }

    pub(crate) fn record_warm(&self, recorder: &mut ComputeRecorder) {
        let warm = self
            .warm
            .as_ref()
            .expect("a warm recording requires a warm entry point");
        self.record_handle(recorder, &warm.pipeline, warm.workgroups);
    }

    pub(crate) fn record_workgroups(&self, recorder: &mut ComputeRecorder, workgroups: u32) {
        assert!(
            matches!(self.dispatch, Dispatch::Workgroups),
            "a workgroup stage takes an explicit dispatch count"
        );
        self.record_handle(recorder, &self.pipeline, workgroups);
    }

    fn record_handle(
        &self,
        recorder: &mut ComputeRecorder,
        pipeline: &PipelineHandle,
        workgroups: u32,
    ) {
        let compiled = pipeline.pipeline();
        match &self.shapes_group {
            Some(shapes) => recorder.record(compiled, &[&self.bind_group, shapes], workgroups),
            None => recorder.record(compiled, &[&self.bind_group], workgroups),
        }
    }
}
