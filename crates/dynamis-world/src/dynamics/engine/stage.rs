use dynamis_gpu::{
    BindingKind, BindingSpec, ComputeProgram, ComputeRecorder, GpuContext, GpuSlot, PipelineHandle,
    ShaderBinding,
};
use std::cell::{RefCell, RefMut};
use std::sync::Arc;
use wgpu::{BindGroup, BindGroupEntry, Device};

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

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResourceId(u32);

impl ResourceId {
    pub(crate) const fn new(index: u32) -> Self {
        Self(index)
    }

    pub(crate) const fn index(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy)]
pub(crate) enum SlotRef {
    Whole(ResourceId),
    Range {
        resource: ResourceId,
        offset: u64,
        size: u64,
    },
}

impl SlotRef {
    pub(crate) const fn whole(resource: ResourceId) -> Self {
        Self::Whole(resource)
    }

    pub(crate) const fn range(resource: ResourceId, offset: u64, size: u64) -> Self {
        Self::Range {
            resource,
            offset,
            size,
        }
    }

    pub(crate) fn resolve<R: Resources>(self, resources: &R) -> GpuSlot<'_> {
        match self {
            Self::Whole(resource) => resources.whole(resource),
            Self::Range {
                resource,
                offset,
                size,
            } => resources.range(resource, offset, size),
        }
    }
}

pub(crate) trait Resources {
    fn generation(&self) -> u64;

    fn slots(&self, resource: ResourceId) -> u32;

    fn whole(&self, resource: ResourceId) -> GpuSlot<'_>;

    fn range(&self, resource: ResourceId, offset: u64, size: u64) -> GpuSlot<'_>;
}

#[derive(Clone, Copy)]
pub(crate) enum Dispatch {
    Rows,
    Stream(ResourceId),
    Workgroups,
}

pub(crate) struct Program {
    pub(crate) source: Arc<str>,
    pub(crate) dispatch: Dispatch,
    pub(crate) warm: bool,
}

#[derive(Clone, Copy)]
struct Binding {
    index: u32,
    slot: SlotRef,
}

struct Bound {
    generation: u64,
    storage: BindGroup,
    shapes: Option<BindGroup>,
}

impl Bound {
    fn of<R: Resources>(
        pipeline: &PipelineHandle,
        device: &Device,
        resources: &R,
        storage: &[Binding],
        shapes: &[Binding],
    ) -> Self {
        let group = pipeline.create_bind_group(device, 0, &entries(resources, storage));
        let shape_group = (!shapes.is_empty())
            .then(|| pipeline.create_bind_group(device, 1, &entries(resources, shapes)));
        Self {
            generation: resources.generation(),
            storage: group,
            shapes: shape_group,
        }
    }
}

fn entries<'a, R: Resources>(resources: &'a R, bindings: &[Binding]) -> Vec<BindGroupEntry<'a>> {
    bindings
        .iter()
        .map(|binding| BindGroupEntry {
            binding: binding.index,
            resource: binding.slot.resolve(resources).as_binding(),
        })
        .collect()
}

pub(crate) struct Stage {
    device: Device,
    pipeline: PipelineHandle,
    dispatch: Dispatch,
    warm: Option<PipelineHandle>,
    storage: Vec<Binding>,
    shapes: Vec<Binding>,
    bound: RefCell<Bound>,
}

impl Stage {
    pub(crate) fn build<R: Resources>(
        context: &GpuContext,
        label: &str,
        program: Program,
        resources: &R,
        slots: &[(&'static str, SlotRef)],
        shapes: &[(&'static str, SlotRef)],
    ) -> Self {
        let Program {
            source,
            dispatch,
            warm,
        } = program;
        let declarations = dynamis_gpu::parse_bindings(&source);
        let storage = storage_bindings(label, &declarations, slots);
        let shapes = shape_bindings(label, &declarations, shapes);
        let mut groups = vec![specs_of(&declarations, 0)];
        if !shapes.is_empty() {
            groups.push(specs_of(&declarations, 1));
        }
        let layout = groups.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let pipeline = context.declare(ComputeProgram::new(label, source.clone(), "main", &layout));
        let warm = warm.then(|| {
            context.declare(ComputeProgram::new(
                &format!("{label} warm"),
                source,
                "warm",
                &layout,
            ))
        });
        let bound = Bound::of(&pipeline, context.device(), resources, &storage, &shapes);
        Self {
            device: context.device().clone(),
            pipeline,
            dispatch,
            warm,
            storage,
            shapes,
            bound: RefCell::new(bound),
        }
    }

    pub(crate) fn record_rows<R: Resources>(
        &self,
        recorder: &mut ComputeRecorder,
        resources: &R,
        rows: u32,
    ) {
        assert!(
            matches!(self.dispatch, Dispatch::Rows),
            "an element stage derives its dispatch from its own row count"
        );
        self.record(recorder, resources, &self.pipeline, workgroups_of(rows));
    }

    pub(crate) fn record_stream<R: Resources>(
        &self,
        recorder: &mut ComputeRecorder,
        resources: &R,
    ) {
        self.record(
            recorder,
            resources,
            &self.pipeline,
            self.stream_workgroups(resources),
        );
    }

    pub(crate) fn record_warm<R: Resources>(&self, recorder: &mut ComputeRecorder, resources: &R) {
        let warm = self
            .warm
            .as_ref()
            .expect("a warm recording requires a warm entry point");
        self.record(recorder, resources, warm, self.stream_workgroups(resources));
    }

    fn stream_workgroups<R: Resources>(&self, resources: &R) -> u32 {
        let Dispatch::Stream(resource) = self.dispatch else {
            panic!("a streaming stage declares its own stream extent");
        };
        workgroups_of(resources.slots(resource)).min(MAX_DISPATCH_WORKGROUPS)
    }

    pub(crate) fn record_workgroups<R: Resources>(
        &self,
        recorder: &mut ComputeRecorder,
        resources: &R,
        workgroups: u32,
    ) {
        assert!(
            matches!(self.dispatch, Dispatch::Workgroups),
            "a workgroup stage takes an explicit dispatch count"
        );
        self.record(recorder, resources, &self.pipeline, workgroups);
    }

    fn record<R: Resources>(
        &self,
        recorder: &mut ComputeRecorder,
        resources: &R,
        pipeline: &PipelineHandle,
        workgroups: u32,
    ) {
        let bound = self.bind(resources);
        let compiled = pipeline.pipeline();
        match &bound.shapes {
            Some(shapes) => recorder.record(compiled, &[&bound.storage, shapes], workgroups),
            None => recorder.record(compiled, &[&bound.storage], workgroups),
        }
    }

    fn bind<R: Resources>(&self, resources: &R) -> RefMut<'_, Bound> {
        let mut current = self.bound.borrow_mut();
        if current.generation != resources.generation() {
            *current = Bound::of(
                &self.pipeline,
                &self.device,
                resources,
                &self.storage,
                &self.shapes,
            );
        }
        current
    }
}

fn storage_bindings(
    label: &str,
    declarations: &[ShaderBinding],
    slots: &[(&'static str, SlotRef)],
) -> Vec<Binding> {
    let mut bindings = Vec::with_capacity(slots.len());
    for (name, slot) in slots {
        let declaration = declarations
            .iter()
            .find(|entry| entry.group == 0 && entry.name == *name)
            .unwrap_or_else(|| panic!("stage {label:?} declares no binding {name:?}"));
        assert!(
            !bindings
                .iter()
                .any(|binding: &Binding| binding.index == declaration.binding),
            "stage {label:?} binds {name:?} twice"
        );
        bindings.push(Binding {
            index: declaration.binding,
            slot: *slot,
        });
    }
    assert!(
        bindings.len() == specs_of(declarations, 0).len(),
        "stage {label:?} declares {} bindings but provides {}",
        specs_of(declarations, 0).len(),
        bindings.len()
    );
    bindings
}

fn shape_bindings(
    label: &str,
    declarations: &[ShaderBinding],
    shapes: &[(&'static str, SlotRef)],
) -> Vec<Binding> {
    if shapes.is_empty() {
        return Vec::new();
    }
    let declared = ordered(declarations, 1);
    assert!(
        declared.len() == shapes.len(),
        "stage {label:?} declares {} shape bindings but provides {}",
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
                        "stage {label:?} declares no shape binding {:?}",
                        declaration.name
                    )
                });
            Binding {
                index: declaration.binding,
                slot: *slot,
            }
        })
        .collect()
}

fn ordered(declarations: &[ShaderBinding], group: u32) -> Vec<&ShaderBinding> {
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

fn specs_of(declarations: &[ShaderBinding], group: u32) -> Vec<BindingSpec> {
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
