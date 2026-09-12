use super::shader::{
    CONSTRAINT_BLOCK_FRAGMENT, CONTACT_BLOCK_FRAGMENT, CONVEX_FRAGMENT, EVENTS_FRAGMENT,
    GRID_INDEX_FRAGMENT, IDENTITY_FRAGMENT, POSITION_CORRECTION_FRAGMENT, SCENE_FRAGMENT,
    WORKGROUP_SIZE, assemble_shader,
};
use crate::dynamics::Frame;
use crate::dynamics::buffers::WorldBuffers;
use dynamis_gpu::{
    BindingKind, BindingSpec, ComputeProgram, ComputeRecorder, GpuContext, GpuSlot, PipelineHandle,
};
use dynamis_layout::StepParamsRecord;
use wgpu::{BindGroup, BindGroupEntry};

pub(super) const CORE: &[&str] = &[];
pub(super) const IDENTITY: &[&str] = &[IDENTITY_FRAGMENT];
pub(super) const CONTACT: &[&str] = &[IDENTITY_FRAGMENT, EVENTS_FRAGMENT];
pub(super) const GEOMETRY: &[&str] = &[CONVEX_FRAGMENT, SCENE_FRAGMENT];
pub(super) const GRID_INDEX: &[&str] = &[GRID_INDEX_FRAGMENT];
pub(super) const GEOMETRY_INDEX: &[&str] = &[GRID_INDEX_FRAGMENT, CONVEX_FRAGMENT, SCENE_FRAGMENT];
pub(super) const BLOCKS: &[&str] = &[CONTACT_BLOCK_FRAGMENT, CONSTRAINT_BLOCK_FRAGMENT];
pub(super) const POSITION_CORRECTION: &[&str] = &[POSITION_CORRECTION_FRAGMENT];

pub(super) const MAX_GRID_WORKGROUPS: u32 = 4096;

pub(super) fn workgroups_of(elements: u32) -> u32 {
    elements.div_ceil(WORKGROUP_SIZE)
}

#[derive(Clone, Copy)]
pub(super) enum Count {
    Bodies,
    Dynamic,
    Colliders,
    Constraints,
    EditRuns,
    BodyMoves,
    ConstraintMoves,
}

impl Count {
    const fn field(self) -> &'static str {
        match self {
            Self::Bodies => "body_count",
            Self::Dynamic => "dynamic_count",
            Self::Colliders => "collider_count",
            Self::Constraints => "constraint_count",
            Self::EditRuns => "edit_run_count",
            Self::BodyMoves => "body_move_count",
            Self::ConstraintMoves => "constraint_move_count",
        }
    }

    fn of(self, params: &StepParamsRecord) -> u32 {
        match self {
            Self::Bodies => params.body_count,
            Self::Dynamic => params.dynamic_count,
            Self::Colliders => params.collider_count,
            Self::Constraints => params.constraint_count,
            Self::EditRuns => params.edit_run_count,
            Self::BodyMoves => params.body_move_count,
            Self::ConstraintMoves => params.constraint_move_count,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Slots {
    Entries,
    Pairs,
    Contacts,
    Archive,
    Resting,
    Blocks,
}

impl Slots {
    fn of(self, buffers: &WorldBuffers) -> u32 {
        match self {
            Self::Entries => buffers.entry_capacity(),
            Self::Pairs => buffers.pair_capacity(),
            Self::Contacts => buffers.contact_capacity(),
            Self::Archive => buffers.archive_capacity(),
            Self::Resting => buffers.resting_capacity(),
            Self::Blocks => buffers.block_capacity(),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Coverage {
    Live(Count),
    Stream { kernel: &'static str, slots: Slots },
    Workgroups,
}

impl Coverage {
    fn workgroups(&self, buffers: &WorldBuffers, frame: &Frame) -> u32 {
        match self {
            Self::Live(count) => workgroups_of(count.of(&frame.params)),
            Self::Stream { slots, .. } => workgroups_of(slots.of(buffers)).min(MAX_GRID_WORKGROUPS),
            Self::Workgroups => panic!("a workgroup stage needs an explicit dispatch count"),
        }
    }
}

fn element_entry(count: Count) -> String {
    let field = count.field();
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

fn stream_entry(entry: &str, kernel: &str) -> String {
    assert!(
        kernel != "main" && kernel != "warm",
        "the streaming kernel {kernel:?} collides with the generated entry point"
    );
    format!(
        "
@compute @workgroup_size(WORKGROUP_SIZE)
fn {entry}(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {{
    let live = extent();
    let stride = grid_stride(groups);
    for (var index = global_index(gid); index < live; index = index + stride) {{
        {kernel}(index);
    }}
}}
"
    )
}

fn assemble_body(
    body: &str,
    per_row: u32,
    fragments: &[&str],
    entries: [Option<Coverage>; 2],
) -> String {
    let mut source = assemble_shader(body, per_row, fragments);
    match entries[0] {
        Some(Coverage::Live(count)) => source.push_str(&element_entry(count)),
        Some(Coverage::Stream { kernel, .. }) => source.push_str(&stream_entry("main", kernel)),
        _ => {}
    }
    if let Some(coverage) = entries[1] {
        match coverage {
            Coverage::Stream { kernel, .. } => source.push_str(&stream_entry("warm", kernel)),
            _ => panic!("a warm entry point must stream"),
        }
    }
    source
}

pub(super) fn whole<'a>(binding: impl Into<GpuSlot<'a>>) -> GpuSlot<'a> {
    binding.into()
}

pub(super) fn shape_resources(buffers: &WorldBuffers) -> [(&'static str, GpuSlot<'_>); 4] {
    [
        ("shape_sources", buffers.shape_sources.slot()),
        ("shape_vertices", buffers.shape_vertices.slot()),
        ("shape_triangles", buffers.shape_triangles.slot()),
        ("shape_nodes", buffers.shape_nodes.slot()),
    ]
}

struct Warm {
    pipeline: PipelineHandle,
    coverage: Coverage,
}

pub(super) struct Stage {
    pipeline: PipelineHandle,
    coverage: Coverage,
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
    pub(super) fn build(
        context: &GpuContext,
        label: &str,
        body: &str,
        fragments: &[&str],
        coverage: Coverage,
        slots: &[(&str, GpuSlot)],
        shapes: &[(&str, GpuSlot)],
    ) -> Self {
        Self::assemble(
            context,
            label,
            body,
            fragments,
            [Some(coverage), None],
            StageBindings { slots, shapes },
        )
    }

    pub(super) fn build_warm(
        context: &GpuContext,
        label: &str,
        body: &str,
        fragments: &[&str],
        entries: [Coverage; 2],
        slots: &[(&str, GpuSlot)],
        shapes: &[(&str, GpuSlot)],
    ) -> Self {
        Self::assemble(
            context,
            label,
            body,
            fragments,
            [Some(entries[0]), Some(entries[1])],
            StageBindings { slots, shapes },
        )
    }

    fn assemble(
        context: &GpuContext,
        label: &str,
        body: &str,
        fragments: &[&str],
        entries: [Option<Coverage>; 2],
        declared: StageBindings,
    ) -> Self {
        let coverage = entries[0].expect("the main entry always declares its coverage");
        let shader: std::sync::Arc<str> =
            assemble_body(body, context.workgroups_per_row(), fragments, entries).into();
        let resolved = Bindings::of(label, &shader, declared);
        let groups = resolved
            .groups
            .iter()
            .map(|group| group.as_slice())
            .collect::<Vec<_>>();
        let pipeline = context.declare(ComputeProgram::new(label, shader.clone(), "main", &groups));
        let warm = entries[1].map(|coverage| Warm {
            pipeline: context.declare(ComputeProgram::new(
                &format!("{label} warm"),
                shader,
                "warm",
                &groups,
            )),
            coverage,
        });
        let bind_group = pipeline.create_bind_group(context.device(), 0, &resolved.storage);
        let shapes_group = (!resolved.shapes.is_empty())
            .then(|| pipeline.create_bind_group(context.device(), 1, &resolved.shapes));
        Self {
            pipeline,
            coverage,
            warm,
            bind_group,
            shapes_group,
        }
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        frame: &Frame,
    ) {
        assert!(
            !matches!(self.coverage, Coverage::Workgroups),
            "a workgroup stage needs an explicit dispatch count"
        );
        let workgroups = self.coverage.workgroups(buffers, frame);
        self.record_handle(recorder, &self.pipeline, workgroups);
    }

    pub(super) fn record_warm(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        frame: &Frame,
    ) {
        let warm = self
            .warm
            .as_ref()
            .expect("a warm recording requires a warm entry point");
        let workgroups = warm.coverage.workgroups(buffers, frame);
        self.record_handle(recorder, &warm.pipeline, workgroups);
    }

    pub(super) fn record_workgroups(&self, recorder: &mut ComputeRecorder, workgroups: u32) {
        assert!(
            matches!(self.coverage, Coverage::Workgroups),
            "an element stage derives its dispatch from the frame"
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
