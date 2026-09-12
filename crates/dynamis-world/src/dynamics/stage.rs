use super::shader::{
    CONSTRAINT_BLOCK_FRAGMENT, CONTACT_BLOCK_FRAGMENT, CONTACT_CORRECTION_FRAGMENT,
    CONVEX_FRAGMENT, EVENTS_FRAGMENT, GRID_INDEX_FRAGMENT, IDENTITY_FRAGMENT, SCENE_FRAGMENT,
    WORKGROUP_SIZE, assemble_shader,
};
use crate::dynamics::buffers::WorldBuffers;
use dynamis_gpu::{
    BindingKind, BindingSpec, ComputeProgram, ComputeRecorder, GpuBuffer, GpuContext, GpuSlot,
    PipelineHandle,
};
use wgpu::{BindGroup, BindGroupEntry};

pub(super) const CORE: &[&str] = &[];
pub(super) const IDENTITY: &[&str] = &[IDENTITY_FRAGMENT];
pub(super) const CONTACT: &[&str] = &[IDENTITY_FRAGMENT, EVENTS_FRAGMENT];
pub(super) const GEOMETRY: &[&str] = &[CONVEX_FRAGMENT, SCENE_FRAGMENT];
pub(super) const GRID_INDEX: &[&str] = &[GRID_INDEX_FRAGMENT];
pub(super) const GEOMETRY_INDEX: &[&str] = &[GRID_INDEX_FRAGMENT, CONVEX_FRAGMENT, SCENE_FRAGMENT];
pub(super) const BLOCKS: &[&str] = &[CONTACT_BLOCK_FRAGMENT, CONSTRAINT_BLOCK_FRAGMENT];
pub(super) const CORRECTIONS: &[&str] = &[CONTACT_CORRECTION_FRAGMENT];

pub(super) const MAX_GRID_WORKGROUPS: u32 = 4096;

pub(super) fn whole(buffer: &GpuBuffer) -> GpuSlot<'_> {
    GpuSlot::whole(buffer)
}

pub(super) fn shape_resources(buffers: &WorldBuffers) -> [&GpuBuffer; 4] {
    [
        &buffers.shapes.sources,
        &buffers.shapes.vertices,
        &buffers.shapes.triangles,
        &buffers.shapes.nodes,
    ]
}

pub(super) struct Stage {
    pipeline: PipelineHandle,
    warm: Option<PipelineHandle>,
    bind_group: BindGroup,
    shapes_group: Option<BindGroup>,
}

struct StageBindings<'a> {
    slots: &'a [(&'a str, GpuSlot<'a>)],
    shapes: &'a [&'a GpuBuffer],
}

struct Bindings<'a> {
    groups: Vec<Vec<BindingSpec>>,
    storage: Vec<BindGroupEntry<'a>>,
    shapes: Vec<BindGroupEntry<'a>>,
}

impl<'a> Bindings<'a> {
    fn of(source: &str, declared: StageBindings<'a>) -> Self {
        let StageBindings { slots, shapes } = declared;
        let declarations = dynamis_gpu::parse_bindings(source);
        let storage = storage_entries(&declarations, slots);
        assert!(
            storage.len() == specs_of(&declarations, 0).len(),
            "the shader declares {} bindings but the stage provides {}",
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
    shapes: &'a [&'a GpuBuffer],
) -> Vec<BindGroupEntry<'a>> {
    let declared = ordered(declarations, 1);
    if shapes.is_empty() {
        return Vec::new();
    }
    assert!(
        declared.len() == shapes.len(),
        "the shader declares {} shape bindings but the stage provides {}",
        declared.len(),
        shapes.len()
    );
    declared
        .iter()
        .zip(shapes)
        .map(|(declaration, buffer)| BindGroupEntry {
            binding: declaration.binding,
            resource: buffer.as_binding(),
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
        per_row: u32,
        fragments: &[&str],
        slots: &[(&str, GpuSlot)],
        shapes: &[&GpuBuffer],
    ) -> Self {
        Self::assemble(
            context,
            label,
            body,
            per_row,
            fragments,
            StageBindings { slots, shapes },
            false,
        )
    }

    pub(super) fn build_warm(
        context: &GpuContext,
        label: &str,
        body: &str,
        per_row: u32,
        fragments: &[&str],
        slots: &[(&str, GpuSlot)],
        shapes: &[&GpuBuffer],
    ) -> Self {
        Self::assemble(
            context,
            label,
            body,
            per_row,
            fragments,
            StageBindings { slots, shapes },
            true,
        )
    }

    fn assemble(
        context: &GpuContext,
        label: &str,
        body: &str,
        per_row: u32,
        fragments: &[&str],
        declared: StageBindings,
        with_warm: bool,
    ) -> Self {
        let shader = assemble_shader(body, per_row, fragments);
        let resolved = Bindings::of(&shader, declared);
        let groups = resolved
            .groups
            .iter()
            .map(|group| group.as_slice())
            .collect::<Vec<_>>();
        let pipeline = context.declare(ComputeProgram::new(label, shader.clone(), "main", &groups));
        let warm = with_warm.then(|| {
            context.declare(ComputeProgram::new(
                &format!("{label} warm"),
                shader,
                "warm",
                &groups,
            ))
        });
        let bind_group = pipeline.create_bind_group(context.device(), 0, &resolved.storage);
        let shapes_group = (!resolved.shapes.is_empty())
            .then(|| pipeline.create_bind_group(context.device(), 1, &resolved.shapes));
        Self {
            pipeline,
            warm,
            bind_group,
            shapes_group,
        }
    }

    pub(super) fn record(&self, recorder: &mut ComputeRecorder, elements: u32) {
        self.record_workgroups(recorder, elements.div_ceil(WORKGROUP_SIZE));
    }

    pub(super) fn record_stride(&self, recorder: &mut ComputeRecorder, bound: u32) {
        let workgroups = bound.div_ceil(WORKGROUP_SIZE).min(MAX_GRID_WORKGROUPS);
        self.record_workgroups(recorder, workgroups);
    }

    pub(super) fn record_workgroups(&self, recorder: &mut ComputeRecorder, workgroups: u32) {
        let compiled = self.pipeline.pipeline();
        match &self.shapes_group {
            Some(shapes) => recorder.record(compiled, &[&self.bind_group, shapes], workgroups),
            None => recorder.record(compiled, &[&self.bind_group], workgroups),
        }
    }

    pub(super) fn record_warm_stride(&self, recorder: &mut ComputeRecorder, bound: u32) {
        let warm = self
            .warm
            .as_ref()
            .expect("warm recording requires a warm entry point")
            .pipeline();
        let workgroups = bound.div_ceil(WORKGROUP_SIZE).min(MAX_GRID_WORKGROUPS);
        match &self.shapes_group {
            Some(shapes) => recorder.record(warm, &[&self.bind_group, shapes], workgroups),
            None => recorder.record(warm, &[&self.bind_group], workgroups),
        }
    }
}
