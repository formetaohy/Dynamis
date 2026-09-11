use super::shader::{
    CONSTRAINT_BLOCK_FRAGMENT, CONTACT_BLOCK_FRAGMENT, CONTACT_CORRECTION_FRAGMENT,
    CONVEX_FRAGMENT, EVENTS_FRAGMENT, IDENTITY_FRAGMENT, SCENE_FRAGMENT, WORKGROUP_SIZE,
    assemble_shader,
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
pub(super) const BLOCKS: &[&str] = &[CONTACT_BLOCK_FRAGMENT, CONSTRAINT_BLOCK_FRAGMENT];
pub(super) const CORRECTIONS: &[&str] = &[CONTACT_CORRECTION_FRAGMENT];

pub(super) const MAX_GRID_WORKGROUPS: u32 = 4096;

pub(super) const RO: BindingKind = BindingKind::ReadOnlyStorage;
pub(super) const RW: BindingKind = BindingKind::ReadWriteStorage;
pub(super) const UNIFORM: BindingKind = BindingKind::Uniform;

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
    storage: &'a [(BindingKind, GpuSlot<'a>)],
    shapes: &'a [&'a GpuBuffer],
}

impl Stage {
    pub(super) fn build(
        context: &GpuContext,
        label: &str,
        body: &str,
        per_row: u32,
        fragments: &[&str],
        bindings: &[(BindingKind, GpuSlot)],
        shape_resources: &[&GpuBuffer],
    ) -> Self {
        Self::assemble(
            context,
            label,
            body,
            per_row,
            fragments,
            StageBindings {
                storage: bindings,
                shapes: shape_resources,
            },
            false,
        )
    }

    pub(super) fn build_warm(
        context: &GpuContext,
        label: &str,
        body: &str,
        per_row: u32,
        fragments: &[&str],
        bindings: &[(BindingKind, GpuSlot)],
        shape_resources: &[&GpuBuffer],
    ) -> Self {
        Self::assemble(
            context,
            label,
            body,
            per_row,
            fragments,
            StageBindings {
                storage: bindings,
                shapes: shape_resources,
            },
            true,
        )
    }

    fn assemble(
        context: &GpuContext,
        label: &str,
        body: &str,
        per_row: u32,
        fragments: &[&str],
        bindings: StageBindings,
        with_warm: bool,
    ) -> Self {
        let shader = assemble_shader(body, per_row, fragments);
        let shape_specs = bindings
            .shapes
            .iter()
            .enumerate()
            .map(|(position, _)| BindingSpec {
                binding: position as u32,
                kind: BindingKind::ReadOnlyStorage,
            })
            .collect::<Vec<_>>();
        let specs = bindings
            .storage
            .iter()
            .enumerate()
            .map(|(position, (kind, _))| BindingSpec {
                binding: position as u32,
                kind: *kind,
            })
            .collect::<Vec<_>>();
        let groups = if shape_specs.is_empty() {
            vec![&specs[..]]
        } else {
            vec![&specs[..], &shape_specs[..]]
        };
        let pipeline = context.declare(ComputeProgram::new(label, shader.clone(), "main", &groups));
        let warm = with_warm.then(|| {
            context.declare(ComputeProgram::new(
                &format!("{label} warm"),
                shader,
                "warm",
                &groups,
            ))
        });
        let entries: Vec<BindGroupEntry> = bindings
            .storage
            .iter()
            .enumerate()
            .map(|(position, (_, slot))| BindGroupEntry {
                binding: position as u32,
                resource: slot.as_binding(),
            })
            .collect();
        let bind_group = pipeline.create_bind_group(context.device(), 0, &entries);
        let shapes_group = if bindings.shapes.is_empty() {
            None
        } else {
            let entries = bindings
                .shapes
                .iter()
                .enumerate()
                .map(|(position, buffer)| BindGroupEntry {
                    binding: position as u32,
                    resource: buffer.as_binding(),
                })
                .collect::<Vec<_>>();
            Some(pipeline.create_bind_group(context.device(), 1, &entries))
        };
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
