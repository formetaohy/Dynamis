use super::shader::{WORKGROUP_SIZE, assemble_shader};
use crate::buffers::WorldBuffers;
use dynamis_gpu::{
    BindingKind, BindingSpec, ComputePipeline, ComputeRecorder, DispatchTable, GpuBuffer,
    GpuContext, GpuSlot,
};
use wgpu::{BindGroup, BindGroupEntry};

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
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    shapes_group: Option<BindGroup>,
}

impl Stage {
    pub(super) fn build(
        context: &GpuContext,
        label: &str,
        body: &str,
        per_row: u32,
        bindings: &[(BindingKind, GpuSlot)],
        shape_resources: &[&GpuBuffer],
    ) -> Self {
        let shader = assemble_shader(body, per_row);
        let shape_specs = shape_resources
            .iter()
            .enumerate()
            .map(|(position, _)| BindingSpec {
                binding: position as u32,
                kind: BindingKind::ReadOnlyStorage,
            })
            .collect::<Vec<_>>();
        let specs = bindings
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
        let pipeline = context.compute_pipeline(label, &shader, "main", &groups, WORKGROUP_SIZE);
        let entries: Vec<BindGroupEntry> = bindings
            .iter()
            .enumerate()
            .map(|(position, (_, slot))| BindGroupEntry {
                binding: position as u32,
                resource: slot.as_binding(),
            })
            .collect();
        let bind_group = pipeline.create_bind_group(context.device(), 0, &entries);
        let shapes_group = if shape_resources.is_empty() {
            None
        } else {
            let entries = shape_resources
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
            bind_group,
            shapes_group,
        }
    }

    /// Dispatches one workgroup per `elements` lanes.
    pub(super) fn record(&self, recorder: &mut ComputeRecorder, elements: u32) {
        let workgroups = elements.div_ceil(WORKGROUP_SIZE);
        self.record_workgroups(recorder, workgroups);
    }

    pub(super) fn record_workgroups(&self, recorder: &mut ComputeRecorder, workgroups: u32) {
        match &self.shapes_group {
            Some(shapes) => {
                recorder.record(&self.pipeline, &[&self.bind_group, shapes], workgroups)
            }
            None => recorder.record(&self.pipeline, &[&self.bind_group], workgroups),
        }
    }

    pub(super) fn record_indirect(
        &self,
        recorder: &mut ComputeRecorder,
        table: &DispatchTable,
        slot: u32,
    ) {
        match &self.shapes_group {
            Some(shapes) => {
                recorder.record_indirect(&self.pipeline, &[&self.bind_group, shapes], table, slot)
            }
            None => recorder.record_indirect(&self.pipeline, &[&self.bind_group], table, slot),
        }
    }
}
