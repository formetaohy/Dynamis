use crate::bindings::{Binding, Bindings};
use dynamis_gpu::{
    ComputeProgram, ComputeRecorder, GpuContext, PipelineHandle, Resources, SlotRef, StorageId,
};
use dynamis_shader::{Dispatch, Program, workgroups_of};
use std::cell::{RefCell, RefMut};
use wgpu::{BindGroup, BindGroupEntry, Device};

pub const MAX_DISPATCH_WORKGROUPS: u32 = 4096;

struct Bound {
    storage: BindGroup,
    shapes: Option<BindGroup>,
    ids: Vec<StorageId>,
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
            storage: group,
            shapes: shape_group,
            ids: ids(resources, storage, shapes),
        }
    }

    fn holds<R: Resources>(&self, resources: &R, storage: &[Binding], shapes: &[Binding]) -> bool {
        self.ids.len() == storage.len() + shapes.len()
            && self.ids.iter().copied().eq(storage
                .iter()
                .chain(shapes)
                .map(|binding| binding.storage_id(resources)))
    }
}

fn ids<R: Resources>(resources: &R, storage: &[Binding], shapes: &[Binding]) -> Vec<StorageId> {
    storage
        .iter()
        .chain(shapes)
        .map(|binding| binding.storage_id(resources))
        .collect()
}

fn entries<'a, R: Resources>(resources: &'a R, bindings: &[Binding]) -> Vec<BindGroupEntry<'a>> {
    bindings
        .iter()
        .map(|binding| BindGroupEntry {
            binding: binding.index,
            resource: binding.slot.resolve(resources).slot().as_binding(),
        })
        .collect()
}

pub struct Stage {
    device: Device,
    pipeline: PipelineHandle,
    dispatch: Dispatch,
    warm: Option<PipelineHandle>,
    storage: Vec<Binding>,
    shapes: Vec<Binding>,
    bound: RefCell<Bound>,
}

impl Stage {
    pub fn build<R: Resources>(
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
        let bindings = Bindings::parse(&source);
        let storage = bindings.table(label, 0, slots);
        let shapes = if shapes.is_empty() {
            Vec::new()
        } else {
            bindings.table(label, 1, shapes)
        };
        let mut groups = vec![bindings.specs(0)];
        if !shapes.is_empty() {
            groups.push(bindings.specs(1));
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

    pub fn record_rows<R: Resources>(
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

    pub fn record_stream<R: Resources>(&self, recorder: &mut ComputeRecorder, resources: &R) {
        self.record(
            recorder,
            resources,
            &self.pipeline,
            self.stream_workgroups(resources),
        );
    }

    pub fn record_warm<R: Resources>(&self, recorder: &mut ComputeRecorder, resources: &R) {
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

    pub fn record_workgroups<R: Resources>(
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
        if !current.holds(resources, &self.storage, &self.shapes) {
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
