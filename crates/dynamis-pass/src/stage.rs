use crate::bindings::{Binding, Bindings};
use dynamis_gpu::{
    ComputeProgram, ComputeRecorder, GpuContext, PipelineHandle, Resources, SlotRef, StorageId,
};
use dynamis_shader::{Dispatch, Program, workgroups_of};
use wgpu::{BindGroup, BindGroupEntry, Device};

pub const MAX_DISPATCH_WORKGROUPS: u32 = 4096;

#[derive(Clone, Copy)]
enum Entry {
    Main,
    Warm,
}

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
    bound: Bound,
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
            bound,
        }
    }

    pub fn record_rows<R: Resources>(
        &mut self,
        recorder: &mut ComputeRecorder,
        resources: &R,
        rows: u32,
    ) {
        assert!(
            matches!(self.dispatch, Dispatch::Rows),
            "an element stage derives its dispatch from its own row count"
        );
        let workgroups = workgroups_of(rows);
        self.record(recorder, resources, Entry::Main, workgroups);
    }

    pub fn record_stream<R: Resources>(&mut self, recorder: &mut ComputeRecorder, resources: &R) {
        let workgroups = self.stream_workgroups(resources);
        self.record(recorder, resources, Entry::Main, workgroups);
    }

    pub fn record_warm<R: Resources>(&mut self, recorder: &mut ComputeRecorder, resources: &R) {
        let workgroups = self.stream_workgroups(resources);
        self.record(recorder, resources, Entry::Warm, workgroups);
    }

    fn stream_workgroups<R: Resources>(&self, resources: &R) -> u32 {
        let Dispatch::Stream(resource) = self.dispatch else {
            panic!("a streaming stage declares its own stream extent");
        };
        workgroups_of(resources.slots(resource)).min(MAX_DISPATCH_WORKGROUPS)
    }

    pub fn record_workgroups<R: Resources>(
        &mut self,
        recorder: &mut ComputeRecorder,
        resources: &R,
        workgroups: u32,
    ) {
        assert!(
            matches!(self.dispatch, Dispatch::Workgroups),
            "a workgroup stage takes an explicit dispatch count"
        );
        self.record(recorder, resources, Entry::Main, workgroups);
    }

    fn record<R: Resources>(
        &mut self,
        recorder: &mut ComputeRecorder,
        resources: &R,
        entry: Entry,
        workgroups: u32,
    ) {
        let Stage {
            device,
            pipeline: main,
            warm,
            storage,
            shapes,
            bound,
            ..
        } = self;
        if !bound.holds(resources, storage, shapes) {
            *bound = Bound::of(main, device, resources, storage, shapes);
        }
        let compiled = match entry {
            Entry::Main => main.pipeline(),
            Entry::Warm => warm
                .as_mut()
                .expect("a warm recording requires a warm entry point")
                .pipeline(),
        };
        match &bound.shapes {
            Some(shape_group) => {
                recorder.record(compiled, &[&bound.storage, shape_group], workgroups)
            }
            None => recorder.record(compiled, &[&bound.storage], workgroups),
        }
    }
}
