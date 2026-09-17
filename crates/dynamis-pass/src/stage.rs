use crate::binding::{Binding, BindingTable};
use dynamis_gpu::{
    ComputeProgram, ComputeRecorder, GpuContext, PipelineHandle, ResourceId, ResourceSource,
    SlotRef, StorageId,
};
use dynamis_shader::{Dispatch, Live, Program, workgroups_of};
use wgpu::{BindGroup, BindGroupEntry, Device};

pub const MAX_DISPATCH_WORKGROUPS: u32 = 4096;

const LIVE_WORKGROUP_FLOOR: u32 = 16;

pub trait PassRuntime<F> {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self
    where
        Self: Sized;

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &F,
    );
}

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
    fn of<R: ResourceSource>(
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

    fn holds<R: ResourceSource>(
        &self,
        resources: &R,
        storage: &[Binding],
        shapes: &[Binding],
    ) -> bool {
        self.ids.len() == storage.len() + shapes.len()
            && self.ids.iter().copied().eq(storage
                .iter()
                .chain(shapes)
                .map(|binding| binding.storage_id(resources)))
    }
}

fn ids<R: ResourceSource>(
    resources: &R,
    storage: &[Binding],
    shapes: &[Binding],
) -> Vec<StorageId> {
    storage
        .iter()
        .chain(shapes)
        .map(|binding| binding.storage_id(resources))
        .collect()
}

fn entries<'a, R: ResourceSource>(
    resources: &'a R,
    bindings: &[Binding],
) -> Vec<BindGroupEntry<'a>> {
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
    source: Option<ResourceId>,
    live: Option<Live>,
    warm: Option<PipelineHandle>,
    storage: Vec<Binding>,
    shapes: Vec<Binding>,
    bound: Bound,
}

impl Stage {
    pub fn build<R: ResourceSource>(
        context: &GpuContext,
        label: &str,
        program: Program,
        resources: &R,
        slots: &[(&'static str, SlotRef)],
        shapes: &[(&'static str, SlotRef)],
    ) -> Self {
        let (source, declarations, dispatch, extent, warm) = program.into_parts();
        let bindings = BindingTable::new(declarations);
        let storage = bindings.table(label, 0, slots);
        let extent_source = match dispatch {
            Dispatch::Stream => Some(
                extent
                    .assert_declared(label, slots)
                    .expect("a streaming stage declares the device fact its work covers"),
            ),
            Dispatch::Rows | Dispatch::Workgroups => {
                assert!(
                    extent.is_none(),
                    "{label:?} declares a work extent without streaming over it",
                );
                None
            }
        };
        let live = match dispatch {
            Dispatch::Stream => Some(extent.counter(label, slots)),
            Dispatch::Rows | Dispatch::Workgroups => None,
        };
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
            source: extent_source,
            live,
            warm,
            storage,
            shapes,
            bound,
        }
    }

    pub fn record_rows<R: ResourceSource>(
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

    pub fn record_stream<R: ResourceSource>(
        &mut self,
        recorder: &mut ComputeRecorder,
        resources: &R,
    ) {
        let workgroups = self.stream_workgroups(resources);
        self.record(recorder, resources, Entry::Main, workgroups);
    }

    pub fn record_warm<R: ResourceSource>(
        &mut self,
        recorder: &mut ComputeRecorder,
        resources: &R,
    ) {
        let workgroups = self.stream_workgroups(resources);
        self.record(recorder, resources, Entry::Warm, workgroups);
    }

    fn stream_workgroups<R: ResourceSource>(&self, resources: &R) -> u32 {
        assert!(
            matches!(self.dispatch, Dispatch::Stream),
            "a streaming stage declares the device fact it covers",
        );
        let source = self
            .source
            .expect("a streaming stage declares the stream its work covers");
        let slots = resources.slots(source);
        let capacity = workgroups_of(slots).min(MAX_DISPATCH_WORKGROUPS);
        let live = self
            .live
            .and_then(|live| live.measured(resources))
            .map_or(capacity, |measured| {
                workgroups_of(measured.min(slots)).min(MAX_DISPATCH_WORKGROUPS)
            });
        live.max(capacity / LIVE_WORKGROUP_FLOOR).max(1)
    }

    pub fn record_workgroups<R: ResourceSource>(
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

    pub fn record_workgroups_warm<R: ResourceSource>(
        &mut self,
        recorder: &mut ComputeRecorder,
        resources: &R,
        workgroups: u32,
    ) {
        assert!(
            matches!(self.dispatch, Dispatch::Workgroups),
            "a workgroup stage takes an explicit dispatch count"
        );
        self.record(recorder, resources, Entry::Warm, workgroups);
    }

    fn record<R: ResourceSource>(
        &mut self,
        recorder: &mut ComputeRecorder,
        resources: &R,
        entry: Entry,
        workgroups: u32,
    ) {
        if workgroups == 0 {
            return;
        }
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
