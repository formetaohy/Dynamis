use dynamis_gpu::{
    ComputeRecorder, GpuContext, GpuSlot, ResourceId, ResourceSource, Retention, STREAM, SlotRef,
    Stream, StreamDesc, StreamElement, WarmupBudget, read_regions,
};
use dynamis_pass::Stage;
use dynamis_shader::{Dispatch, Program};
use std::sync::OnceLock;

const SLOT: ResourceId = ResourceId::new(0, 0);
const ELEMENT: StreamElement = StreamElement::new("u32", 4);

const SOURCE: &str = "
@group(0) @binding(0) var<storage, read_write> stage_out: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    stage_out[gid.x] = gid.x + 1u;
}
";

static CONTEXT: OnceLock<GpuContext> = OnceLock::new();

fn shared() -> &'static GpuContext {
    CONTEXT.get_or_init(|| pollster::block_on(GpuContext::new()))
}

struct Slots {
    storage: Stream,
}

impl ResourceSource for Slots {
    fn slots(&self, resource: ResourceId) -> u32 {
        assert_eq!(resource, SLOT, "the stage names one stream");
        self.storage.slots()
    }

    fn whole(&self, resource: ResourceId) -> GpuSlot<'_> {
        assert_eq!(resource, SLOT, "the stage names one stream");
        self.storage.slot()
    }

    fn range(&self, resource: ResourceId, offset: u64, size: u64) -> GpuSlot<'_> {
        assert_eq!(resource, SLOT, "the stage names one stream");
        GpuSlot::range(self.storage.gpu(), offset, size)
    }
}

fn record(context: &GpuContext, stage: &mut Stage, slots: &Slots, workgroups: u32) {
    let mut encoder = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut recorder =
            ComputeRecorder::begin(&mut encoder, "stage", context.workgroups_per_row());
        stage.record_workgroups(&mut recorder, slots, workgroups);
    }
    context.queue().submit([encoder.finish()]);
}

fn assert_filled(context: &GpuContext, slots: &Slots, words: usize) {
    let bytes = (words * 4) as u64;
    let raw = read_regions(
        context.device(),
        context.queue(),
        "stage readback",
        &[(slots.storage.buffer(), 0, bytes)],
    );
    let (chunks, remainder) = raw.as_chunks::<4>();
    assert!(
        remainder.is_empty(),
        "a stage readback must be a whole number of words"
    );
    let written = chunks
        .iter()
        .enumerate()
        .map(|(index, chunk)| (index as u32, u32::from_le_bytes(*chunk)))
        .collect::<Vec<_>>();
    for (index, word) in written {
        assert_eq!(
            word,
            index + 1,
            "word {index} must hold the storage the stage names"
        );
    }
}

#[test]
fn a_stage_follows_the_storage_it_replaces() {
    let context = shared();
    let device = context.device().clone();
    let queue = context.queue().clone();
    let mut slots = Slots {
        storage: Stream::new(
            &device,
            &queue,
            StreamDesc {
                label: "stage out",
                slots: 64,
                element: ELEMENT,
                elements_per_slot: 1,
                usage: STREAM,
                retention: Retention::Durable,
            },
        ),
    };
    let mut stage = Stage::build(
        context,
        "stage",
        Program::new(SOURCE.to_owned(), Dispatch::Workgroups, false),
        &slots,
        &[("stage_out", SlotRef::whole(SLOT, ELEMENT))],
        &[],
    );
    context.warmup(WarmupBudget::All);

    record(context, &mut stage, &slots, 1);
    assert_filled(context, &slots, 64);

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    assert!(
        slots.storage.reserve(&device, &mut encoder, 128),
        "widening a stream replaces its storage"
    );
    queue.submit([encoder.finish()]);

    record(context, &mut stage, &slots, 2);
    assert_filled(context, &slots, 128);
}

#[test]
fn an_empty_dispatch_resolves_no_kernel() {
    let context = pollster::block_on(GpuContext::new());
    let device = context.device().clone();
    let queue = context.queue().clone();
    let slots = Slots {
        storage: Stream::new(
            &device,
            &queue,
            StreamDesc {
                label: "stage out",
                slots: 64,
                element: ELEMENT,
                elements_per_slot: 1,
                usage: STREAM,
                retention: Retention::Durable,
            },
        ),
    };
    let mut stage = Stage::build(
        &context,
        "stage",
        Program::new(SOURCE.to_owned(), Dispatch::Workgroups, false),
        &slots,
        &[("stage_out", SlotRef::whole(SLOT, ELEMENT))],
        &[],
    );
    record(&context, &mut stage, &slots, 0);
    assert!(
        !context.is_warm(),
        "an empty dispatch must not compile the kernel it would have run"
    );
    record(&context, &mut stage, &slots, 1);
    assert!(
        context.is_warm(),
        "a dispatched stage compiles the kernel it runs"
    );
}
