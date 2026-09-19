use dynamis_gpu::{
    ComputeRecorder, GpuContext, GpuSlot, ResourceId, ResourceSource, Retention, STREAM, SlotRef,
    Stream, StreamDesc, StreamElement, WarmupBudget, read_regions,
};
use dynamis_pass::Stage;
use dynamis_shader::{Dispatch, Extent, Program};
use std::panic::{AssertUnwindSafe, catch_unwind};
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
    declared_device: bool,
}

impl Slots {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, declared_device: bool) -> Self {
        Self {
            storage: Stream::new(
                device,
                queue,
                StreamDesc {
                    label: "stage out",
                    slots: 64,
                    element: ELEMENT,
                    elements_per_slot: 1,
                    usage: STREAM,
                    retention: Retention::Durable,
                },
            ),
            declared_device,
        }
    }
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

    fn device_writes(&self, resource: ResourceId) -> bool {
        assert_eq!(resource, SLOT, "the stage names one stream");
        self.declared_device
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
        declared_device: true,
    };
    let mut stage = Stage::build(
        context,
        "stage",
        Program::new(SOURCE.to_owned(), Dispatch::Workgroups, Extent::None, false),
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
        declared_device: true,
    };
    let mut stage = Stage::build(
        &context,
        "stage",
        Program::new(SOURCE.to_owned(), Dispatch::Workgroups, Extent::None, false),
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

const STREAM_SOURCE: &str = "
@group(0) @binding(0) var<storage, read_write> words: array<u32>;
@group(0) @binding(1) var<storage, read_write> word_count: array<atomic<u32>>;

fn work(index: u32) {
    words[index] = index;
}
";

fn stream_stage(context: &GpuContext, slots: &Slots, counter: usize, extent: Extent) -> Stage {
    Stage::build(
        context,
        "stream stage",
        dynamis_shader::stream(context, STREAM_SOURCE, &[], "work", extent),
        slots,
        &[
            ("words", SlotRef::whole(SLOT, ELEMENT)),
            (
                "word_count",
                SlotRef::range(
                    SLOT,
                    counter as u64 * dynamis_abi::COUNTER_STRIDE,
                    4,
                    ELEMENT,
                ),
            ),
        ],
        &[],
    )
}

#[test]
fn a_streaming_stage_declares_the_counter_and_the_array_its_work_covers() {
    let context = shared();
    let device = context.device().clone();
    let queue = context.queue().clone();
    let slots = Slots {
        storage: Stream::new(
            &device,
            &queue,
            StreamDesc {
                label: "stage stream",
                slots: 1024,
                element: ELEMENT,
                elements_per_slot: 1,
                usage: STREAM,
                retention: Retention::Durable,
            },
        ),
        declared_device: true,
    };
    let counter = dynamis_abi::COUNTER_PAIRS;
    let mut stage = stream_stage(
        context,
        &slots,
        counter,
        Extent::slot(counter, "word_count", "words"),
    );
    assert!(stage_streams_over(&mut stage, &slots));
    assert!(
        catch_unwind(AssertUnwindSafe(|| stream_stage(
            context,
            &slots,
            counter + 1,
            Extent::slot(counter, "word_count", "words"),
        )))
        .is_err(),
        "a work extent must read the counter its stage declares"
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| stream_stage(
            context,
            &slots,
            counter,
            Extent::slot(counter, "word_count", "words_elsewhere"),
        )))
        .is_err(),
        "a work extent must guard an array its stage binds"
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            Stage::build(
                context,
                "rows stage",
                Program::new(SOURCE.to_owned(), Dispatch::Rows, Extent::None, false),
                &slots,
                &[("stage_out", SlotRef::whole(SLOT, ELEMENT))],
                &[],
            );
        }))
        .is_ok(),
        "a row stage declares its bound from the step facts"
    );
}

fn stage_streams_over(stage: &mut Stage, slots: &Slots) -> bool {
    let mut encoder = shared()
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let mut recorder = ComputeRecorder::begin(&mut encoder, "stage", shared().workgroups_per_row());
    stage.record_stream(&mut recorder, slots);
    true
}

#[test]
fn a_stage_that_writes_a_stream_the_host_alone_owns_is_refused() {
    let context = shared();
    let device = context.device().clone();
    let queue = context.queue().clone();
    let declared = Slots::new(&device, &queue, true);
    let host_only = Slots::new(&device, &queue, false);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            Stage::build(
                context,
                "host stage",
                Program::new(SOURCE.to_owned(), Dispatch::Workgroups, Extent::None, false),
                &host_only,
                &[("stage_out", SlotRef::whole(SLOT, ELEMENT))],
                &[],
            );
        }))
        .is_err(),
        "a stage must not write a stream the table hands to the host alone"
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            Stage::build(
                context,
                "declared stage",
                Program::new(SOURCE.to_owned(), Dispatch::Workgroups, Extent::None, false),
                &declared,
                &[("stage_out", SlotRef::whole(SLOT, ELEMENT))],
                &[],
            );
        }))
        .is_ok(),
        "a stage may write a stream the table declares the device writes"
    );
}
