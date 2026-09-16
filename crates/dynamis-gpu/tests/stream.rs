mod common;

use common::shared;
use dynamis_gpu::{
    GpuContext, Retention, Stream, StreamDesc, StreamElement, SubmissionEncoder, read_regions,
};
use wgpu::BufferUsages;

const STREAM_USAGE: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);

fn stream(
    context: &GpuContext,
    label: &'static str,
    slots: u32,
    element: StreamElement,
    elements_per_slot: u64,
    retention: Retention,
) -> Stream {
    Stream::new(
        context.device(),
        context.queue(),
        StreamDesc {
            label,
            slots,
            element,
            elements_per_slot,
            usage: STREAM_USAGE,
            retention,
        },
    )
}

fn seed(stream: &Stream, values: &[u32]) {
    stream.write(shared().queue(), bytemuck::cast_slice(values));
}

fn read(stream: &Stream, words: u64) -> Vec<u32> {
    let context = shared();
    let bytes = read_regions(
        context.device(),
        context.queue(),
        "stream probe",
        &[(stream.buffer(), 0, words * 4)],
    );
    bytemuck::cast_slice(&bytes).to_vec()
}

fn resized(stream: &mut Stream, slots: u32) -> bool {
    let context = shared();
    let mut encoder = SubmissionEncoder::new(context.device(), "stream resize");
    let changed = stream.reserve(context.device(), &mut encoder, slots);
    if changed {
        encoder.submit(context.queue());
    }
    changed
}

#[test]
fn a_preserving_stream_keeps_its_leading_slots_across_resizes() {
    let context = shared();
    let mut stream = stream(
        context,
        "preserving stream",
        4,
        StreamElement::new("u32", 4),
        1,
        Retention::Durable,
    );
    seed(&stream, &[7, 8, 9, 10]);
    assert!(!resized(&mut stream, 4));
    assert!(resized(&mut stream, 6));
    assert_eq!(stream.slots(), 6);
    assert_eq!(read(&stream, 4), vec![7, 8, 9, 10]);
    assert!(resized(&mut stream, 2));
    assert_eq!(read(&stream, 2), vec![7, 8]);
}

#[test]
fn a_reset_stream_drops_its_contents_across_resizes() {
    let context = shared();
    let mut stream = stream(
        context,
        "reset stream",
        4,
        StreamElement::new("u32", 4),
        1,
        Retention::Scratch,
    );
    seed(&stream, &[7, 8, 9, 10]);
    assert!(resized(&mut stream, 8));
    assert_eq!(read(&stream, 4), vec![0, 0, 0, 0]);
}

#[test]
fn a_seeded_stream_starts_with_its_head_word() {
    let context = shared();
    let stream = stream(
        context,
        "seeded stream",
        1,
        StreamElement::new("u32", 4),
        1,
        Retention::Seeded(u32::MAX),
    );
    assert_eq!(read(&stream, 1), vec![u32::MAX]);
}

#[test]
fn a_stream_reports_its_stride_and_bytes() {
    let context = shared();
    let stream = stream(
        context,
        "strided stream",
        3,
        StreamElement::new("vec4f", 16),
        4,
        Retention::Scratch,
    );
    assert_eq!(stream.slots(), 3);
    assert_eq!(stream.element(), StreamElement::new("vec4f", 16));
    assert_eq!(stream.stride(), 64);
    assert_eq!(stream.size(), 192);
}

#[test]
#[should_panic(expected = "requires at least one slot")]
fn a_stream_refuses_an_empty_capacity() {
    let context = shared();
    let _ = stream(
        context,
        "empty stream",
        0,
        StreamElement::new("u32", 4),
        1,
        Retention::Scratch,
    );
}

#[test]
#[should_panic(expected = "requires at least one element per slot")]
fn a_stream_refuses_an_empty_slot() {
    let context = shared();
    let _ = stream(
        context,
        "dense stream",
        4,
        StreamElement::new("u32", 4),
        0,
        Retention::Scratch,
    );
}

#[test]
#[should_panic(expected = "word aligned record")]
fn a_stream_refuses_an_unaligned_record() {
    let context = shared();
    let _ = stream(
        context,
        "unaligned stream",
        4,
        StreamElement::new("u32", 6),
        1,
        Retention::Scratch,
    );
}

#[test]
#[should_panic(expected = "storage binding limit")]
fn a_stream_refuses_to_exceed_the_binding_limit() {
    let context = shared();
    let _ = stream(
        context,
        "oversized stream",
        u32::MAX,
        StreamElement::new("vec4f", 16),
        4,
        Retention::Scratch,
    );
}
