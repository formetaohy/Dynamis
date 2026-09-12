mod common;

use common::shared;
use dynamis_gpu::{BufferReadback, Contents, Stream, SubmissionEncoder};
use wgpu::BufferUsages;

const STREAM_USAGE: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);

fn seed(stream: &Stream, values: &[u32]) {
    stream.write(shared().queue(), bytemuck::cast_slice(values));
}

fn read(stream: &Stream, words: u64) -> Vec<u32> {
    let context = shared();
    let mut readback = BufferReadback::new(context.device(), "stream probe", (words * 4).max(4));
    let bytes = readback.read(context.queue(), stream.buffer(), 0, words * 4);
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
    let mut stream = Stream::new(
        context.device(),
        context.queue(),
        "preserving stream",
        4,
        4,
        STREAM_USAGE,
        Contents::Preserve,
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
    let mut stream = Stream::new(
        context.device(),
        context.queue(),
        "reset stream",
        4,
        4,
        STREAM_USAGE,
        Contents::Reset,
    );
    seed(&stream, &[7, 8, 9, 10]);
    assert!(resized(&mut stream, 8));
    assert_eq!(read(&stream, 4), vec![0, 0, 0, 0]);
}

#[test]
fn a_seeded_stream_starts_with_its_head_word() {
    let context = shared();
    let stream = Stream::new(
        context.device(),
        context.queue(),
        "seeded stream",
        1,
        4,
        STREAM_USAGE,
        Contents::PreserveSeeded(u32::MAX),
    );
    assert_eq!(read(&stream, 1), vec![u32::MAX]);
}

#[test]
fn a_stream_reports_its_stride_and_bytes() {
    let context = shared();
    let stream = Stream::new(
        context.device(),
        context.queue(),
        "strided stream",
        3,
        64,
        STREAM_USAGE,
        Contents::Reset,
    );
    assert_eq!(stream.slots(), 3);
    assert_eq!(stream.stride(), 64);
    assert_eq!(stream.size(), 192);
}

#[test]
#[should_panic(expected = "requires at least one slot")]
fn a_stream_refuses_an_empty_capacity() {
    let context = shared();
    let _ = Stream::new(
        context.device(),
        context.queue(),
        "empty stream",
        0,
        4,
        STREAM_USAGE,
        Contents::Reset,
    );
}

#[test]
#[should_panic(expected = "word aligned stride")]
fn a_stream_refuses_an_unaligned_stride() {
    let context = shared();
    let _ = Stream::new(
        context.device(),
        context.queue(),
        "unaligned stream",
        4,
        6,
        STREAM_USAGE,
        Contents::Reset,
    );
}

#[test]
#[should_panic(expected = "storage binding limit")]
fn a_stream_refuses_to_exceed_the_binding_limit() {
    let context = shared();
    let _ = Stream::new(
        context.device(),
        context.queue(),
        "oversized stream",
        u32::MAX,
        64,
        STREAM_USAGE,
        Contents::Reset,
    );
}
