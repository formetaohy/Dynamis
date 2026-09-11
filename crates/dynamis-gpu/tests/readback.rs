mod common;

use common::shared;
use dynamis_gpu::{BufferReadback, GpuBuffer, ReadbackRing, SubmissionEncoder};
use std::sync::Barrier;
use wgpu::BufferUsages;

fn source() -> GpuBuffer {
    GpuBuffer::new(
        shared().device(),
        "readback source",
        32,
        BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    )
}

#[test]
fn recording_and_polling_never_maps_an_unsubmitted_copy() {
    let context = shared();
    let source = source();
    let expected = [11u8; 8];
    source.write_at(context.queue(), 4, &expected);
    let mut ring = ReadbackRing::new(context.device(), "deferred mapping", 8);
    let mut encoder = SubmissionEncoder::new(context.device(), "deferred copy");
    assert!(
        ring.enqueue(&mut encoder, source.buffer(), 4, 8, 0)
            .is_none()
    );
    context.poll();
    assert!(ring.collect().is_empty());
    encoder.submit(context.queue());
    assert_eq!(ring.drain(), vec![(0, expected.to_vec())]);
}

#[test]
fn readbacks_retain_their_own_submission_and_retire_in_sequence_order() {
    let context = shared();
    let first = source();
    let second = source();
    first.write(context.queue(), &[1u8; 8]);
    second.write(context.queue(), &[2u8; 8]);
    let mut ring = ReadbackRing::new(context.device(), "ordered", 8);
    let mut earlier = SubmissionEncoder::new(context.device(), "earlier sequence");
    let mut later = SubmissionEncoder::new(context.device(), "later sequence");
    ring.enqueue(&mut earlier, first.buffer(), 0, 8, 0);
    ring.enqueue(&mut later, second.buffer(), 0, 8, 1);
    later.submit(context.queue());
    let mut fence = BufferReadback::new(context.device(), "later fence", 8);
    assert_eq!(fence.read(context.queue(), second.buffer(), 0, 8), [2u8; 8]);
    assert!(ring.collect().is_empty());
    earlier.submit(context.queue());
    assert_eq!(ring.drain(), vec![(0, vec![1u8; 8]), (1, vec![2u8; 8])]);
}

#[test]
#[should_panic(expected = "must be submitted before waiting or reusing its slot")]
fn reusing_a_slot_from_the_same_unsubmitted_encoder_fails_fast() {
    let context = shared();
    let source = source();
    let mut ring = ReadbackRing::new(context.device(), "unsubmitted ring", 4);
    let mut encoder = SubmissionEncoder::new(context.device(), "unsubmitted copies");
    for sequence in 0..=ReadbackRing::DEPTH as u64 {
        ring.enqueue(&mut encoder, source.buffer(), 0, 4, sequence);
    }
}

#[test]
#[should_panic(expected = "must be submitted before waiting or reusing its slot")]
fn waiting_on_a_discarded_encoder_fails_fast() {
    let context = shared();
    let source = source();
    let mut ring = ReadbackRing::new(context.device(), "discarded ring", 4);
    let mut encoder = SubmissionEncoder::new(context.device(), "discarded copy");
    ring.enqueue(&mut encoder, source.buffer(), 0, 4, 0);
    drop(encoder);
    ring.drain();
}

#[test]
fn parallel_submissions_recycle_slots_without_losing_or_mixing_results() {
    const WORKERS: usize = 8;
    let context = shared();
    let start = Barrier::new(WORKERS);
    std::thread::scope(|scope| {
        for worker in 0..WORKERS {
            let start = &start;
            scope.spawn(move || {
                let source = source();
                let mut ring = ReadbackRing::new(context.device(), "parallel ring", 16);
                let mut synchronous = BufferReadback::new(context.device(), "parallel sync", 16);
                let mut expected = Vec::new();
                let mut received = Vec::new();
                start.wait();
                for sequence in 0..24u64 {
                    let bytes = 4 * (1 + sequence as usize % 4);
                    let data = vec![(worker * 24 + sequence as usize) as u8; bytes];
                    source.write_at(context.queue(), 4, &data);
                    let mut encoder = SubmissionEncoder::new(context.device(), "parallel copy");
                    if let Some(displaced) =
                        ring.enqueue(&mut encoder, source.buffer(), 4, bytes as u64, sequence)
                    {
                        received.push(displaced);
                    }
                    encoder.submit(context.queue());
                    if sequence % 7 == 0 {
                        assert_eq!(
                            synchronous.read(context.queue(), source.buffer(), 4, bytes as u64),
                            data
                        );
                        context.poll();
                        received.extend(ring.collect());
                    }
                    expected.push((sequence, data));
                }
                received.extend(ring.drain());
                assert_eq!(received, expected);
                assert!(ring.collect().is_empty());
                assert!(ring.drain().is_empty());
            });
        }
    });
}
