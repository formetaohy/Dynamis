mod common;

use common::shared;
use dynamis_gpu::{GpuBuffer, Readback, SubmissionEncoder, read_regions};
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
    let mut readback = Readback::new(context.device(), "deferred mapping", 8, Readback::DEPTH);
    let mut encoder = SubmissionEncoder::new(context.device(), "deferred copy");
    assert!(
        readback
            .enqueue(&mut encoder, source.buffer(), 4, 8, 0)
            .is_none()
    );
    context.poll();
    assert!(readback.collect().is_empty());
    encoder.submit(context.queue());
    assert_eq!(readback.drain(), vec![(0, expected.to_vec())]);
}

#[test]
fn readbacks_retain_their_own_submission_and_retire_in_sequence_order() {
    let context = shared();
    let first = source();
    let second = source();
    first.write(context.queue(), &[1u8; 8]);
    second.write(context.queue(), &[2u8; 8]);
    let mut readback = Readback::new(context.device(), "ordered", 8, Readback::DEPTH);
    let mut earlier = SubmissionEncoder::new(context.device(), "earlier sequence");
    let mut later = SubmissionEncoder::new(context.device(), "later sequence");
    readback.enqueue(&mut earlier, first.buffer(), 0, 8, 0);
    readback.enqueue(&mut later, second.buffer(), 0, 8, 1);
    later.submit(context.queue());
    assert_eq!(
        read_regions(
            context.device(),
            context.queue(),
            "ordered fence",
            &[(second.buffer(), 0, 8)]
        ),
        [2u8; 8]
    );
    assert!(readback.collect().is_empty());
    earlier.submit(context.queue());
    assert_eq!(readback.drain(), vec![(0, vec![1u8; 8]), (1, vec![2u8; 8])]);
}

#[test]
fn a_batch_larger_than_the_pool_grows_the_pool_without_waiting_on_unsubmitted_slots() {
    let context = shared();
    let source = source();
    let expected = [7u8; 4];
    source.write(context.queue(), &expected);
    let mut readback = Readback::new(context.device(), "burst ring", 4, 2);
    let mut encoder = SubmissionEncoder::new(context.device(), "burst copies");
    let reads = Readback::DEPTH as u64 + 2;
    for sequence in 0..reads {
        assert!(
            readback
                .enqueue(&mut encoder, source.buffer(), 0, 4, sequence)
                .is_none(),
            "a slot whose encoder is still being recorded must never be waited on"
        );
    }
    encoder.submit(context.queue());
    assert_eq!(
        readback.drain(),
        (0..reads)
            .map(|sequence| (sequence, expected.to_vec()))
            .collect::<Vec<_>>()
    );

    let mut reused = SubmissionEncoder::new(context.device(), "reused copies");
    readback.enqueue(&mut reused, source.buffer(), 0, 4, reads);
    reused.submit(context.queue());
    assert_eq!(readback.drain(), vec![(reads, expected.to_vec())]);
}

#[test]
#[should_panic(expected = "must be submitted before waiting or reusing its slot")]
fn waiting_on_a_discarded_encoder_fails_fast() {
    let context = shared();
    let source = source();
    let mut readback = Readback::new(context.device(), "discarded ring", 4, Readback::DEPTH);
    let mut encoder = SubmissionEncoder::new(context.device(), "discarded copy");
    readback.enqueue(&mut encoder, source.buffer(), 0, 4, 0);
    drop(encoder);
    readback.drain();
}

#[test]
#[should_panic(expected = "sequences must increase")]
fn a_readback_rejects_a_sequence_that_does_not_advance() {
    let context = shared();
    let source = source();
    source.write(context.queue(), &[5u8; 4]);
    let mut readback = Readback::new(context.device(), "stalled sequence", 4, Readback::DEPTH);
    let mut encoder = SubmissionEncoder::new(context.device(), "stalled copies");
    readback.enqueue(&mut encoder, source.buffer(), 0, 4, 3);
    readback.enqueue(&mut encoder, source.buffer(), 0, 4, 3);
}

#[test]
fn read_regions_concatenates_every_region_in_one_submission() {
    let context = shared();
    let source = source();
    source.write(context.queue(), &[1u8; 32]);
    assert_eq!(
        read_regions(
            context.device(),
            context.queue(),
            "concatenated probe",
            &[(source.buffer(), 8, 4), (source.buffer(), 0, 8)]
        ),
        vec![1u8; 12]
    );
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
                let mut readback = Readback::new(context.device(), "parallel ring", 16, 4);
                let mut expected = Vec::new();
                let mut received = Vec::new();
                start.wait();
                for sequence in 0..24u64 {
                    let bytes = 4 * (1 + sequence as usize % 4);
                    let data = vec![(worker * 24 + sequence as usize) as u8; bytes];
                    source.write_at(context.queue(), 4, &data);
                    let mut encoder = SubmissionEncoder::new(context.device(), "parallel copy");
                    if let Some(displaced) =
                        readback.enqueue(&mut encoder, source.buffer(), 4, bytes as u64, sequence)
                    {
                        received.push(displaced);
                    }
                    encoder.submit(context.queue());
                    if sequence % 7 == 0 {
                        assert_eq!(
                            read_regions(
                                context.device(),
                                context.queue(),
                                "parallel probe",
                                &[(source.buffer(), 4, bytes as u64)]
                            ),
                            data
                        );
                        context.poll();
                        received.extend(readback.collect());
                    }
                    expected.push((sequence, data));
                }
                received.extend(readback.drain());
                assert_eq!(received, expected);
                assert!(readback.collect().is_empty());
                assert!(readback.drain().is_empty());
            });
        }
    });
}

#[test]
fn a_publication_retires_every_declaration_in_order() {
    let context = shared();
    let source = source();
    let mut publication: dynamis_gpu::Publication<u32> = dynamis_gpu::Publication::new(
        "ordered publication",
        dynamis_gpu::Publication::<u32>::DEPTH,
    );
    publication.reserve(context.device(), 16);
    let mut received = Vec::new();
    for sequence in 0..12u64 {
        let data = [sequence as u8; 4];
        source.write_at(context.queue(), 0, &data);
        let regions = [(source.buffer(), 0, 4)];
        let mut encoder = SubmissionEncoder::new(context.device(), "ordered expect");
        if let Some((manifest, bytes)) =
            publication.declare(&mut encoder, &regions, sequence, sequence as u32 * 10)
        {
            received.push((manifest, bytes));
        }
        encoder.submit(context.queue());
    }
    received.extend(publication.drain());
    let expected = (0..12u32)
        .map(|sequence| (sequence * 10, vec![sequence as u8; 4]))
        .collect::<Vec<_>>();
    assert_eq!(received, expected);
    assert!(publication.is_idle());
    publication.clear();
    assert_eq!(publication.budget(), 0);
}

#[test]
fn a_publication_widens_under_a_declaration_in_flight() {
    let context = shared();
    let source = source();
    source.write(context.queue(), &[3u8; 32]);
    let mut publication: dynamis_gpu::Publication<u32> = dynamis_gpu::Publication::new(
        "widening publication",
        dynamis_gpu::Publication::<u32>::DEPTH,
    );
    publication.reserve(context.device(), 4);
    let mut first = SubmissionEncoder::new(context.device(), "narrow declaration");
    assert!(
        publication
            .declare(&mut first, &[(source.buffer(), 0, 4)], 0, 1)
            .is_none()
    );
    assert!(
        publication.reserve(context.device(), 16),
        "a wider publication must widen its ring"
    );
    let mut second = SubmissionEncoder::new(context.device(), "wide declaration");
    assert!(
        publication
            .declare(&mut second, &[(source.buffer(), 0, 16)], 1, 2)
            .is_none()
    );
    first.submit(context.queue());
    second.submit(context.queue());
    assert_eq!(
        publication.drain(),
        vec![(1, vec![3u8; 4]), (2, vec![3u8; 16])],
        "a widened ring must still answer every declaration it carried in flight"
    );
}

#[test]
fn a_widened_ring_recycles_in_flight_slots_at_its_new_width() {
    let context = shared();
    let source = source();
    source.write(context.queue(), &[9u8; 8]);
    let mut readback = Readback::new(context.device(), "widened ring", 4, 2);
    for sequence in 0..2u64 {
        let mut encoder = SubmissionEncoder::new(context.device(), "narrow declaration");
        assert!(
            readback
                .enqueue(&mut encoder, source.buffer(), 0, 4, sequence)
                .is_none(),
            "a ring that still holds a free slot never displaces a declaration"
        );
        encoder.submit(context.queue());
    }
    assert!(
        readback.reserve(8),
        "a wider budget must widen a ring whose slots are all in flight"
    );
    let mut wide = SubmissionEncoder::new(context.device(), "wide declaration");
    let displaced = readback.enqueue(&mut wide, source.buffer(), 0, 8, 2);
    wide.submit(context.queue());
    let mut received = displaced.into_iter().collect::<Vec<_>>();
    received.extend(readback.drain());
    assert_eq!(
        received,
        vec![(0, vec![9u8; 4]), (1, vec![9u8; 4]), (2, vec![9u8; 8])],
        "a recycled slot must carry the width the ring widened to"
    );
    assert!(readback.is_idle());
}

#[test]
fn a_publication_releases_a_wider_ring_only_once_it_is_idle() {
    let context = shared();
    let source = source();
    let mut publication: dynamis_gpu::Publication<u32> = dynamis_gpu::Publication::new(
        "narrowing publication",
        dynamis_gpu::Publication::<u32>::DEPTH,
    );
    publication.reserve(context.device(), 16);
    let mut encoder = SubmissionEncoder::new(context.device(), "in flight declaration");
    publication.declare(&mut encoder, &[(source.buffer(), 0, 16)], 0, 1);
    assert!(
        !publication.reserve(context.device(), 4),
        "a narrower publication must wait for the facts in flight"
    );
    assert_eq!(
        publication.budget(),
        16,
        "a narrowing must keep the ring that serves the in-flight facts"
    );
    encoder.submit(context.queue());
    assert_eq!(publication.drain().len(), 1);
    assert_eq!(
        publication.budget(),
        4,
        "an idle ring must narrow to the budget it was asked for"
    );
    assert!(
        !publication.reserve(context.device(), 4),
        "an idle ring that already holds its budget must not move"
    );
}

#[test]
fn a_publication_refuses_a_declaration_that_outgrows_its_budget() {
    let context = shared();
    let source = source();
    let mut publication: dynamis_gpu::Publication<u32> =
        dynamis_gpu::Publication::new("budgeted publication", 2);
    publication.reserve(context.device(), 4);
    let mut encoder = SubmissionEncoder::new(context.device(), "overrun");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        publication.declare(&mut encoder, &[(source.buffer(), 0, 8)], 0, 0);
    }));
    assert!(
        result.is_err(),
        "a declaration that outruns its budget must be refused"
    );
}
