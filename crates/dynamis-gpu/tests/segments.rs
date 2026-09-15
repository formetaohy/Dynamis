mod common;

use common::shared;
use dynamis_gpu::{
    Contents, SEGMENT_COUNT, STREAM, Segments, Stream, StreamDesc, StreamElement, SubmissionEncoder,
};

const RECORD: u64 = 4;
const SEGMENT_RECORDS: u32 = 4;

fn source() -> Stream {
    Stream::new(
        shared().device(),
        shared().queue(),
        StreamDesc {
            label: "segment source",
            slots: SEGMENT_COUNT * SEGMENT_RECORDS,
            element: StreamElement::new("u32", RECORD),
            elements_per_slot: 1,
            usage: STREAM,
            contents: Contents::Scratch,
        },
    )
}

fn transport(source: &Stream) -> Segments {
    let mut segments = Segments::new("transport");
    segments.reserve(shared().device(), source.size() / SEGMENT_COUNT as u64);
    segments
}

fn write_segment(source: &Stream, step: u64, records: &[u32]) {
    let bytes = records
        .iter()
        .flat_map(|record| record.to_le_bytes())
        .collect::<Vec<_>>();
    let segment = source.size() / SEGMENT_COUNT as u64;
    source.write_at(
        shared().queue(),
        (step % SEGMENT_COUNT as u64) * segment,
        &bytes,
    );
}

fn records(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<{ RECORD as usize }>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect()
}

fn retire(segments: &mut Segments, source: &Stream, now: u64) -> Vec<(u64, u32, Vec<u8>)> {
    let mut encoder = SubmissionEncoder::new(shared().device(), "segment copy");
    assert!(
        segments.copy(&mut encoder, source, now).is_empty(),
        "a ring that has not filled must displace nothing"
    );
    encoder.submit(shared().queue());
    segments.drain()
}

#[test]
fn a_closed_segment_copies_its_own_step_slice() {
    let source = source();
    let mut segments = transport(&source);
    write_segment(&source, 5, &[11, 12, 13]);
    segments.close(5, 3);
    let arrivals = retire(&mut segments, &source, 5);
    assert_eq!(
        arrivals
            .iter()
            .map(|(step, count, _)| (*step, *count))
            .collect::<Vec<_>>(),
        [(5, 3)],
        "a segment retires under the step and count it closed with"
    );
    assert_eq!(records(&arrivals[0].2), [11, 12, 13]);
    assert!(!segments.pending(), "a copied segment leaves no backlog");
}

#[test]
fn segments_retire_in_the_order_they_closed() {
    let source = source();
    let mut segments = transport(&source);
    write_segment(&source, 1, &[21, 22]);
    write_segment(&source, 6, &[31, 32, 33]);
    segments.close(1, 2);
    segments.close(6, 3);
    let arrivals = retire(&mut segments, &source, 6);
    assert_eq!(
        arrivals
            .iter()
            .map(|(step, count, _)| (*step, *count))
            .collect::<Vec<_>>(),
        [(1, 2), (6, 3)],
        "segments must retire in the order they closed"
    );
    assert_eq!(records(&arrivals[0].2), [21, 22]);
    assert_eq!(records(&arrivals[1].2), [31, 32, 33]);
}

#[test]
fn a_count_beyond_the_segment_copies_the_whole_segment() {
    let source = source();
    let mut segments = transport(&source);
    write_segment(&source, 0, &[41, 42, 43, 44]);
    segments.close(0, 5);
    let arrivals = retire(&mut segments, &source, 0);
    assert_eq!(
        records(&arrivals[0].2),
        [41, 42, 43, 44],
        "a count the device refused must still copy every record the segment stores"
    );
    assert_eq!(
        arrivals[0].1, 5,
        "a segment must publish the count the device closed it with"
    );
}

#[test]
fn a_segment_overwritten_by_a_later_step_is_refused() {
    let refused = std::panic::catch_unwind(|| {
        let source = source();
        let mut segments = transport(&source);
        segments.close(0, 1);
        let mut encoder = SubmissionEncoder::new(shared().device(), "overwritten segment copy");
        segments.copy(&mut encoder, &source, SEGMENT_COUNT as u64 + 1);
    });
    assert!(
        refused.is_err(),
        "a segment overwritten by a later step must never be copied"
    );
}
