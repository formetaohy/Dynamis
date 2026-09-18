use super::registry::Streams;
use wgpu::{Buffer, Queue};

struct StreamRange {
    label: &'static str,
    slots: u32,
    stride: u64,
    start: usize,
    end: usize,
}

pub(crate) struct StreamArchive {
    ranges: Vec<StreamRange>,
    bytes: Vec<u8>,
}

impl StreamArchive {
    pub(crate) fn capture(streams: &Streams, bytes: Vec<u8>) -> Self {
        let mut at = 0usize;
        let ranges = streams
            .durable()
            .into_iter()
            .map(|(label, stream)| {
                let end = at + stream.size() as usize;
                let range = StreamRange {
                    label,
                    slots: stream.slots(),
                    stride: stream.stride(),
                    start: at,
                    end,
                };
                at = end;
                range
            })
            .collect();
        assert_eq!(
            at,
            bytes.len(),
            "a snapshot must cover exactly the durable streams it captured"
        );
        Self { ranges, bytes }
    }

    pub(crate) fn slots(&self, label: &'static str) -> Option<u32> {
        self.ranges
            .iter()
            .find(|range| range.label == label)
            .map(|range| range.slots)
    }
}

impl Streams {
    pub(crate) fn durable_regions(&self) -> Vec<(Buffer, u64, u64)> {
        self.durable()
            .into_iter()
            .map(|(_, stream)| (stream.gpu().buffer().clone(), 0, stream.size()))
            .collect()
    }

    pub(crate) fn write(&self, queue: &Queue, archive: &StreamArchive) {
        let durable = self.durable();
        assert_eq!(
            durable.len(),
            archive.ranges.len(),
            "a snapshot must answer every durable stream"
        );
        for ((label, stream), range) in durable.into_iter().zip(&archive.ranges) {
            assert_eq!(
                label, range.label,
                "a durable stream must keep its label across a snapshot"
            );
            assert_eq!(
                stream.stride(),
                range.stride,
                "durable stream {:?} must keep its record layout across a snapshot",
                range.label
            );
            assert_eq!(
                stream.slots(),
                range.slots,
                "durable stream {:?} must be restored at its snapshotted size",
                range.label
            );
            stream.write(queue, &archive.bytes[range.start..range.end]);
        }
    }
}
