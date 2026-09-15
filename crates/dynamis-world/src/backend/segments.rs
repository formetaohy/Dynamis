use super::registry::Streams;
use dynamis_abi::{COUNTER_BREAKS, COUNTER_EVENTS, COUNTER_IMPACTS, COUNTER_SOFT_EVENTS, Counters};
use dynamis_gpu::{SEGMENT_COUNT, Segments as SegmentRing, Stream, SubmissionEncoder};
use wgpu::{BufferAddress, Device};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SegmentKind {
    Events,
    SoftEvents,
    Impacts,
    Breaks,
}

impl SegmentKind {
    pub(crate) const ALL: [Self; 4] = [Self::Events, Self::SoftEvents, Self::Impacts, Self::Breaks];

    const fn label(self) -> &'static str {
        match self {
            Self::Events => "contact event segments",
            Self::SoftEvents => "soft contact event segments",
            Self::Impacts => "impact segments",
            Self::Breaks => "constraint break segments",
        }
    }

    fn count(self, measured: &Counters) -> u32 {
        match self {
            Self::Events => measured[COUNTER_EVENTS],
            Self::SoftEvents => measured[COUNTER_SOFT_EVENTS],
            Self::Impacts => measured[COUNTER_IMPACTS],
            Self::Breaks => measured[COUNTER_BREAKS],
        }
    }

    fn source(self, streams: &Streams) -> &Stream {
        match self {
            Self::Events => &streams.rigid.events,
            Self::SoftEvents => &streams.soft.events,
            Self::Impacts => &streams.rigid.impacts,
            Self::Breaks => &streams.state.constraint_breaks,
        }
    }
}

pub(crate) struct Arrival {
    pub(crate) kind: SegmentKind,
    pub(crate) step: u64,
    pub(crate) count: u32,
    pub(crate) bytes: Vec<u8>,
}

pub(crate) struct Segments {
    rings: [SegmentRing; SegmentKind::ALL.len()],
}

impl Segments {
    pub(crate) fn new(device: &Device, streams: &Streams) -> Self {
        let mut segments = Self {
            rings: SegmentKind::ALL.map(|kind| SegmentRing::new(kind.label())),
        };
        assert!(
            segments.reserve(device, streams),
            "a fresh segment transport must allocate every ring"
        );
        segments
    }

    pub(crate) fn reserve(&mut self, device: &Device, streams: &Streams) -> bool {
        let mut changed = false;
        for kind in SegmentKind::ALL {
            let budget = kind.source(streams).size() / SEGMENT_COUNT as BufferAddress;
            changed |= self.ring(kind).reserve(device, budget);
        }
        changed
    }

    pub(crate) fn pending(&self) -> bool {
        self.rings.iter().any(SegmentRing::pending)
    }

    pub(crate) fn close(&mut self, measured: &Counters, step: u64) {
        for kind in SegmentKind::ALL {
            self.ring(kind).close(step, kind.count(measured));
        }
    }

    pub(crate) fn copy(
        &mut self,
        encoder: &mut SubmissionEncoder,
        streams: &Streams,
        now: u64,
    ) -> Vec<Arrival> {
        let mut arrivals = Vec::new();
        for kind in SegmentKind::ALL {
            let source = kind.source(streams);
            for (step, count, bytes) in self.ring(kind).copy(encoder, source, now) {
                arrivals.push(Arrival {
                    kind,
                    step,
                    count,
                    bytes,
                });
            }
        }
        arrivals
    }

    pub(crate) fn collect(&mut self) -> Vec<Arrival> {
        self.gather(SegmentRing::collect)
    }

    pub(crate) fn drain(&mut self) -> Vec<Arrival> {
        self.gather(SegmentRing::drain)
    }

    fn gather(
        &mut self,
        mut take: impl FnMut(&mut SegmentRing) -> Vec<(u64, u32, Vec<u8>)>,
    ) -> Vec<Arrival> {
        let mut arrivals = Vec::new();
        for kind in SegmentKind::ALL {
            for (step, count, bytes) in take(self.ring(kind)) {
                arrivals.push(Arrival {
                    kind,
                    step,
                    count,
                    bytes,
                });
            }
        }
        arrivals
    }

    fn ring(&mut self, kind: SegmentKind) -> &mut SegmentRing {
        &mut self.rings[kind as usize]
    }
}
