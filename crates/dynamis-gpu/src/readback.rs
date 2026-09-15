use crate::SubmissionEncoder;
use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use wgpu::{
    Buffer, BufferAddress, BufferAsyncError, BufferDescriptor, BufferUsages, Device, MapMode,
    PollType, Queue, SubmissionIndex,
};

const READBACK_TIMEOUT: Duration = Duration::from_secs(30);

struct Pending {
    sequence: u64,
    bytes: BufferAddress,
    submission: Arc<OnceLock<SubmissionIndex>>,
    completion: Mutex<Receiver<Result<(), BufferAsyncError>>>,
}

struct Slot {
    label: String,
    staging: Buffer,
    pending: Option<Pending>,
}

impl Slot {
    fn new(device: &Device, label: &str, size: BufferAddress) -> Self {
        Self {
            label: label.to_owned(),
            staging: device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            pending: None,
        }
    }

    fn is_sealed(&self) -> bool {
        self.pending
            .as_ref()
            .is_some_and(|pending| pending.submission.get().is_some())
    }

    fn record(
        &mut self,
        encoder: &mut SubmissionEncoder,
        regions: &[(&Buffer, BufferAddress, BufferAddress)],
        sequence: u64,
    ) {
        assert!(
            self.pending.is_none(),
            "readback {} is still in use",
            self.label
        );
        let mut at = 0;
        for (source, source_offset, bytes) in regions {
            assert!(
                *bytes > 0 && bytes.is_multiple_of(4),
                "readback length must be positive and word aligned"
            );
            assert!(
                source_offset.is_multiple_of(4),
                "readback source offset must be word aligned"
            );
            assert!(
                *source_offset <= source.size() && *bytes <= source.size() - source_offset,
                "readback exceeds source buffer"
            );
            encoder.copy_buffer_to_buffer(source, *source_offset, &self.staging, at, *bytes);
            at += bytes;
        }
        assert!(
            at > 0 && at <= self.staging.size(),
            "readback exceeds staging capacity"
        );
        let (sender, completion) = mpsc::channel();
        encoder.map_buffer_on_submit(&self.staging, MapMode::Read, ..at, move |result| {
            let _ = sender.send(result);
        });
        self.pending = Some(Pending {
            sequence,
            bytes: at,
            submission: encoder.submission(),
            completion: Mutex::new(completion),
        });
    }

    fn collect(&mut self) -> Option<(u64, Vec<u8>)> {
        let result = {
            let pending = self.pending.as_ref()?;
            pending.submission.get()?;
            match pending
                .completion
                .lock()
                .expect("a readback completion is never poisoned")
                .try_recv()
            {
                Ok(result) => result,
                Err(TryRecvError::Empty) => return None,
                Err(TryRecvError::Disconnected) => {
                    panic!("readback {} completion was dropped", self.label)
                }
            }
        };
        Some(self.consume(result))
    }

    fn wait(&mut self, device: &Device) -> (u64, Vec<u8>) {
        let pending = self
            .pending
            .as_ref()
            .expect("cannot wait for an idle readback");
        let submission = pending.submission.get().cloned().unwrap_or_else(|| {
            panic!(
                "readback {} must be submitted before waiting or reusing its slot",
                self.label
            )
        });
        let deadline = Instant::now() + READBACK_TIMEOUT;
        device
            .poll(PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(READBACK_TIMEOUT),
            })
            .unwrap_or_else(|error| panic!("readback {} submission failed: {error}", self.label));
        let result = pending
            .completion
            .lock()
            .expect("a readback completion is never poisoned")
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .unwrap_or_else(|error| panic!("readback {} completion failed: {error}", self.label));
        self.consume(result)
    }

    fn consume(&mut self, result: Result<(), BufferAsyncError>) -> (u64, Vec<u8>) {
        result.unwrap_or_else(|error| panic!("readback {} mapping failed: {error}", self.label));
        let pending = self
            .pending
            .take()
            .expect("readback completion requires a pending read");
        let bytes = self
            .staging
            .slice(..pending.bytes)
            .get_mapped_range()
            .expect("completed readback must be mapped")
            .to_vec();
        self.staging.unmap();
        (pending.sequence, bytes)
    }
}

pub const SEGMENT_COUNT: u32 = Publication::<()>::DEPTH as u32 + 2;

pub const FACT_LAG: usize = SEGMENT_COUNT as usize;

pub struct Readback {
    device: Device,
    label: String,
    size: BufferAddress,
    slots: VecDeque<Slot>,
    inflight: usize,
    last_sequence: Option<u64>,
}

impl Readback {
    pub const DEPTH: usize = 4;

    pub fn new(device: &Device, label: &str, size: BufferAddress, depth: usize) -> Self {
        assert!(
            size > 0 && size.is_multiple_of(4),
            "readback size must be positive and word aligned"
        );
        assert!(depth > 0, "a readback needs at least one staging slot");
        Self {
            device: device.clone(),
            label: label.to_owned(),
            size,
            slots: (0..depth)
                .map(|index| Slot::new(device, &format!("{label} slot {index}"), size))
                .collect(),
            inflight: 0,
            last_sequence: None,
        }
    }

    pub fn size(&self) -> BufferAddress {
        self.size
    }

    pub fn is_idle(&self) -> bool {
        self.inflight == 0
    }

    pub fn enqueue(
        &mut self,
        encoder: &mut SubmissionEncoder,
        source: &Buffer,
        source_offset: BufferAddress,
        bytes: BufferAddress,
        sequence: u64,
    ) -> Option<(u64, Vec<u8>)> {
        self.enqueue_regions(encoder, &[(source, source_offset, bytes)], sequence)
    }

    pub fn enqueue_regions(
        &mut self,
        encoder: &mut SubmissionEncoder,
        regions: &[(&Buffer, BufferAddress, BufferAddress)],
        sequence: u64,
    ) -> Option<(u64, Vec<u8>)> {
        assert!(
            self.last_sequence
                .is_none_or(|previous| sequence > previous),
            "readback {} sequences must increase: received {sequence} after {:?}",
            self.label,
            self.last_sequence
        );
        let displaced = self.reclaim();
        self.slots[self.inflight].record(encoder, regions, sequence);
        self.inflight += 1;
        self.last_sequence = Some(sequence);
        displaced
    }

    pub fn collect(&mut self) -> Vec<(u64, Vec<u8>)> {
        let mut completed = Vec::new();
        while self.inflight > 0 {
            let Some(entry) = self.slots[0].collect() else {
                break;
            };
            self.retire();
            completed.push(entry);
        }
        completed
    }

    pub fn drain(&mut self) -> Vec<(u64, Vec<u8>)> {
        let mut completed = Vec::with_capacity(self.inflight);
        while self.inflight > 0 {
            let entry = self.slots[0].wait(&self.device);
            self.retire();
            completed.push(entry);
        }
        completed
    }

    fn reclaim(&mut self) -> Option<(u64, Vec<u8>)> {
        if self.inflight < self.slots.len() {
            return None;
        }
        if !self.slots[0].is_sealed() {
            let index = self.slots.len();
            self.slots.push_back(Slot::new(
                &self.device,
                &format!("{} slot {index}", self.label),
                self.size,
            ));
            return None;
        }
        let mut oldest = self.slots.pop_front().expect("readback holds a slot");
        let entry = oldest.wait(&self.device);
        self.slots.push_back(oldest);
        self.inflight -= 1;
        Some(entry)
    }

    fn retire(&mut self) {
        let slot = self.slots.pop_front().expect("readback holds a slot");
        self.slots.push_back(slot);
        self.inflight -= 1;
        if self.inflight == 0 {
            self.last_sequence = None;
        }
    }
}

pub struct Publication<M> {
    label: &'static str,
    depth: usize,
    readback: Option<Readback>,
    declared: VecDeque<M>,
}

impl<M> Publication<M> {
    pub const DEPTH: usize = Readback::DEPTH;

    pub const fn new(label: &'static str, depth: usize) -> Self {
        assert!(depth > 0, "a publication needs at least one staging slot");
        Self {
            label,
            depth,
            readback: None,
            declared: VecDeque::new(),
        }
    }

    pub fn budget(&self) -> BufferAddress {
        self.readback.as_ref().map_or(0, Readback::size)
    }

    pub fn is_idle(&self) -> bool {
        self.readback.as_ref().is_none_or(Readback::is_idle)
    }

    pub fn reserve(&mut self, device: &Device, budget: BufferAddress) -> bool {
        assert!(
            budget > 0 && budget.is_multiple_of(4),
            "publication {:?} requires a positive word aligned budget",
            self.label
        );
        if self.budget() == budget {
            return false;
        }
        self.assert_drained("reallocate");
        self.readback = Some(Readback::new(device, self.label, budget, self.depth));
        true
    }

    pub fn clear(&mut self) {
        self.assert_drained("clear");
        self.readback = None;
    }

    pub fn declare(
        &mut self,
        encoder: &mut SubmissionEncoder,
        regions: &[(&Buffer, BufferAddress, BufferAddress)],
        sequence: u64,
        manifest: M,
    ) -> Option<(M, Vec<u8>)> {
        let bytes: BufferAddress = regions.iter().map(|(_, _, bytes)| *bytes).sum();
        assert!(
            bytes > 0 && bytes <= self.budget(),
            "publication {:?} declares {bytes} bytes within the {} byte budget it is sized for",
            self.label,
            self.budget()
        );
        let displaced = self
            .readback
            .as_mut()
            .expect("a declaration publishes through a budgeted ring")
            .enqueue_regions(encoder, regions, sequence);
        let displaced = displaced.map(|(_, bytes)| (self.retire(), bytes));
        self.declared.push_back(manifest);
        displaced
    }

    pub fn collect(&mut self) -> Vec<(M, Vec<u8>)> {
        let arrivals = match &mut self.readback {
            Some(readback) => readback.collect(),
            None => Vec::new(),
        };
        arrivals
            .into_iter()
            .map(|(_, bytes)| (self.retire(), bytes))
            .collect()
    }

    pub fn drain(&mut self) -> Vec<(M, Vec<u8>)> {
        let arrivals = match &mut self.readback {
            Some(readback) => readback.drain(),
            None => Vec::new(),
        };
        arrivals
            .into_iter()
            .map(|(_, bytes)| (self.retire(), bytes))
            .collect()
    }

    fn retire(&mut self) -> M {
        self.declared.pop_front().unwrap_or_else(|| {
            panic!(
                "publication {:?} retired an arrival it never declared",
                self.label
            )
        })
    }

    fn assert_drained(&self, what: &str) {
        assert!(
            self.is_idle() && self.declared.is_empty(),
            "publication {:?} must drain before it can {what}",
            self.label
        );
    }
}

pub fn read_regions(
    device: &Device,
    queue: &Queue,
    label: &str,
    regions: &[(&Buffer, BufferAddress, BufferAddress)],
) -> Vec<u8> {
    let bytes: BufferAddress = regions.iter().map(|region| region.2).sum();
    let mut readback = Readback::new(device, label, bytes, 1);
    let mut encoder = SubmissionEncoder::new(device, label);
    assert!(
        readback.enqueue_regions(&mut encoder, regions, 0).is_none(),
        "a one shot read requires an idle readback"
    );
    encoder.submit(queue);
    readback
        .drain()
        .pop()
        .expect("a one shot read retires exactly once")
        .1
}
