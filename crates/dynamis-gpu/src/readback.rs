use crate::SubmissionEncoder;
use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, OnceLock};
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
    completion: Receiver<Result<(), BufferAsyncError>>,
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
            completion,
        });
    }

    fn collect(&mut self) -> Option<(u64, Vec<u8>)> {
        let pending = self.pending.as_ref()?;
        pending.submission.get()?;
        match pending.completion.try_recv() {
            Ok(result) => Some(self.consume(result)),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                panic!("readback {} completion was dropped", self.label)
            }
        }
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
            "readback {} sequences must increase",
            self.label
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
