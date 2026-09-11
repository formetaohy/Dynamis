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

struct PendingRead {
    sequence: u64,
    bytes: BufferAddress,
    submission: Arc<OnceLock<SubmissionIndex>>,
    completion: Receiver<Result<(), BufferAsyncError>>,
}

pub struct BufferReadback {
    device: Device,
    label: String,
    staging: Buffer,
    pending: Option<PendingRead>,
}

impl BufferReadback {
    pub fn new(device: &Device, label: &str, size: BufferAddress) -> Self {
        assert!(
            size > 0 && size.is_multiple_of(4),
            "readback size must be positive and word aligned"
        );
        Self {
            device: device.clone(),
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

    pub fn size(&self) -> BufferAddress {
        self.staging.size()
    }

    fn is_sealed(&self) -> bool {
        self.pending
            .as_ref()
            .is_some_and(|pending| pending.submission.get().is_some())
    }

    pub fn read(
        &mut self,
        queue: &Queue,
        source: &Buffer,
        source_offset: BufferAddress,
        bytes: BufferAddress,
    ) -> Vec<u8> {
        let mut encoder = SubmissionEncoder::new(&self.device, &self.label);
        self.record(&mut encoder, source, source_offset, bytes, 0);
        encoder.submit(queue);
        self.wait().1
    }

    fn record(
        &mut self,
        encoder: &mut SubmissionEncoder,
        source: &Buffer,
        source_offset: BufferAddress,
        bytes: BufferAddress,
        sequence: u64,
    ) {
        assert!(
            self.pending.is_none(),
            "readback {} is still in use",
            self.label
        );
        assert!(
            bytes > 0 && bytes.is_multiple_of(4),
            "readback length must be positive and word aligned"
        );
        assert!(
            source_offset.is_multiple_of(4),
            "readback source offset must be word aligned"
        );
        assert!(bytes <= self.size(), "readback exceeds staging capacity");
        assert!(
            source_offset <= source.size() && bytes <= source.size() - source_offset,
            "readback exceeds source buffer"
        );
        let (sender, completion) = mpsc::channel();
        encoder.copy_buffer_to_buffer(source, source_offset, &self.staging, 0, bytes);
        encoder.map_buffer_on_submit(&self.staging, MapMode::Read, ..bytes, move |result| {
            let _ = sender.send(result);
        });
        self.pending = Some(PendingRead {
            sequence,
            bytes,
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

    fn wait(&mut self) -> (u64, Vec<u8>) {
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
        self.device
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

pub struct ReadbackRing {
    device: Device,
    label: String,
    size: BufferAddress,
    slots: VecDeque<BufferReadback>,
    inflight: usize,
    last_sequence: Option<u64>,
}

impl ReadbackRing {
    pub const DEPTH: usize = 4;

    pub fn new(device: &Device, label: &str, size: BufferAddress) -> Self {
        Self {
            device: device.clone(),
            label: label.to_owned(),
            size,
            slots: (0..Self::DEPTH)
                .map(|index| BufferReadback::new(device, &format!("{label} slot {index}"), size))
                .collect(),
            inflight: 0,
            last_sequence: None,
        }
    }

    pub fn size(&self) -> BufferAddress {
        self.size
    }

    pub fn enqueue(
        &mut self,
        encoder: &mut SubmissionEncoder,
        source: &Buffer,
        source_offset: BufferAddress,
        bytes: BufferAddress,
        sequence: u64,
    ) -> Option<(u64, Vec<u8>)> {
        assert!(
            self.last_sequence
                .is_none_or(|previous| sequence > previous),
            "readback sequences must increase"
        );
        let displaced = self.reclaim();
        self.slots[self.inflight].record(encoder, source, source_offset, bytes, sequence);
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
            let entry = self.slots[0].wait();
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
            self.slots.push_back(BufferReadback::new(
                &self.device,
                &format!("{} slot {index}", self.label),
                self.size,
            ));
            return None;
        }
        let mut oldest = self.slots.pop_front().expect("readback ring holds a slot");
        let entry = oldest.wait();
        self.slots.push_back(oldest);
        self.inflight -= 1;
        Some(entry)
    }

    fn retire(&mut self) {
        let slot = self.slots.pop_front().expect("readback ring holds a slot");
        self.slots.push_back(slot);
        self.inflight -= 1;
    }
}
