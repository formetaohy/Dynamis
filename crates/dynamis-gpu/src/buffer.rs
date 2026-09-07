use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use wgpu::{
    Buffer, BufferAddress, BufferAsyncError, BufferDescriptor, BufferUsages, Device, MapMode,
    PollType, Queue,
};

static NEXT_BUFFER_TOKEN: AtomicU64 = AtomicU64::new(1);

pub struct GpuBuffer {
    buffer: Buffer,
    size: BufferAddress,
    usage: BufferUsages,
    token: u64,
}

impl GpuBuffer {
    pub fn new(device: &Device, label: &str, size: BufferAddress, usage: BufferUsages) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            size,
            usage,
            token: NEXT_BUFFER_TOKEN.fetch_add(1, Ordering::Relaxed),
        }
    }

    pub fn token(&self) -> u64 {
        self.token
    }

    pub fn write(&self, queue: &Queue, bytes: &[u8]) {
        assert!(
            self.usage.contains(BufferUsages::COPY_DST),
            "buffer write requires COPY_DST usage"
        );
        assert!(bytes.len() as u64 <= self.size, "write exceeds buffer size");
        queue.write_buffer(&self.buffer, 0, bytes);
    }

    pub fn write_at(&self, queue: &Queue, offset: u64, bytes: &[u8]) {
        assert!(
            self.usage.contains(BufferUsages::COPY_DST),
            "buffer write requires COPY_DST usage"
        );
        assert!(
            offset + bytes.len() as u64 <= self.size,
            "write exceeds buffer size"
        );
        queue.write_buffer(&self.buffer, offset, bytes);
    }

    pub fn as_binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: 0,
            size: None,
        })
    }

    pub fn as_indirect_args(&self) -> &Buffer {
        &self.buffer
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn size(&self) -> BufferAddress {
        self.size
    }
}

struct PendingRead {
    sequence: u64,
    mapping_started: bool,
    result: Arc<Mutex<Option<Result<(), BufferAsyncError>>>>,
}

pub struct GpuReadback {
    staging: [Buffer; 2],
    size: BufferAddress,
    next: usize,
    pending: [Option<PendingRead>; 2],
}

impl GpuReadback {
    pub fn new(device: &Device, label: &str, size: BufferAddress) -> Self {
        assert!(size > 0, "readback size must be positive");
        let staging = std::array::from_fn(|index| {
            let staging_label = format!("{label} staging {index}");
            device.create_buffer(&BufferDescriptor {
                label: Some(&staging_label),
                size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        Self {
            staging,
            size,
            next: 0,
            pending: [None, None],
        }
    }

    pub fn enqueue(
        &mut self,
        device: &Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &Buffer,
        sequence: u64,
    ) -> Option<(u64, Vec<u8>)> {
        let slot = self.next;
        self.next = (self.next + 1) % 2;
        let displaced = if self.pending[slot].is_some() {
            Some(self.drain(device, slot))
        } else {
            None
        };
        encoder.copy_buffer_to_buffer(source, 0, &self.staging[slot], 0, self.size);
        self.pending[slot] = Some(PendingRead {
            sequence,
            mapping_started: false,
            result: Arc::new(Mutex::new(None)),
        });
        displaced
    }

    pub fn arm(&mut self) {
        for slot in 0..2 {
            if self.pending[slot].is_some() {
                self.ensure_mapping(slot);
            }
        }
    }

    pub fn poll(&mut self, device: &Device) -> Vec<(u64, Vec<u8>)> {
        for slot in 0..2 {
            if self.pending[slot].is_some() {
                self.ensure_mapping(slot);
            }
        }
        device
            .poll(PollType::Poll)
            .expect("device lost while polling readback");
        let mut completed = Vec::new();
        for slot in 0..2 {
            if let Some(entry) = self.try_consume(slot) {
                completed.push(entry);
            }
        }
        completed.sort_unstable_by_key(|(sequence, _)| *sequence);
        completed
    }

    fn ensure_mapping(&mut self, slot: usize) {
        let pending = self.pending[slot]
            .as_mut()
            .expect("pending readback just checked");
        if !pending.mapping_started {
            pending.mapping_started = true;
            let result = pending.result.clone();
            self.staging[slot]
                .slice(..)
                .map_async(MapMode::Read, move |value| {
                    *result.lock().unwrap() = Some(value);
                });
        }
    }

    fn try_consume(&mut self, slot: usize) -> Option<(u64, Vec<u8>)> {
        let result = {
            let pending = self.pending[slot].as_ref()?;
            pending.result.lock().unwrap().take()?
        };
        result.unwrap_or_else(|error| panic!("buffer mapping failed: {error}"));
        let pending = self.pending[slot]
            .take()
            .expect("pending readback just checked");
        let bytes = self.staging[slot]
            .slice(..)
            .get_mapped_range()
            .expect("mapped range unavailable")
            .to_vec();
        self.staging[slot].unmap();
        Some((pending.sequence, bytes))
    }

    fn drain(&mut self, device: &Device, slot: usize) -> (u64, Vec<u8>) {
        self.ensure_mapping(slot);
        loop {
            device
                .poll(PollType::wait_indefinitely())
                .expect("device lost while awaiting readback");
            if let Some(entry) = self.try_consume(slot) {
                return entry;
            }
            std::thread::yield_now();
        }
    }
}
