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

    pub fn zeroed(device: &Device, label: &str, size: BufferAddress, usage: BufferUsages) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("mapped at creation range unavailable")
            .slice(..)
            .fill(0);
        buffer.unmap();
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

    pub fn as_binding_at(&self, offset: u64, size: u64) -> wgpu::BindingResource<'_> {
        let length = core::num::NonZeroU64::new(size).expect("binding size must be non-zero");
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset,
            size: Some(length),
        })
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn size(&self) -> BufferAddress {
        self.size
    }
}

#[derive(Clone, Copy)]
pub struct GpuSlot<'a> {
    buffer: &'a GpuBuffer,
    offset: BufferAddress,
    size: BufferAddress,
}

impl<'a> GpuSlot<'a> {
    pub fn whole(buffer: &'a GpuBuffer) -> Self {
        Self {
            buffer,
            offset: 0,
            size: buffer.size(),
        }
    }

    pub fn range(buffer: &'a GpuBuffer, offset: BufferAddress, size: BufferAddress) -> Self {
        assert!(size > 0, "binding range must be non-empty");
        assert_eq!(offset % 4, 0, "binding offset must be word aligned");
        assert_eq!(size % 4, 0, "binding size must be word aligned");
        Self {
            buffer,
            offset,
            size,
        }
    }

    pub fn as_binding(&self) -> wgpu::BindingResource<'a> {
        self.buffer.as_binding_at(self.offset, self.size)
    }

    pub fn identity(&self) -> (u64, BufferAddress, BufferAddress) {
        (self.buffer.token(), self.offset, self.size)
    }
}

struct PendingRead {
    sequence: u64,
    bytes: BufferAddress,
    mapping_started: bool,
    result: Arc<Mutex<Option<Result<(), BufferAsyncError>>>>,
}

pub struct GpuReadback {
    label: String,
    staging: Vec<Buffer>,
    size: BufferAddress,
    next: usize,
    pending: Vec<Option<PendingRead>>,
}

impl GpuReadback {
    pub const DEPTH: usize = 4;

    pub fn new(device: &Device, label: &str, size: BufferAddress) -> Self {
        assert!(size > 0, "readback size must be positive");
        let staging: Vec<Buffer> = (0..Self::DEPTH)
            .map(|index| {
                let staging_label = format!("{label} staging {index}");
                device.create_buffer(&BufferDescriptor {
                    label: Some(&staging_label),
                    size,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let pending = (0..staging.len()).map(|_| None).collect();
        Self {
            label: label.to_owned(),
            staging,
            size,
            next: 0,
            pending,
        }
    }

    pub fn enqueue(
        &mut self,
        device: &Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &Buffer,
        source_offset: BufferAddress,
        bytes: BufferAddress,
        sequence: u64,
    ) -> Option<(u64, Vec<u8>)> {
        assert!(bytes > 0, "an empty readback has nothing to displace");
        assert!(bytes <= self.size, "readback exceeds the staging buffer");
        let slot = self.next;
        self.next = (self.next + 1) % self.staging.len();
        let displaced = if self.pending[slot].is_some() {
            Some(self.await_slot(device, slot))
        } else {
            None
        };
        encoder.copy_buffer_to_buffer(source, source_offset, &self.staging[slot], 0, bytes);
        self.pending[slot] = Some(PendingRead {
            sequence,
            bytes,
            mapping_started: false,
            result: Arc::new(Mutex::new(None)),
        });
        displaced
    }

    pub fn size(&self) -> BufferAddress {
        self.size
    }

    pub fn arm(&mut self) {
        for slot in 0..self.staging.len() {
            if self.pending[slot].is_some() {
                self.ensure_mapping(slot);
            }
        }
    }

    pub fn poll(&mut self, device: &Device) -> Vec<(u64, Vec<u8>)> {
        for slot in 0..self.staging.len() {
            if self.pending[slot].is_some() {
                self.ensure_mapping(slot);
            }
        }
        device
            .poll(PollType::Poll)
            .expect("device lost while polling readback");
        let mut completed = Vec::new();
        for slot in 0..self.staging.len() {
            if let Some(entry) = self.try_consume(slot) {
                completed.push(entry);
            }
        }
        completed.sort_unstable_by_key(|(sequence, _)| *sequence);
        completed
    }

    pub fn drain(&mut self, device: &Device) -> Vec<(u64, Vec<u8>)> {
        let mut arrived = Vec::new();
        while self.pending.iter().any(Option::is_some) {
            device
                .poll(PollType::wait_indefinitely())
                .expect("device lost while draining readback");
            arrived.extend(self.poll(device));
        }
        arrived
    }

    fn ensure_mapping(&mut self, slot: usize) {
        let pending = self.pending[slot]
            .as_mut()
            .expect("pending readback just checked");
        if !pending.mapping_started {
            pending.mapping_started = true;
            let result = pending.result.clone();
            let bytes = self.pending[slot]
                .as_ref()
                .expect("mapping a pending read")
                .bytes;
            self.staging[slot]
                .slice(..bytes)
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
        result.unwrap_or_else(|error| {
            panic!("buffer {} slot {slot} failed to map: {error:?}", self.label)
        });
        let pending = self.pending[slot]
            .take()
            .expect("pending readback just checked");
        let bytes = self.staging[slot]
            .slice(..pending.bytes)
            .get_mapped_range()
            .expect("mapped range unavailable")
            .to_vec();
        self.staging[slot].unmap();
        Some((pending.sequence, bytes))
    }

    fn await_slot(&mut self, device: &Device, slot: usize) -> (u64, Vec<u8>) {
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
