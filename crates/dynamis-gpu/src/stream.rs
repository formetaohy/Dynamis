use crate::{GpuBuffer, GpuSlot, StorageId};
use wgpu::{BufferAddress, BufferUsages, CommandEncoder, Device, Queue};

pub const STREAM: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);
pub const UNIFORM: BufferUsages = BufferUsages::UNIFORM.union(BufferUsages::COPY_DST);
pub const PACK: BufferUsages = BufferUsages::COPY_DST.union(BufferUsages::COPY_SRC);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamElement {
    wgsl: &'static str,
    bytes: u64,
}

impl StreamElement {
    pub const fn new(wgsl: &'static str, bytes: u64) -> Self {
        assert!(bytes > 0, "a stream element must occupy at least one byte");
        Self { wgsl, bytes }
    }

    pub const fn wgsl(self) -> &'static str {
        self.wgsl
    }

    pub const fn bytes(self) -> u64 {
        self.bytes
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Contents {
    Scratch,
    Durable,
    Seeded(u32),
}

impl Contents {
    pub const fn durable(self) -> bool {
        matches!(self, Self::Durable | Self::Seeded(_))
    }

    pub const fn seed(self) -> Option<u32> {
        match self {
            Self::Seeded(word) => Some(word),
            Self::Scratch | Self::Durable => None,
        }
    }
}

pub struct Stream {
    label: &'static str,
    buffer: GpuBuffer,
    slots: u32,
    stride: u64,
    element: StreamElement,
    usage: BufferUsages,
    contents: Contents,
}

fn assert_fits(device: &Device, label: &str, bytes: BufferAddress) {
    assert!(bytes > 0, "stream {label:?} requires at least one byte");
    let limits = device.limits();
    assert!(
        bytes <= limits.max_storage_buffer_binding_size,
        "stream {label:?} requires {bytes} bytes but the device storage binding limit is {}",
        limits.max_storage_buffer_binding_size
    );
    assert!(
        bytes <= limits.max_buffer_size,
        "stream {label:?} requires {bytes} bytes but the device buffer limit is {}",
        limits.max_buffer_size
    );
}

pub struct StreamDesc {
    pub label: &'static str,
    pub slots: u32,
    pub element: StreamElement,
    pub elements_per_slot: u64,
    pub usage: BufferUsages,
    pub contents: Contents,
}

#[derive(Clone, Copy)]
pub struct TypedSlot<'a> {
    slot: GpuSlot<'a>,
    element: StreamElement,
}

impl<'a> TypedSlot<'a> {
    pub fn new(slot: GpuSlot<'a>, element: StreamElement) -> Self {
        Self { slot, element }
    }

    pub fn slot(self) -> GpuSlot<'a> {
        self.slot
    }

    pub fn element(self) -> StreamElement {
        self.element
    }

    pub fn storage_id(self) -> StorageId {
        self.slot.storage_id()
    }
}

impl Stream {
    pub fn new(device: &Device, queue: &Queue, desc: StreamDesc) -> Self {
        let StreamDesc {
            label,
            slots,
            element,
            elements_per_slot,
            usage,
            contents,
        } = desc;
        assert!(slots > 0, "stream {label:?} requires at least one slot");
        assert!(
            elements_per_slot > 0,
            "stream {label:?} requires at least one element per slot"
        );
        assert!(
            element.bytes().is_multiple_of(4),
            "stream {label:?} requires a word aligned record of {}",
            element.wgsl()
        );
        let stride = element.bytes() * elements_per_slot;
        let bytes = slots as BufferAddress * stride;
        assert_fits(device, label, bytes);
        let stream = Self {
            label,
            buffer: GpuBuffer::new(device, label, bytes, usage),
            slots,
            stride,
            element,
            usage,
            contents,
        };
        if let Some(word) = contents.seed() {
            stream.write_at(queue, 0, &word.to_le_bytes());
        }
        stream
    }

    pub fn slots(&self) -> u32 {
        self.slots
    }

    pub fn stride(&self) -> u64 {
        self.stride
    }

    pub fn element(&self) -> StreamElement {
        self.element
    }

    pub fn size(&self) -> BufferAddress {
        self.buffer.size()
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        self.buffer.buffer()
    }

    pub fn gpu(&self) -> &GpuBuffer {
        &self.buffer
    }

    pub fn slot(&self) -> GpuSlot<'_> {
        GpuSlot::whole(&self.buffer)
    }

    pub fn write(&self, queue: &Queue, bytes: &[u8]) {
        self.buffer.write(queue, bytes);
    }

    pub fn write_at(&self, queue: &Queue, offset: BufferAddress, bytes: &[u8]) {
        self.buffer.write_at(queue, offset, bytes);
    }

    pub fn reserve(&mut self, device: &Device, encoder: &mut CommandEncoder, slots: u32) -> bool {
        assert!(
            slots > 0,
            "stream {:?} requires at least one slot",
            self.label
        );
        if slots == self.slots {
            return false;
        }
        let bytes = slots as BufferAddress * self.stride;
        assert_fits(device, self.label, bytes);
        let next = GpuBuffer::new(device, self.label, bytes, self.usage);
        if self.contents.durable() {
            encoder.copy_buffer_to_buffer(
                self.buffer.buffer(),
                0,
                next.buffer(),
                0,
                self.buffer.size().min(next.size()),
            );
        }
        self.buffer = next;
        self.slots = slots;
        true
    }
}

impl<'a> From<&'a Stream> for GpuSlot<'a> {
    fn from(stream: &'a Stream) -> Self {
        stream.slot()
    }
}

impl<'a> From<&'a Stream> for TypedSlot<'a> {
    fn from(stream: &'a Stream) -> Self {
        Self::new(stream.slot(), stream.element())
    }
}
