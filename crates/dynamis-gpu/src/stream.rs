use crate::{GpuBuffer, GpuSlot};
use wgpu::{BufferAddress, BufferUsages, CommandEncoder, Device, Queue};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Contents {
    Reset,
    Preserve,
    PreserveSeeded(u32),
}

impl Contents {
    fn survives_resize(self) -> bool {
        !matches!(self, Self::Reset)
    }
}

pub struct Stream {
    label: &'static str,
    buffer: GpuBuffer,
    slots: u32,
    stride: u64,
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

impl Stream {
    pub fn new(
        device: &Device,
        queue: &Queue,
        label: &'static str,
        slots: u32,
        stride: u64,
        usage: BufferUsages,
        contents: Contents,
    ) -> Self {
        assert!(slots > 0, "stream {label:?} requires at least one slot");
        assert!(
            stride > 0 && stride.is_multiple_of(4),
            "stream {label:?} requires a positive word aligned stride"
        );
        let bytes = slots as BufferAddress * stride;
        assert_fits(device, label, bytes);
        let stream = Self {
            label,
            buffer: GpuBuffer::new(device, label, bytes, usage),
            slots,
            stride,
            usage,
            contents,
        };
        if let Contents::PreserveSeeded(word) = contents {
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
        if self.contents.survives_resize() {
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
