use std::sync::atomic::{AtomicU64, Ordering};
use wgpu::{Buffer, BufferAddress, BufferDescriptor, BufferUsages, Device, Queue};

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

impl<'a> From<&'a GpuBuffer> for GpuSlot<'a> {
    fn from(buffer: &'a GpuBuffer) -> Self {
        Self::whole(buffer)
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
