use std::sync::mpsc::{self, TryRecvError};
use wgpu::{
    Buffer, BufferAddress, BufferDescriptor, BufferUsages, Device, MapMode, PollType, Queue,
};

pub struct GpuBuffer {
    buffer: Buffer,
    size: BufferAddress,
}

impl GpuBuffer {
    pub fn new(device: &Device, label: &str, size: BufferAddress, usage: BufferUsages) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        Self { buffer, size }
    }

    pub fn write(&self, queue: &Queue, bytes: &[u8]) {
        assert!(bytes.len() as u64 <= self.size, "write exceeds buffer size");
        queue.write_buffer(&self.buffer, 0, bytes);
    }

    pub fn as_entire_binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: 0,
            size: None,
        })
    }

    pub fn as_indirect_target(&self) -> &Buffer {
        &self.buffer
    }

    pub fn size(&self) -> BufferAddress {
        self.size
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn read_sync(&self, device: &Device, queue: &Queue) -> Vec<u8> {
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("readback staging"),
            size: self.size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.size);
        queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        loop {
            device
                .poll(PollType::wait_indefinitely())
                .expect("device lost while awaiting readback");
            match receiver.try_recv() {
                Ok(Ok(())) => break,
                Ok(Err(error)) => panic!("buffer mapping failed: {error}"),
                Err(TryRecvError::Empty) => std::thread::yield_now(),
                Err(TryRecvError::Disconnected) => panic!("mapping callback was dropped"),
            }
        }
        let bytes = slice
            .get_mapped_range()
            .expect("mapped range unavailable")
            .to_vec();
        staging.unmap();
        bytes
    }
}
