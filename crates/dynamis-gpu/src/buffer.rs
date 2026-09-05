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
}

pub struct Readback {
    staging: [Buffer; 2],
    size: BufferAddress,
    submitted: u64,
}

impl Readback {
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
            submitted: 0,
        }
    }

    pub fn enqueue(&mut self, encoder: &mut wgpu::CommandEncoder, source: &Buffer) {
        let index = (self.submitted % 2) as usize;
        encoder.copy_buffer_to_buffer(source, 0, &self.staging[index], 0, self.size);
        self.submitted += 1;
    }

    pub fn read(&mut self, device: &Device) -> Vec<u8> {
        assert!(
            self.submitted > 0,
            "no readback has been submitted to the GPU"
        );
        let index = ((self.submitted - 1) % 2) as usize;
        let staging = &self.staging[index];
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
