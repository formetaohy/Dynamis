use dynamis::{GpuBuffer, GpuContext, GpuSort};
use std::sync::{Mutex, MutexGuard};
use wgpu::BufferUsages;

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn serialized_gpu() -> (MutexGuard<'static, ()>, GpuContext) {
    let guard = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let context = pollster::block_on(GpuContext::new());
    (guard, context)
}

fn run_sort(data: &[u32]) -> Vec<u32> {
    let (_guard, gpu) = serialized_gpu();
    let device = gpu.device();
    let queue = gpu.queue();
    let count = data.len() as u64;
    let bytes = count * 4;
    let keys_lo = GpuBuffer::new(
        device,
        "klo",
        bytes,
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    );
    let keys_hi = GpuBuffer::new(
        device,
        "khi",
        bytes,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    let values = GpuBuffer::new(
        device,
        "val",
        bytes,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    let out_lo = GpuBuffer::new(device, "olo", bytes, BufferUsages::STORAGE);
    let out_hi = GpuBuffer::new(device, "ohi", bytes, BufferUsages::STORAGE);
    let out_val = GpuBuffer::new(device, "oval", bytes, BufferUsages::STORAGE);
    let holder = GpuBuffer::new(
        device,
        "holder",
        12,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    let mut count_bytes = vec![0u8; 12];
    count_bytes[..4].copy_from_slice(&(data.len() as u32).to_le_bytes());
    count_bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
    count_bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    holder.write(queue, &count_bytes);
    keys_lo.write(queue, bytemuck::cast_slice(data));
    keys_hi.write(queue, &vec![0u8; bytes as usize]);
    values.write(queue, bytemuck::cast_slice(data));
    let sort = GpuSort::new(device, "test", data.len() as u32);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut sort_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sort"),
    });
    sort.sort_64(
        device,
        &mut sort_encoder,
        &holder,
        data.len() as u32,
        &keys_lo,
        &keys_hi,
        &values,
        &out_lo,
        &out_hi,
        &out_val,
    );
    sort_encoder.copy_buffer_to_buffer(keys_lo.buffer(), 0, &staging, 0, bytes);
    queue.submit([sort_encoder.finish()]);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map").expect("map err");
    let mapped = slice.get_mapped_range().unwrap();
    bytemuck::cast_slice(&mapped).to_vec()
}

#[test]
fn sort_orders_keys() {
    let data = vec![3u32, 5, 1, 0, 7, 2, 2, 9, 4, 6];
    let result = run_sort(&data);
    let mut expected = data.clone();
    expected.sort();
    assert_eq!(result, expected.as_slice());
}

#[test]
fn sort_handles_duplicates() {
    let data = vec![7u32, 3, 7, 3, 7, 3, 1, 1, 1, 9];
    let result = run_sort(&data);
    let mut expected = data.clone();
    expected.sort();
    assert_eq!(result, expected.as_slice());
}

#[test]
fn sort_orders_larger_array() {
    let data: Vec<u32> = (0..64).map(|i| (i * 37 % 64) as u32).collect();
    let result = run_sort(&data);
    let mut expected = data.clone();
    expected.sort();
    assert_eq!(result, expected.as_slice());
}
