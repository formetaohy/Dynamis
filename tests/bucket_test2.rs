use dynamis::{ComputeRecorder, GpuBucketSort, GpuBuffer, GpuContext, GpuCountArgs};
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

fn run_bucket(data: &[u32], buckets: u32) -> (Vec<u32>, Vec<u32>) {
    let (_guard, gpu) = serialized_gpu();
    let device = gpu.device();
    let queue = gpu.queue();
    let bytes = data.len() as u64 * 4;
    let keys = GpuBuffer::new(device, "keys", bytes, BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC);
    let values = GpuBuffer::new(device, "values", bytes, BufferUsages::STORAGE | BufferUsages::COPY_DST);
    let keys_out = GpuBuffer::new(device, "keys_out", bytes, BufferUsages::STORAGE | BufferUsages::COPY_SRC);
    let values_out = GpuBuffer::new(device, "values_out", bytes, BufferUsages::STORAGE | BufferUsages::COPY_SRC);
    let holder = GpuBuffer::new(device, "holder", 12, BufferUsages::STORAGE | BufferUsages::COPY_DST);
    let args = GpuBuffer::new(device, "args", 32, BufferUsages::STORAGE | BufferUsages::INDIRECT);
    let mut count_bytes = vec![0u8; 12];
    count_bytes[..4].copy_from_slice(&(data.len() as u32).to_le_bytes());
    count_bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
    count_bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    holder.write(queue, &count_bytes);
    keys.write(queue, bytemuck::cast_slice(data));
    let vals: Vec<u32> = (0..data.len() as u32).collect();
    values.write(queue, bytemuck::cast_slice(&vals));
    let bucket = GpuBucketSort::new(device, "test", buckets, data.len() as u32);
    let count_args = GpuCountArgs::new(device, "targs");
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes * 2,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sort"),
    });
    {
        let mut recorder = ComputeRecorder::begin(&mut encoder, "bucket sort test");
        count_args.encode(device, &mut recorder, &holder, &args);
        bucket.sort(device, &mut recorder, &holder, &args, &keys, &values, &keys_out, &values_out);
    }
    encoder.copy_buffer_to_buffer(keys_out.buffer(), 0, &staging, 0, bytes);
    encoder.copy_buffer_to_buffer(values_out.buffer(), 0, &staging, bytes, bytes);
    queue.submit([encoder.finish()]);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map").expect("map err");
    let mapped = slice.get_mapped_range().unwrap();
    let k: Vec<u32> = bytemuck::cast_slice(&mapped[..bytes as usize]).to_vec();
    let v: Vec<u32> = bytemuck::cast_slice(&mapped[bytes as usize..bytes as usize * 2]).to_vec();
    (k, v)
}

#[test]
fn bucket_groups_by_key() {
    let data = vec![2u32, 0, 1, 2, 0, 1, 3, 3, 0, 2];
    let (keys, values) = run_bucket(&data, 4);
    let mut expected: Vec<u32> = data.clone();
    expected.sort();
    assert_eq!(keys, expected, "keys must be grouped ascending");
    // stable: values are original indices
    let mut ref_pairs: Vec<(u32, u32)> = data.iter().enumerate().map(|(i, &k)| (k, i as u32)).collect();
    ref_pairs.sort();
    let expected_values: Vec<u32> = ref_pairs.iter().map(|&(_, i)| i).collect();
    assert_eq!(values, expected_values, "values must be stably ordered");
}

#[test]
fn bucket_groups_larger() {
    let data: Vec<u32> = (0..64).map(|i| (i * 5 % 8) as u32).collect();
    let (keys, _values) = run_bucket(&data, 8);
    let mut w: Vec<(u32, u32)> = data.iter().enumerate().map(|(i, &k)| (k, i as u32)).collect();
    w.sort();
    assert_eq!(keys, w.iter().map(|&(k, _)| k).collect::<Vec<_>>());
}
