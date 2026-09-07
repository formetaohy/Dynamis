use dynamis_gpu::{ComputeRecorder, GpuBucketSort, GpuBuffer, GpuContext, GpuCountArgs, GpuSort};
use std::sync::OnceLock;
use wgpu::{Backend, BufferUsages};

static CONTEXT: OnceLock<GpuContext> = OnceLock::new();

fn shared() -> &'static GpuContext {
    CONTEXT.get_or_init(|| pollster::block_on(GpuContext::new()))
}

fn read_u32s(context: &GpuContext, buffer: &GpuBuffer, words: usize) -> Vec<u32> {
    let bytes = (words * 4) as u64;
    let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("sort readback"),
        size: bytes,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer.buffer(), 0, &staging, 0, bytes);
    context.queue().submit([encoder.finish()]);
    let _ = context.device().poll(wgpu::PollType::wait_indefinitely());
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = context.device().poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map").expect("map error");
    let mapped = slice.get_mapped_range().unwrap();
    bytemuck::cast_slice(&mapped).to_vec()
}

struct SortInputs {
    keys_lo: GpuBuffer,
    keys_hi: GpuBuffer,
    values: GpuBuffer,
    out_lo: GpuBuffer,
    out_hi: GpuBuffer,
    out_values: GpuBuffer,
}

fn sort_inputs(context: &GpuContext, lo_data: &[u32], hi_data: &[u32]) -> SortInputs {
    let device = context.device();
    let bytes = (lo_data.len() * 4) as u64;
    let make = |label: &str, usage: wgpu::BufferUsages| GpuBuffer::new(device, label, bytes, usage);
    let keys_lo = GpuBuffer::new(
        device,
        "keys_lo",
        bytes,
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    );
    keys_lo.write(context.queue(), bytemuck::cast_slice(lo_data));
    let keys_hi = GpuBuffer::new(
        device,
        "keys_hi",
        bytes,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    keys_hi.write(context.queue(), bytemuck::cast_slice(hi_data));
    let values_data: Vec<u32> = (0..lo_data.len() as u32).collect();
    let values = GpuBuffer::new(
        device,
        "values",
        bytes,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    values.write(context.queue(), bytemuck::cast_slice(&values_data));
    SortInputs {
        keys_lo,
        keys_hi,
        values,
        out_lo: make("out_lo", BufferUsages::STORAGE | BufferUsages::COPY_SRC),
        out_hi: make("out_hi", BufferUsages::STORAGE | BufferUsages::COPY_SRC),
        out_values: make("out_values", BufferUsages::STORAGE | BufferUsages::COPY_SRC),
    }
}

fn run_sort(
    keys_lo: &[u32],
    keys_hi: &[u32],
    lo_words: u32,
    hi_words: u32,
) -> (Vec<u32>, Vec<u32>) {
    let context = shared().clone();
    let device = context.device();
    let inputs = sort_inputs(&context, keys_lo, keys_hi);
    let holder = GpuBuffer::new(
        device,
        "holder",
        12,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    let args = GpuBuffer::new(
        device,
        "args",
        32,
        BufferUsages::STORAGE | BufferUsages::INDIRECT,
    );
    let mut count_bytes = vec![0u8; 12];
    count_bytes[..4].copy_from_slice(&(keys_lo.len() as u32).to_le_bytes());
    count_bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
    count_bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    holder.write(context.queue(), &count_bytes);
    let sort = GpuSort::new(&context, "test sort", keys_lo.len() as u32);
    let count_args = GpuCountArgs::new(&context, "test count args");
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sort"),
    });
    {
        let mut recorder = ComputeRecorder::begin(&mut encoder, "sort pass");
        count_args.encode(device, &mut recorder, &holder, &args);
        sort.sort_64(
            device,
            &mut recorder,
            &holder,
            &args,
            lo_words,
            hi_words,
            &inputs.keys_lo,
            &inputs.keys_hi,
            &inputs.values,
            &inputs.out_lo,
            &inputs.out_hi,
            &inputs.out_values,
        );
    }
    context.queue().submit([encoder.finish()]);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let keys_out = read_u32s(&context, &inputs.out_lo, keys_lo.len());
    let values_out = read_u32s(&context, &inputs.out_values, keys_lo.len());
    (keys_out, values_out)
}

#[test]
fn count_args_convert_element_counts_to_dispatch_sizes() {
    let context = shared().clone();
    let holder = GpuBuffer::new(
        context.device(),
        "holder",
        12,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    let args = GpuBuffer::new(
        context.device(),
        "args",
        32,
        BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
    );
    let mut count_bytes = vec![0u8; 12];
    count_bytes[..4].copy_from_slice(&1000u32.to_le_bytes());
    count_bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
    count_bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    holder.write(context.queue(), &count_bytes);
    let count_args = GpuCountArgs::new(&context, "count args");
    let mut encoder = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut recorder = ComputeRecorder::begin(&mut encoder, "count");
        count_args.encode(context.device(), &mut recorder, &holder, &args);
    }
    context.queue().submit([encoder.finish()]);
    let _ = context.device().poll(wgpu::PollType::wait_indefinitely());
    let out = read_u32s(&context, &args, 8);
    assert_eq!(
        out[0], 16,
        "1000 elements / 64 threads must dispatch 16 workgroups"
    );
    assert_eq!(out[1], 1);
    assert_eq!(
        out[4], 4,
        "1000 elements / 256 threads must dispatch 4 workgroups"
    );
}

#[test]
fn radix_sort_orders_keys_and_keeps_values_stable() {
    let data = vec![3u32, 5, 1, 0, 7, 2, 2, 9, 4, 6];
    let (keys, values) = run_sort(&data, &vec![0u32; data.len()], 1, 0);
    let mut pairs: Vec<(u32, u32)> = data
        .iter()
        .enumerate()
        .map(|(i, &k)| (k, i as u32))
        .collect();
    pairs.sort();
    let expected_keys: Vec<u32> = pairs.iter().map(|&(k, _)| k).collect();
    let expected_values: Vec<u32> = pairs.iter().map(|&(_, v)| v).collect();
    assert_eq!(keys, expected_keys);
    assert_eq!(
        values, expected_values,
        "values must follow their keys stably"
    );
}

#[test]
fn radix_sort_orders_high_words_alone() {
    let n = 64u32;
    let lo: Vec<u32> = (0..n).map(|i| (i * 37) % n).collect();
    let hi: Vec<u32> = (0..n).map(|i| (i * 7) / n).collect();
    let (keys, _) = run_sort(&lo, &hi, 0, 1);
    let mut order: Vec<u32> = (0..n).collect();
    order.sort_by_key(|&i| (hi[i as usize], i));
    let expected: Vec<u32> = order.iter().map(|&i| lo[i as usize]).collect();
    assert_eq!(
        keys, expected,
        "high-word passes must order keys by the upper bits"
    );
}

#[test]
fn bucket_sort_groups_keys_and_values_stably() {
    let context = shared().clone();
    let device = context.device();
    let data = vec![2u32, 0, 1, 2, 0, 1, 3, 3, 0, 2];
    let bytes = (data.len() * 4) as u64;
    let keys = GpuBuffer::new(
        device,
        "keys",
        bytes,
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    );
    keys.write(context.queue(), bytemuck::cast_slice(&data));
    let indices: Vec<u32> = (0..data.len() as u32).collect();
    let values = GpuBuffer::new(
        device,
        "values",
        bytes,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    values.write(context.queue(), bytemuck::cast_slice(&indices));
    let keys_out = GpuBuffer::new(
        device,
        "keys_out",
        bytes,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let values_out = GpuBuffer::new(
        device,
        "values_out",
        bytes,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let holder = GpuBuffer::new(
        device,
        "holder",
        12,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    let args = GpuBuffer::new(
        device,
        "args",
        32,
        BufferUsages::STORAGE | BufferUsages::INDIRECT,
    );
    let mut count_bytes = vec![0u8; 12];
    count_bytes[..4].copy_from_slice(&(data.len() as u32).to_le_bytes());
    count_bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
    count_bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    holder.write(context.queue(), &count_bytes);
    let bucket = GpuBucketSort::new(&context, "test bucket", 4, data.len() as u32);
    let count_args = GpuCountArgs::new(&context, "bucket args");
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut recorder = ComputeRecorder::begin(&mut encoder, "bucket");
        count_args.encode(device, &mut recorder, &holder, &args);
        bucket.sort(
            device,
            &mut recorder,
            &holder,
            &args,
            &keys,
            &values,
            &keys_out,
            &values_out,
        );
    }
    context.queue().submit([encoder.finish()]);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let keys_sorted = read_u32s(&context, &keys_out, data.len());
    let values_sorted = read_u32s(&context, &values_out, data.len());
    let mut pairs: Vec<(u32, u32)> = data
        .iter()
        .enumerate()
        .map(|(i, &k)| (k, i as u32))
        .collect();
    pairs.sort();
    let expected_keys: Vec<u32> = pairs.iter().map(|&(k, _)| k).collect();
    let expected_values: Vec<u32> = pairs.iter().map(|&(_, v)| v).collect();
    assert_eq!(keys_sorted, expected_keys, "keys must be grouped ascending");
    assert_eq!(
        values_sorted, expected_values,
        "values must be stably ordered"
    );
}

#[test]
fn backend_never_gl() {
    assert_ne!(shared().adapter_info().backend, Backend::Gl);
}

#[cfg(target_os = "windows")]
#[test]
fn windows_prefers_dx12_when_vulkan_available() {
    use wgpu::{Backends, Instance};
    let instance = Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let has_dx12 = !pollster::block_on(instance.enumerate_adapters(Backends::DX12)).is_empty();
    let has_vulkan = !pollster::block_on(instance.enumerate_adapters(Backends::VULKAN)).is_empty();
    if has_dx12 && has_vulkan {
        assert_eq!(shared().adapter_info().backend, Backend::Dx12);
    }
}
