use dynamis_gpu::{ComputeRecorder, DispatchTable, GpuBuffer, GpuContext, GpuSlot};
use dynamis_kernel::{RadixSort, SortChannels};
use dynamis_layout::dispatch;
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

fn fill_args(table: &DispatchTable, context: &GpuContext, count: u32) {
    let row = context.workgroups_per_row();
    let workgroups = count.div_ceil(256);
    let mut args = [0u8; 16];
    args[..4].copy_from_slice(&workgroups.min(row).to_le_bytes());
    args[4..8].copy_from_slice(&workgroups.div_ceil(row).to_le_bytes());
    args[8..12].copy_from_slice(&1u32.to_le_bytes());
    table.buffer().write(context.queue(), &args);
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
    let mut count_bytes = vec![0u8; 12];
    count_bytes[..4].copy_from_slice(&(keys_lo.len() as u32).to_le_bytes());
    count_bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
    count_bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    holder.write(context.queue(), &count_bytes);
    let sort = RadixSort::new(&context, "test sort", keys_lo.len() as u32);
    let table = DispatchTable::new(device, "test dispatch", 1);
    fill_args(&table, &context, keys_lo.len() as u32);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sort"),
    });
    {
        let mut recorder =
            ComputeRecorder::begin(&mut encoder, "sort pass", context.workgroups_per_row());
        sort.sort(
            &mut recorder,
            &SortChannels {
                count: GpuSlot::whole(&holder),
                keys_lo: &inputs.keys_lo,
                keys_hi: &inputs.keys_hi,
                values: &inputs.values,
                scratch_lo: &inputs.out_lo,
                scratch_hi: &inputs.out_hi,
                scratch_values: &inputs.out_values,
            },
            lo_words,
            hi_words,
            &table,
            dispatch(0, 256, 0),
        );
    }
    context.queue().submit([encoder.finish()]);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let keys_out = read_u32s(&context, &inputs.out_lo, keys_lo.len());
    let values_out = read_u32s(&context, &inputs.out_values, keys_lo.len());
    (keys_out, values_out)
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
fn sort_groups_by_bucket_and_keeps_the_payload_order_stable() {
    let context = shared().clone();
    let device = context.device();
    let buckets = vec![2u32, 0, 1, 2, 0, 1, 3, 3, 0, 2];
    let payload: Vec<u32> = (0..buckets.len() as u32).collect();
    let lanes = buckets.len() as u32;
    let bytes = (buckets.len() * 4) as u64;
    let stream = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
    let group = GpuBuffer::new(device, "group", bytes, stream);
    group.write(context.queue(), bytemuck::cast_slice(&buckets));
    let index = GpuBuffer::new(device, "index", bytes, stream);
    index.write(context.queue(), bytemuck::cast_slice(&payload));
    let pad = GpuBuffer::new(device, "pad", bytes, stream);
    let scratch = GpuBuffer::new(device, "scratch", bytes, stream);
    let scratch_hi = GpuBuffer::new(device, "scratch hi", bytes, stream);
    let scratch_values = GpuBuffer::new(device, "scratch values", bytes, stream);
    let holder = GpuBuffer::new(
        device,
        "holder",
        12,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    let mut count_bytes = vec![0u8; 12];
    count_bytes[..4].copy_from_slice(&(buckets.len() as u32).to_le_bytes());
    holder.write(context.queue(), &count_bytes);
    let sort = RadixSort::new(&context, "grouping sort", lanes);
    let table = DispatchTable::new(device, "grouping dispatch", 1);
    fill_args(&table, &context, buckets.len() as u32);
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut recorder =
            ComputeRecorder::begin(&mut encoder, "grouping", context.workgroups_per_row());
        sort.sort(
            &mut recorder,
            &SortChannels {
                count: GpuSlot::whole(&holder),
                keys_lo: &index,
                keys_hi: &group,
                values: &pad,
                scratch_lo: &scratch,
                scratch_hi: &scratch_hi,
                scratch_values: &scratch_values,
            },
            dynamis_kernel::key_words(lanes),
            dynamis_kernel::key_words(4),
            &table,
            dispatch(0, 256, 0),
        );
    }
    context.queue().submit([encoder.finish()]);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let grouped = read_u32s(&context, &group, buckets.len());
    let ordered = read_u32s(&context, &index, buckets.len());
    let mut expected: Vec<(u32, u32)> = buckets
        .iter()
        .enumerate()
        .map(|(at, &bucket)| (bucket, at as u32))
        .collect();
    expected.sort();
    assert_eq!(
        grouped,
        expected
            .iter()
            .map(|&(bucket, _)| bucket)
            .collect::<Vec<_>>(),
        "groups must be ascending"
    );
    assert_eq!(
        ordered,
        expected.iter().map(|&(_, at)| at).collect::<Vec<_>>(),
        "entries within a group must keep their original order"
    );
}

#[test]
fn adapter_uses_a_native_backend() {
    assert!(
        matches!(
            shared().adapter_info().backend,
            Backend::Dx12 | Backend::Metal | Backend::Vulkan
        ),
        "dynamis must never fall back to a legacy backend"
    );
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
