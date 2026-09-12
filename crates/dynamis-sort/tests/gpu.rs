use dynamis_gpu::{BufferReadback, ComputeRecorder, GpuBuffer, GpuContext, GpuSlot, WarmupBudget};
use dynamis_sort::{RadixSort, SortChannels, key_words};
use std::sync::OnceLock;
use wgpu::{Backend, BufferUsages};

const STREAM: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);

static CONTEXT: OnceLock<GpuContext> = OnceLock::new();

fn shared() -> &'static GpuContext {
    CONTEXT.get_or_init(|| pollster::block_on(GpuContext::new()))
}

fn read_u32s(context: &GpuContext, buffer: &GpuBuffer, words: usize) -> Vec<u32> {
    let bytes = (words * 4) as u64;
    let mut readback = BufferReadback::new(context.device(), "sort readback", bytes);
    let data = readback.read(context.queue(), buffer.buffer(), 0, bytes);
    let (chunks, remainder) = data.as_chunks::<4>();
    assert!(
        remainder.is_empty(),
        "readback is not a whole number of words"
    );
    chunks
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

struct Channels {
    major: GpuBuffer,
    minor: GpuBuffer,
    payload: GpuBuffer,
    scratch_major: GpuBuffer,
    scratch_minor: GpuBuffer,
    scratch_payload: GpuBuffer,
    count: GpuBuffer,
}

impl Channels {
    fn new(context: &GpuContext, major: &[u32], minor: &[u32], payload: &[u32]) -> Self {
        let device = context.device();
        let bytes = (major.len() * 4) as u64;
        let lane = |label: &str, data: Option<&[u32]>| {
            let buffer = GpuBuffer::new(device, label, bytes, STREAM);
            if let Some(data) = data {
                buffer.write(context.queue(), bytemuck::cast_slice(data));
            }
            buffer
        };
        let count = GpuBuffer::new(device, "sort count", 16, STREAM);
        count.write(context.queue(), bytemuck::cast_slice(&[major.len() as u32]));
        Self {
            major: lane("major", Some(major)),
            minor: lane("minor", Some(minor)),
            payload: lane("payload", Some(payload)),
            scratch_major: lane("scratch major", None),
            scratch_minor: lane("scratch minor", None),
            scratch_payload: lane("scratch payload", None),
            count,
        }
    }

    fn lanes(&self) -> SortChannels<'_> {
        SortChannels {
            count: GpuSlot::whole(&self.count),
            major: GpuSlot::whole(&self.major),
            minor: GpuSlot::whole(&self.minor),
            payload: GpuSlot::whole(&self.payload),
            scratch_major: GpuSlot::whole(&self.scratch_major),
            scratch_minor: GpuSlot::whole(&self.scratch_minor),
            scratch_payload: GpuSlot::whole(&self.scratch_payload),
        }
    }
}

fn run_sort(
    context: &GpuContext,
    major: &[u32],
    minor: &[u32],
    payload: &[u32],
    major_words: u32,
    minor_words: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let channels = Channels::new(context, major, minor, payload);
    let row = context.workgroups_per_row();
    let sort = RadixSort::new(context, "test sort", major.len() as u32);
    context.warmup(WarmupBudget::All);
    let mut encoder = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut recorder = ComputeRecorder::begin(&mut encoder, "sort", row);
        sort.sort(
            &mut recorder,
            &channels.lanes(),
            major_words,
            minor_words,
            major.len() as u32,
        );
    }
    context.queue().submit([encoder.finish()]);
    (
        read_u32s(context, &channels.major, major.len()),
        read_u32s(context, &channels.minor, minor.len()),
        read_u32s(context, &channels.payload, payload.len()),
    )
}

fn counting_payload(len: usize) -> Vec<u32> {
    (0..len as u32).collect()
}

#[test]
fn sort_orders_the_key_and_carries_the_payload_stably() {
    let context = shared();
    let keys = vec![3u32, 5, 1, 0, 7, 2, 2, 9, 4, 6];
    let payload = counting_payload(keys.len());
    let (sorted, _, carried) = run_sort(context, &keys, &keys, &payload, 1, 0);
    let mut expected: Vec<(u32, u32)> = keys
        .iter()
        .enumerate()
        .map(|(at, &key)| (key, at as u32))
        .collect();
    expected.sort();
    assert_eq!(
        sorted,
        expected.iter().map(|&(key, _)| key).collect::<Vec<_>>()
    );
    assert_eq!(
        carried,
        expected.iter().map(|&(_, at)| at).collect::<Vec<_>>(),
        "the payload must follow its key"
    );
}

#[test]
fn sort_orders_the_major_key_above_the_minor_key() {
    let context = shared();
    let n = 64u32;
    let major: Vec<u32> = (0..n).map(|i| i % 4).collect();
    let minor: Vec<u32> = (0..n).map(|i| (i * 5) % 8).collect();
    let payload = counting_payload(n as usize);
    let (sorted, _, carried) = run_sort(context, &major, &minor, &payload, 1, 1);
    let mut expected: Vec<(u32, u32, u32)> = (0..n)
        .map(|i| (major[i as usize], minor[i as usize], i))
        .collect();
    expected.sort();
    assert_eq!(
        sorted,
        expected
            .iter()
            .map(|&(major, _, _)| major)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        carried,
        expected.iter().map(|&(_, _, at)| at).collect::<Vec<_>>()
    );
}

#[test]
fn payload_digits_never_reach_the_key_order() {
    let context = shared();
    let keys = vec![1u32, 1, 0, 0, 2, 2];
    let payload = vec![u32::MAX, 0, u32::MAX, 0, u32::MAX, 0];
    let (sorted, _, carried) = run_sort(context, &keys, &keys, &payload, 1, 0);
    assert_eq!(sorted, vec![0, 0, 1, 1, 2, 2]);
    assert_eq!(
        carried,
        vec![u32::MAX, 0, u32::MAX, 0, u32::MAX, 0],
        "payload bytes must stay out of the key comparison"
    );
}

#[test]
fn equal_keys_keep_the_payload_in_input_order() {
    let context = shared();
    let keys = vec![2u32, 0, 1, 2, 0, 1, 3, 3, 0, 2];
    let payload = counting_payload(keys.len());
    let (sorted, _, carried) = run_sort(context, &keys, &keys, &payload, 1, 0);
    let mut expected: Vec<(u32, u32)> = keys
        .iter()
        .enumerate()
        .map(|(at, &key)| (key, at as u32))
        .collect();
    expected.sort();
    assert_eq!(
        sorted,
        expected.iter().map(|&(key, _)| key).collect::<Vec<_>>()
    );
    assert_eq!(
        carried,
        expected.iter().map(|&(_, at)| at).collect::<Vec<_>>()
    );
}

#[test]
fn key_words_cover_the_index_space() {
    assert_eq!(key_words(1), 1);
    assert_eq!(key_words(256), 1);
    assert_eq!(key_words(257), 2);
    assert_eq!(key_words(1 << 16), 2);
    assert_eq!(key_words(1 << 24), 3);
    assert_eq!(key_words(u32::MAX), 4);
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
    use wgpu::Backends;
    let has_dx12 = !pollster::block_on(GpuContext::available_adapters(Backends::DX12)).is_empty();
    let has_vulkan =
        !pollster::block_on(GpuContext::available_adapters(Backends::VULKAN)).is_empty();
    if has_dx12 && has_vulkan {
        assert_eq!(shared().adapter_info().backend, Backend::Dx12);
    }
}
