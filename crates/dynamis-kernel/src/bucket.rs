use dynamis_gpu::{
    BindingKind, BindingSpec, ComputePipeline, ComputeRecorder, GpuBuffer, GpuContext,
};
use wgpu::{BindGroup, BindGroupEntry, Device};

const THREADS: u32 = 256;
const BLOCK: u32 = 256;

fn bucketed_shader(source: &str, buckets: u32) -> String {
    source.replace("__BUCKETS__", &buckets.to_string())
}

pub struct BucketChannels<'a> {
    pub count: &'a GpuBuffer,
    pub keys: &'a GpuBuffer,
    pub values: &'a GpuBuffer,
    pub keys_out: &'a GpuBuffer,
    pub values_out: &'a GpuBuffer,
}

#[derive(PartialEq, Eq)]
struct ChannelsKey {
    keys: u64,
    values: u64,
    keys_out: u64,
    values_out: u64,
    count: u64,
}

impl ChannelsKey {
    fn of(channels: &BucketChannels<'_>) -> Self {
        Self {
            keys: channels.keys.token(),
            values: channels.values.token(),
            keys_out: channels.keys_out.token(),
            values_out: channels.values_out.token(),
            count: channels.count.token(),
        }
    }
}

struct BucketBindGroups {
    channels: ChannelsKey,
    histogram: BindGroup,
    scatter: BindGroup,
}

impl BucketBindGroups {
    fn build(device: &Device, bucket: &BucketSort, channels: &BucketChannels<'_>) -> Self {
        let histogram = bucket.histogram_pipeline.create_bind_group(
            device,
            0,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: channels.keys.as_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: bucket.histogram.as_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: bucket.block_histogram.as_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: channels.count.as_binding(),
                },
            ],
        );
        let scatter = bucket.scatter_pipeline.create_bind_group(
            device,
            0,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: channels.keys.as_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: channels.values.as_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: bucket.cursor.as_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: bucket.block_prefix.as_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: channels.keys_out.as_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: channels.values_out.as_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: channels.count.as_binding(),
                },
            ],
        );
        Self {
            channels: ChannelsKey::of(channels),
            histogram,
            scatter,
        }
    }
}

pub struct BucketSort {
    histogram_pipeline: ComputePipeline,
    prefix_pipeline: ComputePipeline,
    block_prefix_pipeline: ComputePipeline,
    scatter_pipeline: ComputePipeline,
    prefix_group: BindGroup,
    block_prefix_group: BindGroup,
    histogram: GpuBuffer,
    cursor: GpuBuffer,
    block_histogram: GpuBuffer,
    block_prefix: GpuBuffer,
    blocks: u32,
    bindings: std::sync::Mutex<Option<BucketBindGroups>>,
}

impl BucketSort {
    pub fn new(context: &GpuContext, label: &str, buckets: u32, data_capacity: u32) -> Self {
        let device = context.device();
        let histogram_spec = [
            BindingSpec {
                binding: 0,
                kind: BindingKind::ReadOnlyStorage,
            },
            BindingSpec {
                binding: 1,
                kind: BindingKind::ReadWriteStorage,
            },
            BindingSpec {
                binding: 2,
                kind: BindingKind::ReadWriteStorage,
            },
            BindingSpec {
                binding: 3,
                kind: BindingKind::ReadOnlyStorage,
            },
        ];
        let scatter_spec = [
            BindingSpec {
                binding: 0,
                kind: BindingKind::ReadOnlyStorage,
            },
            BindingSpec {
                binding: 1,
                kind: BindingKind::ReadOnlyStorage,
            },
            BindingSpec {
                binding: 2,
                kind: BindingKind::ReadWriteStorage,
            },
            BindingSpec {
                binding: 3,
                kind: BindingKind::ReadOnlyStorage,
            },
            BindingSpec {
                binding: 4,
                kind: BindingKind::ReadWriteStorage,
            },
            BindingSpec {
                binding: 5,
                kind: BindingKind::ReadWriteStorage,
            },
            BindingSpec {
                binding: 6,
                kind: BindingKind::ReadOnlyStorage,
            },
        ];
        let prefix_spec = [
            BindingSpec {
                binding: 0,
                kind: BindingKind::ReadWriteStorage,
            },
            BindingSpec {
                binding: 1,
                kind: BindingKind::ReadWriteStorage,
            },
        ];
        let block_prefix_spec = [
            BindingSpec {
                binding: 0,
                kind: BindingKind::ReadWriteStorage,
            },
            BindingSpec {
                binding: 1,
                kind: BindingKind::ReadWriteStorage,
            },
        ];
        let histogram_pipeline = context.compute_pipeline(
            &format!("{label} histogram"),
            &bucketed_shader(include_str!("shaders/bucket_histogram.wgsl"), buckets),
            "main",
            &[&histogram_spec[..]],
            THREADS,
        );
        let prefix_pipeline = context.compute_pipeline(
            &format!("{label} prefix"),
            &bucketed_shader(include_str!("shaders/bucket_prefix.wgsl"), buckets),
            "main",
            &[&prefix_spec[..]],
            THREADS,
        );
        let block_prefix_pipeline = context.compute_pipeline(
            &format!("{label} block prefix"),
            &bucketed_shader(include_str!("shaders/bucket_block_prefix.wgsl"), buckets),
            "main",
            &[&block_prefix_spec[..]],
            THREADS,
        );
        let scatter_pipeline = context.compute_pipeline(
            &format!("{label} scatter"),
            &bucketed_shader(include_str!("shaders/bucket_scatter.wgsl"), buckets),
            "main",
            &[&scatter_spec[..]],
            THREADS,
        );
        let bucket_bytes = (buckets as u64) * 4;
        let blocks = data_capacity.div_ceil(BLOCK);
        let block_bytes = blocks as u64 * buckets as u64 * 4;
        let histogram = GpuBuffer::zeroed(
            device,
            &format!("{label} histogram"),
            bucket_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let cursor = GpuBuffer::new(
            device,
            &format!("{label} cursor"),
            bucket_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let block_histogram = GpuBuffer::zeroed(
            device,
            &format!("{label} block histogram"),
            block_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let block_prefix = GpuBuffer::new(
            device,
            &format!("{label} block prefix"),
            block_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let prefix_group = prefix_pipeline.create_bind_group(
            device,
            0,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: histogram.as_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: cursor.as_binding(),
                },
            ],
        );
        let block_prefix_group = block_prefix_pipeline.create_bind_group(
            device,
            0,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: block_histogram.as_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: block_prefix.as_binding(),
                },
            ],
        );
        Self {
            histogram_pipeline,
            prefix_pipeline,
            block_prefix_pipeline,
            scatter_pipeline,
            prefix_group,
            block_prefix_group,
            histogram,
            cursor,
            block_histogram,
            block_prefix,
            blocks,
            bindings: std::sync::Mutex::new(None),
        }
    }

    fn bindings(
        &self,
        device: &Device,
        channels: &BucketChannels<'_>,
    ) -> std::sync::MutexGuard<'_, Option<BucketBindGroups>> {
        let mut guard = self.bindings.lock().unwrap();
        let key = ChannelsKey::of(channels);
        if guard.as_ref().is_none_or(|cached| cached.channels != key) {
            *guard = Some(BucketBindGroups::build(device, self, channels));
        }
        guard
    }

    pub fn sort(
        &self,
        device: &Device,
        recorder: &mut ComputeRecorder,
        channels: &BucketChannels<'_>,
    ) {
        let guard = self.bindings(device, channels);
        let bindings = guard.as_ref().expect("bindings ensured just above");
        recorder.record(
            &self.histogram_pipeline,
            &[&bindings.histogram],
            self.blocks,
        );
        recorder.record(&self.prefix_pipeline, &[&self.prefix_group], 1);
        recorder.record(&self.block_prefix_pipeline, &[&self.block_prefix_group], 1);
        recorder.record(&self.scatter_pipeline, &[&bindings.scatter], self.blocks);
    }
}
