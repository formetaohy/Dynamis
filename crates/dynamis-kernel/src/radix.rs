use dynamis_gpu::{
    BindingKind, BindingSpec, ComputePipeline, ComputeRecorder, GpuBuffer, GpuContext,
};
use wgpu::{BindGroup, BindGroupEntry, Device};

const THREADS: u32 = 256;
const RADIX_PASSES: usize = 8;
const BIN_COUNT: usize = 256;

fn shifted_shader(source: &str, shift: u32) -> String {
    source.replace("__SHIFT__", &shift.to_string())
}

pub struct SortChannels<'a> {
    pub count: &'a GpuBuffer,
    pub keys_lo: &'a GpuBuffer,
    pub keys_hi: &'a GpuBuffer,
    pub values: &'a GpuBuffer,
    pub scratch_lo: &'a GpuBuffer,
    pub scratch_hi: &'a GpuBuffer,
    pub scratch_values: &'a GpuBuffer,
}

#[derive(PartialEq, Eq)]
struct ChannelsKey {
    count: u64,
    keys_lo: u64,
    keys_hi: u64,
    values: u64,
    scratch_lo: u64,
    scratch_hi: u64,
    scratch_values: u64,
}

impl ChannelsKey {
    fn of(channels: &SortChannels<'_>) -> Self {
        Self {
            count: channels.count.token(),
            keys_lo: channels.keys_lo.token(),
            keys_hi: channels.keys_hi.token(),
            values: channels.values.token(),
            scratch_lo: channels.scratch_lo.token(),
            scratch_hi: channels.scratch_hi.token(),
            scratch_values: channels.scratch_values.token(),
        }
    }
}

struct SortBindGroups {
    channels: ChannelsKey,
    histogram: [BindGroup; 2],
    scatter: [BindGroup; 2],
}

impl SortBindGroups {
    fn build(device: &Device, sort: &RadixSort, channels: &SortChannels<'_>) -> Self {
        let in_lo = [channels.keys_lo, channels.scratch_lo];
        let in_hi = [channels.keys_hi, channels.scratch_hi];
        let in_values = [channels.values, channels.scratch_values];
        let out_lo = [channels.scratch_lo, channels.keys_lo];
        let out_hi = [channels.scratch_hi, channels.keys_hi];
        let out_values = [channels.scratch_values, channels.values];
        let histogram = std::array::from_fn(|parity| {
            sort.histogram_pipelines[0].create_bind_group(
                device,
                0,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: in_lo[parity].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: in_hi[parity].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: sort.histogram.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: sort.block_histogram.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: channels.count.as_binding(),
                    },
                ],
            )
        });
        let scatter = std::array::from_fn(|parity| {
            sort.scatter_pipelines[0].create_bind_group(
                device,
                0,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: in_lo[parity].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: in_hi[parity].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: in_values[parity].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: sort.cursor.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: sort.block_prefix.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: out_lo[parity].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 6,
                        resource: out_hi[parity].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 7,
                        resource: out_values[parity].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 8,
                        resource: channels.count.as_binding(),
                    },
                ],
            )
        });
        Self {
            channels: ChannelsKey::of(channels),
            histogram,
            scatter,
        }
    }
}

pub struct RadixSort {
    histogram_pipelines: [ComputePipeline; RADIX_PASSES],
    scatter_pipelines: [ComputePipeline; RADIX_PASSES],
    prefix_pipeline: ComputePipeline,
    prefix_group: BindGroup,
    histogram: GpuBuffer,
    cursor: GpuBuffer,
    block_histogram: GpuBuffer,
    block_prefix: GpuBuffer,
    blocks: u32,
    bindings: std::sync::Mutex<Option<SortBindGroups>>,
}

impl RadixSort {
    pub fn new(context: &GpuContext, label: &str, capacity: u32) -> Self {
        let device = context.device();
        let blocks = capacity.div_ceil(THREADS).max(1);
        let histogram_spec = [
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
                kind: BindingKind::ReadWriteStorage,
            },
            BindingSpec {
                binding: 4,
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
                kind: BindingKind::ReadOnlyStorage,
            },
            BindingSpec {
                binding: 3,
                kind: BindingKind::ReadOnlyStorage,
            },
            BindingSpec {
                binding: 4,
                kind: BindingKind::ReadOnlyStorage,
            },
            BindingSpec {
                binding: 5,
                kind: BindingKind::ReadWriteStorage,
            },
            BindingSpec {
                binding: 6,
                kind: BindingKind::ReadWriteStorage,
            },
            BindingSpec {
                binding: 7,
                kind: BindingKind::ReadWriteStorage,
            },
            BindingSpec {
                binding: 8,
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
            BindingSpec {
                binding: 2,
                kind: BindingKind::ReadWriteStorage,
            },
            BindingSpec {
                binding: 3,
                kind: BindingKind::ReadWriteStorage,
            },
        ];
        let prefix_pipeline = context.compute_pipeline(
            &format!("{label} prefix"),
            include_str!("shaders/sort_prefix.wgsl"),
            "main",
            &[&prefix_spec[..]],
            THREADS,
        );
        let histogram_shader = include_str!("shaders/sort_histogram.wgsl");
        let histogram_pipelines = std::array::from_fn(|index| {
            context.compute_pipeline(
                &format!("{label} histogram {index}"),
                &shifted_shader(histogram_shader, (index * 8) as u32),
                "main",
                &[&histogram_spec[..]],
                THREADS,
            )
        });
        let scatter_shader = include_str!("shaders/sort_scatter.wgsl");
        let scatter_pipelines = std::array::from_fn(|index| {
            context.compute_pipeline(
                &format!("{label} scatter {index}"),
                &shifted_shader(scatter_shader, (index * 8) as u32),
                "main",
                &[&scatter_spec[..]],
                THREADS,
            )
        });
        let histogram = GpuBuffer::zeroed(
            device,
            &format!("{label} histogram"),
            (BIN_COUNT * 4) as u64,
            wgpu::BufferUsages::STORAGE,
        );
        let cursor = GpuBuffer::new(
            device,
            &format!("{label} cursor"),
            (BIN_COUNT * 4) as u64,
            wgpu::BufferUsages::STORAGE,
        );
        let block_bytes = blocks as u64 * BIN_COUNT as u64 * 4;
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
                BindGroupEntry {
                    binding: 2,
                    resource: block_histogram.as_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: block_prefix.as_binding(),
                },
            ],
        );
        Self {
            histogram_pipelines,
            scatter_pipelines,
            prefix_pipeline,
            prefix_group,
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
        channels: &SortChannels<'_>,
    ) -> std::sync::MutexGuard<'_, Option<SortBindGroups>> {
        let mut guard = self.bindings.lock().unwrap();
        let key = ChannelsKey::of(channels);
        if guard.as_ref().is_none_or(|cached| cached.channels != key) {
            *guard = Some(SortBindGroups::build(device, self, channels));
        }
        guard
    }

    fn encode_all_passes(
        &self,
        recorder: &mut ComputeRecorder,
        bindings: &SortBindGroups,
        lo_words: u32,
        hi_words: u32,
    ) {
        let pass_ranges = [0..lo_words, 4..(4 + hi_words)];
        let passes = pass_ranges.into_iter().flatten().collect::<Vec<_>>();
        for (executed, pass_index) in passes.iter().enumerate() {
            let parity = executed % 2;
            let pass_index = *pass_index as usize;
            recorder.record(
                &self.histogram_pipelines[pass_index],
                &[&bindings.histogram[parity]],
                self.blocks,
            );
            recorder.record(&self.prefix_pipeline, &[&self.prefix_group], 1);
            recorder.record(
                &self.scatter_pipelines[pass_index],
                &[&bindings.scatter[parity]],
                self.blocks,
            );
        }
        if passes.len() % 2 == 1 {
            let parity = passes.len() % 2;
            recorder.record(
                &self.histogram_pipelines[7],
                &[&bindings.histogram[parity]],
                self.blocks,
            );
            recorder.record(&self.prefix_pipeline, &[&self.prefix_group], 1);
            recorder.record(
                &self.scatter_pipelines[7],
                &[&bindings.scatter[parity]],
                self.blocks,
            );
        }
    }

    pub fn sort(
        &self,
        device: &Device,
        recorder: &mut ComputeRecorder,
        channels: &SortChannels<'_>,
        lo_words: u32,
        hi_words: u32,
    ) {
        let guard = self.bindings(device, channels);
        let bindings = guard.as_ref().expect("bindings ensured just above");
        self.encode_all_passes(recorder, bindings, lo_words, hi_words);
    }
}
