use dynamis_gpu::{
    BindingKind, BindingSpec, ComputeProgram, ComputeRecorder, GpuBuffer, GpuContext, GpuSlot,
    PipelineHandle,
};
use wgpu::{BindGroup, BindGroupEntry, Device};

const THREADS: u32 = 256;
const MAX_TILES: u32 = 4096;
const BINS: u32 = 256;
const PASSES: usize = 8;

fn shifted_shader(source: &str, shift: u32, row: u32) -> String {
    source
        .replace("__SHIFT__", &shift.to_string())
        .replace("__ROW__", &row.to_string())
}

pub fn key_words(elements: u32) -> u32 {
    let bits = 32 - elements.saturating_sub(1).leading_zeros();
    bits.div_ceil(8).clamp(1, 4)
}

pub struct SortChannels<'a> {
    pub count: GpuSlot<'a>,
    pub major: &'a GpuBuffer,
    pub minor: &'a GpuBuffer,
    pub payload: &'a GpuBuffer,
    pub scratch_major: &'a GpuBuffer,
    pub scratch_minor: &'a GpuBuffer,
    pub scratch_payload: &'a GpuBuffer,
}

#[derive(PartialEq, Eq)]
struct ChannelsKey {
    count: (u64, u64, u64),
    major: u64,
    minor: u64,
    payload: u64,
    scratch_major: u64,
    scratch_minor: u64,
    scratch_payload: u64,
}

impl ChannelsKey {
    fn of(channels: &SortChannels<'_>) -> Self {
        Self {
            count: channels.count.identity(),
            major: channels.major.token(),
            minor: channels.minor.token(),
            payload: channels.payload.token(),
            scratch_major: channels.scratch_major.token(),
            scratch_minor: channels.scratch_minor.token(),
            scratch_payload: channels.scratch_payload.token(),
        }
    }
}

struct SortBindGroups {
    channels: ChannelsKey,
    histogram: [BindGroup; 2],
    scatter: [BindGroup; 2],
    prefix: BindGroup,
}

impl SortBindGroups {
    fn build(device: &Device, sort: &RadixSort, channels: &SortChannels<'_>) -> Self {
        let in_minor = [channels.minor, channels.scratch_minor];
        let in_major = [channels.major, channels.scratch_major];
        let in_payload = [channels.payload, channels.scratch_payload];
        let out_minor = [channels.scratch_minor, channels.minor];
        let out_major = [channels.scratch_major, channels.major];
        let out_payload = [channels.scratch_payload, channels.payload];
        let histogram = std::array::from_fn(|parity| {
            sort.histogram_pipelines[0].create_bind_group(
                device,
                0,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: in_minor[parity].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: in_major[parity].as_binding(),
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
                        resource: in_minor[parity].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: in_major[parity].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: in_payload[parity].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: sort.block_prefix.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: sort.histogram.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: out_minor[parity].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 6,
                        resource: out_major[parity].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 7,
                        resource: out_payload[parity].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 8,
                        resource: channels.count.as_binding(),
                    },
                ],
            )
        });
        let prefix = sort.prefix_pipeline.create_bind_group(
            device,
            0,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: sort.histogram.as_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: sort.block_histogram.as_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: sort.block_prefix.as_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: channels.count.as_binding(),
                },
            ],
        );
        Self {
            channels: ChannelsKey::of(channels),
            histogram,
            scatter,
            prefix,
        }
    }
}

pub struct RadixSort {
    device: Device,
    histogram_pipelines: [PipelineHandle; PASSES],
    scatter_pipelines: [PipelineHandle; PASSES],
    prefix_pipeline: PipelineHandle,
    histogram: GpuBuffer,
    block_histogram: GpuBuffer,
    block_prefix: GpuBuffer,
    bindings: std::sync::Mutex<Vec<SortBindGroups>>,
}

fn binding(index: u32, kind: BindingKind) -> BindingSpec {
    BindingSpec {
        binding: index,
        kind,
    }
}

const READ: BindingKind = BindingKind::ReadOnlyStorage;
const WRITE: BindingKind = BindingKind::ReadWriteStorage;

impl RadixSort {
    pub fn new(context: &GpuContext, label: &str, reservation: u32) -> Self {
        let device = context.device().clone();
        let row = context.workgroups_per_row();
        let blocks = reservation.div_ceil(THREADS).max(1);
        let histogram_spec = [
            binding(0, READ),
            binding(1, READ),
            binding(2, WRITE),
            binding(3, WRITE),
            binding(4, READ),
        ];
        let scatter_spec = [
            binding(0, READ),
            binding(1, READ),
            binding(2, READ),
            binding(3, READ),
            binding(4, WRITE),
            binding(5, WRITE),
            binding(6, WRITE),
            binding(7, WRITE),
            binding(8, READ),
        ];
        let prefix_spec = [
            binding(0, WRITE),
            binding(1, WRITE),
            binding(2, WRITE),
            binding(3, READ),
        ];
        let prefix_pipeline = context.declare(ComputeProgram::new(
            &format!("{label} prefix"),
            include_str!("shaders/sort_prefix.wgsl"),
            "main",
            &[&prefix_spec[..]],
        ));
        let histogram_shader = include_str!("shaders/sort_histogram.wgsl");
        let histogram_pipelines = std::array::from_fn(|index| {
            context.declare(ComputeProgram::new(
                &format!("{label} histogram {index}"),
                shifted_shader(histogram_shader, (index * 8) as u32, row),
                "main",
                &[&histogram_spec[..]],
            ))
        });
        let scatter_shader = include_str!("shaders/sort_scatter.wgsl");
        let scatter_pipelines = std::array::from_fn(|index| {
            context.declare(ComputeProgram::new(
                &format!("{label} scatter {index}"),
                shifted_shader(scatter_shader, (index * 8) as u32, row),
                "main",
                &[&scatter_spec[..]],
            ))
        });
        let histogram = GpuBuffer::zeroed(
            &device,
            &format!("{label} histogram"),
            (BINS * 4) as u64,
            wgpu::BufferUsages::STORAGE,
        );
        let block_bytes = blocks as u64 * BINS as u64 * 4;
        let block_histogram = GpuBuffer::zeroed(
            &device,
            &format!("{label} block histogram"),
            block_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let block_prefix = GpuBuffer::new(
            &device,
            &format!("{label} block prefix"),
            block_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        Self {
            device,
            histogram_pipelines,
            scatter_pipelines,
            prefix_pipeline,
            histogram,
            block_histogram,
            block_prefix,
            bindings: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn encode_passes(
        &self,
        recorder: &mut ComputeRecorder,
        bindings: &SortBindGroups,
        major_words: u32,
        minor_words: u32,
        elements: u32,
    ) {
        let mut passes = (0..minor_words)
            .chain(4..4 + major_words)
            .collect::<Vec<_>>();
        assert!(
            !passes.is_empty(),
            "a sort needs at least one key digit to permute the payload"
        );
        if passes.len() % 2 == 1 {
            let last = *passes.last().expect("a sort runs at least one pass");
            passes.push(last);
        }
        let workgroups = elements.div_ceil(THREADS).min(MAX_TILES);
        for (order, pass) in passes.iter().enumerate() {
            let parity = order % 2;
            let pass = *pass as usize;
            recorder.record(
                self.histogram_pipelines[pass].pipeline(),
                &[&bindings.histogram[parity]],
                workgroups,
            );
            recorder.record(
                self.prefix_pipeline.pipeline(),
                &[&bindings.prefix],
                THREADS,
            );
            recorder.record(
                self.scatter_pipelines[pass].pipeline(),
                &[&bindings.scatter[parity]],
                workgroups,
            );
        }
    }

    fn with_bindings<R>(
        &self,
        device: &Device,
        channels: &SortChannels<'_>,
        use_groups: impl FnOnce(&SortBindGroups) -> R,
    ) -> R {
        let key = ChannelsKey::of(channels);
        let mut cache = self.bindings.lock().unwrap();
        let index = match cache.iter().position(|entry| entry.channels == key) {
            Some(index) => index,
            None => {
                cache.push(SortBindGroups::build(device, self, channels));
                cache.len() - 1
            }
        };
        use_groups(&cache[index])
    }

    pub fn sort(
        &self,
        recorder: &mut ComputeRecorder,
        channels: &SortChannels<'_>,
        major_words: u32,
        minor_words: u32,
        elements: u32,
    ) {
        self.with_bindings(&self.device, channels, |bindings| {
            self.encode_passes(recorder, bindings, major_words, minor_words, elements)
        })
    }
}
