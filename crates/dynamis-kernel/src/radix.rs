use dynamis_gpu::{
    BindingKind, BindingSpec, ComputePipeline, ComputeRecorder, DispatchTable, GpuBuffer,
    GpuContext, GpuSlot,
};
use dynamis_layout::Dispatcher;
use wgpu::{BindGroup, BindGroupEntry, Device};

const THREADS: u32 = 256;
const BINS: u32 = 256;
const PASSES: usize = 8;

fn shifted_shader(source: &str, shift: u32, row: u32) -> String {
    source
        .replace("__SHIFT__", &shift.to_string())
        .replace("__ROW__", &row.to_string())
}

/// The number of `u32` words needed to order `elements` distinct values.
pub fn key_words(elements: u32) -> u32 {
    let bits = 32 - elements.saturating_sub(1).leading_zeros();
    bits.div_ceil(8).clamp(1, 4)
}

/// The lanes a sort orders: a key split into two words, plus a payload.
pub struct SortChannels<'a> {
    /// How many leading lanes are live. A slot, so counters can share one buffer.
    pub count: GpuSlot<'a>,
    pub keys_lo: &'a GpuBuffer,
    pub keys_hi: &'a GpuBuffer,
    pub values: &'a GpuBuffer,
    pub scratch_lo: &'a GpuBuffer,
    pub scratch_hi: &'a GpuBuffer,
    pub scratch_values: &'a GpuBuffer,
}

#[derive(PartialEq, Eq)]
struct ChannelsKey {
    count: (u64, u64, u64),
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
            count: channels.count.identity(),
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

/// A stable 8-bit radix sort over `u32` key/value lanes.
///
/// The bin count is fixed at 256, so neither shared memory nor per-pass work depends
/// on how many distinct keys exist — only on how many lanes are being ordered.
pub struct RadixSort {
    device: Device,
    histogram_pipelines: [ComputePipeline; PASSES],
    scatter_pipelines: [ComputePipeline; PASSES],
    prefix_pipeline: ComputePipeline,
    prefix_group: BindGroup,
    block_prefix_pipeline: ComputePipeline,
    block_prefix_group: BindGroup,
    histogram: GpuBuffer,
    cursor: GpuBuffer,
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
    /// Build a sort able to order up to `reservation` lanes.
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
            binding(4, READ),
            binding(5, WRITE),
            binding(6, WRITE),
            binding(7, WRITE),
            binding(8, READ),
        ];
        let prefix_spec = [binding(0, WRITE), binding(1, WRITE)];
        let prefix_pipeline = context.compute_pipeline(
            &format!("{label} prefix"),
            include_str!("shaders/sort_prefix.wgsl"),
            "main",
            &[&prefix_spec[..]],
            THREADS,
        );
        let block_prefix_pipeline = context.compute_pipeline(
            &format!("{label} block prefix"),
            include_str!("shaders/sort_block_prefix.wgsl"),
            "main",
            &[&prefix_spec[..]],
            THREADS,
        );
        let histogram_shader = include_str!("shaders/sort_histogram.wgsl");
        let histogram_pipelines = std::array::from_fn(|index| {
            context.compute_pipeline(
                &format!("{label} histogram {index}"),
                &shifted_shader(histogram_shader, (index * 8) as u32, row),
                "main",
                &[&histogram_spec[..]],
                THREADS,
            )
        });
        let scatter_shader = include_str!("shaders/sort_scatter.wgsl");
        let scatter_pipelines = std::array::from_fn(|index| {
            context.compute_pipeline(
                &format!("{label} scatter {index}"),
                &shifted_shader(scatter_shader, (index * 8) as u32, row),
                "main",
                &[&scatter_spec[..]],
                THREADS,
            )
        });
        let histogram = GpuBuffer::zeroed(
            &device,
            &format!("{label} histogram"),
            (BINS * 4) as u64,
            wgpu::BufferUsages::STORAGE,
        );
        let cursor = GpuBuffer::new(
            &device,
            &format!("{label} cursor"),
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
        let prefix_group = prefix_pipeline.create_bind_group(
            &device,
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
            &device,
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
            device,
            histogram_pipelines,
            scatter_pipelines,
            prefix_pipeline,
            prefix_group,
            block_prefix_pipeline,
            block_prefix_group,
            histogram,
            cursor,
            block_histogram,
            block_prefix,
            bindings: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn encode_passes(
        &self,
        recorder: &mut ComputeRecorder,
        bindings: &SortBindGroups,
        lo_words: u32,
        hi_words: u32,
        table: &DispatchTable,
        dispatcher: Dispatcher,
    ) {
        let mut passes = (0..lo_words).chain(4..4 + hi_words).collect::<Vec<_>>();
        // An odd count would leave the result in the scratch lanes. Repeating the most
        // significant pass is a stable no-op on already-ordered data and restores it.
        if passes.len() % 2 == 1 {
            let last = *passes.last().expect("a sort runs at least one pass");
            passes.push(last);
        }
        for (order, pass) in passes.iter().enumerate() {
            let parity = order % 2;
            let pass = *pass as usize;
            recorder.record_indirect(
                &self.histogram_pipelines[pass],
                &[&bindings.histogram[parity]],
                table,
                dispatcher.slot(),
            );
            recorder.record(&self.prefix_pipeline, &[&self.prefix_group], 1);
            recorder.record(
                &self.block_prefix_pipeline,
                &[&self.block_prefix_group],
                BINS,
            );
            recorder.record_indirect(
                &self.scatter_pipelines[pass],
                &[&bindings.scatter[parity]],
                table,
                dispatcher.slot(),
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

    /// Order `channels` by `lo_words` of the low key word then `hi_words` of the high one.
    ///
    /// The sorted lanes always end up back in `channels`, and dispatches run by how many
    /// lanes `dispatcher`'s counter says are live, never by the reservation.
    pub fn sort(
        &self,
        recorder: &mut ComputeRecorder,
        channels: &SortChannels<'_>,
        lo_words: u32,
        hi_words: u32,
        table: &DispatchTable,
        dispatcher: Dispatcher,
    ) {
        self.with_bindings(&self.device, channels, |bindings| {
            self.encode_passes(recorder, bindings, lo_words, hi_words, table, dispatcher)
        });
    }
}
