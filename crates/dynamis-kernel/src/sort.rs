use dynamis_gpu::GpuBuffer;
use dynamis_gpu::{BindingKind, BindingSpec, ComputePipeline, ComputeRecorder, GpuContext};
use wgpu::{BindGroup, BindGroupEntry, Device};

const THREADS: u32 = 256;
const RADIX_PASSES: usize = 8;
const BIN_COUNT: usize = 256;

fn shifted_shader(source: &str, shift: u32) -> String {
    source.replace("__SHIFT__", &shift.to_string())
}

#[derive(PartialEq, Eq)]
struct SortChannels {
    count: u64,
    keys_lo: u64,
    keys_hi: u64,
    values: u64,
    keys_lo_out: u64,
    keys_hi_out: u64,
    values_out: u64,
}

impl SortChannels {
    fn of(
        count: &GpuBuffer,
        keys_lo: &GpuBuffer,
        keys_hi: &GpuBuffer,
        values: &GpuBuffer,
        keys_lo_out: &GpuBuffer,
        keys_hi_out: &GpuBuffer,
        values_out: &GpuBuffer,
    ) -> Self {
        Self {
            count: count.token(),
            keys_lo: keys_lo.token(),
            keys_hi: keys_hi.token(),
            values: values.token(),
            keys_lo_out: keys_lo_out.token(),
            keys_hi_out: keys_hi_out.token(),
            values_out: values_out.token(),
        }
    }
}

#[derive(PartialEq, Eq)]
struct SortArgs {
    args: u64,
}

struct SortBindGroups {
    channels: SortChannels,
    args: SortArgs,
    histogram: [BindGroup; 2],
    scatter: [BindGroup; 2],
}

impl SortBindGroups {
    #[expect(
        clippy::too_many_arguments,
        reason = "radix sort carries three key channels and their outputs explicitly"
    )]
    fn build(
        device: &Device,
        sort: &GpuSort,
        channels: SortChannels,
        args: SortArgs,
        keys_lo: &GpuBuffer,
        keys_hi: &GpuBuffer,
        values: &GpuBuffer,
        keys_lo_out: &GpuBuffer,
        keys_hi_out: &GpuBuffer,
        values_out: &GpuBuffer,
        count_holder: &GpuBuffer,
    ) -> Self {
        let in_lo = [keys_lo, keys_lo_out];
        let in_hi = [keys_hi, keys_hi_out];
        let in_values = [values, values_out];
        let out_lo = [keys_lo_out, keys_lo];
        let out_hi = [keys_hi_out, keys_hi];
        let out_values = [values_out, values];
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
                        resource: count_holder.as_binding(),
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
                        resource: count_holder.as_binding(),
                    },
                ],
            )
        });
        Self {
            channels,
            args,
            histogram,
            scatter,
        }
    }
}

pub struct GpuSort {
    histogram_pipelines: [ComputePipeline; RADIX_PASSES],
    scatter_pipelines: [ComputePipeline; RADIX_PASSES],
    prefix_pipeline: ComputePipeline,
    prefix_group: BindGroup,
    histogram: GpuBuffer,
    cursor: GpuBuffer,
    block_histogram: GpuBuffer,
    block_prefix: GpuBuffer,
    bindings: std::sync::Mutex<Option<SortBindGroups>>,
}

impl GpuSort {
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
            bindings: std::sync::Mutex::new(None),
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "radix sort carries three key channels and their outputs explicitly"
    )]
    fn bindings(
        &self,
        device: &Device,
        channels: SortChannels,
        args: SortArgs,
        keys_lo: &GpuBuffer,
        keys_hi: &GpuBuffer,
        values: &GpuBuffer,
        keys_lo_out: &GpuBuffer,
        keys_hi_out: &GpuBuffer,
        values_out: &GpuBuffer,
        count_holder: &GpuBuffer,
    ) -> std::sync::MutexGuard<'_, Option<SortBindGroups>> {
        let mut guard = self.bindings.lock().unwrap();
        if guard
            .as_ref()
            .is_none_or(|cached| cached.channels != channels || cached.args != args)
        {
            *guard = Some(SortBindGroups::build(
                device,
                self,
                channels,
                args,
                keys_lo,
                keys_hi,
                values,
                keys_lo_out,
                keys_hi_out,
                values_out,
                count_holder,
            ));
        }
        guard
    }

    fn encode_all_passes(
        &self,
        recorder: &mut ComputeRecorder,
        bindings: &SortBindGroups,
        args: &GpuBuffer,
        lo_words: u32,
        hi_words: u32,
    ) {
        let pass_ranges = [0..lo_words, 4..(4 + hi_words)];
        let passes = pass_ranges.into_iter().flatten().collect::<Vec<_>>();
        for (executed, pass_index) in passes.iter().enumerate() {
            let parity = executed % 2;
            let pass_index = *pass_index as usize;
            recorder.record_indirect(
                &self.histogram_pipelines[pass_index],
                &[&bindings.histogram[parity]],
                args,
                16,
            );
            recorder.record(&self.prefix_pipeline, &[&self.prefix_group], 1);
            recorder.record_indirect(
                &self.scatter_pipelines[pass_index],
                &[&bindings.scatter[parity]],
                args,
                16,
            );
        }
        if passes.len() % 2 == 1 {
            let parity = passes.len() % 2;
            recorder.record_indirect(
                &self.histogram_pipelines[7],
                &[&bindings.histogram[parity]],
                args,
                16,
            );
            recorder.record(&self.prefix_pipeline, &[&self.prefix_group], 1);
            recorder.record_indirect(
                &self.scatter_pipelines[7],
                &[&bindings.scatter[parity]],
                args,
                16,
            );
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "radix sort carries three key channels and their outputs explicitly"
    )]
    pub fn sort_64(
        &self,
        device: &Device,
        recorder: &mut ComputeRecorder,
        count_holder: &GpuBuffer,
        args: &GpuBuffer,
        lo_words: u32,
        hi_words: u32,
        keys_lo: &GpuBuffer,
        keys_hi: &GpuBuffer,
        values: &GpuBuffer,
        keys_lo_out: &GpuBuffer,
        keys_hi_out: &GpuBuffer,
        values_out: &GpuBuffer,
    ) {
        let channels = SortChannels::of(
            count_holder,
            keys_lo,
            keys_hi,
            values,
            keys_lo_out,
            keys_hi_out,
            values_out,
        );
        let guard = self.bindings(
            device,
            channels,
            SortArgs { args: args.token() },
            keys_lo,
            keys_hi,
            values,
            keys_lo_out,
            keys_hi_out,
            values_out,
            count_holder,
        );
        let bindings = guard.as_ref().expect("bindings ensured just above");
        self.encode_all_passes(recorder, bindings, args, lo_words, hi_words);
    }

    pub fn debug_histogram(&self) -> &GpuBuffer {
        &self.histogram
    }
}
