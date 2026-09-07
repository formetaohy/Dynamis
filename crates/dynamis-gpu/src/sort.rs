use crate::buffer::GpuBuffer;
use crate::{BindingKind, BindingSpec, ComputePipeline};
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::{BindGroup, BindGroupEntry, CommandEncoder, Device};

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

struct SortBindGroups {
    channels: SortChannels,
    histogram: [BindGroup; 2],
    scatter: [BindGroup; 2],
}

impl SortBindGroups {
    fn build(
        device: &Device,
        sort: &GpuSort,
        channels: SortChannels,
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
            histogram,
            scatter,
        }
    }
}

pub struct GpuSort {
    histogram_pipelines: [ComputePipeline; RADIX_PASSES],
    scatter_pipelines: [ComputePipeline; RADIX_PASSES],
    prefix_pipeline: ComputePipeline,
    block_prefix_pipeline: ComputePipeline,
    prefix_group: BindGroup,
    block_prefix_group: BindGroup,
    histogram: GpuBuffer,
    cursor: GpuBuffer,
    block_histogram: GpuBuffer,
    block_prefix: GpuBuffer,
    cleared: AtomicBool,
    bindings: std::sync::Mutex<Option<SortBindGroups>>,
}

impl GpuSort {
    pub fn new(device: &Device, label: &str, capacity: u32) -> Self {
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
        let histogram_shader = include_str!("shaders/sort_histogram.wgsl");
        let histogram_pipelines = std::array::from_fn(|index| {
            ComputePipeline::new(
                device,
                &format!("{label} histogram {index}"),
                &shifted_shader(histogram_shader, (index * 8) as u32),
                "main",
                &[&histogram_spec[..]],
                THREADS,
            )
        });
        let scatter_shader = include_str!("shaders/sort_scatter.wgsl");
        let scatter_pipelines = std::array::from_fn(|index| {
            ComputePipeline::new(
                device,
                &format!("{label} scatter {index}"),
                &shifted_shader(scatter_shader, (index * 8) as u32),
                "main",
                &[&scatter_spec[..]],
                THREADS,
            )
        });
        let prefix_pipeline = ComputePipeline::new(
            device,
            &format!("{label} prefix"),
            include_str!("shaders/sort_prefix.wgsl"),
            "main",
            &[&prefix_spec[..]],
            THREADS,
        );
        let block_prefix_pipeline = ComputePipeline::new(
            device,
            &format!("{label} block prefix"),
            include_str!("shaders/sort_block_prefix.wgsl"),
            "main",
            &[&block_prefix_spec[..]],
            THREADS,
        );
        let histogram = GpuBuffer::new(
            device,
            &format!("{label} histogram"),
            (BIN_COUNT * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let cursor = GpuBuffer::new(
            device,
            &format!("{label} cursor"),
            (BIN_COUNT * 4) as u64,
            wgpu::BufferUsages::STORAGE,
        );
        let block_bytes = blocks as u64 * BIN_COUNT as u64 * 4;
        let block_histogram = GpuBuffer::new(
            device,
            &format!("{label} block histogram"),
            block_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
            histogram_pipelines,
            scatter_pipelines,
            prefix_pipeline,
            block_prefix_pipeline,
            prefix_group,
            block_prefix_group,
            histogram,
            cursor,
            block_histogram,
            block_prefix,
            cleared: AtomicBool::new(false),
            bindings: std::sync::Mutex::new(None),
        }
    }

    fn bindings(
        &self,
        device: &Device,
        channels: SortChannels,
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
            .is_none_or(|cached| cached.channels != channels)
        {
            *guard = Some(SortBindGroups::build(
                device,
                self,
                channels,
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
        encoder: &mut CommandEncoder,
        bindings: &SortBindGroups,
        workgroups: u32,
        lo_words: u32,
        hi_words: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        let pass_ranges = [0..lo_words, 4..(4 + hi_words)];
        let mut executed = 0usize;
        for pass_index in pass_ranges.into_iter().flatten() {
            let parity = executed % 2;
            executed += 1;
            let pass_index = pass_index as usize;
            pass.set_pipeline(self.histogram_pipelines[pass_index].pipeline());
            pass.set_bind_group(0, &bindings.histogram[parity], &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
            pass.set_pipeline(self.prefix_pipeline.pipeline());
            pass.set_bind_group(0, &self.prefix_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(self.block_prefix_pipeline.pipeline());
            pass.set_bind_group(0, &self.block_prefix_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(self.scatter_pipelines[pass_index].pipeline());
            pass.set_bind_group(0, &bindings.scatter[parity], &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "radix sort carries three key channels and their outputs explicitly"
    )]
    pub fn sort_64(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        count_holder: &GpuBuffer,
        capacity: u32,
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
            count_holder, keys_lo, keys_hi, values, keys_lo_out, keys_hi_out, values_out,
        );
        let guard = self.bindings(
            device,
            channels,
            keys_lo,
            keys_hi,
            values,
            keys_lo_out,
            keys_hi_out,
            values_out,
            count_holder,
        );
        let bindings = guard.as_ref().expect("bindings ensured just above");
        if !self.cleared.swap(true, Ordering::Relaxed) {
            encoder.clear_buffer(self.histogram.buffer(), 0, None);
            encoder.clear_buffer(self.block_histogram.buffer(), 0, None);
        }
        let workgroups = capacity.div_ceil(THREADS);
        self.encode_all_passes(encoder, bindings, workgroups, lo_words, hi_words);
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "radix sort carries three key channels and their outputs explicitly"
    )]
    pub fn sort_pass(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        count_holder: &GpuBuffer,
        capacity: u32,
        pass_index: usize,
        keys_lo: &GpuBuffer,
        keys_hi: &GpuBuffer,
        values: &GpuBuffer,
        keys_lo_out: &GpuBuffer,
        keys_hi_out: &GpuBuffer,
        values_out: &GpuBuffer,
    ) {
        let channels = SortChannels::of(
            count_holder, keys_lo, keys_hi, values, keys_lo_out, keys_hi_out, values_out,
        );
        let guard = self.bindings(
            device,
            channels,
            keys_lo,
            keys_hi,
            values,
            keys_lo_out,
            keys_hi_out,
            values_out,
            count_holder,
        );
        let bindings = guard.as_ref().expect("bindings ensured just above");
        if !self.cleared.swap(true, Ordering::Relaxed) {
            encoder.clear_buffer(self.histogram.buffer(), 0, None);
            encoder.clear_buffer(self.block_histogram.buffer(), 0, None);
        }
        let workgroups = capacity.div_ceil(THREADS);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(self.histogram_pipelines[pass_index as usize].pipeline());
        pass.set_bind_group(0, &bindings.histogram[pass_index % 2], &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        pass.set_pipeline(self.prefix_pipeline.pipeline());
        pass.set_bind_group(0, &self.prefix_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(self.block_prefix_pipeline.pipeline());
        pass.set_bind_group(0, &self.block_prefix_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(self.scatter_pipelines[pass_index].pipeline());
        pass.set_bind_group(0, &bindings.scatter[pass_index % 2], &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    pub fn debug_histogram(&self) -> &GpuBuffer {
        &self.histogram
    }
}
