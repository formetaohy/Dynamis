use crate::buffer::GpuBuffer;
use crate::{BindingKind, BindingSpec, ComputePipeline};
use wgpu::{BindGroup, BindGroupEntry, CommandEncoder, Device};

const THREADS: u32 = 256;
const RADIX_PASSES: usize = 8;

fn shifted_shader(source: &str, shift: u32) -> String {
    source.replace("__SHIFT__", &shift.to_string())
}

pub struct GpuSort {
    histogram_pipelines: [ComputePipeline; RADIX_PASSES],
    assign_pipelines: [ComputePipeline; RADIX_PASSES],
    scatter_pipelines: [ComputePipeline; RADIX_PASSES],
    histogram: GpuBuffer,
    positions: GpuBuffer,
}

impl GpuSort {
    pub fn new(device: &Device, label: &str, capacity: u32) -> Self {
        let histogram_bindings = [
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
        ];
        let assign_bindings = [
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
        ];
        let scatter_bindings = [
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
                kind: BindingKind::ReadWriteStorage,
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
                kind: BindingKind::ReadOnlyStorage,
            },
        ];
        let histogram_shader = include_str!("shaders/sort_histogram.wgsl");
        let assign_shader = include_str!("shaders/sort_assign.wgsl");
        let scatter_shader = include_str!("shaders/sort_scatter.wgsl");
        let histogram_pipelines = std::array::from_fn(|index| {
            ComputePipeline::new(
                device,
                &format!("{label} histogram {index}"),
                &shifted_shader(histogram_shader, (index * 8) as u32),
                "main",
                &[&histogram_bindings[..]],
                THREADS,
            )
        });
        let assign_pipelines = std::array::from_fn(|index| {
            ComputePipeline::new(
                device,
                &format!("{label} assign {index}"),
                &shifted_shader(assign_shader, (index * 8) as u32),
                "main",
                &[&assign_bindings[..]],
                256,
            )
        });
        let scatter_pipelines = std::array::from_fn(|index| {
            ComputePipeline::new(
                device,
                &format!("{label} scatter {index}"),
                &shifted_shader(scatter_shader, (index * 8) as u32),
                "main",
                &[&scatter_bindings[..]],
                THREADS,
            )
        });
        Self {
            histogram_pipelines,
            assign_pipelines,
            scatter_pipelines,
            histogram: GpuBuffer::new(
                device,
                &format!("{label} histogram"),
                256 * 4,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            ),
            positions: GpuBuffer::new(
                device,
                &format!("{label} positions"),
                capacity as u64 * 4,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
        }
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
        pass: usize,
        keys_lo: &GpuBuffer,
        keys_hi: &GpuBuffer,
        values: &GpuBuffer,
        keys_lo_out: &GpuBuffer,
        keys_hi_out: &GpuBuffer,
        values_out: &GpuBuffer,
    ) {
        let workgroups = capacity.div_ceil(THREADS);
        encoder.clear_buffer(self.histogram.buffer(), 0, None);
        let histogram_group = self.histogram_pipelines[pass].create_bind_group(
            device,
            0,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: keys_lo.as_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: keys_hi.as_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.histogram.as_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: count_holder.as_binding(),
                },
            ],
        );
        encode_dispatch(
            encoder,
            &self.histogram_pipelines[pass],
            &histogram_group,
            workgroups,
        );

        let assign_group = self.assign_pipelines[pass].create_bind_group(
            device,
            0,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: keys_lo.as_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: keys_hi.as_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.histogram.as_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: count_holder.as_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: self.positions.as_binding(),
                },
            ],
        );
        encode_dispatch(encoder, &self.assign_pipelines[pass], &assign_group, 1);

        let scatter_group = self.scatter_pipelines[pass].create_bind_group(
            device,
            0,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: keys_lo.as_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: keys_hi.as_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: values.as_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.positions.as_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: keys_lo_out.as_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: keys_hi_out.as_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: values_out.as_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: count_holder.as_binding(),
                },
            ],
        );
        encode_dispatch(
            encoder,
            &self.scatter_pipelines[pass],
            &scatter_group,
            workgroups,
        );
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
        keys_lo: &GpuBuffer,
        keys_hi: &GpuBuffer,
        values: &GpuBuffer,
        keys_lo_out: &GpuBuffer,
        keys_hi_out: &GpuBuffer,
        values_out: &GpuBuffer,
    ) {
        let workgroups = capacity.div_ceil(THREADS);
        let mut out_is_current = false;
        for pass in 0..RADIX_PASSES {
            encoder.clear_buffer(self.histogram.buffer(), 0, None);
            let (in_lo, in_hi, in_values) = if out_is_current {
                (keys_lo_out, keys_hi_out, values_out)
            } else {
                (keys_lo, keys_hi, values)
            };

            let histogram_group = self.histogram_pipelines[pass].create_bind_group(
            device,
            0,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: in_lo.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: in_hi.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: self.histogram.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: count_holder.as_binding(),
                    },
                ],
            );
            encode_dispatch(
                encoder,
                &self.histogram_pipelines[pass],
                &histogram_group,
                workgroups,
            );

            let assign_group = self.assign_pipelines[pass].create_bind_group(
            device,
            0,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: in_lo.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: in_hi.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: self.histogram.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: count_holder.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: self.positions.as_binding(),
                    },
                ],
            );
            encode_dispatch(encoder, &self.assign_pipelines[pass], &assign_group, 1);

            let (out_lo, out_hi, out_values) = if out_is_current {
                (keys_lo, keys_hi, values)
            } else {
                (keys_lo_out, keys_hi_out, values_out)
            };
            let scatter_group = self.scatter_pipelines[pass].create_bind_group(
            device,
            0,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: in_lo.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: in_hi.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: in_values.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: self.positions.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: out_lo.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: out_hi.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 6,
                        resource: out_values.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 7,
                        resource: count_holder.as_binding(),
                    },
                ],
            );
            encode_dispatch(
                encoder,
                &self.scatter_pipelines[pass],
                &scatter_group,
                workgroups,
            );

            out_is_current = !out_is_current;
        }
    }

    pub fn debug_histogram(&self) -> &GpuBuffer {
        &self.histogram
    }

    pub fn debug_positions(&self) -> &GpuBuffer {
        &self.positions
    }
}

fn encode_dispatch(
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    group: &BindGroup,
    workgroups: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline.pipeline());
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}
