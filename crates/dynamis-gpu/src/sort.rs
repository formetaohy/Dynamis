use crate::buffer::GpuBuffer;
use crate::{BindingKind, BindingSpec, ComputePipeline};
use wgpu::{BindGroup, BindGroupEntry, CommandEncoder, Device, Queue};

const THREADS: u32 = 256;
const RADIX_PASSES: usize = 8;

fn shifted_shader(source: &str, shift: u32) -> String {
    source.replace("__SHIFT__", &shift.to_string())
}

pub struct GpuSort {
    histogram_pipelines: [ComputePipeline; RADIX_PASSES],
    scan_pipeline: ComputePipeline,
    scatter_pipelines: [ComputePipeline; RADIX_PASSES],
    histogram: GpuBuffer,
    buckets: GpuBuffer,
}

impl GpuSort {
    pub fn new(device: &Device, label: &str) -> Self {
        let histogram_bindings = [
            BindingSpec { binding: 0, kind: BindingKind::ReadOnlyStorage },
            BindingSpec { binding: 1, kind: BindingKind::ReadOnlyStorage },
            BindingSpec { binding: 2, kind: BindingKind::ReadWriteStorage },
            BindingSpec { binding: 3, kind: BindingKind::ReadOnlyStorage },
        ];
        let scan_bindings = [
            BindingSpec { binding: 0, kind: BindingKind::ReadWriteStorage },
            BindingSpec { binding: 1, kind: BindingKind::ReadWriteStorage },
        ];
        let scatter_bindings = [
            BindingSpec { binding: 0, kind: BindingKind::ReadOnlyStorage },
            BindingSpec { binding: 1, kind: BindingKind::ReadOnlyStorage },
            BindingSpec { binding: 2, kind: BindingKind::ReadOnlyStorage },
            BindingSpec { binding: 3, kind: BindingKind::ReadWriteStorage },
            BindingSpec { binding: 4, kind: BindingKind::ReadWriteStorage },
            BindingSpec { binding: 5, kind: BindingKind::ReadWriteStorage },
            BindingSpec { binding: 6, kind: BindingKind::ReadWriteStorage },
            BindingSpec { binding: 7, kind: BindingKind::ReadOnlyStorage },
        ];
        let histogram_shader = include_str!("shaders/sort_histogram.wgsl");
        let scatter_shader = include_str!("shaders/sort_scatter.wgsl");
        let histogram_pipelines = std::array::from_fn(|index| {
            ComputePipeline::new(
                device,
                &format!("{label} histogram {index}"),
                &shifted_shader(histogram_shader, (index * 8) as u32),
                "main",
                &histogram_bindings,
                THREADS,
            )
        });
        let scatter_pipelines = std::array::from_fn(|index| {
            ComputePipeline::new(
                device,
                &format!("{label} scatter {index}"),
                &shifted_shader(scatter_shader, (index * 8) as u32),
                "main",
                &scatter_bindings,
                THREADS,
            )
        });
        Self {
            histogram_pipelines,
            scan_pipeline: ComputePipeline::new(
                device,
                &format!("{label} scan"),
                include_str!("shaders/sort_scan.wgsl"),
                "main",
                &scan_bindings,
                256,
            ),
            scatter_pipelines,
            histogram: GpuBuffer::new(
                device,
                &format!("{label} histogram"),
                256 * 4,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            ),
            buckets: GpuBuffer::new(
                device,
                &format!("{label} buckets"),
                256 * 4,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
        }
    }

    #[expect(clippy::too_many_arguments, reason = "radix sort carries three key channels and their outputs explicitly")]
    pub fn sort_64(
        &self,
        device: &Device,
        queue: &Queue,
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
            self.histogram.write(queue, &vec![0u8; 256 * 4]);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sort pass"),
            });
            let (in_lo, in_hi, in_values) = if out_is_current {
                (keys_lo_out, keys_hi_out, values_out)
            } else {
                (keys_lo, keys_hi, values)
            };

            let histogram_group = self.histogram_pipelines[pass].create_bind_group(
                device,
                &[
                    BindGroupEntry { binding: 0, resource: in_lo.as_binding() },
                    BindGroupEntry { binding: 1, resource: in_hi.as_binding() },
                    BindGroupEntry { binding: 2, resource: self.histogram.as_binding() },
                    BindGroupEntry { binding: 3, resource: count_holder.as_binding() },
                ],
            );
            encode_dispatch(&mut encoder, &self.histogram_pipelines[pass], &histogram_group, workgroups);

            let scan_group = self.scan_pipeline.create_bind_group(
                device,
                &[
                    BindGroupEntry { binding: 0, resource: self.histogram.as_binding() },
                    BindGroupEntry { binding: 1, resource: self.buckets.as_binding() },
                ],
            );
            encode_dispatch(&mut encoder, &self.scan_pipeline, &scan_group, 1);

            let (out_lo, out_hi, out_values) = if out_is_current {
                (keys_lo, keys_hi, values)
            } else {
                (keys_lo_out, keys_hi_out, values_out)
            };
            let scatter_group = self.scatter_pipelines[pass].create_bind_group(
                device,
                &[
                    BindGroupEntry { binding: 0, resource: in_lo.as_binding() },
                    BindGroupEntry { binding: 1, resource: in_hi.as_binding() },
                    BindGroupEntry { binding: 2, resource: in_values.as_binding() },
                    BindGroupEntry { binding: 3, resource: self.buckets.as_binding() },
                    BindGroupEntry { binding: 4, resource: out_lo.as_binding() },
                    BindGroupEntry { binding: 5, resource: out_hi.as_binding() },
                    BindGroupEntry { binding: 6, resource: out_values.as_binding() },
                    BindGroupEntry { binding: 7, resource: count_holder.as_binding() },
                ],
            );
            encode_dispatch(&mut encoder, &self.scatter_pipelines[pass], &scatter_group, workgroups);

            out_is_current = !out_is_current;
            queue.submit([encoder.finish()]);
        }
    }

    pub fn debug_histogram(&self) -> &GpuBuffer {
        &self.histogram
    }

    pub fn debug_buckets(&self) -> &GpuBuffer {
        &self.buckets
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
