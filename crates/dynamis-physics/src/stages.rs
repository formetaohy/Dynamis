use crate::records::DispatchCount;
use dynamis_gpu::{BindingKind, BindingSpec, ComputePipeline, GpuBuffer};
use wgpu::{BindGroup, BindGroupEntry, CommandEncoder, Device};

const WORKGROUP_SIZE: u32 = 64;

struct Stage {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
}

impl Stage {
    fn build(
        device: &Device,
        label: &str,
        shader: &str,
        entry: &str,
        bindings: &[BindingKind],
        resources: &[&GpuBuffer],
    ) -> Self {
        assert_eq!(bindings.len(), resources.len(), "stage binding mismatch");
        let pipeline = ComputePipeline::new(
            device,
            label,
            shader,
            entry,
            &bindings
                .iter()
                .enumerate()
                .map(|(position, kind)| BindingSpec {
                    binding: position as u32,
                    kind: *kind,
                })
                .collect::<Vec<_>>(),
            WORKGROUP_SIZE,
        );
        let entries: Vec<BindGroupEntry> = bindings
            .iter()
            .zip(resources)
            .enumerate()
            .map(|(position, (_, buffer))| BindGroupEntry {
                binding: position as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bind_group = pipeline.create_bind_group(device, &entries);
        Self {
            pipeline,
            bind_group,
        }
    }

    fn record(&self, encoder: &mut CommandEncoder, elements: u32) {
        let workgroups = self.pipeline.workgroup_count(elements);
        self.pipeline
            .record_passes(encoder, &self.bind_group, workgroups);
    }

    fn record_indirect(&self, encoder: &mut CommandEncoder, target: &GpuBuffer) {
        self.pipeline
            .record_indirect(encoder, &self.bind_group, target);
    }
}

pub(crate) struct StageBuffers {
    pub(crate) params: GpuBuffer,
    pub(crate) bodies: GpuBuffer,
    pub(crate) aabbs: GpuBuffer,
    pub(crate) pairs: GpuBuffer,
    pub(crate) pair_count: GpuBuffer,
    pub(crate) contacts: GpuBuffer,
    pub(crate) contact_count: GpuBuffer,
}

impl StageBuffers {
    pub(crate) fn new(device: &Device, capacity: usize, pair_capacity: usize) -> Self {
        let body_bytes = (capacity * size_of_record::<crate::records::RigidBodyRecord>()) as u64;
        let aabb_bytes = (capacity * size_of::<crate::records::AabbRecord>()) as u64;
        let pair_bytes = (pair_capacity.max(1) * size_of::<crate::records::PairRecord>()) as u64;
        let contact_bytes =
            (pair_capacity.max(1) * size_of::<crate::records::ContactRecord>()) as u64;
        let counter_bytes = size_of::<DispatchCount>() as u64;
        let params_bytes = size_of::<crate::records::SimParamsRecord>() as u64;
        Self {
            params: GpuBuffer::new(
                device,
                "sim params",
                params_bytes,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            bodies: GpuBuffer::new(
                device,
                "bodies",
                body_bytes.max(1),
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            ),
            aabbs: GpuBuffer::new(
                device,
                "broadphase aabbs",
                aabb_bytes.max(1),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            pairs: GpuBuffer::new(
                device,
                "broadphase pairs",
                pair_bytes,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            ),
            pair_count: GpuBuffer::new(
                device,
                "pair count",
                counter_bytes,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            ),
            contacts: GpuBuffer::new(
                device,
                "narrowphase contacts",
                contact_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            contact_count: GpuBuffer::new(
                device,
                "contact count",
                counter_bytes,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            ),
        }
    }

    pub(crate) fn reset_counters(&self, queue: &wgpu::Queue) {
        let idle_value = DispatchCount::idle();
        let idle = bytemuck::cast_slice(std::slice::from_ref(&idle_value));
        self.pair_count.write(queue, idle);
        self.contact_count.write(queue, idle);
    }
}

pub(crate) struct Stages {
    integrate: Stage,
    broadphase_aabb: Stage,
    broadphase_pairs: Stage,
    narrowphase: Stage,
    solve: Stage,
}

pub(crate) fn build_stages(device: &Device, buffers: &StageBuffers) -> Stages {
    Stages {
        integrate: Stage::build(
            device,
            "integrate",
            include_str!("shaders/integrate.wgsl"),
            "main",
            &[BindingKind::Uniform, BindingKind::ReadWriteStorage],
            &[&buffers.params, &buffers.bodies],
        ),
        broadphase_aabb: Stage::build(
            device,
            "broadphase_aabb",
            include_str!("shaders/broadphase_aabb.wgsl"),
            "main",
            &[
                BindingKind::Uniform,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
            ],
            &[&buffers.params, &buffers.bodies, &buffers.aabbs],
        ),
        broadphase_pairs: Stage::build(
            device,
            "broadphase_pairs",
            include_str!("shaders/broadphase_pairs.wgsl"),
            "main",
            &[
                BindingKind::Uniform,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
            ],
            &[
                &buffers.params,
                &buffers.aabbs,
                &buffers.pairs,
                &buffers.pair_count,
            ],
        ),
        narrowphase: Stage::build(
            device,
            "narrowphase",
            include_str!("shaders/narrowphase.wgsl"),
            "main",
            &[
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadOnlyStorage,
            ],
            &[
                &buffers.bodies,
                &buffers.pairs,
                &buffers.contacts,
                &buffers.contact_count,
                &buffers.pair_count,
            ],
        ),
        solve: Stage::build(
            device,
            "solve",
            include_str!("shaders/solve.wgsl"),
            "main",
            &[
                BindingKind::Uniform,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
            ],
            &[
                &buffers.params,
                &buffers.bodies,
                &buffers.contacts,
                &buffers.contact_count,
            ],
        ),
    }
}

pub(crate) fn record_physics(
    stages: &Stages,
    buffers: &StageBuffers,
    encoder: &mut CommandEncoder,
    body_count: u32,
    solve_iterations: u32,
) {
    stages.integrate.record(encoder, body_count);
    stages.broadphase_aabb.record(encoder, body_count);
    let pair_total = if body_count >= 2 {
        body_count * (body_count - 1) / 2
    } else {
        0
    };
    stages.broadphase_pairs.record(encoder, pair_total);
    stages
        .narrowphase
        .record_indirect(encoder, &buffers.pair_count);
    for _ in 0..solve_iterations {
        stages
            .solve
            .record_indirect(encoder, &buffers.contact_count);
    }
}

fn size_of_record<T>() -> usize {
    std::mem::size_of::<T>()
}
