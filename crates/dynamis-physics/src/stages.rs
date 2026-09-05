use crate::buffers::StageBuffers;
use crate::records::{
    COMMAND_ADD, COMMAND_FORCE, COMMAND_IMPULSE, COMMAND_PATCH, COMMAND_REMOVE, COMMAND_TORQUE,
    IMPULSE_AT_POINT, PATCH_ANGULAR_VELOCITY, PATCH_FRICTION, PATCH_INVERSE_MASS,
    PATCH_ORIENTATION, PATCH_POSITION, PATCH_RADIUS, PATCH_RESTITUTION, PATCH_VELOCITY, QUERY_RAY,
    QUERY_SPHERE,
};
use dynamis_gpu::{BindingKind, BindingSpec, ComputePipeline, GpuBuffer};
use wgpu::{BindGroup, BindGroupEntry, CommandEncoder, Device};

const WORKGROUP_SIZE: u32 = 64;
const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");

fn shader_constants() -> String {
    format!(
        "const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;\n\
         const COMMAND_ADD: u32 = {COMMAND_ADD}u;\n\
         const COMMAND_REMOVE: u32 = {COMMAND_REMOVE}u;\n\
         const COMMAND_PATCH: u32 = {COMMAND_PATCH}u;\n\
         const COMMAND_FORCE: u32 = {COMMAND_FORCE}u;\n\
         const COMMAND_TORQUE: u32 = {COMMAND_TORQUE}u;\n\
         const COMMAND_IMPULSE: u32 = {COMMAND_IMPULSE}u;\n\
         const IMPULSE_AT_POINT: u32 = {IMPULSE_AT_POINT}u;\n\
         const PATCH_POSITION: u32 = {PATCH_POSITION}u;\n\
         const PATCH_VELOCITY: u32 = {PATCH_VELOCITY}u;\n\
         const PATCH_INVERSE_MASS: u32 = {PATCH_INVERSE_MASS}u;\n\
         const PATCH_RADIUS: u32 = {PATCH_RADIUS}u;\n\
         const PATCH_RESTITUTION: u32 = {PATCH_RESTITUTION}u;\n\
         const PATCH_ORIENTATION: u32 = {PATCH_ORIENTATION}u;\n\
         const PATCH_ANGULAR_VELOCITY: u32 = {PATCH_ANGULAR_VELOCITY}u;\n\
         const PATCH_FRICTION: u32 = {PATCH_FRICTION}u;\n\
         const QUERY_RAY: u32 = {QUERY_RAY}u;\n\
         const QUERY_SPHERE: u32 = {QUERY_SPHERE}u;\n"
    )
}

fn assemble_shader(body: &str) -> String {
    format!("{COMMON_SHADER}\n{}\n{body}", shader_constants())
}

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
                resource: buffer.as_binding(),
            })
            .collect();
        let bind_group = pipeline.create_bind_group(device, &entries);
        Self {
            pipeline,
            bind_group,
        }
    }

    fn dispatch(&self, encoder: &mut CommandEncoder, elements: u32) {
        let workgroups = self.pipeline.workgroup_count(elements);
        self.pipeline
            .dispatch(encoder, &self.bind_group, workgroups);
    }

    fn dispatch_workgroups(&self, encoder: &mut CommandEncoder, workgroups: u32) {
        self.pipeline
            .dispatch(encoder, &self.bind_group, workgroups);
    }

    fn dispatch_indirect(&self, encoder: &mut CommandEncoder, args: &GpuBuffer) {
        self.pipeline
            .dispatch_indirect(encoder, &self.bind_group, args);
    }
}

pub(crate) struct Stages {
    apply_commands: Stage,
    integrate: Stage,
    broadphase_aabb: Stage,
    broadphase_pairs: Stage,
    narrowphase: Stage,
    solve: Stage,
    query: Stage,
}

pub(crate) fn build_stages(device: &Device, buffers: &StageBuffers) -> Stages {
    Stages {
        apply_commands: Stage::build(
            device,
            "apply_commands",
            &assemble_shader(include_str!("shaders/apply_commands.wgsl")),
            "main",
            &[
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadOnlyStorage,
            ],
            &[&buffers.commands, &buffers.bodies, &buffers.command_count],
        ),
        integrate: Stage::build(
            device,
            "integrate",
            &assemble_shader(include_str!("shaders/integrate.wgsl")),
            "main",
            &[BindingKind::Uniform, BindingKind::ReadWriteStorage],
            &[&buffers.params, &buffers.bodies],
        ),
        broadphase_aabb: Stage::build(
            device,
            "broadphase_aabb",
            &assemble_shader(include_str!("shaders/broadphase_aabb.wgsl")),
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
            &assemble_shader(include_str!("shaders/broadphase_pairs.wgsl")),
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
            &assemble_shader(include_str!("shaders/narrowphase.wgsl")),
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
            &assemble_shader(include_str!("shaders/solve.wgsl")),
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
        query: Stage::build(
            device,
            "query",
            &assemble_shader(include_str!("shaders/queries.wgsl")),
            "main",
            &[
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::Uniform,
            ],
            &[
                &buffers.queries,
                &buffers.bodies,
                &buffers.query_results,
                &buffers.params,
            ],
        ),
    }
}

pub(crate) fn encode_physics(
    stages: &Stages,
    buffers: &StageBuffers,
    encoder: &mut CommandEncoder,
    body_count: u32,
    solve_iterations: u32,
    query_count: u32,
) {
    stages.apply_commands.dispatch(encoder, 1);
    stages.integrate.dispatch(encoder, body_count);
    stages.broadphase_aabb.dispatch(encoder, body_count);
    let pair_total = if body_count >= 2 {
        body_count * (body_count - 1) / 2
    } else {
        0
    };
    stages.broadphase_pairs.dispatch(encoder, pair_total);
    stages
        .narrowphase
        .dispatch_indirect(encoder, &buffers.pair_count);
    for _ in 0..solve_iterations {
        stages
            .solve
            .dispatch_indirect(encoder, &buffers.contact_count);
    }
    if query_count > 0 {
        stages.query.dispatch_workgroups(encoder, query_count);
    }
}
