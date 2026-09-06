use crate::buffers::{MAX_CELLS_PER_BODY, StageBuffers};
use dynamis_gpu::{BindingKind, BindingSpec, ComputePipeline, GpuBuffer, GpuSort};
use dynamis_layout::{
    BODY_KINEMATIC, BODY_SLEEPING, COMMAND_ADD, COMMAND_CONSTRAINT_ADD,
    COMMAND_CONSTRAINT_REMOVE, COMMAND_FORCE, COMMAND_IMPULSE, COMMAND_PATCH, COMMAND_REMOVE,
    COMMAND_SLEEP, COMMAND_TORQUE, COMMAND_WAKE, CONSTRAINT_BALL, CONSTRAINT_DISTANCE,
    CONSTRAINT_FIXED, CONSTRAINT_INVALID, CONSTRAINT_PRISMATIC, CONSTRAINT_REVOLUTE,
    CONTACT_MAX_POINTS, IMPULSE_AT_POINT, ISLAND_ACTIVE, ISLAND_WAKE, NO_BODY,
    PATCH_ANGULAR_VELOCITY, PATCH_COLLIDER, PATCH_FRICTION, PATCH_GROUP, PATCH_INVERSE_MASS,
    PATCH_KINEMATIC, PATCH_MASK, PATCH_ORIENTATION, PATCH_POSITION, PATCH_RESTITUTION,
    PATCH_VELOCITY, QUERY_RAY, QUERY_SPHERE, SHAPE_BOX, SHAPE_CAPSULE, SHAPE_SPHERE,
};
use wgpu::{BindGroup, BindGroupEntry, CommandEncoder, Device};

const WORKGROUP_SIZE: u32 = 64;

fn shader_constants() -> String {
    format!(
        "const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;\n\
         const COMMAND_ADD: u32 = {COMMAND_ADD}u;\n\
         const COMMAND_REMOVE: u32 = {COMMAND_REMOVE}u;\n\
         const COMMAND_PATCH: u32 = {COMMAND_PATCH}u;\n\
         const COMMAND_FORCE: u32 = {COMMAND_FORCE}u;\n\
         const COMMAND_TORQUE: u32 = {COMMAND_TORQUE}u;\n\
         const COMMAND_IMPULSE: u32 = {COMMAND_IMPULSE}u;\n\
         const COMMAND_CONSTRAINT_ADD: u32 = {COMMAND_CONSTRAINT_ADD}u;\n\
         const COMMAND_CONSTRAINT_REMOVE: u32 = {COMMAND_CONSTRAINT_REMOVE}u;\n\
         const COMMAND_SLEEP: u32 = {COMMAND_SLEEP}u;\n\
         const COMMAND_WAKE: u32 = {COMMAND_WAKE}u;\n\
         const IMPULSE_AT_POINT: u32 = {IMPULSE_AT_POINT}u;\n\
         const PATCH_POSITION: u32 = {PATCH_POSITION}u;\n\
         const PATCH_VELOCITY: u32 = {PATCH_VELOCITY}u;\n\
         const PATCH_INVERSE_MASS: u32 = {PATCH_INVERSE_MASS}u;\n\
         const PATCH_COLLIDER: u32 = {PATCH_COLLIDER}u;\n\
         const PATCH_RESTITUTION: u32 = {PATCH_RESTITUTION}u;\n\
         const PATCH_ORIENTATION: u32 = {PATCH_ORIENTATION}u;\n\
         const PATCH_ANGULAR_VELOCITY: u32 = {PATCH_ANGULAR_VELOCITY}u;\n\
         const PATCH_FRICTION: u32 = {PATCH_FRICTION}u;\n\
         const PATCH_GROUP: u32 = {PATCH_GROUP}u;\n\
         const PATCH_MASK: u32 = {PATCH_MASK}u;\n\
         const PATCH_KINEMATIC: u32 = {PATCH_KINEMATIC}u;\n\
         const SHAPE_SPHERE: u32 = {SHAPE_SPHERE}u;\n\
         const SHAPE_BOX: u32 = {SHAPE_BOX}u;\n\
         const SHAPE_CAPSULE: u32 = {SHAPE_CAPSULE}u;\n\
         const BODY_KINEMATIC: u32 = {BODY_KINEMATIC}u;\n\
         const BODY_SLEEPING: u32 = {BODY_SLEEPING}u;\n\
         const ISLAND_WAKE: u32 = {ISLAND_WAKE}u;\n\
         const ISLAND_ACTIVE: u32 = {ISLAND_ACTIVE}u;\n\
         const CONTACT_MAX_POINTS: u32 = {CONTACT_MAX_POINTS}u;\n\
         const CONSTRAINT_BALL: u32 = {CONSTRAINT_BALL}u;\n\
         const CONSTRAINT_DISTANCE: u32 = {CONSTRAINT_DISTANCE}u;\n\
         const CONSTRAINT_REVOLUTE: u32 = {CONSTRAINT_REVOLUTE}u;\n\
         const CONSTRAINT_PRISMATIC: u32 = {CONSTRAINT_PRISMATIC}u;\n\
         const CONSTRAINT_FIXED: u32 = {CONSTRAINT_FIXED}u;\n\
         const CONSTRAINT_INVALID: u32 = {CONSTRAINT_INVALID}u;\n\
         const QUERY_RAY: u32 = {QUERY_RAY}u;\n\
         const QUERY_SPHERE: u32 = {QUERY_SPHERE}u;\n\
         const NO_BODY: u32 = {NO_BODY}u;\n\
         const NO_HIT: f32 = 3.402823466e38;\n\
         const MAX_CELLS_PER_BODY: u32 = {MAX_CELLS_PER_BODY}u;\n"
    )
}

pub(crate) const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");

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
        bindings: &[BindingKind],
        resources: &[&GpuBuffer],
    ) -> Self {
        assert_eq!(bindings.len(), resources.len(), "stage binding mismatch");
        let pipeline = ComputePipeline::new(
            device,
            label,
            shader,
            "main",
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

    fn dispatch_indirect(&self, encoder: &mut CommandEncoder, args: &GpuBuffer) {
        self.pipeline
            .dispatch_indirect(encoder, &self.bind_group, args);
    }

    fn dispatch_indirect_workgrouped(&self, encoder: &mut CommandEncoder, args: &GpuBuffer) {
        self.pipeline
            .dispatch_indirect_workgrouped(encoder, &self.bind_group, args);
    }

    fn dispatch_workgroups(&self, encoder: &mut CommandEncoder, workgroups: u32) {
        self.pipeline
            .dispatch(encoder, &self.bind_group, workgroups);
    }
}

pub(crate) struct Stages {
    apply_commands: Stage,
    apply_constraint_commands: Stage,
    integrate: Stage,
    broadphase_aabb: Stage,
    grid_entries: Stage,
    broadphase_pairs: Stage,
    large_pairs: Stage,
    narrowphase: Stage,
    contact_index: Stage,
    contact_match: Stage,
    contact_archive: Stage,
    island_init: Stage,
    island_link_contacts: Stage,
    island_link_constraints: Stage,
    island_jump: Stage,
    island_aggregate: Stage,
    island_broadcast: Stage,
    ccd_sweep: Stage,
    solve: Stage,
    solve_constraints: Stage,
    solve_position: Stage,
    query: Stage,
    sort: GpuSort,
    sort_hi: GpuBuffer,
    sort_lo: GpuBuffer,
    sort_values: GpuBuffer,
}

pub(crate) fn build_stages(device: &Device, buffers: &StageBuffers) -> Stages {
    Stages {
        apply_commands: Stage::build(
            device,
            "apply_commands",
            &assemble_shader(include_str!("shaders/apply_commands.wgsl")),
            &[
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
            ],
            &[
                &buffers.commands,
                &buffers.bodies_current,
                &buffers.colliders,
                &buffers.command_count,
                &buffers.wake_flags,
            ],
        ),
        apply_constraint_commands: Stage::build(
            device,
            "apply_constraint_commands",
            &assemble_shader(include_str!("shaders/apply_constraint_commands.wgsl")),
            &[
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadOnlyStorage,
            ],
            &[
                &buffers.constraint_commands,
                &buffers.constraints,
                &buffers.constraint_command_count,
            ],
        ),
        integrate: Stage::build(
            device,
            "integrate",
            &assemble_shader(include_str!("shaders/integrate.wgsl")),
            &[BindingKind::Uniform, BindingKind::ReadWriteStorage],
            &[&buffers.params, &buffers.bodies_current],
        ),
        broadphase_aabb: Stage::build(
            device,
            "broadphase_aabb",
            &assemble_shader(include_str!("shaders/broadphase_aabb.wgsl")),
            &[
                BindingKind::Uniform,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
            ],
            &[
                &buffers.params,
                &buffers.bodies_current,
                &buffers.colliders,
                &buffers.aabbs,
            ],
        ),
        grid_entries: Stage::build(
            device,
            "grid_entries",
            &assemble_shader(include_str!("shaders/grid_entries.wgsl")),
            &[
                BindingKind::Uniform,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
            ],
            &[
                &buffers.params,
                &buffers.aabbs,
                &buffers.entries.keys_hi,
                &buffers.entries.keys_lo,
                &buffers.entry_count,
                &buffers.large_bodies,
                &buffers.large_count,
            ],
        ),
        broadphase_pairs: Stage::build(
            device,
            "broadphase_pairs",
            &assemble_shader(include_str!("shaders/broadphase_pairs.wgsl")),
            &[
                BindingKind::Uniform,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
            ],
            &[
                &buffers.params,
                &buffers.entries.keys_hi,
                &buffers.entries.keys_lo,
                &buffers.entry_count,
                &buffers.pairs.keys_hi,
                &buffers.pairs.keys_lo,
                &buffers.pair_count,
            ],
        ),
        large_pairs: Stage::build(
            device,
            "large_pairs",
            &assemble_shader(include_str!("shaders/large_pairs.wgsl")),
            &[
                BindingKind::Uniform,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
            ],
            &[
                &buffers.params,
                &buffers.large_bodies,
                &buffers.large_count,
                &buffers.pairs.keys_hi,
                &buffers.pairs.keys_lo,
                &buffers.pair_count,
            ],
        ),
        narrowphase: Stage::build(
            device,
            "narrowphase",
            &assemble_shader(include_str!("shaders/narrowphase.wgsl")),
            &[
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadOnlyStorage,
            ],
            &[
                &buffers.bodies_current,
                &buffers.colliders,
                &buffers.pairs.keys_hi,
                &buffers.pairs.keys_lo,
                &buffers.contacts,
                &buffers.contact_count,
                &buffers.pair_count,
            ],
        ),
        contact_index: Stage::build(
            device,
            "contact_index",
            &assemble_shader(include_str!("shaders/contact_index.wgsl")),
            &[
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
            ],
            &[
                &buffers.contacts,
                &buffers.contact_count,
                &buffers.contact_keys_hi,
                &buffers.contact_keys_lo,
                &buffers.contact_indices,
            ],
        ),
        contact_match: Stage::build(
            device,
            "contact_match",
            &assemble_shader(include_str!("shaders/contact_match.wgsl")),
            &[
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadOnlyStorage,
            ],
            &[
                &buffers.contact_indices,
                &buffers.contact_keys_hi,
                &buffers.contact_keys_lo,
                &buffers.prev_contacts,
                &buffers.prev_contact_count,
                &buffers.contacts,
                &buffers.contact_count,
            ],
        ),
        contact_archive: Stage::build(
            device,
            "contact_archive",
            &assemble_shader(include_str!("shaders/contact_archive.wgsl")),
            &[
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadOnlyStorage,
            ],
            &[
                &buffers.contact_indices,
                &buffers.contacts,
                &buffers.prev_contacts,
                &buffers.prev_contact_count,
                &buffers.contact_count,
            ],
        ),
        island_init: Stage::build(
            device,
            "island_init",
            &assemble_shader(include_str!("shaders/island_init.wgsl")),
            &[
                BindingKind::Uniform,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
            ],
            &[
                &buffers.params,
                &buffers.island_parents,
                &buffers.island_state,
            ],
        ),
        island_link_contacts: Stage::build(
            device,
            "island_link_contacts",
            &assemble_shader(include_str!("shaders/island_link_contacts.wgsl")),
            &
            [
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
            ],
            &[
                &buffers.bodies_current,
                &buffers.contacts,
                &buffers.contact_count,
                &buffers.island_parents,
            ],
        ),
        island_link_constraints: Stage::build(
            device,
            "island_link_constraints",
            &assemble_shader(include_str!("shaders/island_link_constraints.wgsl")),
            &[
                BindingKind::Uniform,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
            ],
            &[
                &buffers.params,
                &buffers.bodies_current,
                &buffers.constraints,
                &buffers.island_parents,
            ],
        ),
        island_jump: Stage::build(
            device,
            "island_jump",
            &assemble_shader(include_str!("shaders/island_jump.wgsl")),
            &[BindingKind::Uniform, BindingKind::ReadWriteStorage],
            &[&buffers.params, &buffers.island_parents],
        ),
        island_aggregate: Stage::build(
            device,
            "island_aggregate",
            &assemble_shader(include_str!("shaders/island_aggregate.wgsl")),
            &[
                BindingKind::Uniform,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
            ],
            &[
                &buffers.params,
                &buffers.bodies_current,
                &buffers.island_parents,
                &buffers.island_state,
                &buffers.wake_flags,
            ],
        ),
        island_broadcast: Stage::build(
            device,
            "island_broadcast",
            &assemble_shader(include_str!("shaders/island_broadcast.wgsl")),
            &[
                BindingKind::Uniform,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
            ],
            &[
                &buffers.params,
                &buffers.bodies_current,
                &buffers.island_parents,
                &buffers.island_state,
                &buffers.wake_flags,
            ],
        ),
        ccd_sweep: Stage::build(
            device,
            "ccd_sweep",
            &assemble_shader(include_str!("shaders/ccd_sweep.wgsl")),
            &[
                BindingKind::Uniform,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
            ],
            &[
                &buffers.params,
                &buffers.bodies_current,
                &buffers.colliders,
                &buffers.pairs.keys_hi,
                &buffers.pairs.keys_lo,
                &buffers.pair_count,
            ],
        ),
        solve: Stage::build(
            device,
            "solve",
            &assemble_shader(include_str!("shaders/solve.wgsl")),
            &[
                BindingKind::Uniform,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
            ],
            &[
                &buffers.params,
                &buffers.bodies_current,
                &buffers.contacts,
                &buffers.contact_count,
                &buffers.wake_flags,
            ],
        ),
        solve_constraints: Stage::build(
            device,
            "solve_constraints",
            &assemble_shader(include_str!("shaders/constraints.wgsl")),
            &[
                BindingKind::Uniform,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
            ],
            &[
                &buffers.params,
                &buffers.bodies_current,
                &buffers.constraints,
                &buffers.wake_flags,
            ],
        ),
        solve_position: Stage::build(
            device,
            "solve_position",
            &assemble_shader(include_str!("shaders/solve_position.wgsl")),
            &[
                BindingKind::Uniform,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::ReadOnlyStorage,
            ],
            &[
                &buffers.params,
                &buffers.bodies_current,
                &buffers.contacts,
                &buffers.contact_count,
            ],
        ),
        query: Stage::build(
            device,
            "query",
            &assemble_shader(include_str!("shaders/queries.wgsl")),
            &[
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadOnlyStorage,
                BindingKind::ReadWriteStorage,
                BindingKind::Uniform,
            ],
            &[
                &buffers.queries,
                &buffers.bodies_current,
                &buffers.colliders,
                &buffers.query_results,
                &buffers.params,
            ],
        ),
        sort: GpuSort::new(device, "sim sort", buffers.sort_scratch.capacity_u32()),
        sort_hi: GpuBuffer::new(
            device,
            "sort scratch hi",
            buffers.sort_scratch.keys_hi.size(),
            wgpu::BufferUsages::STORAGE,
        ),
        sort_lo: GpuBuffer::new(
            device,
            "sort scratch lo",
            buffers.sort_scratch.keys_lo.size(),
            wgpu::BufferUsages::STORAGE,
        ),
        sort_values: GpuBuffer::new(
            device,
            "sort scratch values",
            buffers.sort_scratch.values.size(),
            wgpu::BufferUsages::STORAGE,
        ),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "physics encoding carries the full stage set, buffers and frame parameters"
)]
pub(crate) fn encode_physics(
    stages: &Stages,
    buffers: &StageBuffers,
    device: &Device,
    encoder: &mut CommandEncoder,
    body_count: u32,
    solve_iterations: u32,
    position_iterations: u32,
    island_rounds: u32,
    query_count: u32,
) {
    stages.apply_commands.dispatch(encoder, 1);
    stages.apply_constraint_commands.dispatch(encoder, 1);
    stages.integrate.dispatch(encoder, body_count);
    stages.broadphase_aabb.dispatch(encoder, body_count);
    stages.grid_entries.dispatch(encoder, body_count);
    stages.sort.sort_64(
        device,
        encoder,
        &buffers.entry_count,
        buffers.entries.capacity_u32(),
        &buffers.entries.keys_lo,
        &buffers.entries.keys_hi,
        &buffers.entries.values,
        &stages.sort_hi,
        &stages.sort_lo,
        &stages.sort_values,
    );
    stages
        .broadphase_pairs
        .dispatch_indirect_workgrouped(encoder, &buffers.entry_count);
    stages.large_pairs.dispatch(encoder, body_count);
    stages.sort.sort_64(
        device,
        encoder,
        &buffers.pair_count,
        buffers.pairs.capacity_u32(),
        &buffers.pairs.keys_lo,
        &buffers.pairs.keys_hi,
        &buffers.pairs.values,
        &stages.sort_hi,
        &stages.sort_lo,
        &stages.sort_values,
    );
    stages
        .ccd_sweep
        .dispatch_indirect(encoder, &buffers.pair_count);
    stages
        .narrowphase
        .dispatch_indirect(encoder, &buffers.pair_count);
    stages
        .contact_index
        .dispatch_indirect(encoder, &buffers.contact_count);
    stages.sort.sort_64(
        device,
        encoder,
        &buffers.contact_count,
        buffers.contact_capacity(),
        &buffers.contact_keys_lo,
        &buffers.contact_keys_hi,
        &buffers.contact_indices,
        &stages.sort_lo,
        &stages.sort_hi,
        &stages.sort_values,
    );
    stages
        .contact_match
        .dispatch_indirect(encoder, &buffers.contact_count);
    stages.island_init.dispatch(encoder, body_count);
    stages
        .island_link_contacts
        .dispatch_indirect(encoder, &buffers.contact_count);
    stages
        .island_link_constraints
        .dispatch(encoder, buffers.constraint_capacity());
    for _ in 0..island_rounds {
        stages.island_jump.dispatch(encoder, body_count);
    }
    stages.island_aggregate.dispatch(encoder, body_count);
    stages.island_broadcast.dispatch(encoder, body_count);
    for _ in 0..solve_iterations {
        stages
            .solve_constraints
            .dispatch(encoder, buffers.constraint_capacity());
        stages.solve.dispatch(encoder, 1);
    }
    for _ in 0..position_iterations {
        stages
            .solve_position
            .dispatch_indirect(encoder, &buffers.contact_count);
    }
    stages
        .contact_archive
        .dispatch_indirect(encoder, &buffers.contact_count);
    if query_count > 0 {
        stages.query.dispatch_workgroups(encoder, query_count);
    }
}
