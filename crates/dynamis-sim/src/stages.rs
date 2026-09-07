use crate::buffers::{COMPACT_BLOCK, MAX_CELLS_PER_COLLIDER, StageBuffers};
use dynamis_gpu::{
    BindingKind, BindingSpec, ComputePipeline, ComputeRecorder, GpuBucketSort, GpuBuffer,
    GpuContext, GpuCountArgs, GpuSort,
};
use dynamis_layout::{
    BODY_CCD, BODY_KINEMATIC, BODY_SLEEPING, COLLIDER_SENSOR, COMMAND_ADD, COMMAND_ANGULAR_IMPULSE,
    COMMAND_CONSTRAINT_ADD, COMMAND_CONSTRAINT_REMOVE, COMMAND_FORCE, COMMAND_FORCE_AT_POINT,
    COMMAND_IMPULSE, COMMAND_PATCH, COMMAND_REMOVE, COMMAND_SLEEP, COMMAND_TORQUE, COMMAND_WAKE,
    CONSTRAINT_BALL, CONSTRAINT_DISABLE_COLLISIONS, CONSTRAINT_DISTANCE, CONSTRAINT_FIXED,
    CONSTRAINT_HAS_LIMIT, CONSTRAINT_HAS_MOTOR, CONSTRAINT_INVALID, CONSTRAINT_IS_SPRING,
    CONSTRAINT_PRISMATIC, CONSTRAINT_REVOLUTE, CONTACT_MAX_POINTS, EVENT_BEGIN, EVENT_END,
    FILTER_IGNORE_KINEMATIC, FILTER_IGNORE_SENSORS, FILTER_IGNORE_SLEEPING, FILTER_IGNORE_STATIC,
    IMPULSE_AT_POINT, ISLAND_ACTIVE, ISLAND_WAKE, MAX_COLLIDERS_PER_BODY, MAX_HITS_PER_QUERY,
    NO_BODY, PATCH_ANGULAR_VELOCITY, PATCH_CCD, PATCH_COLLIDER, PATCH_FRICTION, PATCH_GROUP,
    PATCH_INVERSE_MASS, PATCH_KINEMATIC, PATCH_MASK, PATCH_ORIENTATION, PATCH_POSITION,
    PATCH_RESTITUTION, PATCH_VELOCITY, QUERY_BOX, QUERY_RAY, QUERY_SPHERE, QUERY_SWEEP, SHAPE_BOX,
    SHAPE_CAPSULE, SHAPE_CYLINDER, SHAPE_HEIGHTFIELD, SHAPE_HULL, SHAPE_MESH, SHAPE_NONE,
    SHAPE_SOURCE_HEIGHTFIELD, SHAPE_SOURCE_HULL, SHAPE_SOURCE_MESH, SHAPE_SPHERE,
};
use wgpu::{BindGroup, BindGroupEntry, CommandEncoder, Device};

const WORKGROUP_SIZE: u32 = 64;
const SHAPE_TRIANGLE: u32 = 8;

fn shader_constants() -> String {
    format!(
        "const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;\n\
         const COMMAND_ADD: u32 = {COMMAND_ADD}u;\n\
         const COMMAND_REMOVE: u32 = {COMMAND_REMOVE}u;\n\
         const COMMAND_PATCH: u32 = {COMMAND_PATCH}u;\n\
         const COMMAND_FORCE: u32 = {COMMAND_FORCE}u;\n\
         const COMMAND_FORCE_AT_POINT: u32 = {COMMAND_FORCE_AT_POINT}u;\n\
         const COMMAND_TORQUE: u32 = {COMMAND_TORQUE}u;\n\
         const COMMAND_IMPULSE: u32 = {COMMAND_IMPULSE}u;\n\
         const COMMAND_ANGULAR_IMPULSE: u32 = {COMMAND_ANGULAR_IMPULSE}u;\n\
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
         const PATCH_CCD: u32 = {PATCH_CCD}u;\n\
         const SHAPE_NONE: u32 = {SHAPE_NONE}u;\n\
         const SHAPE_SPHERE: u32 = {SHAPE_SPHERE}u;\n\
         const SHAPE_BOX: u32 = {SHAPE_BOX}u;\n\
         const SHAPE_CAPSULE: u32 = {SHAPE_CAPSULE}u;\n\
         const SHAPE_CYLINDER: u32 = {SHAPE_CYLINDER}u;\n\
         const SHAPE_HULL: u32 = {SHAPE_HULL}u;\n\
         const SHAPE_MESH: u32 = {SHAPE_MESH}u;\n\
         const SHAPE_HEIGHTFIELD: u32 = {SHAPE_HEIGHTFIELD}u;\n\
         const SHAPE_TRIANGLE: u32 = {SHAPE_TRIANGLE}u;\n\
         const SHAPE_SOURCE_HULL: u32 = {SHAPE_SOURCE_HULL}u;\n\
         const SHAPE_SOURCE_MESH: u32 = {SHAPE_SOURCE_MESH}u;\n\
         const SHAPE_SOURCE_HEIGHTFIELD: u32 = {SHAPE_SOURCE_HEIGHTFIELD}u;\n\
         const BODY_KINEMATIC: u32 = {BODY_KINEMATIC}u;\n\
         const BODY_SLEEPING: u32 = {BODY_SLEEPING}u;\n\
         const BODY_CCD: u32 = {BODY_CCD}u;\n\
         const COLLIDER_SENSOR: u32 = {COLLIDER_SENSOR}u;\n\
         const ISLAND_WAKE: u32 = {ISLAND_WAKE}u;\n\
         const ISLAND_ACTIVE: u32 = {ISLAND_ACTIVE}u;\n\
         const CONTACT_MAX_POINTS: u32 = {CONTACT_MAX_POINTS}u;\n\
         const CONSTRAINT_BALL: u32 = {CONSTRAINT_BALL}u;\n\
         const CONSTRAINT_DISTANCE: u32 = {CONSTRAINT_DISTANCE}u;\n\
         const CONSTRAINT_REVOLUTE: u32 = {CONSTRAINT_REVOLUTE}u;\n\
         const CONSTRAINT_PRISMATIC: u32 = {CONSTRAINT_PRISMATIC}u;\n\
         const CONSTRAINT_FIXED: u32 = {CONSTRAINT_FIXED}u;\n\
         const CONSTRAINT_INVALID: u32 = {CONSTRAINT_INVALID}u;\n\
         const CONSTRAINT_DISABLE_COLLISIONS: u32 = {CONSTRAINT_DISABLE_COLLISIONS}u;\n\
         const CONSTRAINT_HAS_LIMIT: u32 = {CONSTRAINT_HAS_LIMIT}u;\n\
         const CONSTRAINT_HAS_MOTOR: u32 = {CONSTRAINT_HAS_MOTOR}u;\n\
         const CONSTRAINT_IS_SPRING: u32 = {CONSTRAINT_IS_SPRING}u;\n\
         const QUERY_RAY: u32 = {QUERY_RAY}u;\n\
         const QUERY_SPHERE: u32 = {QUERY_SPHERE}u;\n\
         const QUERY_BOX: u32 = {QUERY_BOX}u;\n\
         const QUERY_SWEEP: u32 = {QUERY_SWEEP}u;\n\
         const FILTER_IGNORE_SENSORS: u32 = {FILTER_IGNORE_SENSORS}u;\n\
         const FILTER_IGNORE_SLEEPING: u32 = {FILTER_IGNORE_SLEEPING}u;\n\
         const FILTER_IGNORE_STATIC: u32 = {FILTER_IGNORE_STATIC}u;\n\
         const FILTER_IGNORE_KINEMATIC: u32 = {FILTER_IGNORE_KINEMATIC}u;\n\
         const EVENT_BEGIN: u32 = {EVENT_BEGIN}u;\n\
         const EVENT_END: u32 = {EVENT_END}u;\n\
         const NO_BODY: u32 = {NO_BODY}u;\n\
         const NO_HIT: f32 = 3.402823466e38;\n\
         const MAX_CELLS_PER_COLLIDER: u32 = {MAX_CELLS_PER_COLLIDER}u;\n\
         const MAX_COLLIDERS_PER_BODY: u32 = {MAX_COLLIDERS_PER_BODY}u;\n\
         const MAX_HITS_PER_QUERY: u32 = {MAX_HITS_PER_QUERY}u;\n"
    )
}

pub(crate) const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
pub(crate) const SHAPES_FRAGMENT: &str = include_str!("shaders/shapes.wgsl");

fn assemble_shader(body: &str) -> String {
    format!(
        "{COMMON_SHADER}\n{body}\n{SHAPES_FRAGMENT}\n{}",
        shader_constants()
    )
}

struct Stage {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    shapes_group: BindGroup,
}

impl Stage {
    fn build(
        context: &GpuContext,
        label: &str,
        shader: &str,
        bindings: &[BindingKind],
        resources: &[&GpuBuffer],
        shape_resources: &[&GpuBuffer],
    ) -> Self {
        assert_eq!(bindings.len(), resources.len(), "stage binding mismatch");
        let shape_specs = shape_resources
            .iter()
            .enumerate()
            .map(|(position, _)| BindingSpec {
                binding: position as u32,
                kind: BindingKind::ReadOnlyStorage,
            })
            .collect::<Vec<_>>();
        let pipeline = context.compute_pipeline(
            label,
            shader,
            "main",
            &[
                &bindings
                    .iter()
                    .enumerate()
                    .map(|(position, kind)| BindingSpec {
                        binding: position as u32,
                        kind: *kind,
                    })
                    .collect::<Vec<_>>()[..],
                &shape_specs[..],
            ],
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
        let bind_group = pipeline.create_bind_group(context.device(), 0, &entries);
        let shape_entries = shape_resources
            .iter()
            .enumerate()
            .map(|(position, buffer)| BindGroupEntry {
                binding: position as u32,
                resource: buffer.as_binding(),
            })
            .collect::<Vec<_>>();
        let shapes_group = pipeline.create_bind_group(context.device(), 1, &shape_entries);
        Self {
            pipeline,
            bind_group,
            shapes_group,
        }
    }

    fn dispatch(&self, recorder: &mut ComputeRecorder, elements: u32) {
        let workgroups = self.pipeline.workgroup_count(elements);
        recorder.record(
            &self.pipeline,
            &[&self.bind_group, &self.shapes_group],
            workgroups,
        );
    }

    fn dispatch_indirect(&self, recorder: &mut ComputeRecorder, args: &GpuBuffer) {
        recorder.record_indirect(
            &self.pipeline,
            &[&self.bind_group, &self.shapes_group],
            args,
            0,
        );
    }

    fn dispatch_workgroups(&self, recorder: &mut ComputeRecorder, workgroups: u32) {
        recorder.record(
            &self.pipeline,
            &[&self.bind_group, &self.shapes_group],
            workgroups,
        );
    }
}

pub(crate) struct Stages {
    apply_commands: Stage,
    apply_constraint_commands: Stage,
    joint_filter: Stage,
    integrate: Stage,
    broadphase_aabb: Stage,
    grid_entries: Stage,
    broadphase_pairs: Stage,
    large_pairs: Stage,
    narrowphase: Stage,
    compact_scan: Stage,
    compact_offsets: Stage,
    compact_scatter: Stage,
    events_end: Stage,
    contact_archive: Stage,
    prev_count_sync: Stage,
    island_init: Stage,
    island_link_contacts: Stage,
    island_link_constraints: Stage,
    island_jump: Stage,
    island_aggregate: Stage,
    island_broadcast: Stage,
    gather_contact_keys_b: Stage,
    gather_constraint_keys: Stage,
    reset_gather_boundaries: Stage,
    mark_contact_boundaries: Stage,
    mark_constraint_boundaries: Stage,
    ccd_sweep: Stage,
    contact_match: Stage,
    contact_solve_extract: Stage,
    constraint_solve_extract: Stage,
    body_apply_solver: Stage,
    position_solve_extract: Stage,
    body_apply_positions: Stage,
    query: Stage,
    sort: GpuSort,
    sort_hi: GpuBuffer,
    sort_lo: GpuBuffer,
    sort_values: GpuBuffer,
    sort_args: GpuCountArgs,
    contact_bucket: GpuBucketSort,
    constraint_bucket: GpuBucketSort,
}

pub(crate) fn build_stages(
    context: &GpuContext,
    buffers: &StageBuffers,
    body_capacity: u32,
) -> Stages {
    let device = context.device();
    let shape_resources = [
        &buffers.shapes,
        &buffers.shape_vertices,
        &buffers.shape_triangles,
        &buffers.shape_nodes,
    ];
    let apply_commands = Stage::build(
        context,
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
        &shape_resources,
    );
    let apply_constraint_commands = Stage::build(
        context,
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
        &shape_resources,
    );
    let joint_filter = Stage::build(
        context,
        "joint_filter",
        &assemble_shader(include_str!("shaders/joint_filter.wgsl")),
        &[
            BindingKind::Uniform,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
        ],
        &[
            &buffers.params,
            &buffers.constraints,
            &buffers.joint_hi,
            &buffers.joint_lo,
            &buffers.joint_count,
        ],
        &shape_resources,
    );
    let integrate = Stage::build(
        context,
        "integrate",
        &assemble_shader(include_str!("shaders/integrate.wgsl")),
        &[BindingKind::Uniform, BindingKind::ReadWriteStorage],
        &[&buffers.params, &buffers.bodies_current],
        &shape_resources,
    );
    let broadphase_aabb = Stage::build(
        context,
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
        &shape_resources,
    );
    let grid_entries = Stage::build(
        context,
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
        &shape_resources,
    );
    let broadphase_pairs = Stage::build(
        context,
        "broadphase_pairs",
        &assemble_shader(include_str!("shaders/broadphase_pairs.wgsl")),
        &[
            BindingKind::Uniform,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
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
        &shape_resources,
    );
    let large_pairs = Stage::build(
        context,
        "large_pairs",
        &assemble_shader(include_str!("shaders/large_pairs.wgsl")),
        &[
            BindingKind::Uniform,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadOnlyStorage,
        ],
        &[
            &buffers.params,
            &buffers.large_bodies,
            &buffers.large_count,
            &buffers.pairs.keys_hi,
            &buffers.pairs.keys_lo,
            &buffers.pair_count,
            &buffers.colliders,
        ],
        &shape_resources,
    );
    let narrowphase = Stage::build(
        context,
        "narrowphase",
        &assemble_shader(include_str!("shaders/narrowphase.wgsl")),
        &[
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
        ],
        &[
            &buffers.bodies_current,
            &buffers.colliders,
            &buffers.pairs.keys_hi,
            &buffers.pairs.keys_lo,
            &buffers.contacts_raw,
            &buffers.contact_valid,
            &buffers.pair_count,
            &buffers.joint_hi,
            &buffers.joint_lo,
            &buffers.joint_count,
        ],
        &shape_resources,
    );
    let compact_scan = Stage::build(
        context,
        "compact_scan",
        &assemble_shader(include_str!("shaders/compact_scan.wgsl")),
        &[
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadOnlyStorage,
        ],
        &[
            &buffers.contact_valid,
            &buffers.compact_ranks,
            &buffers.compact_block_sums,
            &buffers.pair_count,
        ],
        &shape_resources,
    );
    let compact_offsets = Stage::build(
        context,
        "compact_offsets",
        &assemble_shader(include_str!("shaders/compact_offsets.wgsl")),
        &[
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
        ],
        &[
            &buffers.compact_block_sums,
            &buffers.compact_block_offsets,
            &buffers.contact_count,
        ],
        &shape_resources,
    );
    let compact_scatter = Stage::build(
        context,
        "compact_scatter",
        &assemble_shader(include_str!("shaders/compact_scatter.wgsl")),
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
            &buffers.contacts_raw,
            &buffers.contact_valid,
            &buffers.compact_ranks,
            &buffers.compact_block_offsets,
            &buffers.contacts,
            &buffers.contact_a_body,
            &buffers.pair_count,
        ],
        &shape_resources,
    );
    let events_end = Stage::build(
        context,
        "events_end",
        &assemble_shader(include_str!("shaders/events_end.wgsl")),
        &[
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
        ],
        &[
            &buffers.prev_contacts,
            &buffers.prev_contact_count,
            &buffers.contacts,
            &buffers.contact_count,
            &buffers.events,
            &buffers.event_count,
        ],
        &shape_resources,
    );
    let contact_archive = Stage::build(
        context,
        "contact_archive",
        &assemble_shader(include_str!("shaders/contact_archive.wgsl")),
        &[
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadOnlyStorage,
        ],
        &[
            &buffers.contacts,
            &buffers.prev_contacts,
            &buffers.contact_count,
        ],
        &shape_resources,
    );
    let prev_count_sync = Stage::build(
        context,
        "prev_count_sync",
        &assemble_shader(include_str!("shaders/prev_count_sync.wgsl")),
        &[BindingKind::ReadOnlyStorage, BindingKind::ReadWriteStorage],
        &[&buffers.contact_count, &buffers.prev_contact_count],
        &shape_resources,
    );
    let island_init = Stage::build(
        context,
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
        &shape_resources,
    );
    let island_link_contacts = Stage::build(
        context,
        "island_link_contacts",
        &assemble_shader(include_str!("shaders/island_link_contacts.wgsl")),
        &[
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
        ],
        &[
            &buffers.bodies_current,
            &buffers.contacts,
            &buffers.contact_count,
            &buffers.island_parents,
        ],
        &shape_resources,
    );
    let island_link_constraints = Stage::build(
        context,
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
        &shape_resources,
    );
    let island_jump = Stage::build(
        context,
        "island_jump",
        &assemble_shader(include_str!("shaders/island_jump.wgsl")),
        &[BindingKind::Uniform, BindingKind::ReadWriteStorage],
        &[&buffers.params, &buffers.island_parents],
        &shape_resources,
    );
    let island_aggregate = Stage::build(
        context,
        "island_aggregate",
        &assemble_shader(include_str!("shaders/island_aggregate.wgsl")),
        &[
            BindingKind::Uniform,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
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
        &shape_resources,
    );
    let island_broadcast = Stage::build(
        context,
        "island_broadcast",
        &assemble_shader(include_str!("shaders/island_broadcast.wgsl")),
        &[
            BindingKind::Uniform,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
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
        &shape_resources,
    );
    let gather_contact_keys_b = Stage::build(
        context,
        "gather_contact_keys_b",
        &assemble_shader(include_str!("shaders/gather_contact_keys_b.wgsl")),
        &[
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
        ],
        &[
            &buffers.contacts,
            &buffers.contact_count,
            &buffers.contact_b_keys,
            &buffers.contact_b_values,
        ],
        &shape_resources,
    );
    let gather_constraint_keys = Stage::build(
        context,
        "gather_constraint_keys",
        &assemble_shader(include_str!("shaders/gather_constraint_keys.wgsl")),
        &[
            BindingKind::Uniform,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
        ],
        &[
            &buffers.params,
            &buffers.constraints,
            &buffers.constraint_gather_a_keys,
            &buffers.constraint_gather_a_values,
            &buffers.constraint_gather_b_keys,
            &buffers.constraint_gather_b_values,
        ],
        &shape_resources,
    );
    let reset_gather_boundaries = Stage::build(
        context,
        "reset_gather_boundaries",
        &assemble_shader(include_str!("shaders/reset_gather_boundaries.wgsl")),
        &[
            BindingKind::Uniform,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
        ],
        &[
            &buffers.params,
            &buffers.contact_first_a,
            &buffers.contact_first_b,
            &buffers.constraint_first_a,
            &buffers.constraint_first_b,
        ],
        &shape_resources,
    );
    let mark_contact_boundaries = Stage::build(
        context,
        "mark_contact_boundaries",
        &assemble_shader(include_str!("shaders/mark_contact_boundaries.wgsl")),
        &[
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
        ],
        &[
            &buffers.contact_a_body,
            &buffers.contact_b_keys_out,
            &buffers.contact_count,
            &buffers.contact_first_a,
            &buffers.contact_first_b,
        ],
        &shape_resources,
    );
    let mark_constraint_boundaries = Stage::build(
        context,
        "mark_constraint_boundaries",
        &assemble_shader(include_str!("shaders/mark_constraint_boundaries.wgsl")),
        &[
            BindingKind::Uniform,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
        ],
        &[
            &buffers.params,
            &buffers.constraint_gather_a_keys_out,
            &buffers.constraint_gather_b_keys_out,
            &buffers.constraint_first_a,
            &buffers.constraint_first_b,
        ],
        &shape_resources,
    );
    let ccd_sweep = Stage::build(
        context,
        "ccd_sweep",
        &assemble_shader(include_str!("shaders/ccd_sweep.wgsl")),
        &[
            BindingKind::Uniform,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
        ],
        &[
            &buffers.params,
            &buffers.bodies_current,
            &buffers.colliders,
            &buffers.pairs.keys_hi,
            &buffers.pairs.keys_lo,
            &buffers.pair_count,
        ],
        &shape_resources,
    );
    let contact_match = Stage::build(
        context,
        "contact_match",
        &assemble_shader(include_str!("shaders/contact_match.wgsl")),
        &[
            BindingKind::ReadWriteStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
        ],
        &[
            &buffers.contacts,
            &buffers.prev_contacts,
            &buffers.prev_contact_count,
            &buffers.contact_count,
            &buffers.events,
            &buffers.event_count,
        ],
        &shape_resources,
    );
    let contact_solve_extract = Stage::build(
        context,
        "contact_solve_extract",
        &assemble_shader(include_str!("shaders/contact_solve_extract.wgsl")),
        &[
            BindingKind::Uniform,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
        ],
        &[
            &buffers.params,
            &buffers.bodies_current,
            &buffers.contacts,
            &buffers.contact_count,
            &buffers.wake_flags,
            &buffers.contact_deltas,
        ],
        &shape_resources,
    );
    let constraint_solve_extract = Stage::build(
        context,
        "constraint_solve_extract",
        &assemble_shader(include_str!("shaders/constraint_solve_extract.wgsl")),
        &[
            BindingKind::Uniform,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
        ],
        &[
            &buffers.params,
            &buffers.bodies_current,
            &buffers.constraints,
            &buffers.wake_flags,
            &buffers.constraint_deltas,
        ],
        &shape_resources,
    );
    let body_apply_solver = Stage::build(
        context,
        "body_apply_solver",
        &assemble_shader(include_str!("shaders/body_apply_solver.wgsl")),
        &[
            BindingKind::Uniform,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
        ],
        &[
            &buffers.params,
            &buffers.bodies_current,
            &buffers.contact_first_a,
            &buffers.contact_first_b,
            &buffers.contact_a_body,
            &buffers.contact_b_keys_out,
            &buffers.contact_b_values_out,
            &buffers.constraint_first_a,
            &buffers.constraint_first_b,
            &buffers.constraint_gather_a_keys_out,
            &buffers.constraint_gather_a_values_out,
            &buffers.constraint_gather_b_keys_out,
            &buffers.constraint_gather_b_values_out,
            &buffers.contact_count,
            &buffers.contact_deltas,
            &buffers.constraint_deltas,
        ],
        &shape_resources,
    );
    let position_solve_extract = Stage::build(
        context,
        "position_solve_extract",
        &assemble_shader(include_str!("shaders/position_solve_extract.wgsl")),
        &[
            BindingKind::Uniform,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
        ],
        &[
            &buffers.params,
            &buffers.bodies_current,
            &buffers.contacts,
            &buffers.contact_count,
            &buffers.contact_deltas,
        ],
        &shape_resources,
    );
    let body_apply_positions = Stage::build(
        context,
        "body_apply_positions",
        &assemble_shader(include_str!("shaders/body_apply_positions.wgsl")),
        &[
            BindingKind::Uniform,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
        ],
        &[
            &buffers.params,
            &buffers.bodies_current,
            &buffers.contact_first_a,
            &buffers.contact_first_b,
            &buffers.contact_a_body,
            &buffers.contact_b_keys_out,
            &buffers.contact_b_values_out,
            &buffers.contact_count,
            &buffers.contact_deltas,
        ],
        &shape_resources,
    );
    let query = Stage::build(
        context,
        "query",
        &assemble_shader(include_str!("shaders/queries.wgsl")),
        &[
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadWriteStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::ReadOnlyStorage,
            BindingKind::Uniform,
        ],
        &[
            &buffers.queries,
            &buffers.bodies_current,
            &buffers.colliders,
            &buffers.aabbs,
            &buffers.entries.keys_hi,
            &buffers.entries.keys_lo,
            &buffers.entry_count,
            &buffers.query_headers,
            &buffers.query_hits,
            &buffers.large_bodies,
            &buffers.large_count,
            &buffers.params,
        ],
        &shape_resources,
    );
    let sort = GpuSort::new(context, "sim sort", buffers.sort_scratch.capacity_u32());
    let sort_hi = GpuBuffer::new(
        device,
        "sort scratch hi",
        buffers.sort_scratch.keys_hi.size(),
        wgpu::BufferUsages::STORAGE,
    );
    let sort_lo = GpuBuffer::new(
        device,
        "sort scratch lo",
        buffers.sort_scratch.keys_lo.size(),
        wgpu::BufferUsages::STORAGE,
    );
    let sort_values = GpuBuffer::new(
        device,
        "sort scratch values",
        buffers.sort_scratch.values.size(),
        wgpu::BufferUsages::STORAGE,
    );
    let sort_args = GpuCountArgs::new(context, "sim sort args");
    let contact_bucket = GpuBucketSort::new(
        context,
        "contact bucket",
        body_capacity,
        buffers.contact_capacity(),
    );
    let constraint_bucket = GpuBucketSort::new(
        context,
        "constraint bucket",
        body_capacity,
        buffers.constraint_capacity(),
    );
    Stages {
        apply_commands,
        apply_constraint_commands,
        joint_filter,
        integrate,
        broadphase_aabb,
        grid_entries,
        broadphase_pairs,
        large_pairs,
        narrowphase,
        compact_scan,
        compact_offsets,
        compact_scatter,
        events_end,
        contact_archive,
        prev_count_sync,
        island_init,
        island_link_contacts,
        island_link_constraints,
        island_jump,
        island_aggregate,
        island_broadcast,
        gather_contact_keys_b,
        gather_constraint_keys,
        reset_gather_boundaries,
        mark_contact_boundaries,
        mark_constraint_boundaries,
        ccd_sweep,
        contact_match,
        contact_solve_extract,
        constraint_solve_extract,
        body_apply_solver,
        position_solve_extract,
        body_apply_positions,
        query,
        sort,
        sort_hi,
        sort_lo,
        sort_values,
        sort_args,
        contact_bucket,
        constraint_bucket,
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
    constraint_count: u32,
) {
    let constraint_active = constraint_count > 0;
    let mut recorder = ComputeRecorder::begin(encoder, "physics frame");
    stages.apply_commands.dispatch(&mut recorder, 1);
    stages.apply_constraint_commands.dispatch(&mut recorder, 1);
    if constraint_active {
        stages
            .joint_filter
            .dispatch(&mut recorder, buffers.constraint_capacity());
        stages.sort_args.encode(
            device,
            &mut recorder,
            &buffers.joint_count,
            &buffers.joint_args,
        );
        let joint_words = sort_words(buffers.constraint_capacity());
        stages.sort.sort_64(
            device,
            &mut recorder,
            &buffers.joint_count,
            &buffers.joint_args,
            joint_words,
            joint_words,
            &buffers.joint_lo,
            &buffers.joint_hi,
            &buffers.entries.values,
            &stages.sort_lo,
            &stages.sort_hi,
            &stages.sort_values,
        );
    }
    stages.integrate.dispatch(&mut recorder, body_count);
    let slot_words = sort_words(body_count);
    stages.broadphase_aabb.dispatch(&mut recorder, body_count);
    stages.grid_entries.dispatch(&mut recorder, body_count);
    stages.sort_args.encode(
        device,
        &mut recorder,
        &buffers.entry_count,
        &buffers.entry_args,
    );
    stages.sort.sort_64(
        device,
        &mut recorder,
        &buffers.entry_count,
        &buffers.entry_args,
        slot_words,
        4,
        &buffers.entries.keys_lo,
        &buffers.entries.keys_hi,
        &buffers.entries.values,
        &stages.sort_lo,
        &stages.sort_hi,
        &stages.sort_values,
    );
    stages
        .broadphase_pairs
        .dispatch_indirect(&mut recorder, &buffers.entry_args);
    stages.large_pairs.dispatch(&mut recorder, body_count);
    stages.sort_args.encode(
        device,
        &mut recorder,
        &buffers.pair_count,
        &buffers.pair_args,
    );
    stages.sort.sort_64(
        device,
        &mut recorder,
        &buffers.pair_count,
        &buffers.pair_args,
        slot_words,
        slot_words,
        &buffers.pairs.keys_lo,
        &buffers.pairs.keys_hi,
        &buffers.pairs.values,
        &stages.sort_lo,
        &stages.sort_hi,
        &stages.sort_values,
    );
    stages
        .ccd_sweep
        .dispatch_indirect(&mut recorder, &buffers.pair_args);
    stages
        .narrowphase
        .dispatch_indirect(&mut recorder, &buffers.pair_args);
    let compact_blocks = buffers.pairs.capacity_u32().div_ceil(COMPACT_BLOCK);
    stages
        .compact_scan
        .dispatch_workgroups(&mut recorder, compact_blocks);
    stages.compact_offsets.dispatch_workgroups(&mut recorder, 1);
    stages
        .compact_scatter
        .dispatch_indirect(&mut recorder, &buffers.pair_args);
    stages.sort_args.encode(
        device,
        &mut recorder,
        &buffers.contact_count,
        &buffers.contact_args,
    );
    stages
        .contact_match
        .dispatch_indirect(&mut recorder, &buffers.contact_args);
    stages.island_init.dispatch(&mut recorder, body_count);
    stages
        .island_link_contacts
        .dispatch_indirect(&mut recorder, &buffers.contact_args);
    if constraint_active {
        stages
            .island_link_constraints
            .dispatch(&mut recorder, buffers.constraint_capacity());
    }
    for _ in 0..island_rounds {
        stages.island_jump.dispatch(&mut recorder, body_count);
    }
    stages
        .gather_contact_keys_b
        .dispatch_indirect(&mut recorder, &buffers.contact_args);
    stages.contact_bucket.sort(
        device,
        &mut recorder,
        &buffers.contact_count,
        &buffers.contact_args,
        &buffers.contact_b_keys,
        &buffers.contact_b_values,
        &buffers.contact_b_keys_out,
        &buffers.contact_b_values_out,
    );
    if constraint_active {
        stages
            .gather_constraint_keys
            .dispatch(&mut recorder, buffers.constraint_capacity());
        stages.sort_args.encode(
            device,
            &mut recorder,
            &buffers.constraint_count_state,
            &buffers.constraint_args,
        );
        stages.constraint_bucket.sort(
            device,
            &mut recorder,
            &buffers.constraint_count_state,
            &buffers.constraint_args,
            &buffers.constraint_gather_a_keys,
            &buffers.constraint_gather_a_values,
            &buffers.constraint_gather_a_keys_out,
            &buffers.constraint_gather_a_values_out,
        );
        stages.constraint_bucket.sort(
            device,
            &mut recorder,
            &buffers.constraint_count_state,
            &buffers.constraint_args,
            &buffers.constraint_gather_b_keys,
            &buffers.constraint_gather_b_values,
            &buffers.constraint_gather_b_keys_out,
            &buffers.constraint_gather_b_values_out,
        );
    }
    stages
        .reset_gather_boundaries
        .dispatch(&mut recorder, body_count);
    stages
        .mark_contact_boundaries
        .dispatch_indirect(&mut recorder, &buffers.contact_args);
    if constraint_active {
        stages
            .mark_constraint_boundaries
            .dispatch(&mut recorder, buffers.constraint_capacity());
    }
    stages.island_aggregate.dispatch(&mut recorder, body_count);
    stages.island_broadcast.dispatch(&mut recorder, body_count);
    for _ in 0..solve_iterations {
        if constraint_active {
            stages
                .constraint_solve_extract
                .dispatch(&mut recorder, buffers.constraint_capacity());
        }
        stages
            .contact_solve_extract
            .dispatch_indirect(&mut recorder, &buffers.contact_args);
        stages.body_apply_solver.dispatch(&mut recorder, body_count);
    }
    for _ in 0..position_iterations {
        stages
            .position_solve_extract
            .dispatch_indirect(&mut recorder, &buffers.contact_args);
        stages
            .body_apply_positions
            .dispatch(&mut recorder, body_count);
    }
    drop(recorder);
    let mut tail = ComputeRecorder::begin(encoder, "physics tail");
    stages
        .events_end
        .dispatch_indirect(&mut tail, &buffers.prev_args);
    stages
        .contact_archive
        .dispatch_indirect(&mut tail, &buffers.contact_args);
    stages.prev_count_sync.dispatch(&mut tail, 1);
    stages.sort_args.encode(
        device,
        &mut tail,
        &buffers.prev_contact_count,
        &buffers.prev_args,
    );
    if query_count > 0 {
        stages.query.dispatch_workgroups(&mut tail, query_count);
    }
}

fn sort_words(capacity: u32) -> u32 {
    let value = capacity.saturating_mul(4).max(2);
    let width = (32 - value.leading_zeros()).max(1);
    width.div_ceil(8).clamp(1, 4)
}
