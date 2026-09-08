use crate::buffers::{COMPACT_BLOCK, WorldBuffers};
#[cfg(feature = "profile")]
use dynamis_gpu::GpuTimer;
use dynamis_gpu::{
    BindingKind, BindingSpec, ComputePipeline, ComputeRecorder, GpuBuffer, GpuContext,
};
use dynamis_kernel::{BucketChannels, BucketSort, RadixSort, SortChannels};
use dynamis_layout::{
    BODY_CCD, BODY_KINEMATIC, BODY_SLEEPING, COLLIDER_EVENT_BEGIN_END, COLLIDER_EVENT_PERSIST,
    COLLIDER_SENSOR, COMMAND_ADD, COMMAND_ANGULAR_IMPULSE, COMMAND_CONSTRAINT_ADD,
    COMMAND_CONSTRAINT_PATCH, COMMAND_CONSTRAINT_REMOVE, COMMAND_FORCE, COMMAND_FORCE_AT_POINT,
    COMMAND_IMPULSE, COMMAND_PATCH, COMMAND_REMOVE, COMMAND_SLEEP, COMMAND_SWAP, COMMAND_TORQUE,
    COMMAND_WAKE, CONSTRAINT_BALL, CONSTRAINT_BROKEN, CONSTRAINT_CONE,
    CONSTRAINT_DISABLE_COLLISIONS, CONSTRAINT_DISTANCE, CONSTRAINT_FIXED, CONSTRAINT_GEAR,
    CONSTRAINT_HAS_BREAK, CONSTRAINT_HAS_LIMIT, CONSTRAINT_HAS_MOTOR, CONSTRAINT_HAS_SWING,
    CONSTRAINT_INVALID, CONSTRAINT_IS_SPRING, CONSTRAINT_PRISMATIC, CONSTRAINT_PULLEY,
    CONSTRAINT_REVOLUTE, CONSTRAINT_SIXDOF, CONSTRAINT_WARM_START, CONTACT_MAX_POINTS, DOF_DRIVEN,
    DOF_FREE, DOF_LIMITED, DOF_LOCKED, EVENT_BEGIN, EVENT_END, EVENT_PERSIST,
    FILTER_IGNORE_KINEMATIC, FILTER_IGNORE_SENSORS, FILTER_IGNORE_SLEEPING, FILTER_IGNORE_STATIC,
    IMPULSE_AT_POINT, ISLAND_ACTIVE, ISLAND_WAKE, MAX_CELLS_PER_COLLIDER, MAX_HITS_PER_QUERY,
    NO_BODY, NO_COLLISION_FILTER, NO_HIT, OVERFLOW_EVENTS, OVERFLOW_PAIRS, PATCH_ANGULAR_VELOCITY,
    PATCH_CCD, PATCH_COLLIDER, PATCH_DYNAMICS, PATCH_FRICTION, PATCH_GROUP, PATCH_KINEMATIC,
    PATCH_MASK, PATCH_MASS, PATCH_ORIENTATION, PATCH_POSITION, PATCH_RESTITUTION, PATCH_VELOCITY,
    QUERY_CONVEX, QUERY_CUBOID, QUERY_POINT, QUERY_RAY, QUERY_SPHERE, QUERY_SWEEP, SHAPE_CAPSULE,
    SHAPE_CUBOID, SHAPE_CYLINDER, SHAPE_HEIGHTFIELD, SHAPE_HULL, SHAPE_MESH, SHAPE_NONE,
    SHAPE_PLANE, SHAPE_SPHERE, SHAPE_TRIANGLE,
};
use dynamis_model::MAX_COLLIDERS_PER_BODY;
use wgpu::{BindGroup, BindGroupEntry, CommandEncoder, Device};

const WORKGROUP_SIZE: u32 = 64;

pub(crate) const SIM_PASSES: &[&str] = &[
    "commands",
    "integrate",
    "broadphase",
    "narrowphase",
    "islands",
    "sleep",
    "velocity_solve",
    "position_solve",
    "tail",
];

const PASS_COMMANDS: usize = 0;
const PASS_INTEGRATE: usize = 1;
const PASS_BROADPHASE: usize = 2;
const PASS_NARROWPHASE: usize = 3;
const PASS_ISLANDS: usize = 4;
const PASS_SLEEP: usize = 5;
const PASS_VELOCITY_SOLVE: usize = 6;
const PASS_POSITION_SOLVE: usize = 7;
const PASS_TAIL: usize = 8;

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
         const COMMAND_CONSTRAINT_PATCH: u32 = {COMMAND_CONSTRAINT_PATCH}u;\n\
         const COMMAND_SWAP: u32 = {COMMAND_SWAP}u;\n\
         const COMMAND_SLEEP: u32 = {COMMAND_SLEEP}u;\n\
         const COMMAND_WAKE: u32 = {COMMAND_WAKE}u;\n\
         const IMPULSE_AT_POINT: u32 = {IMPULSE_AT_POINT}u;\n\
         const PATCH_POSITION: u32 = {PATCH_POSITION}u;\n\
         const PATCH_VELOCITY: u32 = {PATCH_VELOCITY}u;\n\
         const PATCH_MASS: u32 = {PATCH_MASS}u;\n\
         const PATCH_COLLIDER: u32 = {PATCH_COLLIDER}u;\n\
         const PATCH_RESTITUTION: u32 = {PATCH_RESTITUTION}u;\n\
         const PATCH_ORIENTATION: u32 = {PATCH_ORIENTATION}u;\n\
         const PATCH_ANGULAR_VELOCITY: u32 = {PATCH_ANGULAR_VELOCITY}u;\n\
         const PATCH_FRICTION: u32 = {PATCH_FRICTION}u;\n\
         const PATCH_GROUP: u32 = {PATCH_GROUP}u;\n\
         const PATCH_MASK: u32 = {PATCH_MASK}u;\n\
         const PATCH_KINEMATIC: u32 = {PATCH_KINEMATIC}u;\n\
         const PATCH_CCD: u32 = {PATCH_CCD}u;\n\
         const PATCH_DYNAMICS: u32 = {PATCH_DYNAMICS}u;\n\
         const NO_COLLISION_FILTER: u32 = {NO_COLLISION_FILTER}u;\n\
         const OVERFLOW_PAIRS: u32 = {OVERFLOW_PAIRS}u;\n\
         const OVERFLOW_EVENTS: u32 = {OVERFLOW_EVENTS}u;\n\
         const SHAPE_NONE: u32 = {SHAPE_NONE}u;\n\
         const SHAPE_SPHERE: u32 = {SHAPE_SPHERE}u;\n\
         const SHAPE_CUBOID: u32 = {SHAPE_CUBOID}u;\n\
         const SHAPE_CAPSULE: u32 = {SHAPE_CAPSULE}u;\n\
         const SHAPE_CYLINDER: u32 = {SHAPE_CYLINDER}u;\n\
         const SHAPE_HULL: u32 = {SHAPE_HULL}u;\n\
         const SHAPE_MESH: u32 = {SHAPE_MESH}u;\n\
         const SHAPE_HEIGHTFIELD: u32 = {SHAPE_HEIGHTFIELD}u;\n\
         const SHAPE_PLANE: u32 = {SHAPE_PLANE}u;\n\
         const SHAPE_TRIANGLE: u32 = {SHAPE_TRIANGLE}u;\n\
         const BODY_KINEMATIC: u32 = {BODY_KINEMATIC}u;\n\
         const BODY_SLEEPING: u32 = {BODY_SLEEPING}u;\n\
         const BODY_CCD: u32 = {BODY_CCD}u;\n\
         const COLLIDER_SENSOR: u32 = {COLLIDER_SENSOR}u;\n\
         const COLLIDER_EVENT_BEGIN_END: u32 = {COLLIDER_EVENT_BEGIN_END}u;\n\
         const COLLIDER_EVENT_PERSIST: u32 = {COLLIDER_EVENT_PERSIST}u;\n\
         const ISLAND_WAKE: u32 = {ISLAND_WAKE}u;\n\
         const ISLAND_ACTIVE: u32 = {ISLAND_ACTIVE}u;\n\
         const CONTACT_MAX_POINTS: u32 = {CONTACT_MAX_POINTS}u;\n\
         const CONSTRAINT_BALL: u32 = {CONSTRAINT_BALL}u;\n\
         const CONSTRAINT_DISTANCE: u32 = {CONSTRAINT_DISTANCE}u;\n\
         const CONSTRAINT_REVOLUTE: u32 = {CONSTRAINT_REVOLUTE}u;\n\
         const CONSTRAINT_PRISMATIC: u32 = {CONSTRAINT_PRISMATIC}u;\n\
         const CONSTRAINT_FIXED: u32 = {CONSTRAINT_FIXED}u;\n\
         const CONSTRAINT_GEAR: u32 = {CONSTRAINT_GEAR}u;\n\
         const CONSTRAINT_PULLEY: u32 = {CONSTRAINT_PULLEY}u;\n\
         const CONSTRAINT_CONE: u32 = {CONSTRAINT_CONE}u;\n\
         const CONSTRAINT_SIXDOF: u32 = {CONSTRAINT_SIXDOF}u;\n\
         const CONSTRAINT_INVALID: u32 = {CONSTRAINT_INVALID}u;\n\
         const CONSTRAINT_DISABLE_COLLISIONS: u32 = {CONSTRAINT_DISABLE_COLLISIONS}u;\n\
         const CONSTRAINT_HAS_LIMIT: u32 = {CONSTRAINT_HAS_LIMIT}u;\n\
         const CONSTRAINT_HAS_MOTOR: u32 = {CONSTRAINT_HAS_MOTOR}u;\n\
         const CONSTRAINT_IS_SPRING: u32 = {CONSTRAINT_IS_SPRING}u;\n\
         const CONSTRAINT_HAS_SWING: u32 = {CONSTRAINT_HAS_SWING}u;\n\
         const CONSTRAINT_HAS_BREAK: u32 = {CONSTRAINT_HAS_BREAK}u;\n\
         const CONSTRAINT_BROKEN: u32 = {CONSTRAINT_BROKEN}u;\n\
         const CONSTRAINT_WARM_START: u32 = {CONSTRAINT_WARM_START}u;\n\
         const DOF_FREE: u32 = {DOF_FREE}u;\n\
         const DOF_LOCKED: u32 = {DOF_LOCKED}u;\n\
         const DOF_LIMITED: u32 = {DOF_LIMITED}u;\n\
         const DOF_DRIVEN: u32 = {DOF_DRIVEN}u;\n\
         const QUERY_RAY: u32 = {QUERY_RAY}u;\n\
         const QUERY_SPHERE: u32 = {QUERY_SPHERE}u;\n\
         const QUERY_CUBOID: u32 = {QUERY_CUBOID}u;\n\
         const QUERY_SWEEP: u32 = {QUERY_SWEEP}u;\n\
         const QUERY_POINT: u32 = {QUERY_POINT}u;\n\
         const QUERY_CONVEX: u32 = {QUERY_CONVEX}u;\n\
         const FILTER_IGNORE_SENSORS: u32 = {FILTER_IGNORE_SENSORS}u;\n\
         const FILTER_IGNORE_SLEEPING: u32 = {FILTER_IGNORE_SLEEPING}u;\n\
         const FILTER_IGNORE_STATIC: u32 = {FILTER_IGNORE_STATIC}u;\n\
         const FILTER_IGNORE_KINEMATIC: u32 = {FILTER_IGNORE_KINEMATIC}u;\n\
         const EVENT_BEGIN: u32 = {EVENT_BEGIN}u;\n\
         const EVENT_END: u32 = {EVENT_END}u;\n\
         const EVENT_PERSIST: u32 = {EVENT_PERSIST}u;\n\
         const NO_BODY: u32 = {NO_BODY}u;\n\
         const NO_HIT: f32 = {NO_HIT:e};\n\
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
    workgroups: u32,
}

impl Stage {
    fn build(
        context: &GpuContext,
        label: &str,
        shader: &str,
        bindings: &[(BindingKind, &GpuBuffer)],
        shape_resources: &[&GpuBuffer],
        elements: u32,
        threads: u32,
    ) -> Self {
        let shape_specs = shape_resources
            .iter()
            .enumerate()
            .map(|(position, _)| BindingSpec {
                binding: position as u32,
                kind: BindingKind::ReadOnlyStorage,
            })
            .collect::<Vec<_>>();
        let specs = bindings
            .iter()
            .enumerate()
            .map(|(position, (kind, _))| BindingSpec {
                binding: position as u32,
                kind: *kind,
            })
            .collect::<Vec<_>>();
        let pipeline = context.compute_pipeline(
            label,
            shader,
            "main",
            &[&specs[..], &shape_specs[..]],
            WORKGROUP_SIZE,
        );
        let entries: Vec<BindGroupEntry> = bindings
            .iter()
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
            workgroups: elements.div_ceil(threads),
        }
    }

    fn dispatch(&self, recorder: &mut ComputeRecorder) {
        recorder.record(
            &self.pipeline,
            &[&self.bind_group, &self.shapes_group],
            self.workgroups,
        );
    }

    fn dispatch_at(&self, recorder: &mut ComputeRecorder, elements: u32) {
        recorder.record(
            &self.pipeline,
            &[&self.bind_group, &self.shapes_group],
            elements.div_ceil(WORKGROUP_SIZE),
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

pub(crate) struct FrameParams {
    pub(crate) dynamic_count: u32,
    pub(crate) body_count: u32,
    pub(crate) solve_iterations: u32,
    pub(crate) position_iterations: u32,
    pub(crate) island_rounds: u32,
    pub(crate) query_count: u32,
    pub(crate) constraint_count: u32,
}

pub(crate) struct Pipeline {
    device: Device,
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
    constraints_warm_end: Stage,
    static_wake_clear: Stage,
    sort: RadixSort,
    sort_hi: GpuBuffer,
    sort_lo: GpuBuffer,
    sort_values: GpuBuffer,
    contact_bucket: BucketSort,
    constraint_bucket: BucketSort,
    #[cfg(feature = "profile")]
    timer: Option<GpuTimer>,
}

impl Pipeline {
    pub(crate) fn new(context: &GpuContext, buffers: &WorldBuffers, body_capacity: u32) -> Self {
        let device = context.device().clone();
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
                (BindingKind::ReadOnlyStorage, &buffers.commands),
                (BindingKind::ReadWriteStorage, &buffers.bodies),
                (BindingKind::ReadOnlyStorage, &buffers.command_count),
                (BindingKind::ReadWriteStorage, &buffers.wake_flags),
            ],
            &shape_resources,
            1,
            WORKGROUP_SIZE,
        );

        let apply_constraint_commands = Stage::build(
            context,
            "apply_constraint_commands",
            &assemble_shader(include_str!("shaders/apply_constraint_commands.wgsl")),
            &[
                (BindingKind::ReadOnlyStorage, &buffers.constraint_commands),
                (BindingKind::ReadWriteStorage, &buffers.constraints),
                (
                    BindingKind::ReadOnlyStorage,
                    &buffers.constraint_command_count,
                ),
            ],
            &shape_resources,
            1,
            WORKGROUP_SIZE,
        );

        let joint_filter = Stage::build(
            context,
            "joint_filter",
            &assemble_shader(include_str!("shaders/joint_filter.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadOnlyStorage, &buffers.constraints),
                (BindingKind::ReadWriteStorage, &buffers.joint_hi),
                (BindingKind::ReadWriteStorage, &buffers.joint_lo),
                (BindingKind::ReadWriteStorage, &buffers.joint_count),
            ],
            &shape_resources,
            buffers.constraint_capacity(),
            WORKGROUP_SIZE,
        );

        let integrate = Stage::build(
            context,
            "integrate",
            &assemble_shader(include_str!("shaders/integrate.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadWriteStorage, &buffers.bodies),
            ],
            &shape_resources,
            body_capacity,
            WORKGROUP_SIZE,
        );

        let broadphase_aabb = Stage::build(
            context,
            "broadphase_aabb",
            &assemble_shader(include_str!("shaders/broadphase_aabb.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadOnlyStorage, &buffers.bodies),
                (BindingKind::ReadOnlyStorage, &buffers.colliders),
                (BindingKind::ReadWriteStorage, &buffers.aabbs),
            ],
            &shape_resources,
            body_capacity,
            WORKGROUP_SIZE,
        );

        let grid_entries = Stage::build(
            context,
            "grid_entries",
            &assemble_shader(include_str!("shaders/grid_entries.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadOnlyStorage, &buffers.aabbs),
                (BindingKind::ReadWriteStorage, &buffers.entries.keys_hi),
                (BindingKind::ReadWriteStorage, &buffers.entries.keys_lo),
                (BindingKind::ReadWriteStorage, &buffers.entry_count),
                (BindingKind::ReadWriteStorage, &buffers.large_bodies),
                (BindingKind::ReadWriteStorage, &buffers.large_count),
            ],
            &shape_resources,
            body_capacity,
            WORKGROUP_SIZE,
        );

        let broadphase_pairs = Stage::build(
            context,
            "broadphase_pairs",
            &assemble_shader(include_str!("shaders/broadphase_pairs.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadOnlyStorage, &buffers.entries.keys_hi),
                (BindingKind::ReadOnlyStorage, &buffers.entries.keys_lo),
                (BindingKind::ReadWriteStorage, &buffers.entry_count),
                (BindingKind::ReadWriteStorage, &buffers.pairs.keys_hi),
                (BindingKind::ReadWriteStorage, &buffers.pairs.keys_lo),
                (BindingKind::ReadWriteStorage, &buffers.pair_count),
                (BindingKind::ReadWriteStorage, &buffers.overflow_flags),
            ],
            &shape_resources,
            buffers.entry_capacity(),
            WORKGROUP_SIZE,
        );

        let large_pairs = Stage::build(
            context,
            "large_pairs",
            &assemble_shader(include_str!("shaders/large_pairs.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadOnlyStorage, &buffers.large_bodies),
                (BindingKind::ReadWriteStorage, &buffers.large_count),
                (BindingKind::ReadWriteStorage, &buffers.pairs.keys_hi),
                (BindingKind::ReadWriteStorage, &buffers.pairs.keys_lo),
                (BindingKind::ReadWriteStorage, &buffers.pair_count),
                (BindingKind::ReadOnlyStorage, &buffers.colliders),
                (BindingKind::ReadWriteStorage, &buffers.overflow_flags),
            ],
            &shape_resources,
            body_capacity,
            WORKGROUP_SIZE,
        );

        let narrowphase = Stage::build(
            context,
            "narrowphase",
            &assemble_shader(include_str!("shaders/narrowphase.wgsl")),
            &[
                (BindingKind::ReadOnlyStorage, &buffers.bodies),
                (BindingKind::ReadOnlyStorage, &buffers.colliders),
                (BindingKind::ReadOnlyStorage, &buffers.pairs.keys_hi),
                (BindingKind::ReadOnlyStorage, &buffers.pairs.keys_lo),
                (BindingKind::ReadWriteStorage, &buffers.contacts_raw),
                (BindingKind::ReadWriteStorage, &buffers.contact_valid),
                (BindingKind::ReadWriteStorage, &buffers.pair_count),
                (BindingKind::ReadOnlyStorage, &buffers.joint_hi),
                (BindingKind::ReadOnlyStorage, &buffers.joint_lo),
                (BindingKind::ReadOnlyStorage, &buffers.joint_count),
                (BindingKind::Uniform, &buffers.params),
            ],
            &shape_resources,
            buffers.pair_capacity(),
            WORKGROUP_SIZE,
        );

        let compact_scan = Stage::build(
            context,
            "compact_scan",
            &assemble_shader(include_str!("shaders/compact_scan.wgsl")),
            &[
                (BindingKind::ReadOnlyStorage, &buffers.contact_valid),
                (BindingKind::ReadWriteStorage, &buffers.compact_ranks),
                (BindingKind::ReadWriteStorage, &buffers.compact_block_sums),
                (BindingKind::ReadOnlyStorage, &buffers.pair_count),
            ],
            &shape_resources,
            buffers.pair_capacity(),
            COMPACT_BLOCK,
        );

        let compact_offsets = Stage::build(
            context,
            "compact_offsets",
            &assemble_shader(include_str!("shaders/compact_offsets.wgsl")),
            &[
                (BindingKind::ReadOnlyStorage, &buffers.compact_block_sums),
                (
                    BindingKind::ReadWriteStorage,
                    &buffers.compact_block_offsets,
                ),
                (BindingKind::ReadWriteStorage, &buffers.contact_count),
            ],
            &shape_resources,
            1,
            COMPACT_BLOCK,
        );

        let compact_scatter = Stage::build(
            context,
            "compact_scatter",
            &assemble_shader(include_str!("shaders/compact_scatter.wgsl")),
            &[
                (BindingKind::ReadOnlyStorage, &buffers.contacts_raw),
                (BindingKind::ReadOnlyStorage, &buffers.contact_valid),
                (BindingKind::ReadOnlyStorage, &buffers.compact_ranks),
                (BindingKind::ReadOnlyStorage, &buffers.compact_block_offsets),
                (BindingKind::ReadWriteStorage, &buffers.contacts),
                (BindingKind::ReadWriteStorage, &buffers.contact_a_body),
                (BindingKind::ReadOnlyStorage, &buffers.pair_count),
            ],
            &shape_resources,
            buffers.pair_capacity(),
            WORKGROUP_SIZE,
        );

        let events_end = Stage::build(
            context,
            "events_end",
            &assemble_shader(include_str!("shaders/events_end.wgsl")),
            &[
                (BindingKind::ReadOnlyStorage, &buffers.prev_contacts),
                (BindingKind::ReadOnlyStorage, &buffers.prev_contact_count),
                (BindingKind::ReadOnlyStorage, &buffers.contacts),
                (BindingKind::ReadOnlyStorage, &buffers.contact_count),
                (BindingKind::ReadWriteStorage, &buffers.events),
                (BindingKind::ReadWriteStorage, &buffers.event_count),
                (BindingKind::ReadWriteStorage, &buffers.overflow_flags),
            ],
            &shape_resources,
            buffers.contact_capacity(),
            WORKGROUP_SIZE,
        );

        let contact_archive = Stage::build(
            context,
            "contact_archive",
            &assemble_shader(include_str!("shaders/contact_archive.wgsl")),
            &[
                (BindingKind::ReadOnlyStorage, &buffers.contacts),
                (BindingKind::ReadWriteStorage, &buffers.prev_contacts),
                (BindingKind::ReadOnlyStorage, &buffers.contact_count),
            ],
            &shape_resources,
            buffers.contact_capacity(),
            WORKGROUP_SIZE,
        );

        let prev_count_sync = Stage::build(
            context,
            "prev_count_sync",
            &assemble_shader(include_str!("shaders/prev_count_sync.wgsl")),
            &[
                (BindingKind::ReadOnlyStorage, &buffers.contact_count),
                (BindingKind::ReadWriteStorage, &buffers.prev_contact_count),
            ],
            &shape_resources,
            1,
            WORKGROUP_SIZE,
        );

        let island_init = Stage::build(
            context,
            "island_init",
            &assemble_shader(include_str!("shaders/island_init.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadWriteStorage, &buffers.island_parents),
                (BindingKind::ReadWriteStorage, &buffers.island_state),
            ],
            &shape_resources,
            body_capacity,
            WORKGROUP_SIZE,
        );

        let island_link_contacts = Stage::build(
            context,
            "island_link_contacts",
            &assemble_shader(include_str!("shaders/island_link_contacts.wgsl")),
            &[
                (BindingKind::ReadOnlyStorage, &buffers.bodies),
                (BindingKind::ReadOnlyStorage, &buffers.contacts),
                (BindingKind::ReadWriteStorage, &buffers.contact_count),
                (BindingKind::ReadWriteStorage, &buffers.island_parents),
                (BindingKind::ReadWriteStorage, &buffers.wake_flags),
            ],
            &shape_resources,
            buffers.contact_capacity(),
            WORKGROUP_SIZE,
        );

        let island_link_constraints = Stage::build(
            context,
            "island_link_constraints",
            &assemble_shader(include_str!("shaders/island_link_constraints.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadOnlyStorage, &buffers.bodies),
                (BindingKind::ReadOnlyStorage, &buffers.constraints),
                (BindingKind::ReadWriteStorage, &buffers.island_parents),
                (BindingKind::ReadWriteStorage, &buffers.wake_flags),
            ],
            &shape_resources,
            buffers.constraint_capacity(),
            WORKGROUP_SIZE,
        );

        let island_jump = Stage::build(
            context,
            "island_jump",
            &assemble_shader(include_str!("shaders/island_jump.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadWriteStorage, &buffers.island_parents),
            ],
            &shape_resources,
            body_capacity,
            WORKGROUP_SIZE,
        );

        let island_aggregate = Stage::build(
            context,
            "island_aggregate",
            &assemble_shader(include_str!("shaders/island_aggregate.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadOnlyStorage, &buffers.bodies),
                (BindingKind::ReadWriteStorage, &buffers.island_parents),
                (BindingKind::ReadWriteStorage, &buffers.island_state),
                (BindingKind::ReadWriteStorage, &buffers.wake_flags),
            ],
            &shape_resources,
            body_capacity,
            WORKGROUP_SIZE,
        );

        let island_broadcast = Stage::build(
            context,
            "island_broadcast",
            &assemble_shader(include_str!("shaders/island_broadcast.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadWriteStorage, &buffers.bodies),
                (BindingKind::ReadWriteStorage, &buffers.island_parents),
                (BindingKind::ReadWriteStorage, &buffers.island_state),
                (BindingKind::ReadWriteStorage, &buffers.wake_flags),
            ],
            &shape_resources,
            body_capacity,
            WORKGROUP_SIZE,
        );

        let gather_contact_keys_b = Stage::build(
            context,
            "gather_contact_keys_b",
            &assemble_shader(include_str!("shaders/gather_contact_keys_b.wgsl")),
            &[
                (BindingKind::ReadOnlyStorage, &buffers.contacts),
                (BindingKind::ReadOnlyStorage, &buffers.contact_count),
                (BindingKind::ReadWriteStorage, &buffers.contact_b_keys),
                (BindingKind::ReadWriteStorage, &buffers.contact_b_values),
            ],
            &shape_resources,
            buffers.contact_capacity(),
            WORKGROUP_SIZE,
        );

        let gather_constraint_keys = Stage::build(
            context,
            "gather_constraint_keys",
            &assemble_shader(include_str!("shaders/gather_constraint_keys.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadOnlyStorage, &buffers.constraints),
                (
                    BindingKind::ReadWriteStorage,
                    &buffers.constraint_gather_a_keys,
                ),
                (
                    BindingKind::ReadWriteStorage,
                    &buffers.constraint_gather_a_values,
                ),
                (
                    BindingKind::ReadWriteStorage,
                    &buffers.constraint_gather_b_keys,
                ),
                (
                    BindingKind::ReadWriteStorage,
                    &buffers.constraint_gather_b_values,
                ),
            ],
            &shape_resources,
            buffers.constraint_capacity(),
            WORKGROUP_SIZE,
        );

        let reset_gather_boundaries = Stage::build(
            context,
            "reset_gather_boundaries",
            &assemble_shader(include_str!("shaders/reset_gather_boundaries.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadWriteStorage, &buffers.contact_first_a),
                (BindingKind::ReadWriteStorage, &buffers.contact_first_b),
                (BindingKind::ReadWriteStorage, &buffers.constraint_first_a),
                (BindingKind::ReadWriteStorage, &buffers.constraint_first_b),
            ],
            &shape_resources,
            body_capacity,
            WORKGROUP_SIZE,
        );

        let mark_contact_boundaries = Stage::build(
            context,
            "mark_contact_boundaries",
            &assemble_shader(include_str!("shaders/mark_contact_boundaries.wgsl")),
            &[
                (BindingKind::ReadOnlyStorage, &buffers.contact_a_body),
                (BindingKind::ReadOnlyStorage, &buffers.contact_b_keys_out),
                (BindingKind::ReadOnlyStorage, &buffers.contact_count),
                (BindingKind::ReadWriteStorage, &buffers.contact_first_a),
                (BindingKind::ReadWriteStorage, &buffers.contact_first_b),
            ],
            &shape_resources,
            buffers.contact_capacity(),
            WORKGROUP_SIZE,
        );

        let mark_constraint_boundaries = Stage::build(
            context,
            "mark_constraint_boundaries",
            &assemble_shader(include_str!("shaders/mark_constraint_boundaries.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (
                    BindingKind::ReadOnlyStorage,
                    &buffers.constraint_gather_a_keys_out,
                ),
                (
                    BindingKind::ReadOnlyStorage,
                    &buffers.constraint_gather_b_keys_out,
                ),
                (BindingKind::ReadWriteStorage, &buffers.constraint_first_a),
                (BindingKind::ReadWriteStorage, &buffers.constraint_first_b),
            ],
            &shape_resources,
            buffers.constraint_capacity(),
            WORKGROUP_SIZE,
        );

        let ccd_sweep = Stage::build(
            context,
            "ccd_sweep",
            &assemble_shader(include_str!("shaders/ccd_sweep.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadWriteStorage, &buffers.bodies),
                (BindingKind::ReadOnlyStorage, &buffers.colliders),
                (BindingKind::ReadOnlyStorage, &buffers.pairs.keys_hi),
                (BindingKind::ReadOnlyStorage, &buffers.pairs.keys_lo),
                (BindingKind::ReadWriteStorage, &buffers.pair_count),
            ],
            &shape_resources,
            buffers.pair_capacity(),
            WORKGROUP_SIZE,
        );

        let contact_match = Stage::build(
            context,
            "contact_match",
            &assemble_shader(include_str!("shaders/contact_match.wgsl")),
            &[
                (BindingKind::ReadWriteStorage, &buffers.contacts),
                (BindingKind::ReadOnlyStorage, &buffers.prev_contacts),
                (BindingKind::ReadOnlyStorage, &buffers.prev_contact_count),
                (BindingKind::ReadOnlyStorage, &buffers.contact_count),
                (BindingKind::ReadWriteStorage, &buffers.events),
                (BindingKind::ReadWriteStorage, &buffers.event_count),
                (BindingKind::ReadWriteStorage, &buffers.overflow_flags),
            ],
            &shape_resources,
            buffers.contact_capacity(),
            WORKGROUP_SIZE,
        );

        let contact_solve_extract = Stage::build(
            context,
            "contact_solve_extract",
            &assemble_shader(include_str!("shaders/contact_solve_extract.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadOnlyStorage, &buffers.bodies),
                (BindingKind::ReadWriteStorage, &buffers.contacts),
                (BindingKind::ReadOnlyStorage, &buffers.contact_count),
                (BindingKind::ReadWriteStorage, &buffers.wake_flags),
                (BindingKind::ReadWriteStorage, &buffers.contact_deltas),
            ],
            &shape_resources,
            buffers.contact_capacity(),
            WORKGROUP_SIZE,
        );

        let constraint_solve_extract = Stage::build(
            context,
            "constraint_solve_extract",
            &assemble_shader(include_str!("shaders/constraint_solve_extract.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadOnlyStorage, &buffers.bodies),
                (BindingKind::ReadWriteStorage, &buffers.constraints),
                (BindingKind::ReadWriteStorage, &buffers.wake_flags),
                (BindingKind::ReadWriteStorage, &buffers.constraint_deltas),
            ],
            &shape_resources,
            buffers.constraint_capacity(),
            WORKGROUP_SIZE,
        );

        let body_apply_solver = Stage::build(
            context,
            "body_apply_solver",
            &assemble_shader(include_str!("shaders/body_apply_solver.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadWriteStorage, &buffers.bodies),
                (BindingKind::ReadOnlyStorage, &buffers.contact_first_a),
                (BindingKind::ReadOnlyStorage, &buffers.contact_first_b),
                (BindingKind::ReadOnlyStorage, &buffers.contact_a_body),
                (BindingKind::ReadOnlyStorage, &buffers.contact_b_keys_out),
                (BindingKind::ReadOnlyStorage, &buffers.contact_b_values_out),
                (BindingKind::ReadOnlyStorage, &buffers.constraint_first_a),
                (BindingKind::ReadOnlyStorage, &buffers.constraint_first_b),
                (
                    BindingKind::ReadOnlyStorage,
                    &buffers.constraint_gather_a_keys_out,
                ),
                (
                    BindingKind::ReadOnlyStorage,
                    &buffers.constraint_gather_a_values_out,
                ),
                (
                    BindingKind::ReadOnlyStorage,
                    &buffers.constraint_gather_b_keys_out,
                ),
                (
                    BindingKind::ReadOnlyStorage,
                    &buffers.constraint_gather_b_values_out,
                ),
                (BindingKind::ReadOnlyStorage, &buffers.contact_count),
                (BindingKind::ReadOnlyStorage, &buffers.contact_deltas),
                (BindingKind::ReadOnlyStorage, &buffers.constraint_deltas),
            ],
            &shape_resources,
            body_capacity,
            WORKGROUP_SIZE,
        );

        let position_solve_extract = Stage::build(
            context,
            "position_solve_extract",
            &assemble_shader(include_str!("shaders/position_solve_extract.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadOnlyStorage, &buffers.bodies),
                (BindingKind::ReadOnlyStorage, &buffers.contacts),
                (BindingKind::ReadOnlyStorage, &buffers.contact_count),
                (BindingKind::ReadWriteStorage, &buffers.contact_deltas),
            ],
            &shape_resources,
            buffers.contact_capacity(),
            WORKGROUP_SIZE,
        );

        let body_apply_positions = Stage::build(
            context,
            "body_apply_positions",
            &assemble_shader(include_str!("shaders/body_apply_positions.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadWriteStorage, &buffers.bodies),
                (BindingKind::ReadOnlyStorage, &buffers.contact_first_a),
                (BindingKind::ReadOnlyStorage, &buffers.contact_first_b),
                (BindingKind::ReadOnlyStorage, &buffers.contact_a_body),
                (BindingKind::ReadOnlyStorage, &buffers.contact_b_keys_out),
                (BindingKind::ReadOnlyStorage, &buffers.contact_b_values_out),
                (BindingKind::ReadOnlyStorage, &buffers.contact_count),
                (BindingKind::ReadOnlyStorage, &buffers.contact_deltas),
            ],
            &shape_resources,
            body_capacity,
            WORKGROUP_SIZE,
        );

        let query = Stage::build(
            context,
            "query",
            &assemble_shader(include_str!("shaders/queries.wgsl")),
            &[
                (BindingKind::ReadOnlyStorage, &buffers.queries),
                (BindingKind::ReadOnlyStorage, &buffers.bodies),
                (BindingKind::ReadOnlyStorage, &buffers.colliders),
                (BindingKind::ReadOnlyStorage, &buffers.aabbs),
                (BindingKind::ReadOnlyStorage, &buffers.entries.keys_hi),
                (BindingKind::ReadOnlyStorage, &buffers.entries.keys_lo),
                (BindingKind::ReadOnlyStorage, &buffers.entry_count),
                (BindingKind::ReadWriteStorage, &buffers.query_headers),
                (BindingKind::ReadWriteStorage, &buffers.query_hits),
                (BindingKind::ReadOnlyStorage, &buffers.large_bodies),
                (BindingKind::ReadOnlyStorage, &buffers.large_count),
                (BindingKind::Uniform, &buffers.params),
            ],
            &shape_resources,
            1,
            WORKGROUP_SIZE,
        );

        let constraints_warm_end = Stage::build(
            context,
            "constraints_warm_end",
            &assemble_shader(include_str!("shaders/constraints_warm_end.wgsl")),
            &[(BindingKind::ReadWriteStorage, &buffers.constraints)],
            &shape_resources,
            buffers.constraint_capacity(),
            WORKGROUP_SIZE,
        );

        let static_wake_clear = Stage::build(
            context,
            "static_wake_clear",
            &assemble_shader(include_str!("shaders/static_wake_clear.wgsl")),
            &[
                (BindingKind::Uniform, &buffers.params),
                (BindingKind::ReadWriteStorage, &buffers.wake_flags),
            ],
            &shape_resources,
            body_capacity,
            WORKGROUP_SIZE,
        );

        let sort = RadixSort::new(context, "sim sort", buffers.sort_scratch.capacity_u32());
        let sort_hi = GpuBuffer::new(
            &device,
            "sort scratch hi",
            buffers.sort_scratch.keys_hi.size(),
            wgpu::BufferUsages::STORAGE,
        );
        let sort_lo = GpuBuffer::new(
            &device,
            "sort scratch lo",
            buffers.sort_scratch.keys_lo.size(),
            wgpu::BufferUsages::STORAGE,
        );
        let sort_values = GpuBuffer::new(
            &device,
            "sort scratch values",
            buffers.sort_scratch.values.size(),
            wgpu::BufferUsages::STORAGE,
        );
        let contact_bucket = BucketSort::new(
            context,
            "contact bucket",
            body_capacity,
            buffers.contact_capacity(),
        );
        let constraint_bucket = BucketSort::new(
            context,
            "constraint bucket",
            body_capacity,
            buffers.constraint_capacity(),
        );
        #[cfg(feature = "profile")]
        let timer = context.supports_pass_timing().then(|| {
            GpuTimer::new(
                context.device(),
                SIM_PASSES,
                context.timestamp_period_ns(),
                "dynamis step",
            )
        });
        Self {
            device,
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
            constraints_warm_end,
            static_wake_clear,
            sort,
            sort_hi,
            sort_lo,
            sort_values,
            contact_bucket,
            constraint_bucket,
            #[cfg(feature = "profile")]
            timer,
        }
    }

    fn open<'a>(&'a self, encoder: &'a mut CommandEncoder, slot: usize) -> ComputeRecorder<'a> {
        let label = SIM_PASSES[slot];
        #[cfg(feature = "profile")]
        if let Some(timer) = &self.timer {
            return ComputeRecorder::begin_timed(encoder, label, Some(timer.writes(slot)));
        }
        ComputeRecorder::begin(encoder, label)
    }

    pub(crate) fn encode(
        &self,
        encoder: &mut CommandEncoder,
        buffers: &WorldBuffers,
        params: &FrameParams,
    ) {
        let constraint_active = params.constraint_count > 0;
        let collider_words = key_bytes(buffers.collider_capacity());
        let joint_words = key_bytes(buffers.constraint_capacity());

        let mut commands = self.open(encoder, PASS_COMMANDS);
        self.apply_commands.dispatch(&mut commands);
        self.apply_constraint_commands.dispatch(&mut commands);
        if constraint_active {
            self.joint_filter.dispatch(&mut commands);
            let channels = SortChannels {
                count: &buffers.joint_count,
                keys_lo: &buffers.joint_lo,
                keys_hi: &buffers.joint_hi,
                values: &buffers.entries.values,
                scratch_lo: &self.sort_lo,
                scratch_hi: &self.sort_hi,
                scratch_values: &self.sort_values,
            };
            self.sort.sort(
                &self.device,
                &mut commands,
                &channels,
                joint_words,
                joint_words,
            );
        }
        drop(commands);

        let mut integrate = self.open(encoder, PASS_INTEGRATE);
        self.integrate
            .dispatch_at(&mut integrate, params.dynamic_count);
        self.broadphase_aabb
            .dispatch_at(&mut integrate, params.dynamic_count);
        drop(integrate);

        let mut broadphase = self.open(encoder, PASS_BROADPHASE);
        self.grid_entries
            .dispatch_at(&mut broadphase, params.body_count);
        let channels = SortChannels {
            count: &buffers.entry_count,
            keys_lo: &buffers.entries.keys_lo,
            keys_hi: &buffers.entries.keys_hi,
            values: &buffers.entries.values,
            scratch_lo: &self.sort_lo,
            scratch_hi: &self.sort_hi,
            scratch_values: &self.sort_values,
        };
        self.sort
            .sort(&self.device, &mut broadphase, &channels, collider_words, 4);
        self.broadphase_pairs.dispatch(&mut broadphase);
        self.large_pairs
            .dispatch_at(&mut broadphase, params.body_count);
        let channels = SortChannels {
            count: &buffers.pair_count,
            keys_lo: &buffers.pairs.keys_lo,
            keys_hi: &buffers.pairs.keys_hi,
            values: &buffers.pairs.values,
            scratch_lo: &self.sort_lo,
            scratch_hi: &self.sort_hi,
            scratch_values: &self.sort_values,
        };
        self.sort.sort(
            &self.device,
            &mut broadphase,
            &channels,
            collider_words,
            collider_words,
        );
        self.ccd_sweep.dispatch(&mut broadphase);
        drop(broadphase);

        let mut narrowphase = self.open(encoder, PASS_NARROWPHASE);
        self.narrowphase.dispatch(&mut narrowphase);
        self.compact_scan.dispatch(&mut narrowphase);
        self.compact_offsets.dispatch(&mut narrowphase);
        self.compact_scatter.dispatch(&mut narrowphase);
        self.contact_match.dispatch(&mut narrowphase);
        drop(narrowphase);

        let mut islands = self.open(encoder, PASS_ISLANDS);
        self.island_init
            .dispatch_at(&mut islands, params.dynamic_count);
        self.island_link_contacts.dispatch(&mut islands);
        if constraint_active {
            self.island_link_constraints.dispatch(&mut islands);
        }
        for _ in 0..params.island_rounds {
            self.island_jump
                .dispatch_at(&mut islands, params.dynamic_count);
        }
        self.gather_contact_keys_b.dispatch(&mut islands);
        self.contact_bucket.sort(
            &self.device,
            &mut islands,
            &BucketChannels {
                count: &buffers.contact_count,
                keys: &buffers.contact_b_keys,
                values: &buffers.contact_b_values,
                keys_out: &buffers.contact_b_keys_out,
                values_out: &buffers.contact_b_values_out,
            },
        );
        if constraint_active {
            self.gather_constraint_keys.dispatch(&mut islands);
            self.constraint_bucket.sort(
                &self.device,
                &mut islands,
                &BucketChannels {
                    count: &buffers.constraint_count_state,
                    keys: &buffers.constraint_gather_a_keys,
                    values: &buffers.constraint_gather_a_values,
                    keys_out: &buffers.constraint_gather_a_keys_out,
                    values_out: &buffers.constraint_gather_a_values_out,
                },
            );
            self.constraint_bucket.sort(
                &self.device,
                &mut islands,
                &BucketChannels {
                    count: &buffers.constraint_count_state,
                    keys: &buffers.constraint_gather_b_keys,
                    values: &buffers.constraint_gather_b_values,
                    keys_out: &buffers.constraint_gather_b_keys_out,
                    values_out: &buffers.constraint_gather_b_values_out,
                },
            );
        }
        self.reset_gather_boundaries
            .dispatch_at(&mut islands, params.dynamic_count);
        self.mark_contact_boundaries.dispatch(&mut islands);
        if constraint_active {
            self.mark_constraint_boundaries.dispatch(&mut islands);
        }
        drop(islands);

        let mut sleep = self.open(encoder, PASS_SLEEP);
        self.island_aggregate
            .dispatch_at(&mut sleep, params.dynamic_count);
        self.island_broadcast
            .dispatch_at(&mut sleep, params.dynamic_count);
        drop(sleep);

        let mut velocity_solve = self.open(encoder, PASS_VELOCITY_SOLVE);
        for _ in 0..params.solve_iterations {
            if constraint_active {
                self.constraint_solve_extract.dispatch(&mut velocity_solve);
            }
            self.contact_solve_extract.dispatch(&mut velocity_solve);
            self.body_apply_solver
                .dispatch_at(&mut velocity_solve, params.dynamic_count);
        }
        drop(velocity_solve);

        let mut position_solve = self.open(encoder, PASS_POSITION_SOLVE);
        for _ in 0..params.position_iterations {
            self.position_solve_extract.dispatch(&mut position_solve);
            self.body_apply_positions
                .dispatch_at(&mut position_solve, params.dynamic_count);
        }
        drop(position_solve);

        let mut tail = self.open(encoder, PASS_TAIL);
        self.events_end.dispatch(&mut tail);
        self.contact_archive.dispatch(&mut tail);
        self.prev_count_sync.dispatch(&mut tail);
        self.constraints_warm_end.dispatch(&mut tail);
        self.static_wake_clear
            .dispatch_at(&mut tail, params.body_count);
        if params.query_count > 0 {
            self.query
                .dispatch_workgroups(&mut tail, params.query_count);
        }
        drop(tail);
    }

    #[cfg(feature = "profile")]
    pub(crate) fn capture_timings(
        &mut self,
        encoder: &mut CommandEncoder,
        sequence: u64,
    ) -> Option<(u64, Vec<dynamis_gpu::GpuPassTiming>)> {
        self.timer
            .as_mut()
            .and_then(|timer| timer.capture(&self.device, encoder, sequence))
    }

    #[cfg(feature = "profile")]
    pub(crate) fn poll_timings(
        &mut self,
        device: &Device,
    ) -> Vec<(u64, Vec<dynamis_gpu::GpuPassTiming>)> {
        match &mut self.timer {
            Some(timer) => timer.poll(device),
            None => Vec::new(),
        }
    }

    #[cfg(feature = "profile")]
    pub(crate) fn arm_timings(&mut self) {
        if let Some(timer) = &mut self.timer {
            timer.arm();
        }
    }

    pub(crate) fn encode_queries(&self, encoder: &mut CommandEncoder, query_count: u32) {
        let mut recorder = ComputeRecorder::begin(encoder, "query flush");
        self.query.dispatch_workgroups(&mut recorder, query_count);
    }
}

fn key_bytes(values: u32) -> u32 {
    let bits = 32 - values.saturating_sub(1).leading_zeros();
    bits.div_ceil(8).clamp(1, 4)
}
