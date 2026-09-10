use crate::buffers::{COMPACT_BLOCK, WorldBuffers};
use crate::reservation::Reservation;
#[cfg(feature = "profile")]
use dynamis_gpu::GpuTimer;
use dynamis_gpu::{
    BindingKind, BindingSpec, ComputePipeline, ComputeRecorder, DispatchTable, GpuBuffer,
    GpuContext, GpuReadback, GpuSlot,
};
use dynamis_kernel::{RadixSort, SortChannels, key_words};
use dynamis_layout::{
    BODY_CCD, BODY_KINEMATIC, COLLIDER_EVENT_BEGIN_END, COLLIDER_EVENT_PERSIST, COLLIDER_SENSOR,
    COMMAND_ADD, COMMAND_ANGULAR_IMPULSE, COMMAND_CONSTRAINT_ADD, COMMAND_CONSTRAINT_SWAP,
    COMMAND_FORCE, COMMAND_FORCE_AT_POINT, COMMAND_IMPULSE, COMMAND_IMPULSE_AT_POINT,
    COMMAND_PATCH, COMMAND_REMOVE, COMMAND_SLEEP, COMMAND_SWAP, COMMAND_TORQUE, COMMAND_WAKE,
    CONSTRAINT_BALL, CONSTRAINT_CONE, CONSTRAINT_DISABLE_COLLISIONS, CONSTRAINT_DISTANCE,
    CONSTRAINT_FIXED, CONSTRAINT_GEAR, CONSTRAINT_HAS_BREAK, CONSTRAINT_HAS_LIMIT,
    CONSTRAINT_HAS_MOTOR, CONSTRAINT_HAS_SWING, CONSTRAINT_IS_SPRING, CONSTRAINT_PRISMATIC,
    CONSTRAINT_PULLEY, CONSTRAINT_REVOLUTE, CONSTRAINT_SIXDOF, CONSTRAINT_WARM_START,
    CONTACT_MAX_POINTS, COUNTER_CONSTRAINTS, COUNTER_CONTACTS, COUNTER_ENTRIES, COUNTER_EVENTS,
    COUNTER_JOINTS, COUNTER_LARGE, COUNTER_PAIRS, COUNTER_PREV_CONTACTS, COUNTER_SPILLOVER_ENTRIES,
    COUNTER_SPILLOVER_EVENTS, COUNTER_SPILLOVER_PAIRS, DOF_DRIVEN, DOF_FREE, DOF_LIMITED,
    DOF_LOCKED, EVENT_BEGIN, EVENT_END, EVENT_PERSIST, FILTER_IGNORE_KINEMATIC,
    FILTER_IGNORE_SENSORS, FILTER_IGNORE_SLEEPING, FILTER_IGNORE_STATIC, ISLAND_ACTIVE,
    ISLAND_WAKE, MAX_CELLS_PER_COLLIDER, MAX_HITS_PER_QUERY, NO_BODY, NO_COLLISION_FILTER, NO_HIT,
    OVERRIDE_SLEEP_ANGULAR, OVERRIDE_SLEEP_LINEAR, PATCH_ANGULAR_VELOCITY, PATCH_ORIENTATION,
    PATCH_POSITION, PATCH_VELOCITY, QUERY_CONVEX, QUERY_CUBOID, QUERY_POINT, QUERY_RAY,
    QUERY_SPHERE, QUERY_SWEEP, SHAPE_CAPSULE, SHAPE_CUBOID, SHAPE_CYLINDER, SHAPE_HEIGHTFIELD,
    SHAPE_HULL, SHAPE_MESH, SHAPE_NONE, SHAPE_PLANE, SHAPE_SPHERE, SHAPE_TRIANGLE, dispatch,
};
use dynamis_model::MAX_COLLIDERS_PER_BODY;
use wgpu::{BindGroup, BindGroupEntry, CommandEncoder};

const WORKGROUP_SIZE: u32 = 64;

pub(crate) const SIM_PASSES: &[&str] = &[
    "commands",
    "integrate",
    "grid",
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
const PASS_GRID: usize = 2;
const PASS_BROADPHASE: usize = 3;
const PASS_NARROWPHASE: usize = 4;
const PASS_ISLANDS: usize = 5;
const PASS_SLEEP: usize = 6;
const PASS_VELOCITY_SOLVE: usize = 7;
const PASS_POSITION_SOLVE: usize = 8;
const PASS_TAIL: usize = 9;

fn shader_constants(per_row: u32) -> String {
    let mut source = String::new();
    let mut emit = |name: &str, value: u32| {
        source.push_str(&format!("const {name}: u32 = {value}u;\n"));
    };
    emit("WORKGROUP_SIZE", WORKGROUP_SIZE);
    emit("WORKGROUPS_PER_ROW", per_row);
    emit("COMMAND_ADD", COMMAND_ADD);
    emit("COMMAND_REMOVE", COMMAND_REMOVE);
    emit("COMMAND_PATCH", COMMAND_PATCH);
    emit("COMMAND_FORCE", COMMAND_FORCE);
    emit("COMMAND_FORCE_AT_POINT", COMMAND_FORCE_AT_POINT);
    emit("COMMAND_TORQUE", COMMAND_TORQUE);
    emit("COMMAND_IMPULSE", COMMAND_IMPULSE);
    emit("COMMAND_IMPULSE_AT_POINT", COMMAND_IMPULSE_AT_POINT);
    emit("COMMAND_ANGULAR_IMPULSE", COMMAND_ANGULAR_IMPULSE);
    emit("COMMAND_CONSTRAINT_ADD", COMMAND_CONSTRAINT_ADD);
    emit("COMMAND_CONSTRAINT_SWAP", COMMAND_CONSTRAINT_SWAP);
    emit("COMMAND_SWAP", COMMAND_SWAP);
    emit("COMMAND_SLEEP", COMMAND_SLEEP);
    emit("COMMAND_WAKE", COMMAND_WAKE);
    emit("PATCH_POSITION", PATCH_POSITION);
    emit("PATCH_VELOCITY", PATCH_VELOCITY);
    emit("PATCH_ORIENTATION", PATCH_ORIENTATION);
    emit("PATCH_ANGULAR_VELOCITY", PATCH_ANGULAR_VELOCITY);
    emit("NO_COLLISION_FILTER", NO_COLLISION_FILTER);
    emit("SHAPE_NONE", SHAPE_NONE);
    emit("SHAPE_SPHERE", SHAPE_SPHERE);
    emit("SHAPE_CUBOID", SHAPE_CUBOID);
    emit("SHAPE_CAPSULE", SHAPE_CAPSULE);
    emit("SHAPE_CYLINDER", SHAPE_CYLINDER);
    emit("SHAPE_HULL", SHAPE_HULL);
    emit("SHAPE_MESH", SHAPE_MESH);
    emit("SHAPE_HEIGHTFIELD", SHAPE_HEIGHTFIELD);
    emit("SHAPE_PLANE", SHAPE_PLANE);
    emit("SHAPE_TRIANGLE", SHAPE_TRIANGLE);
    emit("BODY_KINEMATIC", BODY_KINEMATIC);
    emit("BODY_CCD", BODY_CCD);
    emit("OVERRIDE_SLEEP_LINEAR", OVERRIDE_SLEEP_LINEAR);
    emit("OVERRIDE_SLEEP_ANGULAR", OVERRIDE_SLEEP_ANGULAR);
    emit("COLLIDER_SENSOR", COLLIDER_SENSOR);
    emit("COLLIDER_EVENT_BEGIN_END", COLLIDER_EVENT_BEGIN_END);
    emit("COLLIDER_EVENT_PERSIST", COLLIDER_EVENT_PERSIST);
    emit("ISLAND_WAKE", ISLAND_WAKE);
    emit("ISLAND_ACTIVE", ISLAND_ACTIVE);
    emit("CONTACT_MAX_POINTS", CONTACT_MAX_POINTS);
    emit("CONSTRAINT_BALL", CONSTRAINT_BALL);
    emit("CONSTRAINT_DISTANCE", CONSTRAINT_DISTANCE);
    emit("CONSTRAINT_REVOLUTE", CONSTRAINT_REVOLUTE);
    emit("CONSTRAINT_PRISMATIC", CONSTRAINT_PRISMATIC);
    emit("CONSTRAINT_FIXED", CONSTRAINT_FIXED);
    emit("CONSTRAINT_GEAR", CONSTRAINT_GEAR);
    emit("CONSTRAINT_PULLEY", CONSTRAINT_PULLEY);
    emit("CONSTRAINT_CONE", CONSTRAINT_CONE);
    emit("CONSTRAINT_SIXDOF", CONSTRAINT_SIXDOF);
    emit(
        "CONSTRAINT_DISABLE_COLLISIONS",
        CONSTRAINT_DISABLE_COLLISIONS,
    );
    emit("CONSTRAINT_HAS_LIMIT", CONSTRAINT_HAS_LIMIT);
    emit("CONSTRAINT_HAS_MOTOR", CONSTRAINT_HAS_MOTOR);
    emit("CONSTRAINT_IS_SPRING", CONSTRAINT_IS_SPRING);
    emit("CONSTRAINT_HAS_SWING", CONSTRAINT_HAS_SWING);
    emit("CONSTRAINT_HAS_BREAK", CONSTRAINT_HAS_BREAK);
    emit("CONSTRAINT_WARM_START", CONSTRAINT_WARM_START);
    emit("DOF_FREE", DOF_FREE);
    emit("DOF_LOCKED", DOF_LOCKED);
    emit("DOF_LIMITED", DOF_LIMITED);
    emit("DOF_DRIVEN", DOF_DRIVEN);
    emit("QUERY_RAY", QUERY_RAY);
    emit("QUERY_SPHERE", QUERY_SPHERE);
    emit("QUERY_CUBOID", QUERY_CUBOID);
    emit("QUERY_SWEEP", QUERY_SWEEP);
    emit("QUERY_POINT", QUERY_POINT);
    emit("QUERY_CONVEX", QUERY_CONVEX);
    emit("FILTER_IGNORE_SENSORS", FILTER_IGNORE_SENSORS);
    emit("FILTER_IGNORE_SLEEPING", FILTER_IGNORE_SLEEPING);
    emit("FILTER_IGNORE_STATIC", FILTER_IGNORE_STATIC);
    emit("FILTER_IGNORE_KINEMATIC", FILTER_IGNORE_KINEMATIC);
    emit("EVENT_BEGIN", EVENT_BEGIN);
    emit("EVENT_END", EVENT_END);
    emit("EVENT_PERSIST", EVENT_PERSIST);
    emit("NO_BODY", NO_BODY);
    emit("MAX_CELLS_PER_COLLIDER", MAX_CELLS_PER_COLLIDER);
    emit("MAX_COLLIDERS_PER_BODY", MAX_COLLIDERS_PER_BODY as u32);
    emit("MAX_HITS_PER_QUERY", MAX_HITS_PER_QUERY);
    emit("EVENT_SLOTS", GpuReadback::DEPTH as u32);
    emit("COMPACT_BLOCK", COMPACT_BLOCK);
    emit(
        "COUNTER_STRIDE_WORDS",
        (dynamis_layout::COUNTER_STRIDE / 4) as u32,
    );
    emit("COUNTER_ENTRIES", dynamis_layout::COUNTER_ENTRIES as u32);
    emit("COUNTER_PAIRS", dynamis_layout::COUNTER_PAIRS as u32);
    emit("COUNTER_LARGE", dynamis_layout::COUNTER_LARGE as u32);
    emit("COUNTER_CONTACTS", dynamis_layout::COUNTER_CONTACTS as u32);
    emit("COUNTER_JOINTS", dynamis_layout::COUNTER_JOINTS as u32);
    emit("COUNTER_EVENTS", dynamis_layout::COUNTER_EVENTS as u32);
    emit(
        "COUNTER_SPILLOVER_PAIRS",
        dynamis_layout::COUNTER_SPILLOVER_PAIRS as u32,
    );
    emit(
        "COUNTER_SPILLOVER_EVENTS",
        dynamis_layout::COUNTER_SPILLOVER_EVENTS as u32,
    );
    emit(
        "COUNTER_SPILLOVER_ENTRIES",
        dynamis_layout::COUNTER_SPILLOVER_ENTRIES as u32,
    );
    source.push_str(&format!("const NO_HIT: f32 = {NO_HIT:e};\n"));
    source
}

pub(crate) const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
pub(crate) const SHAPES_FRAGMENT: &str = include_str!("shaders/shapes.wgsl");

fn assemble_shader(body: &str, per_row: u32) -> String {
    format!(
        "{COMMON_SHADER}\n{body}\n{SHAPES_FRAGMENT}\n{}",
        shader_constants(per_row)
    )
}

fn whole(buffer: &GpuBuffer) -> GpuSlot<'_> {
    GpuSlot::whole(buffer)
}

const RO: BindingKind = BindingKind::ReadOnlyStorage;
const RW: BindingKind = BindingKind::ReadWriteStorage;
const UNIFORM: BindingKind = BindingKind::Uniform;

struct Stage {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    shapes_group: Option<BindGroup>,
}

impl Stage {
    fn build(
        context: &GpuContext,
        label: &str,
        shader: &str,
        bindings: &[(BindingKind, GpuSlot)],
        shape_resources: &[&GpuBuffer],
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
        let groups = if shape_specs.is_empty() {
            vec![&specs[..]]
        } else {
            vec![&specs[..], &shape_specs[..]]
        };
        let pipeline = context.compute_pipeline(label, shader, "main", &groups, WORKGROUP_SIZE);
        let entries: Vec<BindGroupEntry> = bindings
            .iter()
            .enumerate()
            .map(|(position, (_, slot))| BindGroupEntry {
                binding: position as u32,
                resource: slot.as_binding(),
            })
            .collect();
        let bind_group = pipeline.create_bind_group(context.device(), 0, &entries);
        let shapes_group = if shape_resources.is_empty() {
            None
        } else {
            let entries = shape_resources
                .iter()
                .enumerate()
                .map(|(position, buffer)| BindGroupEntry {
                    binding: position as u32,
                    resource: buffer.as_binding(),
                })
                .collect::<Vec<_>>();
            Some(pipeline.create_bind_group(context.device(), 1, &entries))
        };
        Self {
            pipeline,
            bind_group,
            shapes_group,
        }
    }

    /// Dispatches one workgroup per `elements` lanes.
    fn record(&self, recorder: &mut ComputeRecorder, elements: u32) {
        let workgroups = elements.div_ceil(WORKGROUP_SIZE);
        match &self.shapes_group {
            Some(shapes) => {
                recorder.record(&self.pipeline, &[&self.bind_group, shapes], workgroups)
            }
            None => recorder.record(&self.pipeline, &[&self.bind_group], workgroups),
        }
    }

    fn record_workgroups(&self, recorder: &mut ComputeRecorder, workgroups: u32) {
        match &self.shapes_group {
            Some(shapes) => {
                recorder.record(&self.pipeline, &[&self.bind_group, shapes], workgroups)
            }
            None => recorder.record(&self.pipeline, &[&self.bind_group], workgroups),
        }
    }

    fn record_indirect(&self, recorder: &mut ComputeRecorder, table: &DispatchTable, slot: u32) {
        match &self.shapes_group {
            Some(shapes) => {
                recorder.record_indirect(&self.pipeline, &[&self.bind_group, shapes], table, slot)
            }
            None => recorder.record_indirect(&self.pipeline, &[&self.bind_group], table, slot),
        }
    }
}

/// The dispatch table the device writes once per step, one slot per indirect stage.
///
/// Sort tiles cover 256 lanes, simulation workgroups 64, compaction blocks 256, so a
/// count becomes a workgroup count of the right density for each consumer.
const SORT_JOINTS: u32 = 0;
const SORT_ENTRIES: u32 = 1;
const SORT_PAIRS: u32 = 2;
const SORT_CONTACTS: u32 = 3;
const SORT_CONSTRAINTS: u32 = 4;
const BROADPHASE_PAIRS: u32 = 5;
const NARROWPHASE: u32 = 6;
const COMPACT_SCAN: u32 = 7;
const COMPACT_SCATTER: u32 = 8;
const CCD_SWEEP: u32 = 9;
const CONTACT_ARCHIVE: u32 = 10;
const ISLAND_LINK_CONTACTS: u32 = 11;
const GATHER_CONTACT_KEYS_B: u32 = 12;
const MARK_CONTACT_BOUNDARIES: u32 = 13;
const CONTACT_MATCH: u32 = 14;
const CONTACT_SOLVE_EXTRACT: u32 = 15;
const POSITION_SOLVE_EXTRACT: u32 = 16;
const EVENTS_END: u32 = 17;

pub(crate) const DISPATCH_SLOTS: u32 = 18;

const KERNEL_TILE: u32 = 256;

/// One table entry: the slot it fills, the counter it reads, the lanes a workgroup
/// covers. A entry point writes one batch, at the point of the step where every
/// counter in it is final.
type DispatchBatch = &'static [(u32, usize, u32)];

const DISPATCH_BATCHES: &[DispatchBatch] = &[
    // After commands: joints were counted, constraints and the previous contact count
    // were already final last step.
    &[
        (SORT_JOINTS, COUNTER_JOINTS, KERNEL_TILE),
        (SORT_CONSTRAINTS, COUNTER_CONSTRAINTS, KERNEL_TILE),
        (EVENTS_END, COUNTER_PREV_CONTACTS, WORKGROUP_SIZE),
    ],
    // After the grid emitted its entries.
    &[
        (SORT_ENTRIES, COUNTER_ENTRIES, KERNEL_TILE),
        (BROADPHASE_PAIRS, COUNTER_ENTRIES, WORKGROUP_SIZE),
    ],
    // After entries were paired and the pair count is final.
    &[
        (SORT_PAIRS, COUNTER_PAIRS, KERNEL_TILE),
        (NARROWPHASE, COUNTER_PAIRS, WORKGROUP_SIZE),
        (COMPACT_SCAN, COUNTER_PAIRS, COMPACT_BLOCK),
        (COMPACT_SCATTER, COUNTER_PAIRS, WORKGROUP_SIZE),
        (CCD_SWEEP, COUNTER_PAIRS, WORKGROUP_SIZE),
    ],
    // After pairs were compacted into contacts and the contact count is final.
    &[
        (SORT_CONTACTS, COUNTER_CONTACTS, KERNEL_TILE),
        (CONTACT_ARCHIVE, COUNTER_CONTACTS, WORKGROUP_SIZE),
        (ISLAND_LINK_CONTACTS, COUNTER_CONTACTS, WORKGROUP_SIZE),
        (GATHER_CONTACT_KEYS_B, COUNTER_CONTACTS, WORKGROUP_SIZE),
        (MARK_CONTACT_BOUNDARIES, COUNTER_CONTACTS, WORKGROUP_SIZE),
        (CONTACT_MATCH, COUNTER_CONTACTS, WORKGROUP_SIZE),
        (CONTACT_SOLVE_EXTRACT, COUNTER_CONTACTS, WORKGROUP_SIZE),
        (POSITION_SOLVE_EXTRACT, COUNTER_CONTACTS, WORKGROUP_SIZE),
    ],
];

/// The shader filling the dispatch table: one entry point per batch, one lane per
/// table slot, a slot only written once its counter is final.
fn dispatch_source() -> String {
    let mut source = String::from(
        "struct Args { per_row: u32, rows: u32, layers: u32, _pad: u32 }\n\
@group(0) @binding(0) var<storage, read> counters: array<u32>;\n\
@group(0) @binding(1) var<storage, read_write> table: array<Args>;\n\
fn write_args(index: u32, counter: u32, lanes: u32) {\n\
    let count = counters[counter * COUNTER_STRIDE_WORDS];\n\
    let workgroups = (count + lanes - 1u) / lanes;\n\
    let per_row = min(workgroups, WORKGROUPS_PER_ROW);\n\
    let rows = (workgroups + WORKGROUPS_PER_ROW - 1u) / WORKGROUPS_PER_ROW;\n\
    table[index] = Args(per_row, rows, 1u, 0u);\n\
}\n\n",
    );
    for (batch, entries) in DISPATCH_BATCHES.iter().enumerate() {
        let entry = ['a', 'b', 'c', 'd'][batch];
        source.push_str(&format!(
            "@compute @workgroup_size(64u)\nfn dispatch_{entry}(@builtin(local_invocation_id) lid: vec3u) {{\n"
        ));
        for (index, (slot, counter, lanes)) in entries.iter().enumerate() {
            source.push_str(&format!(
                "    if (lid.x == {index}u) {{ write_args({slot}u, {counter}u, {lanes}u); }}\n"
            ));
        }
        source.push_str("}\n\n");
    }
    source
}

pub(crate) struct FrameParams {
    pub(crate) dynamic_count: u32,
    pub(crate) body_count: u32,
    pub(crate) solve_iterations: u32,
    pub(crate) position_iterations: u32,
    pub(crate) island_rounds: u32,
    pub(crate) query_count: u32,
    pub(crate) constraint_count: u32,
    /// Whether structural row moves were compiled into the command stream.
    pub(crate) body_structural: bool,
    pub(crate) constraint_structural: bool,
    /// Whether any per-slot body edit was compiled.
    pub(crate) has_body_edits: bool,
}

pub(crate) struct Pipeline {
    per_row: u32,
    reset_counters: Stage,
    body_edits: Stage,
    body_gather: Stage,
    body_scatter: Stage,
    constraint_gather: Stage,
    constraint_scatter: Stage,
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
    dispatch_args: [ComputePipeline; 4],
    dispatch_group: [BindGroup; 4],
    sort: RadixSort,
    #[cfg(feature = "profile")]
    timer: Option<GpuTimer>,
}

impl Pipeline {
    pub(crate) fn new(context: &GpuContext, buffers: &WorldBuffers, plan: &Reservation) -> Self {
        let per_row = context.workgroups_per_row();
        let shape_resources = [
            &buffers.shapes,
            &buffers.shape_vertices,
            &buffers.shape_triangles,
            &buffers.shape_nodes,
        ];
        let reset_counters = Stage::build(
            context,
            "reset_counters",
            &assemble_shader(include_str!("shaders/reset_counters.wgsl"), per_row),
            &[(RW, whole(&buffers.counters))],
            &[],
        );

        let body_edits = Stage::build(
            context,
            "body_edits",
            &assemble_shader(include_str!("shaders/body_edits.wgsl"), per_row),
            &[
                (RO, whole(&buffers.commands)),
                (RO, whole(&buffers.command_first)),
                (RW, whole(&buffers.body_states)),
                (RO, whole(&buffers.body_descs)),
                (RW, whole(&buffers.wake_flags)),
                (UNIFORM, whole(&buffers.params)),
            ],
            &[],
        );

        let body_gather = Stage::build(
            context,
            "body_gather",
            &assemble_shader(include_str!("shaders/body_gather.wgsl"), per_row),
            &[
                (RO, whole(&buffers.body_states)),
                (RW, whole(&buffers.state_scratch)),
                (RO, whole(&buffers.row_src)),
                (RO, whole(&buffers.row_fresh)),
                (RO, whole(&buffers.fresh_states)),
                (UNIFORM, whole(&buffers.params)),
            ],
            &[],
        );
        let body_scatter = Stage::build(
            context,
            "body_scatter",
            &assemble_shader(include_str!("shaders/body_scatter.wgsl"), per_row),
            &[
                (RW, whole(&buffers.body_states)),
                (RO, whole(&buffers.state_scratch)),
                (UNIFORM, whole(&buffers.params)),
            ],
            &[],
        );
        let constraint_gather = Stage::build(
            context,
            "constraint_gather",
            &assemble_shader(include_str!("shaders/constraint_gather.wgsl"), per_row),
            &[
                (RO, whole(&buffers.constraint_runtime)),
                (RW, whole(&buffers.constraint_scratch)),
                (RO, whole(&buffers.constraint_row_src)),
                (RO, whole(&buffers.constraint_row_fresh)),
                (RO, whole(&buffers.constraint_fresh)),
                (UNIFORM, whole(&buffers.params)),
            ],
            &[],
        );
        let constraint_scatter = Stage::build(
            context,
            "constraint_scatter",
            &assemble_shader(include_str!("shaders/constraint_scatter.wgsl"), per_row),
            &[
                (RW, whole(&buffers.constraint_runtime)),
                (RO, whole(&buffers.constraint_scratch)),
                (UNIFORM, whole(&buffers.params)),
            ],
            &[],
        );

        let joint_filter = Stage::build(
            context,
            "joint_filter",
            &assemble_shader(include_str!("shaders/joint_filter.wgsl"), per_row),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RO, whole(&buffers.constraint_descs)),
                (RO, whole(&buffers.constraint_runtime)),
                (RW, whole(&buffers.joint_hi)),
                (RW, whole(&buffers.joint_lo)),
                (RW, buffers.counter(COUNTER_JOINTS)),
            ],
            &[],
        );

        let integrate = Stage::build(
            context,
            "integrate",
            &assemble_shader(include_str!("shaders/integrate.wgsl"), per_row),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RW, whole(&buffers.body_states)),
                (RO, whole(&buffers.body_descs)),
            ],
            &[],
        );

        let broadphase_aabb = Stage::build(
            context,
            "broadphase_aabb",
            &assemble_shader(include_str!("shaders/broadphase_aabb.wgsl"), per_row),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RO, whole(&buffers.body_states)),
                (RO, whole(&buffers.colliders)),
                (RW, whole(&buffers.aabbs)),
            ],
            &shape_resources,
        );

        let grid_entries = Stage::build(
            context,
            "grid_entries",
            &assemble_shader(include_str!("shaders/grid_entries.wgsl"), per_row),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RO, whole(&buffers.aabbs)),
                (RW, whole(&buffers.entries.keys_hi)),
                (RW, whole(&buffers.entries.keys_lo)),
                (RW, buffers.counter(COUNTER_ENTRIES)),
                (RW, whole(&buffers.large_bodies)),
                (RW, buffers.counter(COUNTER_LARGE)),
                (RW, buffers.counter(COUNTER_SPILLOVER_ENTRIES)),
            ],
            &[],
        );

        let broadphase_pairs = Stage::build(
            context,
            "broadphase_pairs",
            &assemble_shader(include_str!("shaders/broadphase_pairs.wgsl"), per_row),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RO, whole(&buffers.entries.keys_hi)),
                (RO, whole(&buffers.entries.keys_lo)),
                (RW, buffers.counter(COUNTER_ENTRIES)),
                (RW, whole(&buffers.pairs.keys_hi)),
                (RW, whole(&buffers.pairs.keys_lo)),
                (RW, buffers.counter(COUNTER_PAIRS)),
                (RW, buffers.counter(COUNTER_SPILLOVER_PAIRS)),
            ],
            &[],
        );

        let large_pairs = Stage::build(
            context,
            "large_pairs",
            &assemble_shader(include_str!("shaders/large_pairs.wgsl"), per_row),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RO, whole(&buffers.large_bodies)),
                (RW, buffers.counter(COUNTER_LARGE)),
                (RW, whole(&buffers.pairs.keys_hi)),
                (RW, whole(&buffers.pairs.keys_lo)),
                (RW, buffers.counter(COUNTER_PAIRS)),
                (RO, whole(&buffers.colliders)),
                (RW, buffers.counter(COUNTER_SPILLOVER_PAIRS)),
            ],
            &[],
        );

        let narrowphase = Stage::build(
            context,
            "narrowphase",
            &assemble_shader(include_str!("shaders/narrowphase.wgsl"), per_row),
            &[
                (RO, whole(&buffers.body_states)),
                (RO, whole(&buffers.body_descs)),
                (RO, whole(&buffers.colliders)),
                (RO, whole(&buffers.pairs.keys_hi)),
                (RO, whole(&buffers.pairs.keys_lo)),
                (RW, whole(&buffers.contacts_raw)),
                (RW, whole(&buffers.contact_valid)),
                (RW, buffers.counter(COUNTER_PAIRS)),
                (RO, whole(&buffers.joint_hi)),
                (RO, whole(&buffers.joint_lo)),
                (RW, buffers.counter(COUNTER_JOINTS)),
                (UNIFORM, whole(&buffers.params)),
            ],
            &shape_resources,
        );

        let compact_scan = Stage::build(
            context,
            "compact_scan",
            &assemble_shader(include_str!("shaders/compact_scan.wgsl"), per_row),
            &[
                (RO, whole(&buffers.contact_valid)),
                (RW, whole(&buffers.compact_ranks)),
                (RW, whole(&buffers.compact_block_sums)),
                (RW, buffers.counter(COUNTER_PAIRS)),
            ],
            &[],
        );

        let compact_offsets = Stage::build(
            context,
            "compact_offsets",
            &assemble_shader(include_str!("shaders/compact_offsets.wgsl"), per_row),
            &[
                (RO, whole(&buffers.compact_block_sums)),
                (RW, whole(&buffers.compact_block_offsets)),
                (RW, buffers.counter(COUNTER_CONTACTS)),
                (RW, buffers.counter(COUNTER_PAIRS)),
            ],
            &[],
        );

        let compact_scatter = Stage::build(
            context,
            "compact_scatter",
            &assemble_shader(include_str!("shaders/compact_scatter.wgsl"), per_row),
            &[
                (RO, whole(&buffers.contacts_raw)),
                (RO, whole(&buffers.contact_valid)),
                (RO, whole(&buffers.compact_ranks)),
                (RO, whole(&buffers.compact_block_offsets)),
                (RW, whole(&buffers.contacts)),
                (RW, whole(&buffers.contact_a_body)),
                (RW, buffers.counter(COUNTER_PAIRS)),
            ],
            &[],
        );

        let events_end = Stage::build(
            context,
            "events_end",
            &assemble_shader(include_str!("shaders/events_end.wgsl"), per_row),
            &[
                (RO, whole(&buffers.prev_contacts)),
                (RW, buffers.counter(COUNTER_PREV_CONTACTS)),
                (RO, whole(&buffers.contacts)),
                (RW, buffers.counter(COUNTER_CONTACTS)),
                (RW, whole(&buffers.events)),
                (RW, buffers.counter(COUNTER_EVENTS)),
                (RW, buffers.counter(COUNTER_SPILLOVER_EVENTS)),
                (UNIFORM, whole(&buffers.params)),
            ],
            &[],
        );

        let contact_archive = Stage::build(
            context,
            "contact_archive",
            &assemble_shader(include_str!("shaders/contact_archive.wgsl"), per_row),
            &[
                (RO, whole(&buffers.contacts)),
                (RW, whole(&buffers.prev_contacts)),
                (RW, buffers.counter(COUNTER_CONTACTS)),
            ],
            &[],
        );

        let prev_count_sync = Stage::build(
            context,
            "prev_count_sync",
            &assemble_shader(include_str!("shaders/prev_count_sync.wgsl"), per_row),
            &[
                (RW, buffers.counter(COUNTER_CONTACTS)),
                (RW, buffers.counter(COUNTER_PREV_CONTACTS)),
                (RO, whole(&buffers.contacts)),
            ],
            &[],
        );

        let island_init = Stage::build(
            context,
            "island_init",
            &assemble_shader(include_str!("shaders/island_init.wgsl"), per_row),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RW, whole(&buffers.island_parents)),
                (RW, whole(&buffers.island_state)),
            ],
            &[],
        );

        let island_link_contacts = Stage::build(
            context,
            "island_link_contacts",
            &assemble_shader(include_str!("shaders/island_link_contacts.wgsl"), per_row),
            &[
                (RO, whole(&buffers.body_states)),
                (RO, whole(&buffers.body_descs)),
                (RO, whole(&buffers.contacts)),
                (RW, buffers.counter(COUNTER_CONTACTS)),
                (RW, whole(&buffers.island_parents)),
                (RW, whole(&buffers.wake_flags)),
            ],
            &[],
        );

        let island_link_constraints = Stage::build(
            context,
            "island_link_constraints",
            &assemble_shader(
                include_str!("shaders/island_link_constraints.wgsl"),
                per_row,
            ),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RO, whole(&buffers.body_states)),
                (RO, whole(&buffers.body_descs)),
                (RO, whole(&buffers.constraint_descs)),
                (RO, whole(&buffers.constraint_runtime)),
                (RW, whole(&buffers.island_parents)),
                (RW, whole(&buffers.wake_flags)),
            ],
            &[],
        );

        let island_jump = Stage::build(
            context,
            "island_jump",
            &assemble_shader(include_str!("shaders/island_jump.wgsl"), per_row),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RW, whole(&buffers.island_parents)),
            ],
            &[],
        );

        let island_aggregate = Stage::build(
            context,
            "island_aggregate",
            &assemble_shader(include_str!("shaders/island_aggregate.wgsl"), per_row),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RO, whole(&buffers.body_states)),
                (RO, whole(&buffers.body_descs)),
                (RW, whole(&buffers.island_parents)),
                (RW, whole(&buffers.island_state)),
                (RW, whole(&buffers.wake_flags)),
            ],
            &[],
        );

        let island_broadcast = Stage::build(
            context,
            "island_broadcast",
            &assemble_shader(include_str!("shaders/island_broadcast.wgsl"), per_row),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RW, whole(&buffers.body_states)),
                (RO, whole(&buffers.body_descs)),
                (RW, whole(&buffers.island_parents)),
                (RW, whole(&buffers.island_state)),
                (RW, whole(&buffers.wake_flags)),
            ],
            &[],
        );

        let gather_contact_keys_b = Stage::build(
            context,
            "gather_contact_keys_b",
            &assemble_shader(include_str!("shaders/gather_contact_keys_b.wgsl"), per_row),
            &[
                (RO, whole(&buffers.contacts)),
                (RW, buffers.counter(COUNTER_CONTACTS)),
                (RW, whole(&buffers.contact_b_keys)),
                (RW, whole(&buffers.contact_b_values)),
            ],
            &[],
        );

        let gather_constraint_keys = Stage::build(
            context,
            "gather_constraint_keys",
            &assemble_shader(include_str!("shaders/gather_constraint_keys.wgsl"), per_row),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RO, whole(&buffers.constraint_runtime)),
                (RW, whole(&buffers.constraint_a_keys)),
                (RW, whole(&buffers.constraint_a_values)),
                (RW, whole(&buffers.constraint_b_keys)),
                (RW, whole(&buffers.constraint_b_values)),
                (RO, whole(&buffers.constraint_descs)),
            ],
            &[],
        );

        let reset_gather_boundaries = Stage::build(
            context,
            "reset_gather_boundaries",
            &assemble_shader(
                include_str!("shaders/reset_gather_boundaries.wgsl"),
                per_row,
            ),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RW, whole(&buffers.contact_first_a)),
                (RW, whole(&buffers.contact_first_b)),
                (RW, whole(&buffers.constraint_first_a)),
                (RW, whole(&buffers.constraint_first_b)),
            ],
            &[],
        );

        let mark_contact_boundaries = Stage::build(
            context,
            "mark_contact_boundaries",
            &assemble_shader(
                include_str!("shaders/mark_contact_boundaries.wgsl"),
                per_row,
            ),
            &[
                (RO, whole(&buffers.contact_a_body)),
                (RO, whole(&buffers.contact_b_keys)),
                (RW, buffers.counter(COUNTER_CONTACTS)),
                (RW, whole(&buffers.contact_first_a)),
                (RW, whole(&buffers.contact_first_b)),
            ],
            &[],
        );

        let mark_constraint_boundaries = Stage::build(
            context,
            "mark_constraint_boundaries",
            &assemble_shader(
                include_str!("shaders/mark_constraint_boundaries.wgsl"),
                per_row,
            ),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RO, whole(&buffers.constraint_a_keys)),
                (RO, whole(&buffers.constraint_b_keys)),
                (RW, whole(&buffers.constraint_first_a)),
                (RW, whole(&buffers.constraint_first_b)),
            ],
            &[],
        );

        let ccd_sweep = Stage::build(
            context,
            "ccd_sweep",
            &assemble_shader(include_str!("shaders/ccd_sweep.wgsl"), per_row),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RW, whole(&buffers.body_states)),
                (RO, whole(&buffers.body_descs)),
                (RO, whole(&buffers.colliders)),
                (RO, whole(&buffers.pairs.keys_hi)),
                (RO, whole(&buffers.pairs.keys_lo)),
                (RW, buffers.counter(COUNTER_PAIRS)),
            ],
            &shape_resources,
        );

        let contact_match = Stage::build(
            context,
            "contact_match",
            &assemble_shader(include_str!("shaders/contact_match.wgsl"), per_row),
            &[
                (RW, whole(&buffers.contacts)),
                (RO, whole(&buffers.prev_contacts)),
                (RW, buffers.counter(COUNTER_PREV_CONTACTS)),
                (RW, buffers.counter(COUNTER_CONTACTS)),
                (RW, whole(&buffers.events)),
                (RW, buffers.counter(COUNTER_EVENTS)),
                (RW, buffers.counter(COUNTER_SPILLOVER_EVENTS)),
                (UNIFORM, whole(&buffers.params)),
            ],
            &[],
        );

        let contact_solve_extract = Stage::build(
            context,
            "contact_solve_extract",
            &assemble_shader(include_str!("shaders/contact_solve_extract.wgsl"), per_row),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RO, whole(&buffers.body_states)),
                (RO, whole(&buffers.body_descs)),
                (RW, whole(&buffers.contacts)),
                (RW, buffers.counter(COUNTER_CONTACTS)),
                (RW, whole(&buffers.wake_flags)),
                (RW, whole(&buffers.contact_deltas)),
            ],
            &[],
        );

        let constraint_solve_extract = Stage::build(
            context,
            "constraint_solve_extract",
            &assemble_shader(
                include_str!("shaders/constraint_solve_extract.wgsl"),
                per_row,
            ),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RO, whole(&buffers.body_states)),
                (RO, whole(&buffers.body_descs)),
                (RO, whole(&buffers.constraint_descs)),
                (RW, whole(&buffers.constraint_runtime)),
                (RW, whole(&buffers.wake_flags)),
                (RW, whole(&buffers.constraint_deltas)),
            ],
            &[],
        );

        let body_apply_solver = Stage::build(
            context,
            "body_apply_solver",
            &assemble_shader(include_str!("shaders/body_apply_solver.wgsl"), per_row),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RW, whole(&buffers.body_states)),
                (RO, whole(&buffers.contact_first_a)),
                (RO, whole(&buffers.contact_first_b)),
                (RO, whole(&buffers.contact_a_body)),
                (RO, whole(&buffers.contact_b_keys)),
                (RO, whole(&buffers.contact_b_values)),
                (RO, whole(&buffers.constraint_first_a)),
                (RO, whole(&buffers.constraint_first_b)),
                (RO, whole(&buffers.constraint_a_keys)),
                (RO, whole(&buffers.constraint_a_values)),
                (RO, whole(&buffers.constraint_b_keys)),
                (RO, whole(&buffers.constraint_b_values)),
                (RW, buffers.counter(COUNTER_CONTACTS)),
                (RO, whole(&buffers.contact_deltas)),
                (RO, whole(&buffers.constraint_deltas)),
            ],
            &[],
        );

        let position_solve_extract = Stage::build(
            context,
            "position_solve_extract",
            &assemble_shader(include_str!("shaders/position_solve_extract.wgsl"), per_row),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RO, whole(&buffers.body_states)),
                (RO, whole(&buffers.body_descs)),
                (RO, whole(&buffers.contacts)),
                (RW, buffers.counter(COUNTER_CONTACTS)),
                (RW, whole(&buffers.contact_deltas)),
            ],
            &[],
        );

        let body_apply_positions = Stage::build(
            context,
            "body_apply_positions",
            &assemble_shader(include_str!("shaders/body_apply_positions.wgsl"), per_row),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RW, whole(&buffers.body_states)),
                (RO, whole(&buffers.contact_first_a)),
                (RO, whole(&buffers.contact_first_b)),
                (RO, whole(&buffers.contact_a_body)),
                (RO, whole(&buffers.contact_b_keys)),
                (RO, whole(&buffers.contact_b_values)),
                (RW, buffers.counter(COUNTER_CONTACTS)),
                (RO, whole(&buffers.contact_deltas)),
            ],
            &[],
        );

        let query = Stage::build(
            context,
            "query",
            &assemble_shader(include_str!("shaders/queries.wgsl"), per_row),
            &[
                (RO, whole(&buffers.queries)),
                (RO, whole(&buffers.body_states)),
                (RO, whole(&buffers.body_descs)),
                (RO, whole(&buffers.colliders)),
                (RO, whole(&buffers.aabbs)),
                (RO, whole(&buffers.entries.keys_hi)),
                (RO, whole(&buffers.entries.keys_lo)),
                (RW, buffers.counter(COUNTER_ENTRIES)),
                (RW, whole(&buffers.query_results)),
                (RO, whole(&buffers.large_bodies)),
                (RW, buffers.counter(COUNTER_LARGE)),
                (UNIFORM, whole(&buffers.params)),
            ],
            &shape_resources,
        );

        let constraints_warm_end = Stage::build(
            context,
            "constraints_warm_end",
            &assemble_shader(include_str!("shaders/constraints_warm_end.wgsl"), per_row),
            &[
                (RO, whole(&buffers.constraint_descs)),
                (RW, whole(&buffers.constraint_runtime)),
                (UNIFORM, whole(&buffers.params)),
            ],
            &[],
        );

        let static_wake_clear = Stage::build(
            context,
            "static_wake_clear",
            &assemble_shader(include_str!("shaders/static_wake_clear.wgsl"), per_row),
            &[
                (UNIFORM, whole(&buffers.params)),
                (RW, whole(&buffers.wake_flags)),
            ],
            &[],
        );

        let dispatch_shader = assemble_shader(&dispatch_source(), per_row);
        const DISPATCH_BINDINGS: &[BindingSpec] = &[
            BindingSpec {
                binding: 0,
                kind: BindingKind::ReadOnlyStorage,
            },
            BindingSpec {
                binding: 1,
                kind: BindingKind::ReadWriteStorage,
            },
        ];
        let dispatch_args = std::array::from_fn(|batch| {
            let entry = ["dispatch_a", "dispatch_b", "dispatch_c", "dispatch_d"][batch];
            context.compute_pipeline(
                "dispatch args",
                &dispatch_shader,
                entry,
                &[DISPATCH_BINDINGS],
                64,
            )
        });
        let dispatch_group = std::array::from_fn(|batch| {
            dispatch_args[batch].create_bind_group(
                context.device(),
                0,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: buffers.counters.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: buffers.dispatch.buffer().as_binding(),
                    },
                ],
            )
        });

        let sort = RadixSort::new(context, "sim sort", plan.sort());
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
            per_row,
            reset_counters,
            body_edits,
            body_gather,
            body_scatter,
            constraint_gather,
            constraint_scatter,
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
            dispatch_args,
            dispatch_group,
            sort,
            #[cfg(feature = "profile")]
            timer,
        }
    }

    fn open<'a>(&'a self, encoder: &'a mut CommandEncoder, slot: usize) -> ComputeRecorder<'a> {
        let label = SIM_PASSES[slot];
        #[cfg(feature = "profile")]
        if let Some(timer) = &self.timer {
            return ComputeRecorder::begin_timed(
                encoder,
                label,
                Some(timer.writes(slot)),
                self.per_row,
            );
        }
        ComputeRecorder::begin(encoder, label, self.per_row)
    }

    fn sort_lanes<'a>(
        buffers: &'a WorldBuffers,
        count: GpuSlot<'a>,
        keys_lo: &'a GpuBuffer,
        keys_hi: &'a GpuBuffer,
        values: &'a GpuBuffer,
    ) -> SortChannels<'a> {
        SortChannels {
            count,
            keys_lo,
            keys_hi,
            values,
            scratch_lo: &buffers.sort_scratch.keys_lo,
            scratch_hi: &buffers.sort_scratch.keys_hi,
            scratch_values: &buffers.sort_scratch.values,
        }
    }

    /// One step, recorded from the host-owned counts and the dispatch table alike:
    /// stages whose work the device counted run by indirect arguments, stages whose
    /// work the host knows run by direct counts, and a batch of arguments is written
    /// at every point of the step where its counters become final.
    pub(crate) fn encode(
        &self,
        encoder: &mut CommandEncoder,
        buffers: &WorldBuffers,
        params: &FrameParams,
    ) {
        let constraint_active = params.constraint_count > 0;
        let collider_words = key_words(buffers.colliders());
        let joint_words = key_words(buffers.constraints().max(1));
        let index_words = key_words(buffers.contacts().max(1));
        let body_words = key_words(buffers.bodies().max(1));
        let gather_words = key_words(buffers.constraints().max(1));

        let mut commands = self.open(encoder, PASS_COMMANDS);
        self.reset_counters
            .record(&mut commands, dynamis_layout::COUNTER_COUNT as u32);
        if params.body_structural {
            self.body_gather.record(&mut commands, params.body_count);
            self.body_scatter.record(&mut commands, params.body_count);
        }
        if params.has_body_edits {
            self.body_edits.record(&mut commands, params.body_count);
        }
        if params.constraint_structural {
            self.constraint_gather
                .record(&mut commands, params.constraint_count);
            self.constraint_scatter
                .record(&mut commands, params.constraint_count);
        }
        if constraint_active {
            self.joint_filter
                .record(&mut commands, params.constraint_count);
        }
        drop(commands);
        self.encode_dispatch(encoder, 0);

        let mut integrate = self.open(encoder, PASS_INTEGRATE);
        if constraint_active {
            let channels = Self::sort_lanes(
                buffers,
                buffers.counter(COUNTER_JOINTS),
                &buffers.joint_lo,
                &buffers.joint_hi,
                &buffers.sort_values,
            );
            self.sort.sort(
                &mut integrate,
                &channels,
                joint_words,
                joint_words,
                &buffers.dispatch,
                dispatch(COUNTER_JOINTS as u32, KERNEL_TILE, SORT_JOINTS),
            );
        }
        self.integrate.record(&mut integrate, params.dynamic_count);
        self.broadphase_aabb
            .record(&mut integrate, params.dynamic_count);
        drop(integrate);

        let mut grid = self.open(encoder, PASS_GRID);
        self.grid_entries.record(&mut grid, params.body_count);
        drop(grid);
        self.encode_dispatch(encoder, 1);

        let mut broadphase = self.open(encoder, PASS_BROADPHASE);
        let channels = Self::sort_lanes(
            buffers,
            buffers.counter(COUNTER_ENTRIES),
            &buffers.entries.keys_lo,
            &buffers.entries.keys_hi,
            &buffers.sort_values,
        );
        self.sort.sort(
            &mut broadphase,
            &channels,
            collider_words,
            4,
            &buffers.dispatch,
            dispatch(COUNTER_ENTRIES as u32, KERNEL_TILE, SORT_ENTRIES),
        );
        self.broadphase_pairs
            .record_indirect(&mut broadphase, &buffers.dispatch, BROADPHASE_PAIRS);
        self.large_pairs.record(&mut broadphase, params.body_count);
        drop(broadphase);
        self.encode_dispatch(encoder, 2);

        let mut narrowphase = self.open(encoder, PASS_NARROWPHASE);
        let channels = Self::sort_lanes(
            buffers,
            buffers.counter(COUNTER_PAIRS),
            &buffers.pairs.keys_lo,
            &buffers.pairs.keys_hi,
            &buffers.sort_values,
        );
        self.sort.sort(
            &mut narrowphase,
            &channels,
            collider_words,
            collider_words,
            &buffers.dispatch,
            dispatch(COUNTER_PAIRS as u32, KERNEL_TILE, SORT_PAIRS),
        );
        self.ccd_sweep
            .record_indirect(&mut narrowphase, &buffers.dispatch, CCD_SWEEP);
        self.narrowphase
            .record_indirect(&mut narrowphase, &buffers.dispatch, NARROWPHASE);
        self.compact_scan
            .record_indirect(&mut narrowphase, &buffers.dispatch, COMPACT_SCAN);
        self.compact_offsets.record_workgroups(&mut narrowphase, 1);
        self.compact_scatter
            .record_indirect(&mut narrowphase, &buffers.dispatch, COMPACT_SCATTER);
        drop(narrowphase);
        self.encode_dispatch(encoder, 3);

        let mut islands = self.open(encoder, PASS_ISLANDS);
        self.contact_match
            .record_indirect(&mut islands, &buffers.dispatch, CONTACT_MATCH);
        self.island_init.record(&mut islands, params.dynamic_count);
        self.island_link_contacts.record_indirect(
            &mut islands,
            &buffers.dispatch,
            ISLAND_LINK_CONTACTS,
        );
        if constraint_active {
            self.island_link_constraints
                .record(&mut islands, params.constraint_count);
        }
        for _ in 0..params.island_rounds {
            self.island_jump.record(&mut islands, params.dynamic_count);
        }
        self.gather_contact_keys_b.record_indirect(
            &mut islands,
            &buffers.dispatch,
            GATHER_CONTACT_KEYS_B,
        );
        let channels = Self::sort_lanes(
            buffers,
            buffers.counter(COUNTER_CONTACTS),
            &buffers.contact_b_values,
            &buffers.contact_b_keys,
            &buffers.sort_pad,
        );
        self.sort.sort(
            &mut islands,
            &channels,
            index_words,
            body_words,
            &buffers.dispatch,
            dispatch(COUNTER_CONTACTS as u32, KERNEL_TILE, SORT_CONTACTS),
        );
        if constraint_active {
            self.gather_constraint_keys
                .record(&mut islands, params.constraint_count);
            for (keys, values) in [
                (&buffers.constraint_a_keys, &buffers.constraint_a_values),
                (&buffers.constraint_b_keys, &buffers.constraint_b_values),
            ] {
                let channels = Self::sort_lanes(
                    buffers,
                    buffers.counter(COUNTER_CONSTRAINTS),
                    values,
                    keys,
                    &buffers.sort_pad,
                );
                self.sort.sort(
                    &mut islands,
                    &channels,
                    gather_words,
                    body_words,
                    &buffers.dispatch,
                    dispatch(COUNTER_CONSTRAINTS as u32, KERNEL_TILE, SORT_CONSTRAINTS),
                );
            }
        }
        self.reset_gather_boundaries
            .record(&mut islands, params.dynamic_count);
        self.mark_contact_boundaries.record_indirect(
            &mut islands,
            &buffers.dispatch,
            MARK_CONTACT_BOUNDARIES,
        );
        if constraint_active {
            self.mark_constraint_boundaries
                .record(&mut islands, params.constraint_count);
        }
        drop(islands);

        let mut sleep = self.open(encoder, PASS_SLEEP);
        self.island_aggregate
            .record(&mut sleep, params.dynamic_count);
        self.island_broadcast
            .record(&mut sleep, params.dynamic_count);
        drop(sleep);

        let mut velocity_solve = self.open(encoder, PASS_VELOCITY_SOLVE);
        for _ in 0..params.solve_iterations {
            if constraint_active {
                self.constraint_solve_extract
                    .record(&mut velocity_solve, params.constraint_count);
            }
            self.contact_solve_extract.record_indirect(
                &mut velocity_solve,
                &buffers.dispatch,
                CONTACT_SOLVE_EXTRACT,
            );
            self.body_apply_solver
                .record(&mut velocity_solve, params.dynamic_count);
        }
        drop(velocity_solve);

        let mut position_solve = self.open(encoder, PASS_POSITION_SOLVE);
        for _ in 0..params.position_iterations {
            self.position_solve_extract.record_indirect(
                &mut position_solve,
                &buffers.dispatch,
                POSITION_SOLVE_EXTRACT,
            );
            self.body_apply_positions
                .record(&mut position_solve, params.dynamic_count);
        }
        drop(position_solve);

        let mut tail = self.open(encoder, PASS_TAIL);
        self.events_end
            .record_indirect(&mut tail, &buffers.dispatch, EVENTS_END);
        self.contact_archive
            .record_indirect(&mut tail, &buffers.dispatch, CONTACT_ARCHIVE);
        self.prev_count_sync.record_workgroups(&mut tail, 1);
        self.constraints_warm_end
            .record(&mut tail, params.constraint_count);
        self.static_wake_clear.record(&mut tail, params.body_count);
        if params.query_count > 0 {
            self.query.record_workgroups(&mut tail, params.query_count);
        }
        drop(tail);
    }

    /// Writes one batch of the dispatch table, after the counters it reads are final.
    fn encode_dispatch(&self, encoder: &mut CommandEncoder, batch: usize) {
        let mut recorder = ComputeRecorder::begin(encoder, "dispatch", self.per_row);
        recorder.record(
            &self.dispatch_args[batch],
            &[&self.dispatch_group[batch]],
            1,
        );
    }
    #[cfg(feature = "profile")]
    pub(crate) fn capture_timings(
        &mut self,
        encoder: &mut CommandEncoder,
        sequence: u64,
    ) -> Option<(u64, Vec<dynamis_gpu::GpuPassTiming>)> {
        self.timer
            .as_mut()
            .and_then(|timer| timer.capture(encoder, sequence))
    }

    #[cfg(feature = "profile")]
    pub(crate) fn poll_timings(&mut self) -> Vec<(u64, Vec<dynamis_gpu::GpuPassTiming>)> {
        match &mut self.timer {
            Some(timer) => timer.poll(),
            None => Vec::new(),
        }
    }

    #[cfg(feature = "profile")]
    pub(crate) fn arm_timings(&mut self) {
        if let Some(timer) = &mut self.timer {
            timer.arm();
        }
    }

    /// Refreshes the broadphase scene the queries run against, then runs them.
    ///
    /// A query outside a step must see the world as the device holds it now, so the
    /// grid is rebuilt from the current states rather than trusting the last step's
    /// cells: stale entries would let a moved body slip through a ray unanswered.
    pub(crate) fn encode_queries(
        &self,
        encoder: &mut CommandEncoder,
        buffers: &WorldBuffers,
        frame: &FrameParams,
        query_count: u32,
    ) {
        let mut recorder = ComputeRecorder::begin(encoder, "query commands", self.per_row);
        if frame.body_structural {
            self.body_gather.record(&mut recorder, frame.body_count);
            self.body_scatter.record(&mut recorder, frame.body_count);
        }
        if frame.has_body_edits {
            self.body_edits.record(&mut recorder, frame.body_count);
        }
        if frame.constraint_structural {
            self.constraint_gather
                .record(&mut recorder, frame.constraint_count);
            self.constraint_scatter
                .record(&mut recorder, frame.constraint_count);
        }
        self.reset_counters.record_workgroups(&mut recorder, 1);
        self.broadphase_aabb
            .record(&mut recorder, frame.dynamic_count);
        self.grid_entries.record(&mut recorder, frame.body_count);
        drop(recorder);
        self.encode_dispatch(encoder, 1);
        let mut recorder = ComputeRecorder::begin(encoder, "query flush", self.per_row);
        let collider_words = key_words(buffers.colliders());
        let channels = Self::sort_lanes(
            buffers,
            buffers.counter(COUNTER_ENTRIES),
            &buffers.entries.keys_lo,
            &buffers.entries.keys_hi,
            &buffers.sort_values,
        );
        self.sort.sort(
            &mut recorder,
            &channels,
            collider_words,
            4,
            &buffers.dispatch,
            dispatch(COUNTER_ENTRIES as u32, KERNEL_TILE, SORT_ENTRIES),
        );
        self.query.record_workgroups(&mut recorder, query_count);
    }
}
