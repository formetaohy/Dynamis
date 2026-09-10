use super::shader::{WORKGROUP_SIZE, assemble_shader};
use super::stage::CORE;
use crate::buffers::{COMPACT_BLOCK, WorldBuffers};
use dynamis_gpu::{BindingKind, BindingSpec, ComputePipeline, ComputeRecorder, GpuContext};
use dynamis_layout::{
    COUNTER_ARCHIVED, COUNTER_BODY_MOVES, COUNTER_CONSTRAINTS, COUNTER_CONTACTS, COUNTER_ENTRIES,
    COUNTER_JOINTS, COUNTER_PAIRS, COUNTER_RESTING, COUNTER_RESTING_GATHER,
    COUNTER_RESTING_PENDING, COUNTER_SLEPT, COUNTER_WOKE,
};
use wgpu::{BindGroup, BindGroupEntry, CommandEncoder};

pub(super) const SORT_JOINTS: u32 = 0;
pub(super) const SORT_ENTRIES: u32 = 1;
pub(super) const SORT_PAIRS: u32 = 2;
pub(super) const SORT_CONTACTS: u32 = 3;
pub(super) const SORT_CONSTRAINTS: u32 = 4;
pub(super) const BROADPHASE_PAIRS: u32 = 5;
pub(super) const NARROWPHASE: u32 = 6;
pub(super) const COMPACT_SCAN: u32 = 7;
pub(super) const COMPACT_SCATTER: u32 = 8;
pub(super) const CCD_SWEEP: u32 = 9;
pub(super) const CONTACT_ARCHIVE: u32 = 10;
pub(super) const ISLAND_LINK_CONTACTS: u32 = 11;
pub(super) const GATHER_CONTACT_KEYS_B: u32 = 12;
pub(super) const MARK_CONTACT_BOUNDARIES: u32 = 13;
pub(super) const CONTACT_BEGIN: u32 = 14;
pub(super) const CONTACT_SOLVE_EXTRACT: u32 = 15;
pub(super) const POSITION_SOLVE_EXTRACT: u32 = 16;
pub(super) const CONTACT_RELAY: u32 = 17;
pub(super) const THAW_CONTACTS: u32 = 18;
pub(super) const FREEZE_CONTACTS: u32 = 19;
pub(super) const SORT_RESTING: u32 = 20;
pub(super) const RESTING_GATHER: u32 = 21;
const KERNEL_TILE: u32 = 256;

struct DispatchEntry {
    slot: u32,
    counter: usize,
    gates: &'static [usize],
    lanes: u32,
}

const fn entry(slot: u32, counter: usize, lanes: u32) -> DispatchEntry {
    DispatchEntry {
        slot,
        counter,
        gates: &[],
        lanes,
    }
}

const fn transition(
    slot: u32,
    counter: usize,
    gates: &'static [usize],
    lanes: u32,
) -> DispatchEntry {
    DispatchEntry {
        slot,
        counter,
        gates,
        lanes,
    }
}

const DISPATCH_BATCHES: &[&[DispatchEntry]] = &[
    &[
        entry(SORT_JOINTS, COUNTER_JOINTS, KERNEL_TILE),
        entry(SORT_CONSTRAINTS, COUNTER_CONSTRAINTS, KERNEL_TILE),
    ],
    &[
        entry(SORT_ENTRIES, COUNTER_ENTRIES, KERNEL_TILE),
        entry(BROADPHASE_PAIRS, COUNTER_ENTRIES, WORKGROUP_SIZE),
    ],
    &[
        entry(SORT_PAIRS, COUNTER_PAIRS, KERNEL_TILE),
        entry(NARROWPHASE, COUNTER_PAIRS, WORKGROUP_SIZE),
        entry(COMPACT_SCAN, COUNTER_PAIRS, COMPACT_BLOCK),
        entry(COMPACT_SCATTER, COUNTER_PAIRS, WORKGROUP_SIZE),
        entry(CCD_SWEEP, COUNTER_PAIRS, WORKGROUP_SIZE),
    ],
    &[
        entry(CONTACT_RELAY, COUNTER_ARCHIVED, WORKGROUP_SIZE),
        entry(SORT_CONTACTS, COUNTER_CONTACTS, KERNEL_TILE),
        entry(CONTACT_ARCHIVE, COUNTER_CONTACTS, WORKGROUP_SIZE),
        entry(ISLAND_LINK_CONTACTS, COUNTER_CONTACTS, WORKGROUP_SIZE),
        entry(GATHER_CONTACT_KEYS_B, COUNTER_CONTACTS, WORKGROUP_SIZE),
        entry(MARK_CONTACT_BOUNDARIES, COUNTER_CONTACTS, WORKGROUP_SIZE),
        entry(CONTACT_BEGIN, COUNTER_CONTACTS, WORKGROUP_SIZE),
        entry(CONTACT_SOLVE_EXTRACT, COUNTER_CONTACTS, WORKGROUP_SIZE),
        entry(POSITION_SOLVE_EXTRACT, COUNTER_CONTACTS, WORKGROUP_SIZE),
    ],
    &[
        transition(
            THAW_CONTACTS,
            COUNTER_RESTING,
            &[COUNTER_WOKE, COUNTER_RESTING_PENDING, COUNTER_BODY_MOVES],
            WORKGROUP_SIZE,
        ),
        transition(
            FREEZE_CONTACTS,
            COUNTER_CONTACTS,
            &[COUNTER_SLEPT],
            WORKGROUP_SIZE,
        ),
    ],
    &[transition(
        RESTING_GATHER,
        COUNTER_RESTING,
        &[COUNTER_SLEPT],
        WORKGROUP_SIZE,
    )],
    &[transition(
        SORT_RESTING,
        COUNTER_RESTING_GATHER,
        &[COUNTER_RESTING_GATHER],
        KERNEL_TILE,
    )],
];

pub(super) const COMMANDS_BATCH: usize = 0;
pub(super) const GRID_BATCH: usize = 1;
pub(super) const BROADPHASE_BATCH: usize = 2;
pub(super) const ISLANDS_BATCH: usize = 3;
pub(super) const COMMIT_BATCH: usize = 4;
pub(super) const RESTING_GATHER_BATCH: usize = 5;
pub(super) const RESTING_SORT_BATCH: usize = 6;

pub(crate) const DISPATCH_SLOTS: u32 = dispatch_slots();

const fn dispatch_slots() -> u32 {
    let mut slots = 0u32;
    let mut batch = 0usize;
    while batch < DISPATCH_BATCHES.len() {
        let entries = DISPATCH_BATCHES[batch];
        let mut index = 0usize;
        while index < entries.len() {
            let slot = entries[index].slot;
            if slot >= slots {
                slots = slot + 1;
            }
            index += 1;
        }
        batch += 1;
    }
    slots
}

fn dispatch_source() -> String {
    let mut source = String::from(
        "struct Args { per_row: u32, rows: u32, layers: u32, _pad: u32 }\n\
@group(0) @binding(0) var<storage, read> counters: array<u32>;\n\
@group(0) @binding(1) var<storage, read_write> table: array<Args>;\n\
fn write_count_args(index: u32, count: u32, lanes: u32) {\n\
    let workgroups = (count + lanes - 1u) / lanes;\n\
    let per_row = min(workgroups, WORKGROUPS_PER_ROW);\n\
    let rows = (workgroups + WORKGROUPS_PER_ROW - 1u) / WORKGROUPS_PER_ROW;\n\
    table[index] = Args(per_row, rows, 1u, 0u);\n\
}\n\
fn write_sweep_args(index: u32, counter: u32, lanes: u32) {\n\
    write_count_args(index, counters[counter * COUNTER_STRIDE_WORDS], lanes);\n\
}\n\
fn write_transition_args(index: u32, counter: u32, lanes: u32, run: bool) {\n\
    let held = counters[counter * COUNTER_STRIDE_WORDS];\n\
    write_count_args(index, select(0u, held, run), lanes);\n\
}\n\n",
    );
    for (batch, entries) in DISPATCH_BATCHES.iter().enumerate() {
        source.push_str(&format!(
            "@compute @workgroup_size({WORKGROUP_SIZE}u)\nfn dispatch_{batch}(@builtin(local_invocation_id) lid: vec3u) {{\n"
        ));
        for (index, entry) in entries.iter().enumerate() {
            let call = if entry.gates.is_empty() {
                format!(
                    "write_sweep_args({}, {}, {})",
                    entry.slot, entry.counter, entry.lanes
                )
            } else {
                let condition = entry
                    .gates
                    .iter()
                    .map(|gate| format!("counters[{gate} * COUNTER_STRIDE_WORDS] > 0u"))
                    .collect::<Vec<_>>()
                    .join(" || ");
                format!(
                    "write_transition_args({}, {}, {}, {condition})",
                    entry.slot, entry.counter, entry.lanes
                )
            };
            source.push_str(&format!("    if (lid.x == {index}u) {{ {call}; }}\n"));
        }
        source.push_str("}\n\n");
    }
    source
}

struct DispatchStage {
    pipeline: ComputePipeline,
    group: BindGroup,
}

pub(super) struct Dispatch {
    per_row: u32,
    stages: Vec<DispatchStage>,
}

impl Dispatch {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        const BINDINGS: &[BindingSpec] = &[
            BindingSpec {
                binding: 0,
                kind: BindingKind::ReadOnlyStorage,
            },
            BindingSpec {
                binding: 1,
                kind: BindingKind::ReadWriteStorage,
            },
        ];
        let shader = assemble_shader(&dispatch_source(), per_row, CORE);
        let stages = (0..DISPATCH_BATCHES.len())
            .map(|batch| {
                let pipeline = context.compute_pipeline(
                    "dispatch args",
                    &shader,
                    &format!("dispatch_{batch}"),
                    &[BINDINGS],
                    WORKGROUP_SIZE,
                );
                let group = pipeline.create_bind_group(
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
                );
                DispatchStage { pipeline, group }
            })
            .collect();
        Self { per_row, stages }
    }

    pub(super) fn write(&self, encoder: &mut CommandEncoder, batch: usize) {
        let stage = &self.stages[batch];
        let mut recorder = ComputeRecorder::begin(encoder, "dispatch", self.per_row);
        recorder.record(&stage.pipeline, &[&stage.group], 1);
    }
}
