use super::shader::{WORKGROUP_SIZE, assemble_shader};
use crate::buffers::{COMPACT_BLOCK, WorldBuffers};
use dynamis_gpu::{BindingKind, BindingSpec, ComputePipeline, ComputeRecorder, GpuContext};
use dynamis_layout::{
    COUNTER_CONSTRAINTS, COUNTER_CONTACTS, COUNTER_ENTRIES, COUNTER_JOINTS, COUNTER_PAIRS,
    COUNTER_PREV_CONTACTS,
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
pub(super) const CONTACT_MATCH: u32 = 14;
pub(super) const CONTACT_SOLVE_EXTRACT: u32 = 15;
pub(super) const POSITION_SOLVE_EXTRACT: u32 = 16;
pub(super) const EVENTS_END: u32 = 17;
const KERNEL_TILE: u32 = 256;

struct DispatchEntry {
    slot: u32,
    counter: usize,
    lanes: u32,
}

const fn entry(slot: u32, counter: usize, lanes: u32) -> DispatchEntry {
    DispatchEntry {
        slot,
        counter,
        lanes,
    }
}

const DISPATCH_BATCHES: &[&[DispatchEntry]] = &[
    &[
        entry(SORT_JOINTS, COUNTER_JOINTS, KERNEL_TILE),
        entry(SORT_CONSTRAINTS, COUNTER_CONSTRAINTS, KERNEL_TILE),
        entry(EVENTS_END, COUNTER_PREV_CONTACTS, WORKGROUP_SIZE),
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
        entry(SORT_CONTACTS, COUNTER_CONTACTS, KERNEL_TILE),
        entry(CONTACT_ARCHIVE, COUNTER_CONTACTS, WORKGROUP_SIZE),
        entry(ISLAND_LINK_CONTACTS, COUNTER_CONTACTS, WORKGROUP_SIZE),
        entry(GATHER_CONTACT_KEYS_B, COUNTER_CONTACTS, WORKGROUP_SIZE),
        entry(MARK_CONTACT_BOUNDARIES, COUNTER_CONTACTS, WORKGROUP_SIZE),
        entry(CONTACT_MATCH, COUNTER_CONTACTS, WORKGROUP_SIZE),
        entry(CONTACT_SOLVE_EXTRACT, COUNTER_CONTACTS, WORKGROUP_SIZE),
        entry(POSITION_SOLVE_EXTRACT, COUNTER_CONTACTS, WORKGROUP_SIZE),
    ],
];

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
fn write_args(index: u32, counter: u32, lanes: u32) {\n\
    let count = counters[counter * COUNTER_STRIDE_WORDS];\n\
    let workgroups = (count + lanes - 1u) / lanes;\n\
    let per_row = min(workgroups, WORKGROUPS_PER_ROW);\n\
    let rows = (workgroups + WORKGROUPS_PER_ROW - 1u) / WORKGROUPS_PER_ROW;\n\
    table[index] = Args(per_row, rows, 1u, 0u);\n\
}\n\n",
    );
    for (batch, entries) in DISPATCH_BATCHES.iter().enumerate() {
        source.push_str(&format!(
            "@compute @workgroup_size({WORKGROUP_SIZE}u)\nfn dispatch_{batch}(@builtin(local_invocation_id) lid: vec3u) {{\n"
        ));
        for (index, entry) in entries.iter().enumerate() {
            source.push_str(&format!(
                "    if (lid.x == {index}u) {{ write_args({}, {}, {}); }}\n",
                entry.slot, entry.counter, entry.lanes
            ));
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
        let shader = assemble_shader(&dispatch_source(), per_row);
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
