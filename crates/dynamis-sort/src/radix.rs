use dynamis_gpu::{
    BindingKind, BindingSpec, ComputeProgram, ComputeRecorder, GpuBuffer, GpuContext, GpuSlot,
    PipelineHandle,
};
use wgpu::{BindGroup, BindGroupEntry, Device};

const THREADS: u32 = 256;
const BINS: u32 = 256;
const BINS_ALL: usize = BINS as usize * 8;
const REGIONS: u32 = 8;
const SLOTS: usize = 64;

const READ: BindingKind = BindingKind::ReadOnlyStorage;
const WRITE: BindingKind = BindingKind::ReadWriteStorage;

const PREPARE_BINDINGS: [BindingSpec; 6] = [
    BindingSpec {
        binding: 0,
        kind: READ,
    },
    BindingSpec {
        binding: 1,
        kind: READ,
    },
    BindingSpec {
        binding: 2,
        kind: WRITE,
    },
    BindingSpec {
        binding: 3,
        kind: WRITE,
    },
    BindingSpec {
        binding: 4,
        kind: WRITE,
    },
    BindingSpec {
        binding: 5,
        kind: READ,
    },
];

const BINNING_BINDINGS: [BindingSpec; 9] = [
    BindingSpec {
        binding: 0,
        kind: READ,
    },
    BindingSpec {
        binding: 1,
        kind: READ,
    },
    BindingSpec {
        binding: 2,
        kind: READ,
    },
    BindingSpec {
        binding: 3,
        kind: READ,
    },
    BindingSpec {
        binding: 4,
        kind: WRITE,
    },
    BindingSpec {
        binding: 5,
        kind: WRITE,
    },
    BindingSpec {
        binding: 6,
        kind: WRITE,
    },
    BindingSpec {
        binding: 7,
        kind: WRITE,
    },
    BindingSpec {
        binding: 8,
        kind: READ,
    },
];

const AGGREGATE_BINDINGS: [BindingSpec; 4] = [
    BindingSpec {
        binding: 0,
        kind: READ,
    },
    BindingSpec {
        binding: 1,
        kind: READ,
    },
    BindingSpec {
        binding: 2,
        kind: WRITE,
    },
    BindingSpec {
        binding: 3,
        kind: READ,
    },
];

const COPY_BINDINGS: [BindingSpec; 7] = [
    BindingSpec {
        binding: 0,
        kind: READ,
    },
    BindingSpec {
        binding: 1,
        kind: READ,
    },
    BindingSpec {
        binding: 2,
        kind: READ,
    },
    BindingSpec {
        binding: 3,
        kind: WRITE,
    },
    BindingSpec {
        binding: 4,
        kind: WRITE,
    },
    BindingSpec {
        binding: 5,
        kind: WRITE,
    },
    BindingSpec {
        binding: 6,
        kind: READ,
    },
];

pub fn key_words(elements: u32) -> u32 {
    let bits = 32 - elements.saturating_sub(1).leading_zeros();
    bits.div_ceil(8).clamp(1, 4)
}

pub struct SortChannels<'a> {
    pub count: GpuSlot<'a>,
    pub major: &'a GpuBuffer,
    pub minor: &'a GpuBuffer,
    pub payload: &'a GpuBuffer,
    pub scratch_major: &'a GpuBuffer,
    pub scratch_minor: &'a GpuBuffer,
    pub scratch_payload: &'a GpuBuffer,
}

#[derive(PartialEq, Eq)]
struct ChannelsKey {
    count: (u64, u64, u64),
    major: u64,
    minor: u64,
    payload: u64,
    scratch_major: u64,
    scratch_minor: u64,
    scratch_payload: u64,
}

impl ChannelsKey {
    fn of(channels: &SortChannels<'_>) -> Self {
        Self {
            count: channels.count.identity(),
            major: channels.major.token(),
            minor: channels.minor.token(),
            payload: channels.payload.token(),
            scratch_major: channels.scratch_major.token(),
            scratch_minor: channels.scratch_minor.token(),
            scratch_payload: channels.scratch_payload.token(),
        }
    }
}

struct SortPipelines<'a> {
    prepare: &'a PipelineHandle,
    aggregate: &'a PipelineHandle,
    binning: &'a PipelineHandle,
    copy: &'a PipelineHandle,
}

struct SortBindGroups {
    channels: ChannelsKey,
    prepare: [BindGroup; 2],
    aggregate: [BindGroup; 2],
    binning: [[BindGroup; 2]; 2],
    copy: BindGroup,
}

struct State {
    groups: Vec<SortBindGroups>,
    generation: u32,
}

pub struct RadixSort {
    device: Device,
    prepare: PipelineHandle,
    aggregates: [PipelineHandle; 8],
    binnings: [PipelineHandle; 8],
    copy: PipelineHandle,
    rows: GpuBuffer,
    histograms: [GpuBuffer; 2],
    state: std::sync::Mutex<State>,
}

fn region_offset(pass: usize) -> usize {
    pass * SLOTS * BINS as usize
}

fn prepare_program(label: &str, row: u32) -> ComputeProgram {
    let source = include_str!("shaders/sort_prepare.wgsl")
        .replace("__SLOTS__", &SLOTS.to_string())
        .replace("__ROW__", &row.to_string());
    ComputeProgram::new(
        &format!("{label} prepare"),
        source,
        "main",
        &[&PREPARE_BINDINGS[..]],
    )
}

fn binning_program(label: &str, row: u32, pass: usize) -> ComputeProgram {
    let source = include_str!("shaders/sort_binning.wgsl")
        .replace("__SHIFT__", &(pass * 8).to_string())
        .replace("__DIGIT__", &pass.to_string())
        .replace("__REGION__", &region_offset(pass).to_string())
        .replace("__ROW__", &row.to_string());
    ComputeProgram::new(
        &format!("{label} binning {pass}"),
        source,
        "main",
        &[&BINNING_BINDINGS[..]],
    )
}

fn aggregate_program(label: &str, row: u32, pass: usize) -> ComputeProgram {
    let source = include_str!("shaders/sort_aggregate.wgsl")
        .replace("__SHIFT__", &(pass * 8).to_string())
        .replace("__REGION__", &region_offset(pass).to_string())
        .replace("__ROW__", &row.to_string());
    ComputeProgram::new(
        &format!("{label} aggregate {pass}"),
        source,
        "main",
        &[&AGGREGATE_BINDINGS[..]],
    )
}

fn copy_program(label: &str, row: u32) -> ComputeProgram {
    let source = include_str!("shaders/sort_copy.wgsl").replace("__ROW__", &row.to_string());
    ComputeProgram::new(
        &format!("{label} copy"),
        source,
        "main",
        &[&COPY_BINDINGS[..]],
    )
}

impl SortBindGroups {
    fn build(
        device: &Device,
        channels: &SortChannels<'_>,
        rows: &GpuBuffer,
        histograms: &[GpuBuffer; 2],
        pipelines: &SortPipelines<'_>,
    ) -> Self {
        let prepare = pipelines.prepare;
        let aggregate = pipelines.aggregate;
        let binning = pipelines.binning;
        let copy = pipelines.copy;
        let in_minor = [channels.minor, channels.scratch_minor];
        let in_major = [channels.major, channels.scratch_major];
        let in_payload = [channels.payload, channels.scratch_payload];
        let out_minor = [channels.scratch_minor, channels.minor];
        let out_major = [channels.scratch_major, channels.major];
        let out_payload = [channels.scratch_payload, channels.payload];
        let prepare_groups = std::array::from_fn(|parity| {
            prepare.create_bind_group(
                device,
                0,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: channels.minor.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: channels.major.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: histograms[parity].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: histograms[1 - parity].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: rows.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: channels.count.as_binding(),
                    },
                ],
            )
        });
        let aggregate_groups = std::array::from_fn(|source: usize| {
            aggregate.create_bind_group(
                device,
                0,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: in_minor[source].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: in_major[source].as_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: rows.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: channels.count.as_binding(),
                    },
                ],
            )
        });
        let binning_groups = std::array::from_fn(|source: usize| {
            std::array::from_fn(|parity: usize| {
                binning.create_bind_group(
                    device,
                    0,
                    &[
                        BindGroupEntry {
                            binding: 0,
                            resource: in_minor[source].as_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: in_major[source].as_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: in_payload[source].as_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: histograms[parity].as_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: rows.as_binding(),
                        },
                        BindGroupEntry {
                            binding: 5,
                            resource: out_minor[source].as_binding(),
                        },
                        BindGroupEntry {
                            binding: 6,
                            resource: out_major[source].as_binding(),
                        },
                        BindGroupEntry {
                            binding: 7,
                            resource: out_payload[source].as_binding(),
                        },
                        BindGroupEntry {
                            binding: 8,
                            resource: channels.count.as_binding(),
                        },
                    ],
                )
            })
        });
        let copy_group = copy.create_bind_group(
            device,
            0,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: channels.scratch_minor.as_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: channels.scratch_major.as_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: channels.scratch_payload.as_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: channels.minor.as_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: channels.major.as_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: channels.payload.as_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: channels.count.as_binding(),
                },
            ],
        );
        Self {
            channels: ChannelsKey::of(channels),
            prepare: prepare_groups,
            aggregate: aggregate_groups,
            binning: binning_groups,
            copy: copy_group,
        }
    }
}

impl RadixSort {
    pub fn new(context: &GpuContext, label: &str, reservation: u32) -> Self {
        let device = context.device().clone();
        let row = context.workgroups_per_row();
        let prepare = context.declare(prepare_program(label, row));
        let aggregates =
            std::array::from_fn(|pass| context.declare(aggregate_program(label, row, pass)));
        let binnings =
            std::array::from_fn(|pass| context.declare(binning_program(label, row, pass)));
        let copy = context.declare(copy_program(label, row));
        let rows = GpuBuffer::zeroed(
            &device,
            &format!("{label} rows"),
            (REGIONS as usize * SLOTS * BINS as usize * 4) as u64,
            wgpu::BufferUsages::STORAGE,
        );
        let histograms = std::array::from_fn(|index| {
            GpuBuffer::zeroed(
                &device,
                &format!("{label} histogram {index}"),
                (BINS_ALL * 4) as u64,
                wgpu::BufferUsages::STORAGE,
            )
        });
        assert!(
            reservation < (1 << 30),
            "a sort reserves at most 2^30 elements, got {reservation}"
        );
        Self {
            device,
            prepare,
            aggregates,
            binnings,
            copy,
            rows,
            histograms,
            state: std::sync::Mutex::new(State {
                groups: Vec::new(),
                generation: 0,
            }),
        }
    }

    pub fn sort(
        &self,
        recorder: &mut ComputeRecorder,
        channels: &SortChannels<'_>,
        major_words: u32,
        minor_words: u32,
        elements: u32,
    ) {
        let passes = (0..minor_words)
            .chain(4..4 + major_words)
            .map(|pass| pass as usize)
            .collect::<Vec<_>>();
        assert!(
            !passes.is_empty(),
            "a sort needs at least one key digit to permute the payload"
        );
        let units = (elements.div_ceil(THREADS) as usize).clamp(1, SLOTS);
        let mut state = self.state.lock().unwrap();
        let generation = state.generation;
        let parity = generation as usize % 2;
        let key = ChannelsKey::of(channels);
        let index = match state.groups.iter().position(|entry| entry.channels == key) {
            Some(index) => index,
            None => {
                state.groups.push(SortBindGroups::build(
                    &self.device,
                    channels,
                    &self.rows,
                    &self.histograms,
                    &SortPipelines {
                        prepare: &self.prepare,
                        aggregate: &self.aggregates[passes[0]],
                        binning: &self.binnings[passes[0]],
                        copy: &self.copy,
                    },
                ));
                state.groups.len() - 1
            }
        };
        let groups = &state.groups[index];
        recorder.record(
            self.prepare.pipeline(),
            &[&groups.prepare[parity]],
            units as u32,
        );
        for (order, pass) in passes.iter().enumerate() {
            if order > 0 {
                let group = &groups.aggregate[order % 2];
                recorder.record(self.aggregates[*pass].pipeline(), &[group], units as u32);
            }
            let group = &groups.binning[order % 2][parity];
            recorder.record(self.binnings[*pass].pipeline(), &[group], units as u32);
        }
        if passes.len() % 2 == 1 {
            recorder.record(self.copy.pipeline(), &[&groups.copy], units as u32);
        }
        state.generation = generation + 1;
    }
}
