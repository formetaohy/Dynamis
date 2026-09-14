use dynamis_gpu::{
    ComputePipeline, ComputeProgram, ComputeRecorder, GpuBuffer, GpuContext, GpuSlot,
    PipelineHandle, StreamElement, TypedSlot,
};
use dynamis_pass::Bindings;
use wgpu::{BindGroup, Device};

const BINS: u32 = 256;
const BINS_ALL: usize = BINS as usize * 8;
const REGIONS: u32 = 8;
const SLOTS: usize = 64;

struct Declared {
    label: String,
    handle: PipelineHandle,
    bindings: Bindings,
}

impl Declared {
    fn declare(context: &GpuContext, label: String, source: String, entry: &str) -> Self {
        let bindings = Bindings::parse(&source);
        let specs = bindings.specs(0);
        let handle = context.declare(ComputeProgram::new(&label, source, entry, &[&specs]));
        Self {
            label,
            handle,
            bindings,
        }
    }

    fn pipeline(&self) -> &ComputePipeline {
        self.handle.pipeline()
    }

    fn group(&self, device: &Device, slots: &[(&'static str, TypedSlot<'_>)]) -> BindGroup {
        self.bindings
            .group(&self.label, &self.handle, device, 0, slots)
    }
}

pub fn key_words(elements: u32) -> u32 {
    let bits = 32 - elements.saturating_sub(1).leading_zeros();
    bits.div_ceil(8).clamp(1, 4)
}

pub struct SortChannels<'a> {
    pub generation: u64,
    pub count: TypedSlot<'a>,
    pub major: TypedSlot<'a>,
    pub minor: TypedSlot<'a>,
    pub payload: TypedSlot<'a>,
    pub scratch_major: TypedSlot<'a>,
    pub scratch_minor: TypedSlot<'a>,
    pub scratch_payload: TypedSlot<'a>,
}

#[derive(PartialEq, Eq)]
struct ChannelsKey {
    count: (u64, u64, u64),
    major: (u64, u64, u64),
    minor: (u64, u64, u64),
    payload: (u64, u64, u64),
    scratch_major: (u64, u64, u64),
    scratch_minor: (u64, u64, u64),
    scratch_payload: (u64, u64, u64),
}

impl ChannelsKey {
    fn of(channels: &SortChannels<'_>) -> Self {
        Self {
            count: channels.count.identity(),
            major: channels.major.identity(),
            minor: channels.minor.identity(),
            payload: channels.payload.identity(),
            scratch_major: channels.scratch_major.identity(),
            scratch_minor: channels.scratch_minor.identity(),
            scratch_payload: channels.scratch_payload.identity(),
        }
    }
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
    storage: Option<u64>,
    parity: u32,
}

pub struct RadixSort {
    device: Device,
    prepare: Declared,
    aggregates: [Declared; 8],
    binnings: [Declared; 8],
    copy: Declared,
    rows: GpuBuffer,
    histograms: [GpuBuffer; 2],
    state: std::sync::Mutex<State>,
}

fn region_offset(pass: usize) -> usize {
    pass * SLOTS * BINS as usize
}

fn declare_prepare(context: &GpuContext, label: &str, row: u32) -> Declared {
    let source = include_str!("shaders/sort_prepare.wgsl")
        .replace("__SLOTS__", &SLOTS.to_string())
        .replace("__ROW__", &row.to_string());
    Declared::declare(context, format!("{label} prepare"), source, "main")
}

fn declare_binning(context: &GpuContext, label: &str, row: u32, pass: usize) -> Declared {
    let source = include_str!("shaders/sort_binning.wgsl")
        .replace("__SHIFT__", &(pass * 8).to_string())
        .replace("__DIGIT__", &pass.to_string())
        .replace("__REGION__", &region_offset(pass).to_string())
        .replace("__ROW__", &row.to_string());
    Declared::declare(context, format!("{label} binning {pass}"), source, "main")
}

fn declare_aggregate(context: &GpuContext, label: &str, row: u32, pass: usize) -> Declared {
    let source = include_str!("shaders/sort_aggregate.wgsl")
        .replace("__SHIFT__", &(pass * 8).to_string())
        .replace("__REGION__", &region_offset(pass).to_string())
        .replace("__ROW__", &row.to_string());
    Declared::declare(context, format!("{label} aggregate {pass}"), source, "main")
}

fn declare_copy(context: &GpuContext, label: &str, row: u32) -> Declared {
    let source = include_str!("shaders/sort_copy.wgsl").replace("__ROW__", &row.to_string());
    Declared::declare(context, format!("{label} copy"), source, "main")
}

impl SortBindGroups {
    fn build(
        device: &Device,
        channels: &SortChannels<'_>,
        rows: &GpuBuffer,
        histograms: &[GpuBuffer; 2],
        programs: &RadixSort,
        pass: usize,
    ) -> Self {
        let prepare = &programs.prepare;
        let aggregate = &programs.aggregates[pass];
        let binning = &programs.binnings[pass];
        let copy = &programs.copy;
        let in_minor = [channels.minor, channels.scratch_minor];
        let in_major = [channels.major, channels.scratch_major];
        let in_payload = [channels.payload, channels.scratch_payload];
        let out_minor = [channels.scratch_minor, channels.minor];
        let out_major = [channels.scratch_major, channels.major];
        let out_payload = [channels.scratch_payload, channels.payload];
        let histogram = |index: usize| {
            TypedSlot::new(
                GpuSlot::whole(&histograms[index]),
                StreamElement::new("u32", 4),
            )
        };
        let rows = TypedSlot::new(GpuSlot::whole(rows), StreamElement::new("u32", 4));
        let prepare_groups = std::array::from_fn(|parity| {
            prepare.group(
                device,
                &[
                    ("keys_lo", channels.minor),
                    ("keys_hi", channels.major),
                    ("histogram", histogram(parity)),
                    ("histogram_free", histogram(1 - parity)),
                    ("rows", rows),
                    ("count_holder", channels.count),
                ],
            )
        });
        let aggregate_groups = std::array::from_fn(|source: usize| {
            aggregate.group(
                device,
                &[
                    ("keys_lo", in_minor[source]),
                    ("keys_hi", in_major[source]),
                    ("rows", rows),
                    ("count_holder", channels.count),
                ],
            )
        });
        let binning_groups = std::array::from_fn(|source: usize| {
            std::array::from_fn(|parity: usize| {
                binning.group(
                    device,
                    &[
                        ("keys_lo", in_minor[source]),
                        ("keys_hi", in_major[source]),
                        ("payload_in", in_payload[source]),
                        ("histogram", histogram(parity)),
                        ("rows", rows),
                        ("keys_lo_out", out_minor[source]),
                        ("keys_hi_out", out_major[source]),
                        ("payload_out", out_payload[source]),
                        ("count_holder", channels.count),
                    ],
                )
            })
        });
        let copy_group = copy.group(
            device,
            &[
                ("keys_lo_in", channels.scratch_minor),
                ("keys_hi_in", channels.scratch_major),
                ("payload_in", channels.scratch_payload),
                ("keys_lo_out", channels.minor),
                ("keys_hi_out", channels.major),
                ("payload_out", channels.payload),
                ("count_holder", channels.count),
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
    pub fn new(context: &GpuContext, label: &str, capacity: u32) -> Self {
        let device = context.device().clone();
        let row = context.workgroups_per_row();
        let prepare = declare_prepare(context, label, row);
        let aggregates = std::array::from_fn(|pass| declare_aggregate(context, label, row, pass));
        let binnings = std::array::from_fn(|pass| declare_binning(context, label, row, pass));
        let copy = declare_copy(context, label, row);
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
            capacity < (1 << 30),
            "a sort stream holds at most 2^30 elements, got {capacity}"
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
                storage: None,
                parity: 0,
            }),
        }
    }

    pub fn sort(
        &self,
        recorder: &mut ComputeRecorder,
        channels: &SortChannels<'_>,
        major_words: u32,
        minor_words: u32,
    ) {
        let passes = (0..minor_words)
            .chain(4..4 + major_words)
            .map(|pass| pass as usize)
            .collect::<Vec<_>>();
        assert!(
            !passes.is_empty(),
            "a sort needs at least one key digit to permute the payload"
        );
        let units = SLOTS as u32;
        let mut state = self.state.lock().unwrap();
        if state.storage != Some(channels.generation) {
            state.groups.clear();
            state.storage = Some(channels.generation);
        }
        let parity = state.parity as usize % 2;
        let key = ChannelsKey::of(channels);
        let index = match state.groups.iter().position(|entry| entry.channels == key) {
            Some(index) => index,
            None => {
                state.groups.push(SortBindGroups::build(
                    &self.device,
                    channels,
                    &self.rows,
                    &self.histograms,
                    self,
                    passes[0],
                ));
                state.groups.len() - 1
            }
        };
        let groups = &state.groups[index];
        recorder.record(self.prepare.pipeline(), &[&groups.prepare[parity]], units);
        for (order, pass) in passes.iter().enumerate() {
            if order > 0 {
                let group = &groups.aggregate[order % 2];
                recorder.record(self.aggregates[*pass].pipeline(), &[group], units);
            }
            let group = &groups.binning[order % 2][parity];
            recorder.record(self.binnings[*pass].pipeline(), &[group], units);
        }
        if passes.len() % 2 == 1 {
            recorder.record(self.copy.pipeline(), &[&groups.copy], units);
        }
        state.parity = state.parity.wrapping_add(1);
    }
}
