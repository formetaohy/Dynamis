use dynamis_gpu::{
    BindingSpec, ComputePipeline, ComputeProgram, ComputeRecorder, GpuBuffer, GpuContext, GpuSlot,
    PipelineHandle, ShaderBinding, parse_bindings,
};
use wgpu::{BindGroup, BindGroupEntry, Device};

const THREADS: u32 = 256;
const BINS: u32 = 256;
const BINS_ALL: usize = BINS as usize * 8;
const REGIONS: u32 = 8;
const SLOTS: usize = 64;

struct Declared {
    handle: PipelineHandle,
    bindings: Vec<ShaderBinding>,
}

impl Declared {
    fn declare(context: &GpuContext, label: String, source: String, entry: &str) -> Self {
        let bindings = parse_bindings(&source);
        let specs = binding_specs(&bindings);
        let handle = context.declare(ComputeProgram::new(&label, source, entry, &[&specs]));
        Self { handle, bindings }
    }

    fn pipeline(&self) -> &ComputePipeline {
        self.handle.pipeline()
    }

    fn group(&self, device: &Device, resources: &[(&str, GpuSlot<'_>)]) -> BindGroup {
        assert!(
            self.bindings.len() == resources.len(),
            "the shader declares {} bindings but the sort provides {}",
            self.bindings.len(),
            resources.len()
        );
        let entries = self
            .bindings
            .iter()
            .map(|binding| {
                let resource = resources
                    .iter()
                    .find(|(name, _)| *name == binding.name)
                    .unwrap_or_else(|| {
                        panic!("the shader never declares the binding {:?}", binding.name)
                    });
                BindGroupEntry {
                    binding: binding.binding,
                    resource: resource.1.as_binding(),
                }
            })
            .collect::<Vec<_>>();
        self.handle.create_bind_group(device, 0, &entries)
    }
}

fn binding_specs(bindings: &[ShaderBinding]) -> Vec<BindingSpec> {
    let mut ordered = bindings
        .iter()
        .filter(|binding| binding.group == 0)
        .collect::<Vec<_>>();
    ordered.sort_by_key(|binding| binding.binding);
    for (position, binding) in ordered.iter().enumerate() {
        assert!(
            binding.binding == position as u32,
            "the shader binds index {} where {position} is required",
            binding.binding
        );
    }
    ordered
        .into_iter()
        .map(|binding| BindingSpec {
            binding: binding.binding,
            kind: binding.kind,
        })
        .collect()
}

pub fn key_words(elements: u32) -> u32 {
    let bits = 32 - elements.saturating_sub(1).leading_zeros();
    bits.div_ceil(8).clamp(1, 4)
}

pub struct SortChannels<'a> {
    pub generation: u64,
    pub count: GpuSlot<'a>,
    pub major: GpuSlot<'a>,
    pub minor: GpuSlot<'a>,
    pub payload: GpuSlot<'a>,
    pub scratch_major: GpuSlot<'a>,
    pub scratch_minor: GpuSlot<'a>,
    pub scratch_payload: GpuSlot<'a>,
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
        let prepare_groups = std::array::from_fn(|parity| {
            prepare.group(
                device,
                &[
                    ("keys_lo", channels.minor),
                    ("keys_hi", channels.major),
                    ("histogram", GpuSlot::whole(&histograms[parity])),
                    ("histogram_free", GpuSlot::whole(&histograms[1 - parity])),
                    ("rows", GpuSlot::whole(rows)),
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
                    ("rows", GpuSlot::whole(rows)),
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
                        ("histogram", GpuSlot::whole(&histograms[parity])),
                        ("rows", GpuSlot::whole(rows)),
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
        state.parity = state.parity.wrapping_add(1);
    }
}
