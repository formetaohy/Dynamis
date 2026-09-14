use dynamis_gpu::{
    ComputePipeline, ComputeProgram, ComputeRecorder, GpuBuffer, GpuContext, GpuSlot,
    PipelineHandle, StorageId, StreamElement, TypedSlot,
};
use dynamis_pass::Bindings;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use wgpu::{BindGroup, Device};

const BINS: u32 = 256;
const BINS_ALL: usize = BINS as usize * 8;
const REGIONS: u32 = 8;
const SLOTS: usize = 64;
const CACHED_CHANNELS: usize = 4;

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
    count: StorageId,
    major: StorageId,
    minor: StorageId,
    payload: StorageId,
    scratch_major: StorageId,
    scratch_minor: StorageId,
    scratch_payload: StorageId,
}

impl ChannelsKey {
    fn of(channels: &SortChannels<'_>) -> Self {
        Self {
            count: channels.count.storage_id(),
            major: channels.major.storage_id(),
            minor: channels.minor.storage_id(),
            payload: channels.payload.storage_id(),
            scratch_major: channels.scratch_major.storage_id(),
            scratch_minor: channels.scratch_minor.storage_id(),
            scratch_payload: channels.scratch_payload.storage_id(),
        }
    }
}

struct SortBindGroups {
    channels: ChannelsKey,
    prepare: [BindGroup; 2],
    binning: [[BindGroup; 2]; 2],
    copy: BindGroup,
}

struct State {
    groups: Vec<SortBindGroups>,
    parity: u32,
}

type Digit = usize;
type NextDigit = Option<Digit>;
type Plan = (Digit, NextDigit, NextDigit);

fn digit_passes(minor_words: u32, major_words: u32) -> Vec<Digit> {
    (0..minor_words)
        .chain(4..4 + major_words)
        .map(|pass| pass as Digit)
        .collect()
}

fn digit_declarations() -> (BTreeSet<(Digit, NextDigit)>, BTreeSet<Plan>) {
    let mut leading = BTreeSet::new();
    let mut plans = BTreeSet::new();
    for minor_words in 0..=4u32 {
        for major_words in 0..=4u32 {
            let passes = digit_passes(minor_words, major_words);
            if let Some(first) = passes.first() {
                leading.insert((*first, passes.get(1).copied()));
            }
            for (order, digit) in passes.iter().enumerate() {
                plans.insert((
                    *digit,
                    passes.get(order + 1).copied(),
                    passes.get(order + 2).copied(),
                ));
            }
        }
    }
    (leading, plans)
}

fn region(offset: usize) -> u32 {
    (offset * SLOTS * BINS as usize) as u32
}

fn declare_prepare(
    context: &GpuContext,
    label: &str,
    row: u32,
    first: Digit,
    second: Option<Digit>,
) -> Declared {
    let source = include_str!("shaders/sort_prepare.wgsl")
        .replace("__SLOTS__", &SLOTS.to_string())
        .replace("__ROW__", &row.to_string())
        .replace("__FIRST_DIGIT__", &first.to_string())
        .replace("__FIRST_REGION__", &region(first).to_string())
        .replace(
            "__SECOND_REGION__",
            &second.map(region).unwrap_or_default().to_string(),
        )
        .replace("__HAS_SECOND__", &u32::from(second.is_some()).to_string());
    Declared::declare(
        context,
        format!("{label} prepare {first} -> {second:?}"),
        source,
        "main",
    )
}

fn declare_binning(context: &GpuContext, label: &str, row: u32, plan: Plan) -> Declared {
    let (digit, next, zero) = plan;
    let source = include_str!("shaders/sort_binning.wgsl")
        .replace("__SHIFT__", &(digit * 8).to_string())
        .replace("__DIGIT__", &digit.to_string())
        .replace("__REGION__", &region(digit).to_string())
        .replace("__ROW__", &row.to_string())
        .replace(
            "__NEXT_REGION__",
            &next.map(region).unwrap_or_default().to_string(),
        )
        .replace(
            "__NEXT_SHIFT__",
            &next.map(|digit| digit * 8).unwrap_or_default().to_string(),
        )
        .replace(
            "__NEXT_WORD__",
            &next
                .map(|digit| u32::from(digit >= 4))
                .unwrap_or_default()
                .to_string(),
        )
        .replace("__HAS_NEXT__", &u32::from(next.is_some()).to_string())
        .replace(
            "__ZERO_REGION__",
            &zero.map(region).unwrap_or_default().to_string(),
        )
        .replace("__HAS_ZERO__", &u32::from(zero.is_some()).to_string());
    Declared::declare(context, format!("{label} binning {plan:?}"), source, "main")
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
        prepare: &Declared,
        binning: &Declared,
        copy: &Declared,
    ) -> Self {
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
            binning: binning_groups,
            copy: copy_group,
        }
    }
}

pub struct RadixSort {
    device: Device,
    prepare: BTreeMap<(Digit, NextDigit), Declared>,
    binning: BTreeMap<Plan, Declared>,
    copy: Declared,
    rows: GpuBuffer,
    histograms: [GpuBuffer; 2],
    state: std::sync::Mutex<State>,
}

impl RadixSort {
    pub fn new(context: &GpuContext, label: &str, capacity: u32) -> Self {
        let device = context.device().clone();
        let row = context.workgroups_per_row();
        assert!(
            capacity < (1 << 30),
            "a sort stream holds at most 2^30 elements, got {capacity}"
        );
        let (leading, plans) = digit_declarations();
        assert!(
            !leading.is_empty() && !plans.is_empty(),
            "a sort declares at least one digit plan"
        );
        let prepare = leading
            .into_iter()
            .map(|plan| (plan, declare_prepare(context, label, row, plan.0, plan.1)))
            .collect::<BTreeMap<_, _>>();
        let binning = plans
            .into_iter()
            .map(|plan| (plan, declare_binning(context, label, row, plan)))
            .collect::<BTreeMap<_, _>>();
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
        Self {
            device,
            prepare,
            binning,
            copy,
            rows,
            histograms,
            state: std::sync::Mutex::new(State {
                groups: Vec::new(),
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
        let passes = digit_passes(minor_words, major_words);
        assert!(
            !passes.is_empty(),
            "a sort needs at least one key digit to permute the payload"
        );
        let units = SLOTS as u32;
        let mut state = self.state.lock().unwrap();
        let parity = state.parity as usize % 2;
        let key = ChannelsKey::of(channels);
        let index = state
            .groups
            .iter()
            .position(|entry| entry.channels == key)
            .unwrap_or_else(|| {
                if state.groups.len() == CACHED_CHANNELS {
                    state.groups.remove(0);
                }
                state.groups.push(SortBindGroups::build(
                    &self.device,
                    channels,
                    &self.rows,
                    &self.histograms,
                    self.prepare
                        .values()
                        .next()
                        .expect("a sort declares its leading digits"),
                    self.binning
                        .values()
                        .next()
                        .expect("a sort declares its digit plans"),
                    &self.copy,
                ));
                state.groups.len() - 1
            });
        let groups = &state.groups[index];
        let leading = (passes[0], passes.get(1).copied());
        recorder.record(
            self.prepare[&leading].pipeline(),
            &[&groups.prepare[parity]],
            units,
        );
        for (order, digit) in passes.iter().enumerate() {
            let plan = (
                *digit,
                passes.get(order + 1).copied(),
                passes.get(order + 2).copied(),
            );
            let group = &groups.binning[order % 2][parity];
            recorder.record(self.binning[&plan].pipeline(), &[group], units);
        }
        if passes.len() % 2 == 1 {
            recorder.record(self.copy.pipeline(), &[&groups.copy], units);
        }
        state.parity = state.parity.wrapping_add(1);
    }
}
