use dynamis_gpu::{
    ComputePipeline, ComputeProgram, ComputeRecorder, GpuBuffer, GpuContext, GpuSlot,
    PipelineHandle, StorageId, StreamElement, TypedSlot,
};
use dynamis_pass::Bindings;
use std::collections::BTreeMap;
use wgpu::{BindGroup, Device};

const BINS: u32 = 256;
const TILE: u32 = 256;
const UNITS: u32 = 512;
const CHUNK: u32 = 32;
const CHUNKS: u32 = UNITS / CHUNK;
const CACHED_CHANNELS: usize = 4;

const LENGTH: &str = include_str!("shaders/sort_length.wgsl");
const HISTOGRAM: &str = include_str!("shaders/sort_histogram.wgsl");
const PREFIX: &str = include_str!("shaders/sort_prefix.wgsl");
const SCATTER: &str = include_str!("shaders/sort_scatter.wgsl");
const COPY: &str = include_str!("shaders/sort_copy.wgsl");

type Digit = usize;

const _: () = assert!(TILE == BINS, "a radix tile ranks one bin per lane");
const _: () = assert!(
    CHUNK > 0 && UNITS.is_multiple_of(CHUNK),
    "the radix units must divide into whole scan blocks"
);

fn words(buffer: &GpuBuffer) -> TypedSlot<'_> {
    TypedSlot::new(GpuSlot::whole(buffer), StreamElement::new("u32", 4))
}

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

struct ChannelGroups {
    channels: ChannelsKey,
    digits: Vec<Digit>,
    length: BindGroup,
    histogram: Vec<[BindGroup; 2]>,
    prefix: BindGroup,
    scatter: Vec<[BindGroup; 2]>,
    copy: BindGroup,
}

impl ChannelGroups {
    fn holds(&self, channels: &ChannelsKey, digits: &[Digit]) -> bool {
        &self.channels == channels && self.digits == digits
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        device: &Device,
        channels: &SortChannels<'_>,
        length: &GpuBuffer,
        counts: &GpuBuffer,
        offsets: &GpuBuffer,
        block_totals: &GpuBuffer,
        digits: &[Digit],
        declared: &DeclaredSet<'_>,
    ) -> Self {
        let length = words(length);
        let counts = words(counts);
        let offsets = words(offsets);
        let block_totals = words(block_totals);
        let in_minor = [channels.minor, channels.scratch_minor];
        let in_major = [channels.major, channels.scratch_major];
        let in_payload = [channels.payload, channels.scratch_payload];
        let out_minor = [channels.scratch_minor, channels.minor];
        let out_major = [channels.scratch_major, channels.major];
        let out_payload = [channels.scratch_payload, channels.payload];
        let length_group = declared.length.group(
            device,
            &[
                ("keys_len", channels.major),
                ("count_holder", channels.count),
                ("length_holder", length),
            ],
        );
        let histogram = digits
            .iter()
            .map(|digit| {
                std::array::from_fn(|source| {
                    declared.histogram[digit].group(
                        device,
                        &[
                            ("keys_lo", in_minor[source]),
                            ("keys_hi", in_major[source]),
                            ("counts", counts),
                            ("length_holder", length),
                        ],
                    )
                })
            })
            .collect();
        let prefix_group = declared.prefix.group(
            device,
            &[
                ("counts", counts),
                ("offsets", offsets),
                ("block_totals", block_totals),
                ("length_holder", length),
            ],
        );
        let scatter = digits
            .iter()
            .map(|digit| {
                std::array::from_fn(|source| {
                    declared.scatter[digit].group(
                        device,
                        &[
                            ("keys_lo", in_minor[source]),
                            ("keys_hi", in_major[source]),
                            ("payload_in", in_payload[source]),
                            ("offsets", offsets),
                            ("block_totals", block_totals),
                            ("keys_lo_out", out_minor[source]),
                            ("keys_hi_out", out_major[source]),
                            ("payload_out", out_payload[source]),
                            ("length_holder", length),
                        ],
                    )
                })
            })
            .collect();
        let copy_group = declared.copy.group(
            device,
            &[
                ("keys_lo", channels.scratch_minor),
                ("keys_hi", channels.scratch_major),
                ("payload_in", channels.scratch_payload),
                ("keys_lo_out", channels.minor),
                ("keys_hi_out", channels.major),
                ("payload_out", channels.payload),
                ("length_holder", length),
            ],
        );
        Self {
            channels: ChannelsKey::of(channels),
            digits: digits.to_vec(),
            length: length_group,
            histogram,
            prefix: prefix_group,
            scatter,
            copy: copy_group,
        }
    }
}

struct DeclaredSet<'a> {
    length: &'a Declared,
    histogram: &'a BTreeMap<Digit, Declared>,
    scatter: &'a BTreeMap<Digit, Declared>,
    prefix: &'a Declared,
    copy: &'a Declared,
}

fn constants(row: u32, digit: Digit) -> String {
    format!(
        "\
const TILE: u32 = {TILE}u;
const BINS: u32 = {BINS}u;
const BINS_MASK: u32 = {}u;
const UNITS: u32 = {UNITS}u;
const UNITS_PER_ROW: u32 = {row}u;
const CHUNK: u32 = {CHUNK}u;
const CHUNKS: u32 = {CHUNKS}u;
const DIGIT_SHIFT: u32 = {}u;
const DIGIT_WORD: u32 = {}u;
",
        BINS - 1,
        (digit % 4) * 8,
        u32::from(digit >= 4),
    )
}

fn partition(row: u32, digit: Digit) -> String {
    format!(
        "{}
struct SortTiles {{
    first: u32,
    last: u32,
}}

fn sort_length() -> u32 {{
    return length_holder[0];
}}

fn sort_spans(length: u32) -> u32 {{
    return (length + TILE - 1u) / TILE;
}}

fn sort_units(length: u32) -> u32 {{
    return min(UNITS, max(sort_spans(length), 1u));
}}

fn sort_tiles(unit: u32, length: u32) -> SortTiles {{
    let spans = sort_spans(length);
    let units = sort_units(length);
    let span = (spans + units - 1u) / units;
    var tiles: SortTiles;
    tiles.first = min(unit * span, spans);
    tiles.last = min(tiles.first + span, spans);
    return tiles;
}}
",
        constants(row, digit),
    )
}

fn declare_digit(
    context: &GpuContext,
    label: &str,
    body: &str,
    row: u32,
    digit: Digit,
) -> Declared {
    let source = body.replace("__PARTITION__", &partition(row, digit));
    Declared::declare(context, format!("{label} digit {digit}"), source, "main")
}

fn digit_passes(minor_words: u32, major_words: u32) -> Vec<Digit> {
    (0..minor_words)
        .chain(4..4 + major_words)
        .map(|pass| pass as Digit)
        .collect()
}

pub struct RadixSort {
    device: Device,
    length: Declared,
    histogram: BTreeMap<Digit, Declared>,
    scatter: BTreeMap<Digit, Declared>,
    prefix: Declared,
    copy: Declared,
    length_holder: GpuBuffer,
    counts: GpuBuffer,
    offsets: GpuBuffer,
    block_totals: GpuBuffer,
    groups: Vec<ChannelGroups>,
}

impl RadixSort {
    pub fn new(context: &GpuContext, label: &str) -> Self {
        assert!(
            CHUNKS <= context.workgroups_per_row(),
            "the radix scan blocks must fit one dispatch row"
        );
        let device = context.device().clone();
        let row = context.workgroups_per_row();
        let length = Declared::declare(
            context,
            format!("{label} length"),
            LENGTH.to_owned(),
            "main",
        );
        let histogram = (0..8)
            .map(|digit| (digit, declare_digit(context, label, HISTOGRAM, row, digit)))
            .collect();
        let scatter = (0..8)
            .map(|digit| (digit, declare_digit(context, label, SCATTER, row, digit)))
            .collect();
        let prefix = Declared::declare(
            context,
            format!("{label} prefix"),
            PREFIX.replace("__PARTITION__", &partition(row, 0)),
            "main",
        );
        let copy = Declared::declare(
            context,
            format!("{label} copy"),
            COPY.replace("__PARTITION__", &partition(row, 0)),
            "main",
        );
        let storage = wgpu::BufferUsages::STORAGE;
        let table = (UNITS * BINS * 4) as u64;
        let block_table = (CHUNKS * BINS * 4) as u64;
        let length_holder = GpuBuffer::new(&device, &format!("{label} length"), 4, storage);
        let counts = GpuBuffer::new(&device, &format!("{label} counts"), table, storage);
        let offsets = GpuBuffer::new(&device, &format!("{label} offsets"), table, storage);
        let block_totals = GpuBuffer::new(
            &device,
            &format!("{label} block totals"),
            block_table,
            storage,
        );
        Self {
            device,
            length,
            histogram,
            scatter,
            prefix,
            copy,
            length_holder,
            counts,
            offsets,
            block_totals,
            groups: Vec::new(),
        }
    }

    pub fn sort(
        &mut self,
        recorder: &mut ComputeRecorder,
        channels: &SortChannels<'_>,
        major_words: u32,
        minor_words: u32,
    ) {
        let digits = digit_passes(minor_words, major_words);
        assert!(
            !digits.is_empty(),
            "a sort needs at least one key digit to permute the payload"
        );
        let Self {
            device,
            length,
            histogram,
            scatter,
            prefix,
            copy,
            length_holder,
            counts,
            offsets,
            block_totals,
            groups,
        } = self;
        let key = ChannelsKey::of(channels);
        let index = groups
            .iter()
            .position(|entry| entry.holds(&key, &digits))
            .unwrap_or_else(|| {
                if groups.len() == CACHED_CHANNELS {
                    groups.remove(0);
                }
                let declared = DeclaredSet {
                    length,
                    histogram,
                    scatter,
                    prefix,
                    copy,
                };
                groups.push(ChannelGroups::build(
                    device,
                    channels,
                    length_holder,
                    counts,
                    offsets,
                    block_totals,
                    &digits,
                    &declared,
                ));
                groups.len() - 1
            });
        let bound = &groups[index];
        recorder.record(length.pipeline(), &[&bound.length], 1);
        for (order, digit) in digits.iter().enumerate() {
            let source = order % 2;
            recorder.record(
                histogram[digit].pipeline(),
                &[&bound.histogram[order][source]],
                UNITS,
            );
            recorder.record(prefix.pipeline(), &[&bound.prefix], CHUNKS);
            recorder.record(
                scatter[digit].pipeline(),
                &[&bound.scatter[order][source]],
                UNITS,
            );
        }
        if digits.len() % 2 == 1 {
            recorder.record(copy.pipeline(), &[&bound.copy], UNITS);
        }
    }
}
