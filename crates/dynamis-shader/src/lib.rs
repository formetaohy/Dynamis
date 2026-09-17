mod reflection;

use dynamis_abi::{Bound, COUNTER_STRIDE, RECORDS_WGSL, constants_wgsl};
use dynamis_gpu::{GpuContext, ResourceId, SEGMENT_COUNT, SlotRef};
use std::sync::Arc;

pub use reflection::{ShaderBinding, reflect};

const CORE_FRAGMENT: &str = include_str!("../shaders/core.wgsl");
pub const SCENE_CAST: &str = include_str!("../shaders/scene_cast.wgsl");
pub const COUNTER_ACCESS: &str = include_str!("../shaders/counters.wgsl");
const GRID_INDEX_FRAGMENT: &str = include_str!("../shaders/grid_index.wgsl");
const CONVEX_FRAGMENT: &str = include_str!("../shaders/convex.wgsl");
const SCENE_FRAGMENT: &str = include_str!("../shaders/scene.wgsl");
const SHAPES_FRAGMENT: &str = include_str!("../shaders/shapes.wgsl");
const JOINTS_FRAGMENT: &str = include_str!("../shaders/joints.wgsl");
const CONTACT_FACT_FRAGMENT: &str = include_str!("../shaders/contact_fact.wgsl");

pub const CORE: &[&str] = &[];
pub const CONTACT_FACT: &str = CONTACT_FACT_FRAGMENT;
pub const COUNTERS: &[&str] = &[COUNTER_ACCESS];
pub const GEOMETRY: &[&str] = &[CONVEX_FRAGMENT, SCENE_FRAGMENT];
pub const GRID_INDEX: &[&str] = &[COUNTER_ACCESS, GRID_INDEX_FRAGMENT];
pub const GEOMETRY_INDEX: &[&str] = &[
    COUNTER_ACCESS,
    GRID_INDEX_FRAGMENT,
    CONVEX_FRAGMENT,
    SCENE_FRAGMENT,
];
pub const JOINTS: &[&str] = &[JOINTS_FRAGMENT];

pub const WORKGROUP_SIZE: u32 = 64;

pub fn workgroups_of(elements: u32) -> u32 {
    elements.div_ceil(WORKGROUP_SIZE)
}

pub fn entry_rows(bound: Bound) -> String {
    let bound = bound.expression();
    format!(
        "
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {{
    let index = global_index(gid);
    if (index >= {bound}) {{
        return;
    }}
    work(index);
}}
"
    )
}

pub fn entry_stream(name: &str, kernel: &str) -> String {
    assert!(
        kernel != "main" && kernel != "warm",
        "the streaming kernel {kernel:?} collides with the generated entry point"
    );
    format!(
        "
@compute @workgroup_size(WORKGROUP_SIZE)
fn {name}(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {{
    let live = extent();
    let stride = grid_stride(groups);
    for (var index = global_index(gid); index < live; index = index + stride) {{
        {kernel}(index);
    }}
}}
"
    )
}

#[derive(Clone, Copy)]
pub enum Extent {
    None,
    Shared {
        counter: usize,
        array: &'static str,
        lanes: u32,
    },
    Slot {
        counter: usize,
        count: &'static str,
        array: &'static str,
        lanes: u32,
    },
}

impl Extent {
    pub const fn of_shared_counter(counter: usize, array: &'static str) -> Self {
        Self::Shared {
            counter,
            array,
            lanes: 1,
        }
    }

    pub const fn slot(counter: usize, count: &'static str, array: &'static str) -> Self {
        Self::Slot {
            counter,
            count,
            array,
            lanes: 1,
        }
    }

    pub const fn lanes(self, lanes: u32) -> Self {
        assert!(lanes > 0, "an extent lane width must be positive");
        match self {
            Self::None => Self::None,
            Self::Shared { counter, array, .. } => Self::Shared {
                counter,
                array,
                lanes,
            },
            Self::Slot {
                counter,
                count,
                array,
                ..
            } => Self::Slot {
                counter,
                count,
                array,
                lanes,
            },
        }
    }

    pub const fn is_none(self) -> bool {
        matches!(self, Self::None)
    }

    fn read(self) -> String {
        match self {
            Self::None => panic!("a stage without a declared extent reads no counter"),
            Self::Shared { counter, .. } => {
                format!("counter_load({})", dynamis_abi::COUNTERS[counter].name)
            }
            Self::Slot { count, .. } => format!("atomicLoad(&{count}[0])"),
        }
    }

    fn bounds(self) -> String {
        let (array, lanes) = match self {
            Self::None => panic!("a stage without a declared extent guards no array"),
            Self::Shared { array, lanes, .. } | Self::Slot { array, lanes, .. } => (array, lanes),
        };
        match lanes {
            1 => format!("arrayLength(&{array})"),
            lanes => format!("arrayLength(&{array}) / {lanes}u"),
        }
    }

    pub fn entry(self) -> String {
        format!(
            "
fn extent() -> u32 {{
    return min({}, {});
}}
",
            self.read(),
            self.bounds(),
        )
    }

    pub fn assert_declared(
        self,
        label: &str,
        slots: &[(&'static str, SlotRef)],
    ) -> Option<ResourceId> {
        match self {
            Self::None => None,
            Self::Shared { counter, array, .. } => {
                let spec = dynamis_abi::COUNTERS[counter];
                assert!(
                    matches!(slot(slots, label, "counters"), SlotRef::Whole { .. }),
                    "{label:?} bounds {} by the whole counter stream",
                    spec.label,
                );
                Some(slot(slots, label, array).resource())
            }
            Self::Slot { array, .. } => {
                assert_declared_counter(self, label, slots);
                Some(slot(slots, label, array).resource())
            }
        }
    }

    pub fn counter(self, label: &str, slots: &[(&'static str, SlotRef)]) -> usize {
        match self {
            Self::None => {
                panic!("{label:?} streams over no device counter while its dispatch is streamed")
            }
            Self::Shared { counter, .. } => {
                let spec = dynamis_abi::COUNTERS[counter];
                assert!(
                    matches!(slot(slots, label, "counters"), SlotRef::Whole { .. }),
                    "{label:?} bounds {} by the whole counter stream",
                    spec.label,
                );
                counter
            }
            Self::Slot { counter, .. } => {
                assert_declared_counter(self, label, slots);
                counter
            }
        }
    }
}

fn assert_declared_counter(extent: Extent, label: &str, slots: &[(&'static str, SlotRef)]) {
    let Extent::Slot { counter, count, .. } = extent else {
        panic!("{label:?} streams over a shared counter while its array is absent")
    };
    let spec = dynamis_abi::COUNTERS[counter];
    let SlotRef::Range { offset, size, .. } = *slot(slots, label, count) else {
        panic!(
            "{label:?} bounds {} by the whole counter stream instead of the {} counter",
            spec.label, spec.name,
        );
    };
    assert!(
        offset == counter as u64 * dynamis_abi::COUNTER_STRIDE && size == 4,
        "{label:?} bounds {} by {offset} bytes into the counter stream instead of {}",
        spec.label,
        spec.name,
    );
}

fn slot<'a>(slots: &'a [(&'static str, SlotRef)], label: &str, name: &str) -> &'a SlotRef {
    slots
        .iter()
        .find(|(bound, _)| *bound == name)
        .map(|(_, slot)| slot)
        .unwrap_or_else(|| panic!("{label:?} declares no binding {name:?}"))
}

#[derive(Clone, Copy)]
pub enum Dispatch {
    Rows,
    Stream,
    Workgroups,
}

pub struct Program {
    source: Arc<str>,
    bindings: Vec<ShaderBinding>,
    dispatch: Dispatch,
    extent: Extent,
    warm: bool,
}

impl Program {
    pub fn new(source: String, dispatch: Dispatch, extent: Extent, warm: bool) -> Self {
        Self {
            bindings: reflect(&source),
            source: source.into(),
            dispatch,
            extent,
            warm,
        }
    }

    pub fn into_parts(self) -> (Arc<str>, Vec<ShaderBinding>, Dispatch, Extent, bool) {
        (
            self.source,
            self.bindings,
            self.dispatch,
            self.extent,
            self.warm,
        )
    }
}

pub fn assemble(context: &GpuContext, body: &str, fragments: &[&str]) -> String {
    let mut source = shader_constants(context.workgroups_per_row());
    source.push_str(RECORDS_WGSL);
    source.push('\n');
    source.push_str(CORE_FRAGMENT);
    source.push('\n');
    for fragment in fragments {
        source.push_str(fragment);
        source.push('\n');
    }
    source.push_str(body);
    source.push('\n');
    source.push_str(SHAPES_FRAGMENT);
    source.push('\n');
    source
}

fn shader_constants(per_row: u32) -> String {
    let mut source = constants_wgsl();
    source.push_str(&format!("const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;\n"));
    source.push_str(&format!("const WORKGROUPS_PER_ROW: u32 = {per_row}u;\n"));
    source.push_str(&format!("const SEGMENT_COUNT: u32 = {SEGMENT_COUNT}u;\n"));
    source.push_str(&format!(
        "const COUNTER_STRIDE_WORDS: u32 = {}u;\n",
        COUNTER_STRIDE / 4
    ));
    source
}

pub fn rows(context: &GpuContext, body: &str, fragments: &[&str], bound: Bound) -> Program {
    let mut source = assemble(context, body, fragments);
    source.push_str(&entry_rows(bound));
    Program::new(source, Dispatch::Rows, Extent::None, false)
}

pub fn stream(
    context: &GpuContext,
    body: &str,
    fragments: &[&str],
    kernel: &str,
    extent: Extent,
) -> Program {
    assert!(
        !extent.is_none(),
        "a streaming stage must declare the device fact its work covers",
    );
    let mut source = assemble(context, body, fragments);
    source.push_str(&extent.entry());
    source.push_str(&entry_stream("main", kernel));
    Program::new(source, Dispatch::Stream, extent, false)
}

pub fn stream_warm(
    context: &GpuContext,
    body: &str,
    fragments: &[&str],
    kernel: &str,
    extent: Extent,
) -> Program {
    assert!(
        !extent.is_none(),
        "a streaming stage must declare the device fact its work covers",
    );
    let mut source = assemble(context, body, fragments);
    source.push_str(&extent.entry());
    source.push_str(&entry_stream("main", kernel));
    source.push_str(&entry_stream("warm", "warm_start"));
    Program::new(source, Dispatch::Stream, extent, true)
}

pub fn workgroups(context: &GpuContext, body: &str, fragments: &[&str]) -> Program {
    Program::new(
        assemble(context, body, fragments),
        Dispatch::Workgroups,
        Extent::None,
        false,
    )
}
