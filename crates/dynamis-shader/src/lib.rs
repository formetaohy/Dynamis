mod reflection;

use dynamis_abi::{Bound, COUNTER_STRIDE, RECORDS_WGSL, constants_wgsl};
use dynamis_gpu::{GpuContext, ResourceId, SEGMENT_COUNT};
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
pub enum Dispatch {
    Rows,
    Stream(ResourceId),
    Workgroups,
}

pub struct Program {
    source: Arc<str>,
    bindings: Vec<ShaderBinding>,
    dispatch: Dispatch,
    warm: bool,
}

impl Program {
    pub fn new(source: String, dispatch: Dispatch, warm: bool) -> Self {
        Self {
            bindings: reflect(&source),
            source: source.into(),
            dispatch,
            warm,
        }
    }

    pub fn into_parts(self) -> (Arc<str>, Vec<ShaderBinding>, Dispatch, bool) {
        (self.source, self.bindings, self.dispatch, self.warm)
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
    Program::new(source, Dispatch::Rows, false)
}

pub fn stream(
    context: &GpuContext,
    body: &str,
    fragments: &[&str],
    kernel: &str,
    extent: impl Into<ResourceId>,
) -> Program {
    let mut source = assemble(context, body, fragments);
    source.push_str(&entry_stream("main", kernel));
    Program::new(source, Dispatch::Stream(extent.into()), false)
}

pub fn stream_warm(
    context: &GpuContext,
    body: &str,
    fragments: &[&str],
    kernel: &str,
    extent: impl Into<ResourceId>,
) -> Program {
    let mut source = assemble(context, body, fragments);
    source.push_str(&entry_stream("main", kernel));
    source.push_str(&entry_stream("warm", "warm_start"));
    Program::new(source, Dispatch::Stream(extent.into()), true)
}

pub fn workgroups(context: &GpuContext, body: &str, fragments: &[&str]) -> Program {
    Program::new(
        assemble(context, body, fragments),
        Dispatch::Workgroups,
        false,
    )
}
