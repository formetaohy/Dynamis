use dynamis_abi::{COUNTER_STRIDE, RECORDS_WGSL, constants_wgsl};
use dynamis_gpu::{EVENT_SLOTS, GpuContext, ResourceId};
use std::sync::Arc;

const CORE_FRAGMENT: &str = include_str!("../shaders/core.wgsl");
const GRID_INDEX_FRAGMENT: &str = include_str!("../shaders/grid_index.wgsl");
const CONVEX_FRAGMENT: &str = include_str!("../shaders/convex.wgsl");
const SCENE_FRAGMENT: &str = include_str!("../shaders/scene.wgsl");
const SHAPES_FRAGMENT: &str = include_str!("../shaders/shapes.wgsl");
const JOINTS_FRAGMENT: &str = include_str!("../shaders/joints.wgsl");

pub const CORE: &[&str] = &[];
pub const GEOMETRY: &[&str] = &[CONVEX_FRAGMENT, SCENE_FRAGMENT];
pub const GRID_INDEX: &[&str] = &[GRID_INDEX_FRAGMENT];
pub const GEOMETRY_INDEX: &[&str] = &[GRID_INDEX_FRAGMENT, CONVEX_FRAGMENT, SCENE_FRAGMENT];
pub const JOINTS: &[&str] = &[JOINTS_FRAGMENT];

pub const WORKGROUP_SIZE: u32 = 64;

pub fn workgroups_of(elements: u32) -> u32 {
    elements.div_ceil(WORKGROUP_SIZE)
}

pub fn entry_rows(field: &str) -> String {
    format!(
        "
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {{
    let index = global_index(gid);
    if (index >= params.{field}) {{
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
    pub source: Arc<str>,
    pub dispatch: Dispatch,
    pub warm: bool,
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
    source.push_str(&format!("const EVENT_SLOTS: u32 = {EVENT_SLOTS}u;\n"));
    source.push_str(&format!(
        "const COUNTER_STRIDE_WORDS: u32 = {}u;\n",
        COUNTER_STRIDE / 4
    ));
    source
}

pub fn rows(context: &GpuContext, body: &str, fragments: &[&str], field: &str) -> Program {
    let mut source = assemble(context, body, fragments);
    source.push_str(&entry_rows(field));
    Program {
        source: source.into(),
        dispatch: Dispatch::Rows,
        warm: false,
    }
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
    Program {
        source: source.into(),
        dispatch: Dispatch::Stream(extent.into()),
        warm: false,
    }
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
    Program {
        source: source.into(),
        dispatch: Dispatch::Stream(extent.into()),
        warm: true,
    }
}

pub fn workgroups(context: &GpuContext, body: &str, fragments: &[&str]) -> Program {
    Program {
        source: assemble(context, body, fragments).into(),
        dispatch: Dispatch::Workgroups,
        warm: false,
    }
}
