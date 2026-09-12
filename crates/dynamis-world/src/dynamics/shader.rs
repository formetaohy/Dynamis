use crate::dynamics::engine::{
    Dispatch, Program, ResourceId, WORKGROUP_SIZE, entry_rows, entry_stream,
};
use crate::dynamics::rigid::Count;
use dynamis_gpu::GpuContext;
use dynamis_layout::{ABI_WGSL, COUNTER_STRIDE, constants_wgsl};

pub(crate) const CORE_FRAGMENT: &str = include_str!("shaders/core.wgsl");
pub(crate) const GRID_INDEX_FRAGMENT: &str = include_str!("shaders/grid_index.wgsl");
pub(crate) const IDENTITY_FRAGMENT: &str = include_str!("shaders/identity.wgsl");
pub(crate) const EVENTS_FRAGMENT: &str = include_str!("shaders/events.wgsl");
pub(crate) const CONVEX_FRAGMENT: &str = include_str!("shaders/convex.wgsl");
pub(crate) const SCENE_FRAGMENT: &str = include_str!("shaders/scene.wgsl");
pub(crate) const CONTACT_BLOCK_FRAGMENT: &str = include_str!("shaders/solver_contact_block.wgsl");
pub(crate) const CONSTRAINT_BLOCK_FRAGMENT: &str =
    include_str!("shaders/solver_constraint_block.wgsl");
pub(crate) const POSITION_CORRECTION_FRAGMENT: &str =
    include_str!("shaders/position_correction.wgsl");
pub(crate) const SHAPES_FRAGMENT: &str = include_str!("shaders/shapes.wgsl");

pub(crate) const CORE: &[&str] = &[];
pub(crate) const IDENTITY: &[&str] = &[IDENTITY_FRAGMENT];
pub(crate) const CONTACT: &[&str] = &[IDENTITY_FRAGMENT, EVENTS_FRAGMENT];
pub(crate) const GEOMETRY: &[&str] = &[CONVEX_FRAGMENT, SCENE_FRAGMENT];
pub(crate) const GRID_INDEX: &[&str] = &[GRID_INDEX_FRAGMENT];
pub(crate) const GEOMETRY_INDEX: &[&str] = &[GRID_INDEX_FRAGMENT, CONVEX_FRAGMENT, SCENE_FRAGMENT];
pub(crate) const BLOCKS: &[&str] = &[CONTACT_BLOCK_FRAGMENT, CONSTRAINT_BLOCK_FRAGMENT];
pub(crate) const POSITION_CORRECTION: &[&str] = &[POSITION_CORRECTION_FRAGMENT];

pub(crate) fn assemble(context: &GpuContext, body: &str, fragments: &[&str]) -> String {
    let mut source = shader_constants(context.workgroups_per_row());
    source.push_str(ABI_WGSL);
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
    source.push_str(&format!(
        "const EVENT_SLOTS: u32 = {}u;\n",
        crate::dynamics::EVENT_SLOTS
    ));
    source.push_str(&format!(
        "const COUNTER_STRIDE_WORDS: u32 = {}u;\n",
        COUNTER_STRIDE / 4
    ));
    source
}

pub(crate) fn rows(context: &GpuContext, body: &str, fragments: &[&str], count: Count) -> Program {
    let mut source = assemble(context, body, fragments);
    source.push_str(&entry_rows(count.field()));
    Program {
        source: source.into(),
        dispatch: Dispatch::Rows,
        warm: false,
    }
}

pub(crate) fn stream(
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

pub(crate) fn stream_warm(
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

pub(crate) fn workgroups(context: &GpuContext, body: &str, fragments: &[&str]) -> Program {
    Program {
        source: assemble(context, body, fragments).into(),
        dispatch: Dispatch::Workgroups,
        warm: false,
    }
}
