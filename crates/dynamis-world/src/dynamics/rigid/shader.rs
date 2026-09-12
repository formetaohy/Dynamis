use super::Count;
use crate::dynamics::engine::{Dispatch, Program, WORKGROUP_SIZE, entry_rows, entry_stream};
use dynamis_gpu::GpuContext;
use dynamis_layout::{ABI_WGSL, COUNTER_STRIDE, constants_wgsl};

pub(super) const CORE_FRAGMENT: &str = include_str!("shaders/core.wgsl");
pub(super) const GRID_INDEX_FRAGMENT: &str = include_str!("shaders/grid_index.wgsl");
pub(super) const IDENTITY_FRAGMENT: &str = include_str!("shaders/identity.wgsl");
pub(super) const EVENTS_FRAGMENT: &str = include_str!("shaders/events.wgsl");
pub(super) const CONVEX_FRAGMENT: &str = include_str!("shaders/convex.wgsl");
pub(super) const SCENE_FRAGMENT: &str = include_str!("shaders/scene.wgsl");
pub(super) const CONTACT_BLOCK_FRAGMENT: &str = include_str!("shaders/solver_contact_block.wgsl");
pub(super) const CONSTRAINT_BLOCK_FRAGMENT: &str =
    include_str!("shaders/solver_constraint_block.wgsl");
pub(super) const POSITION_CORRECTION_FRAGMENT: &str =
    include_str!("shaders/position_correction.wgsl");
pub(super) const SHAPES_FRAGMENT: &str = include_str!("shaders/shapes.wgsl");

pub(super) const CORE: &[&str] = &[];
pub(super) const IDENTITY: &[&str] = &[IDENTITY_FRAGMENT];
pub(super) const CONTACT: &[&str] = &[IDENTITY_FRAGMENT, EVENTS_FRAGMENT];
pub(super) const GEOMETRY: &[&str] = &[CONVEX_FRAGMENT, SCENE_FRAGMENT];
pub(super) const GRID_INDEX: &[&str] = &[GRID_INDEX_FRAGMENT];
pub(super) const GEOMETRY_INDEX: &[&str] = &[GRID_INDEX_FRAGMENT, CONVEX_FRAGMENT, SCENE_FRAGMENT];
pub(super) const BLOCKS: &[&str] = &[CONTACT_BLOCK_FRAGMENT, CONSTRAINT_BLOCK_FRAGMENT];
pub(super) const POSITION_CORRECTION: &[&str] = &[POSITION_CORRECTION_FRAGMENT];

pub(super) fn assemble(context: &GpuContext, body: &str, fragments: &[&str]) -> String {
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
        crate::dynamics::rigid::buffers::EVENT_SLOTS
    ));
    source.push_str(&format!(
        "const COUNTER_STRIDE_WORDS: u32 = {}u;\n",
        COUNTER_STRIDE / 4
    ));
    source
}

pub(super) fn rows(context: &GpuContext, body: &str, fragments: &[&str], count: Count) -> Program {
    let mut source = assemble(context, body, fragments);
    source.push_str(&entry_rows(count.field()));
    Program {
        source: source.into(),
        dispatch: Dispatch::rows(),
        warm: false,
    }
}

pub(super) fn stream(
    context: &GpuContext,
    body: &str,
    fragments: &[&str],
    kernel: &str,
    slots: u32,
) -> Program {
    let mut source = assemble(context, body, fragments);
    source.push_str(&entry_stream("main", kernel));
    Program {
        source: source.into(),
        dispatch: Dispatch::stream(slots),
        warm: false,
    }
}

pub(super) fn stream_warm(
    context: &GpuContext,
    body: &str,
    fragments: &[&str],
    kernel: &str,
    slots: u32,
) -> Program {
    let mut source = assemble(context, body, fragments);
    source.push_str(&entry_stream("main", kernel));
    source.push_str(&entry_stream("warm", "warm_start"));
    Program {
        source: source.into(),
        dispatch: Dispatch::stream(slots),
        warm: true,
    }
}

pub(super) fn workgroups(context: &GpuContext, body: &str, fragments: &[&str]) -> Program {
    Program {
        source: assemble(context, body, fragments).into(),
        dispatch: Dispatch::workgroups(),
        warm: false,
    }
}
