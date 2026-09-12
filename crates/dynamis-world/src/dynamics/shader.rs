use dynamis_layout::{ABI_WGSL, COUNTER_STRIDE, constants_wgsl};

pub(super) const WORKGROUP_SIZE: u32 = 64;

pub(super) const CORE_FRAGMENT: &str = include_str!("shaders/core.wgsl");
pub(super) const GRID_INDEX_FRAGMENT: &str = include_str!("shaders/grid_index.wgsl");
pub(super) const IDENTITY_FRAGMENT: &str = include_str!("shaders/identity.wgsl");
pub(super) const EVENTS_FRAGMENT: &str = include_str!("shaders/events.wgsl");
pub(super) const CONVEX_FRAGMENT: &str = include_str!("shaders/convex.wgsl");
pub(super) const SCENE_FRAGMENT: &str = include_str!("shaders/scene.wgsl");
pub(super) const CONTACT_BLOCK_FRAGMENT: &str = include_str!("shaders/solver_contact_block.wgsl");
pub(super) const CONSTRAINT_BLOCK_FRAGMENT: &str =
    include_str!("shaders/solver_constraint_block.wgsl");
pub(super) const CONTACT_CORRECTION_FRAGMENT: &str =
    include_str!("shaders/position_contact_block.wgsl");
pub(super) const SHAPES_FRAGMENT: &str = include_str!("shaders/shapes.wgsl");

pub(super) fn assemble_shader(body: &str, per_row: u32, fragments: &[&str]) -> String {
    let mut source = shader_constants(per_row);
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
        crate::dynamics::buffers::EVENT_SLOTS
    ));
    source.push_str(&format!(
        "const COUNTER_STRIDE_WORDS: u32 = {}u;\n",
        COUNTER_STRIDE / 4
    ));
    source
}
