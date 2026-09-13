use super::common::{gravity_config, new_world};

const STEP: &[&str] = &[
    "commands",
    "prepare",
    "soft_bounds",
    "entries",
    "soft_entries",
    "broadphase",
    "narrowphase",
    "islands",
    "wake",
    "live",
    "solver_prepare",
    "substeps",
    "ccd_sweep",
    "ccd_apply",
    "soft_settle",
    "soft_substeps",
    "soft_apply",
    "sleep",
    "commit",
    "resting_gather",
    "resting_index",
];

#[test]
fn the_step_resolves_the_declared_domain_coupling() {
    let world = new_world(gravity_config());
    assert_eq!(world.pass_labels(), STEP);
}
