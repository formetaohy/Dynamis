use dynamis_pass::{
    PassEdges, PassGroup, PassGroupEdges, PassSpec, PipelineBuilder, assert_declared,
};

const FIRST: u32 = 0;
const SECOND: u32 = 1;

#[test]
fn independent_passes_follow_their_declared_order() {
    let mut builder = PipelineBuilder::new();
    builder.declare(
        SECOND,
        PassGroup {
            passes: &[PassSpec {
                label: "beta",
                after: &[],
            }],
        },
    );
    builder.declare(
        FIRST,
        PassGroup {
            passes: &[PassSpec {
                label: "alpha",
                after: &[],
            }],
        },
    );
    let pipeline = builder.resolve();
    assert_eq!(
        pipeline
            .passes()
            .iter()
            .map(|pass| pass.label)
            .collect::<Vec<_>>(),
        &["beta", "alpha"]
    );
    assert_eq!(pipeline.index("beta"), 0);
    assert_eq!(pipeline.pass(1).domain, FIRST);
}

#[test]
fn a_pass_follows_the_passes_it_declares() {
    let mut builder = PipelineBuilder::new();
    builder.declare(
        FIRST,
        PassGroup {
            passes: &[
                PassSpec {
                    label: "earlier",
                    after: &[],
                },
                PassSpec {
                    label: "later",
                    after: &["earlier"],
                },
            ],
        },
    );
    let pipeline = builder.resolve();
    assert_eq!(
        pipeline
            .passes()
            .iter()
            .map(|pass| pass.label)
            .collect::<Vec<_>>(),
        &["earlier", "later"]
    );
}

#[test]
fn a_declared_dependency_outweighs_the_declared_order() {
    let mut builder = PipelineBuilder::new();
    builder.declare(
        FIRST,
        PassGroup {
            passes: &[PassSpec {
                label: "consumer",
                after: &["producer"],
            }],
        },
    );
    builder.declare(
        SECOND,
        PassGroup {
            passes: &[PassSpec {
                label: "producer",
                after: &[],
            }],
        },
    );
    let pipeline = builder.resolve();
    assert_eq!(
        pipeline
            .passes()
            .iter()
            .map(|pass| pass.label)
            .collect::<Vec<_>>(),
        &["producer", "consumer"]
    );
}

#[test]
fn a_repeated_dependency_resolves_once() {
    let mut builder = PipelineBuilder::new();
    builder.declare(
        FIRST,
        PassGroup {
            passes: &[
                PassSpec {
                    label: "first",
                    after: &[],
                },
                PassSpec {
                    label: "second",
                    after: &["first", "first"],
                },
            ],
        },
    );
    let pipeline = builder.resolve();
    assert_eq!(
        pipeline
            .passes()
            .iter()
            .map(|pass| pass.label)
            .collect::<Vec<_>>(),
        &["first", "second"]
    );
}

#[test]
#[should_panic(expected = "twice")]
fn a_repeated_label_is_refused() {
    let mut builder = PipelineBuilder::new();
    builder.declare(
        FIRST,
        PassGroup {
            passes: &[
                PassSpec {
                    label: "shared",
                    after: &[],
                },
                PassSpec {
                    label: "shared",
                    after: &[],
                },
            ],
        },
    );
    builder.resolve();
}

#[test]
#[should_panic(expected = "undeclared")]
fn an_undeclared_dependency_is_refused() {
    let mut builder = PipelineBuilder::new();
    builder.declare(
        FIRST,
        PassGroup {
            passes: &[PassSpec {
                label: "consumer",
                after: &["absent"],
            }],
        },
    );
    builder.resolve();
}

#[test]
#[should_panic(expected = "follow itself")]
fn a_pass_that_follows_itself_is_refused() {
    let mut builder = PipelineBuilder::new();
    builder.declare(
        FIRST,
        PassGroup {
            passes: &[PassSpec {
                label: "loop",
                after: &["loop"],
            }],
        },
    );
    builder.resolve();
}

#[test]
#[should_panic(expected = "cyclic")]
fn a_dependency_cycle_is_refused() {
    let mut builder = PipelineBuilder::new();
    builder.declare(
        FIRST,
        PassGroup {
            passes: &[
                PassSpec {
                    label: "first",
                    after: &["second"],
                },
                PassSpec {
                    label: "second",
                    after: &["first"],
                },
            ],
        },
    );
    builder.resolve();
}

#[test]
#[should_panic(expected = "run \"producer\" first")]
fn a_group_that_contradicts_its_own_declared_order_is_refused() {
    let mut builder = PipelineBuilder::new();
    builder.declare(
        FIRST,
        PassGroup {
            passes: &[
                PassSpec {
                    label: "consumer",
                    after: &["producer"],
                },
                PassSpec {
                    label: "producer",
                    after: &[],
                },
            ],
        },
    );
    builder.resolve();
}

#[test]
#[should_panic(expected = "outside the step pipeline")]
fn a_pass_beyond_the_pipeline_is_refused() {
    let mut builder = PipelineBuilder::new();
    builder.declare(
        FIRST,
        PassGroup {
            passes: &[PassSpec {
                label: "only",
                after: &[],
            }],
        },
    );
    builder.resolve().pass(1);
}

#[test]
fn a_declared_pass_graph_is_accepted() {
    const RIGID: PassGroupEdges = &[("commands", &[]), ("entries", &["commands"])];
    const SOFT: PassGroupEdges = &[
        ("soft_bounds", &[]),
        ("soft_entries", &["soft_bounds", "entries"]),
    ];
    const EDGES: PassEdges = &[RIGID, SOFT];
    assert_declared(&[EDGES]);
}

#[test]
#[should_panic(expected = "no domain declares")]
fn an_undeclared_pass_label_is_refused() {
    const LONELY: PassGroupEdges = &[("consumer", &["producer"])];
    const EDGES: PassEdges = &[LONELY];
    assert_declared(&[EDGES]);
}

#[test]
#[should_panic(expected = "declares a pass label twice")]
fn a_label_declared_by_two_domains_is_refused() {
    const FIRST: PassEdges = &[&[("shared", &[])]];
    const SECOND: PassEdges = &[&[("shared", &[])]];
    assert_declared(&[FIRST, SECOND]);
}
