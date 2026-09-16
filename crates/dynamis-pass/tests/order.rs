use dynamis_pass::{
    Execution, PassEdges, PassGroup, PassGroupEdges, PassSpec, PipelineBuilder, Run,
    assert_declared,
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
                execution: Execution::ALWAYS,
            }],
        },
    );
    builder.declare(
        FIRST,
        PassGroup {
            passes: &[PassSpec {
                label: "alpha",
                after: &[],
                execution: Execution::ALWAYS,
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
                    execution: Execution::ALWAYS,
                },
                PassSpec {
                    label: "later",
                    after: &["earlier"],
                    execution: Execution::ALWAYS,
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
                execution: Execution::ALWAYS,
            }],
        },
    );
    builder.declare(
        SECOND,
        PassGroup {
            passes: &[PassSpec {
                label: "producer",
                after: &[],
                execution: Execution::ALWAYS,
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
                    execution: Execution::ALWAYS,
                },
                PassSpec {
                    label: "second",
                    after: &["first", "first"],
                    execution: Execution::ALWAYS,
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
                    execution: Execution::ALWAYS,
                },
                PassSpec {
                    label: "shared",
                    after: &[],
                    execution: Execution::ALWAYS,
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
                execution: Execution::ALWAYS,
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
                execution: Execution::ALWAYS,
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
                    execution: Execution::ALWAYS,
                },
                PassSpec {
                    label: "second",
                    after: &["first"],
                    execution: Execution::ALWAYS,
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
                    execution: Execution::ALWAYS,
                },
                PassSpec {
                    label: "producer",
                    after: &[],
                    execution: Execution::ALWAYS,
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
                execution: Execution::ALWAYS,
            }],
        },
    );
    builder.resolve().pass(1);
}

#[test]
fn a_declared_pass_graph_is_accepted() {
    const RIGID: PassGroupEdges = &[
        ("apply_commands", &[]),
        ("emit_entries", &["apply_commands"]),
    ];
    const SOFT: PassGroupEdges = &[
        ("update_soft_bounds", &[]),
        ("emit_soft_entries", &["update_soft_bounds", "emit_entries"]),
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

#[test]
fn a_pass_carries_its_declared_execution() {
    let mut builder = PipelineBuilder::new();
    builder.declare(
        FIRST,
        PassGroup {
            passes: &[PassSpec {
                label: "gated",
                after: &[],
                execution: Execution::AWAKE,
            }],
        },
    );
    let pipeline = builder.resolve();
    assert_eq!(pipeline.pass(0).execution, Execution::AWAKE);
}

#[test]
fn a_pass_runs_exactly_when_every_declared_fact_holds() {
    let idle = Execution::facts(Run::Step, true, true, 0);
    let stopped = Execution::facts(Run::Step, false, false, 0);
    let query = Execution::facts(Run::Query, true, false, 0);
    assert!(Execution::ALWAYS.held(stopped));
    assert!(Execution::INDEXING.held(idle));
    assert!(!Execution::INDEXING.held(stopped));
    assert!(Execution::AWAKE.held(idle));
    assert!(!Execution::AWAKE.held(Execution::facts(Run::Step, true, false, 0)));
    assert!(!Execution::AWAKE.and(Execution::gate(0)).held(idle));
    assert!(
        Execution::AWAKE
            .and(Execution::gate(0))
            .held(Execution::facts(Run::Step, true, true, 1 << 4))
    );
    assert!(Execution::STEP.held(idle));
    assert!(!Execution::STEP.held(query));
    assert!(!Execution::QUERY.held(idle));
    assert!(Execution::QUERY.held(query));
    assert!(Execution::INDEXING.held(query));
    assert!(!Execution::AWAKE.held(query));
}

#[test]
fn a_publication_run_holds_only_the_publishing_passes() {
    let step = Execution::facts(Run::Step, false, false, 0);
    let query = Execution::facts(Run::Query, false, false, 0);
    let publish = Execution::facts(Run::Publish, false, false, 0);
    assert!(Execution::PUBLISH.held(step));
    assert!(Execution::PUBLISH.held(publish));
    assert!(!Execution::PUBLISH.held(query));
    assert!(Execution::GRAPH.held(step));
    assert!(Execution::GRAPH.held(query));
    assert!(!Execution::GRAPH.held(publish));
    assert!(Execution::STEP.held(step));
    assert!(!Execution::STEP.held(publish));
    assert!(!Execution::STEP.held(query));
    assert!(!Execution::INDEXING.held(publish));
    assert!(!Execution::AWAKE.held(publish));
    assert!(!Execution::gate(0).held(publish));
}

#[test]
#[should_panic(expected = "gate")]
fn a_gate_beyond_the_declared_field_is_refused() {
    let _ = Execution::gate(std::hint::black_box(6));
}
