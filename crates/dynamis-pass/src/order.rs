use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::collections::BinaryHeap;
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassSpec {
    pub label: &'static str,
    pub after: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassGroup {
    pub passes: &'static [PassSpec],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pass {
    pub label: &'static str,
    pub domain: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pipeline {
    passes: Vec<Pass>,
}

impl Pipeline {
    pub fn passes(&self) -> &[Pass] {
        &self.passes
    }

    pub fn len(&self) -> usize {
        self.passes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    pub fn pass(&self, index: u32) -> Pass {
        *self
            .passes
            .get(index as usize)
            .unwrap_or_else(|| panic!("pass {index} is outside the step pipeline"))
    }

    pub fn index(&self, label: &str) -> u32 {
        self.passes
            .iter()
            .position(|pass| pass.label == label)
            .unwrap_or_else(|| panic!("the step pipeline declares no pass {label:?}"))
            .try_into()
            .expect("a step pipeline cannot hold more passes than an index holds")
    }
}

struct Declared {
    domain: u32,
    label: &'static str,
    after: &'static [&'static str],
}

#[derive(Default)]
pub struct PipelineBuilder {
    declared: Vec<Declared>,
    groups: Vec<Range<usize>>,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn declare(&mut self, domain: u32, group: PassGroup) {
        assert!(
            !group.passes.is_empty(),
            "a declared pass group must hold at least one pass"
        );
        let start = self.declared.len();
        for spec in group.passes {
            self.declared.push(Declared {
                domain,
                label: spec.label,
                after: spec.after,
            });
        }
        self.groups.push(start..self.declared.len());
    }

    pub fn resolve(self) -> Pipeline {
        assert!(
            !self.declared.is_empty(),
            "a step pipeline needs at least one pass"
        );
        let labels = self.label_positions();
        let (mut successors, pending) = self.dependencies(&labels);
        let order = self.ordering(&mut successors, pending);
        self.assert_group_order(&order);
        Pipeline {
            passes: order
                .into_iter()
                .map(|position| Pass {
                    label: self.declared[position].label,
                    domain: self.declared[position].domain,
                })
                .collect(),
        }
    }

    fn label_positions(&self) -> BTreeMap<&'static str, usize> {
        let mut positions = BTreeMap::new();
        for (position, pass) in self.declared.iter().enumerate() {
            assert!(
                positions.insert(pass.label, position).is_none(),
                "pass {:?} is declared twice in one step pipeline",
                pass.label
            );
        }
        positions
    }

    fn dependencies(
        &self,
        positions: &BTreeMap<&'static str, usize>,
    ) -> (Vec<Vec<usize>>, Vec<u32>) {
        let mut successors = vec![Vec::new(); self.declared.len()];
        let mut pending = vec![0u32; self.declared.len()];
        for (position, pass) in self.declared.iter().enumerate() {
            assert!(
                !pass.after.contains(&pass.label),
                "pass {:?} must not follow itself",
                pass.label
            );
            for (slot, after) in pass.after.iter().enumerate() {
                if pass.after[..slot].contains(after) {
                    continue;
                }
                let dependency = *positions.get(after).unwrap_or_else(|| {
                    panic!(
                        "pass {:?} must follow the undeclared pass {after:?}; the step declares {}",
                        pass.label,
                        self.labels(|_| true),
                    )
                });
                successors[dependency].push(position);
                pending[position] += 1;
            }
        }
        (successors, pending)
    }

    fn ordering(&self, successors: &mut [Vec<usize>], mut pending: Vec<u32>) -> Vec<usize> {
        let mut ready = BinaryHeap::new();
        for (position, outstanding) in pending.iter().enumerate() {
            if *outstanding == 0 {
                ready.push(Reverse(position));
            }
        }
        let mut order = Vec::with_capacity(self.declared.len());
        while let Some(Reverse(position)) = ready.pop() {
            order.push(position);
            for successor in std::mem::take(&mut successors[position]) {
                pending[successor] -= 1;
                if pending[successor] == 0 {
                    ready.push(Reverse(successor));
                }
            }
        }
        assert!(
            order.len() == self.declared.len(),
            "the step pipeline is cyclic across {}",
            self.blocked(&order)
        );
        order
    }

    fn blocked(&self, order: &[usize]) -> String {
        self.labels(|position| !order.contains(&position))
    }

    fn labels(&self, wanted: impl Fn(usize) -> bool) -> String {
        self.declared
            .iter()
            .enumerate()
            .filter(|(position, _)| wanted(*position))
            .map(|(_, pass)| format!("{:?}", pass.label))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn assert_group_order(&self, order: &[usize]) {
        let mut resolved = vec![0usize; self.declared.len()];
        for (position, declared) in order.iter().enumerate() {
            resolved[*declared] = position;
        }
        for group in &self.groups {
            let mut previous: Option<usize> = None;
            for declared in group.clone() {
                if let Some(first) = previous {
                    assert!(
                        resolved[first] < resolved[declared],
                        "a pass group declares {:?} before {:?} while the dependencies run {:?} first",
                        self.declared[first].label,
                        self.declared[declared].label,
                        self.declared[declared].label,
                    );
                }
                previous = Some(declared);
            }
        }
    }
}

#[macro_export]
macro_rules! domain_passes {
    (
        $name:ident,
        $( $field:ident => $after:expr ),+ $(,)?
    ) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name {
            $( pub $field: u32, )+
        }

        impl $name {
            pub const GROUP: $crate::PassGroup = $crate::PassGroup {
                passes: &[
                    $( $crate::PassSpec { label: stringify!($field), after: $after }, )+
                ],
            };

            pub fn resolve(pipeline: &$crate::Pipeline) -> Self {
                Self {
                    $( $field: pipeline.index(stringify!($field)), )+
                }
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
