use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::collections::BinaryHeap;
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassSpec {
    pub label: &'static str,
    pub after: &'static [&'static str],
    pub execution: Execution,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Run {
    Step,
    Query,
    Publish,
}

impl Run {
    pub const fn facts(self) -> u16 {
        match self {
            Self::Step => Execution::STEP.0 | Execution::GRAPH.0 | Execution::PUBLISH.0,
            Self::Query => Execution::QUERY.0 | Execution::GRAPH.0,
            Self::Publish => Execution::PUBLISH.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Execution(u16);

impl Execution {
    pub const ALWAYS: Self = Self(0);

    pub const INDEXING: Self = Self(1);

    pub const AWAKE: Self = Self(1 << 1);

    pub const STEP: Self = Self(1 << 2);

    pub const QUERY: Self = Self(1 << 3);

    pub const PUBLISH: Self = Self(1 << 10);

    pub const GRAPH: Self = Self(1 << 11);

    const GATE_BASE: u32 = 4;
    const GATE_COUNT: u32 = 6;
    const GATE_MASK: u16 = ((1 << Self::GATE_COUNT) - 1) << Self::GATE_BASE;

    pub const fn gate(index: u32) -> Self {
        assert!(
            index < Self::GATE_COUNT,
            "a pass gate must fit the declared gate field"
        );
        Self(1 << (index + Self::GATE_BASE))
    }

    pub const fn and(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn bits(self) -> u16 {
        self.0
    }

    pub const fn facts(run: Run, indexing: bool, awake: bool, gates: u16) -> u16 {
        run.facts() | (indexing as u16) | ((awake as u16) << 1) | (gates & Self::GATE_MASK)
    }

    pub const fn held(self, facts: u16) -> bool {
        self.0 & !facts == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassGroup {
    pub passes: &'static [PassSpec],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pass {
    pub label: &'static str,
    pub domain: u32,
    pub execution: Execution,
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
    execution: Execution,
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
                execution: spec.execution,
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
                    execution: self.declared[position].execution,
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
        $( $field:ident => $execution:expr => $after:expr ),+ $(,)?
    ) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name {
            $( pub $field: u32, )+
        }

        impl $name {
            pub const GROUP: $crate::PassGroup = $crate::PassGroup {
                passes: &[
                    $( $crate::PassSpec {
                        label: stringify!($field),
                        after: $after,
                        execution: $execution,
                    }, )+
                ],
            };

            pub const EDGES: $crate::PassGroupEdges = &[
                $( (stringify!($field), $after as &'static [&'static str]), )+
            ];

            pub fn resolve(pipeline: &$crate::Pipeline) -> Self {
                Self {
                    $( $field: pipeline.index(stringify!($field)), )+
                }
            }
        }
    };
}

pub type PassGroupEdges = &'static [(&'static str, &'static [&'static str])];

pub type PassEdges = &'static [PassGroupEdges];

pub const fn assert_declared(domains: &[PassEdges]) {
    let mut domain = 0;
    while domain < domains.len() {
        let groups = domains[domain];
        let mut group = 0;
        while group < groups.len() {
            let edges = groups[group];
            let mut edge = 0;
            while edge < edges.len() {
                let (label, after) = edges[edge];
                assert!(
                    occurrences(domains, label) == 1,
                    "a step pipeline declares a pass label twice"
                );
                let mut dependency = 0;
                while dependency < after.len() {
                    assert!(
                        occurrences(domains, after[dependency]) == 1,
                        "a pass follows a pass label that no domain declares"
                    );
                    dependency += 1;
                }
                edge += 1;
            }
            group += 1;
        }
        domain += 1;
    }
}

const fn occurrences(domains: &[PassEdges], label: &str) -> usize {
    let mut count = 0;
    let mut domain = 0;
    while domain < domains.len() {
        let groups = domains[domain];
        let mut group = 0;
        while group < groups.len() {
            let edges = groups[group];
            let mut edge = 0;
            while edge < edges.len() {
                if equal(edges[edge].0, label) {
                    count += 1;
                }
                edge += 1;
            }
            group += 1;
        }
        domain += 1;
    }
    count
}

const fn equal(left: &str, right: &str) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let left = left.as_bytes();
    let right = right.as_bytes();
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}
