use std::cmp::Ordering;
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Commands,
    Prepare,
    Entries,
    Index,
    Contacts,
    Continuous,
    Deform,
    Project,
}

impl Phase {
    pub const ALL: &'static [Self] = &[
        Self::Commands,
        Self::Prepare,
        Self::Entries,
        Self::Index,
        Self::Contacts,
        Self::Continuous,
        Self::Deform,
        Self::Project,
    ];

    pub const fn rank(self) -> usize {
        match self {
            Self::Commands => 0,
            Self::Prepare => 1,
            Self::Entries => 2,
            Self::Index => 3,
            Self::Contacts => 4,
            Self::Continuous => 5,
            Self::Deform => 6,
            Self::Project => 7,
        }
    }
}

impl PartialOrd for Phase {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Phase {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank().cmp(&other.rank())
    }
}

const _: () = {
    let mut index = 0;
    while index < Phase::ALL.len() {
        assert!(
            Phase::ALL[index].rank() == index,
            "phase order must follow Phase::ALL"
        );
        index += 1;
    }
    assert!(Phase::ALL.len() == 8, "every phase must enter Phase::ALL");
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pass {
    pub label: &'static str,
    pub phase: Phase,
    pub domain: &'static str,
}

#[derive(Default)]
pub struct PassOrder {
    passes: Vec<Pass>,
}

#[derive(Clone)]
pub struct DomainPasses {
    span: Range<usize>,
}

#[macro_export]
macro_rules! domain_passes {
    ($name:ident, $domain:literal, $($field:ident: $phase:expr => $label:literal),+ $(,)?) => {
        pub struct $name {
            $( pub $field: usize, )+
        }

        impl $name {
            pub const PASSES: &'static [$crate::Pass] = &[
                $( $crate::Pass { label: $label, phase: $phase, domain: $domain }, )+
            ];

            pub fn claim(order: &mut $crate::PassOrder) -> Self {
                let claimed = order.claim($domain, Self::PASSES);
                let mut cursor = claimed.start();
                Self { $( $field: claimed.take(&mut cursor), )+ }
            }
        }
    };
}

impl PassOrder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn claim(&mut self, domain: &'static str, passes: &'static [Pass]) -> DomainPasses {
        assert!(
            !passes.is_empty(),
            "domain {domain:?} must claim at least one pass"
        );
        let mut previous: Option<Phase> = None;
        for pass in passes {
            assert_eq!(
                pass.domain, domain,
                "pass {:?} declares the domain {:?} while {:?} claims it",
                pass.label, pass.domain, domain
            );
            assert!(
                !self
                    .passes
                    .iter()
                    .any(|claimed| claimed.label == pass.label),
                "pass {:?} is already claimed by the pass order",
                pass.label
            );
            assert!(
                previous.is_none_or(|claimed| claimed <= pass.phase),
                "domain {domain:?} claims {:?} in {:?} after {:?}",
                pass.label,
                pass.phase,
                previous.expect("a later pass declares an earlier phase")
            );
            previous = Some(pass.phase);
        }
        let start = self.passes.len();
        self.passes.extend_from_slice(passes);
        DomainPasses {
            span: start..self.passes.len(),
        }
    }

    pub fn passes(&self) -> &[Pass] {
        &self.passes
    }
}

impl DomainPasses {
    pub fn start(&self) -> usize {
        self.span.start
    }

    pub fn take(&self, cursor: &mut usize) -> usize {
        assert!(
            *cursor < self.span.end,
            "a domain claims more passes than it declared"
        );
        let index = *cursor;
        *cursor += 1;
        index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Phase;

    domain_passes!(
        FirstPasses,
        "first",
        alpha: Phase::Prepare => "alpha",
        beta: Phase::Prepare => "beta"
    );
    domain_passes!(SecondPasses, "second", gamma: Phase::Index => "gamma");

    #[test]
    fn domains_claim_contiguous_pass_regions_in_order() {
        let mut order = PassOrder::new();
        let first = FirstPasses::claim(&mut order);
        let second = SecondPasses::claim(&mut order);
        assert_eq!(
            order
                .passes()
                .iter()
                .map(|pass| pass.label)
                .collect::<Vec<_>>(),
            &["alpha", "beta", "gamma"]
        );
        assert_eq!(first.alpha, 0);
        assert_eq!(first.beta, 1);
        assert_eq!(second.gamma, 2);
    }

    #[test]
    #[should_panic(expected = "already claimed")]
    fn a_pass_label_belongs_to_exactly_one_domain() {
        let mut order = PassOrder::new();
        let _ = FirstPasses::claim(&mut order);
        let _ = FirstPasses::claim(&mut order);
    }

    #[test]
    #[should_panic(expected = "after")]
    fn a_domain_declares_its_passes_in_phase_order() {
        let mut order = PassOrder::new();
        let _ = order.claim(
            "backwards",
            &[
                Pass {
                    label: "later",
                    phase: Phase::Project,
                    domain: "backwards",
                },
                Pass {
                    label: "earlier",
                    phase: Phase::Commands,
                    domain: "backwards",
                },
            ],
        );
    }

    #[test]
    #[should_panic(expected = "declares the domain")]
    fn a_pass_answers_to_the_domain_that_claims_it() {
        let mut order = PassOrder::new();
        let _ = order.claim(
            "claimer",
            &[Pass {
                label: "stray",
                phase: Phase::Commands,
                domain: "declarer",
            }],
        );
    }
}
