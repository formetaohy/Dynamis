use std::ops::Range;

#[derive(Default)]
pub struct PassOrder {
    labels: Vec<&'static str>,
}

#[derive(Clone)]
pub struct DomainPasses {
    span: Range<usize>,
}

#[macro_export]
macro_rules! domain_passes {
    ($name:ident, $domain:literal, $($field:ident => $label:literal),+ $(,)?) => {
        pub struct $name {
            $( pub $field: usize, )+
        }

        impl $name {
            pub const LABELS: &'static [&'static str] = &[$($label,)+];

            pub fn claim(order: &mut $crate::PassOrder) -> Self {
                let claimed = order.claim($domain, Self::LABELS);
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

    pub fn claim(&mut self, domain: &'static str, passes: &'static [&'static str]) -> DomainPasses {
        assert!(
            !passes.is_empty(),
            "domain {domain:?} must claim at least one pass"
        );
        for pass in passes {
            assert!(
                !self.labels.contains(pass),
                "pass {pass:?} is already claimed by the pass order"
            );
        }
        let start = self.labels.len();
        self.labels.extend_from_slice(passes);
        DomainPasses {
            span: start..self.labels.len(),
        }
    }

    pub fn labels(&self) -> &[&'static str] {
        &self.labels
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

    domain_passes!(FirstPasses, "first", alpha => "alpha", beta => "beta");
    domain_passes!(SecondPasses, "second", gamma => "gamma");

    #[test]
    fn domains_claim_contiguous_pass_regions_in_order() {
        let mut order = PassOrder::new();
        let first = FirstPasses::claim(&mut order);
        let second = SecondPasses::claim(&mut order);
        assert_eq!(order.labels(), &["alpha", "beta", "gamma"]);
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
}
