#[derive(Default)]
pub struct Ledger {
    idle: Option<bool>,
    pairs: Option<u32>,
    bodies: Option<u32>,
}

impl Ledger {
    pub fn idle(&self) -> bool {
        self.idle
            .expect("a domain plans after the broadphase settles the idle state")
    }

    pub fn pairs(&self) -> u32 {
        self.pairs
            .expect("a domain plans after the broadphase settles the pair capacity")
    }

    pub fn bodies(&self) -> u32 {
        self.bodies
            .expect("a domain plans after the state settles the body capacity")
    }

    pub fn set_idle(&mut self, idle: bool) {
        assert!(
            self.idle.replace(idle).is_none(),
            "the idle state settles once per plan"
        );
    }

    pub fn set_pairs(&mut self, pairs: u32) {
        assert!(
            self.pairs.replace(pairs).is_none(),
            "the pair capacity settles once per plan"
        );
    }

    pub fn set_bodies(&mut self, bodies: u32) {
        assert!(
            self.bodies.replace(bodies).is_none(),
            "the body capacity settles once per plan"
        );
    }

    pub fn assert_complete(&self) {
        assert!(
            self.idle.is_some() && self.pairs.is_some() && self.bodies.is_some(),
            "a plan must settle every cross-domain fact it declares"
        );
    }
}
