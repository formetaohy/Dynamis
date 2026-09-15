use crate::body::BodyHandle;
use crate::soft::SoftBodyHandle;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryTargets(u32);

impl QueryTargets {
    pub const COLLIDERS: Self = Self(1 << 0);
    pub const PARTICLES: Self = Self(1 << 1);
    pub const ALL: Self = Self(Self::COLLIDERS.0 | Self::PARTICLES.0);

    pub const fn with(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn holds(self, target: Self) -> bool {
        self.0 & target.0 == target.0
    }

    pub const fn bits(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QueryFilter {
    pub targets: QueryTargets,
    pub group: u32,
    pub mask: u32,
    pub ignore_sensors: bool,
    pub ignore_sleeping: bool,
    pub ignore_static: bool,
    pub ignore_kinematic: bool,
    pub exclude: Option<BodyHandle>,
    pub include: Option<BodyHandle>,
    pub exclude_soft: Option<SoftBodyHandle>,
    pub include_soft: Option<SoftBodyHandle>,
    pub max_hits: u32,
}

impl Default for QueryFilter {
    fn default() -> Self {
        Self {
            targets: QueryTargets::ALL,
            group: 0,
            mask: u32::MAX,
            ignore_sensors: true,
            ignore_sleeping: false,
            ignore_static: false,
            ignore_kinematic: false,
            exclude: None,
            include: None,
            exclude_soft: None,
            include_soft: None,
            max_hits: 4,
        }
    }
}
