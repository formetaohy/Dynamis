#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollisionFilter {
    group: u32,
    mask: u32,
}

impl CollisionFilter {
    pub const DEFAULT: Self = Self {
        group: 1,
        mask: u32::MAX,
    };

    pub const fn new(group: u32, mask: u32) -> Self {
        Self { group, mask }
    }

    pub const fn group(self) -> u32 {
        self.group
    }

    pub const fn mask(self) -> u32 {
        self.mask
    }

    pub const fn with_group(self, group: u32) -> Self {
        Self { group, ..self }
    }

    pub const fn with_mask(self, mask: u32) -> Self {
        Self { mask, ..self }
    }

    pub const fn intersects(self, other: Self) -> bool {
        (self.group & other.mask) != 0 && (other.group & self.mask) != 0
    }
}

impl Default for CollisionFilter {
    fn default() -> Self {
        Self::DEFAULT
    }
}
