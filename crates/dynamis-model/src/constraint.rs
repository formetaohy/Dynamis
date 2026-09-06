#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintKind {
    Ball,
    Distance,
    Revolute,
    Prismatic,
    Fixed,
}

#[derive(Clone, Copy, Debug)]
pub struct ConstraintDesc {
    pub kind: ConstraintKind,
    pub anchor_a: [f32; 3],
    pub anchor_b: [f32; 3],
    pub axis: [f32; 3],
    pub distance: f32,
}

impl ConstraintDesc {
    pub fn ball(anchor_a: [f32; 3], anchor_b: [f32; 3]) -> Self {
        Self {
            kind: ConstraintKind::Ball,
            anchor_a,
            anchor_b,
            axis: [0.0, 1.0, 0.0],
            distance: 0.0,
        }
    }

    pub fn distance(anchor_a: [f32; 3], anchor_b: [f32; 3], distance: f32) -> Self {
        assert!(distance >= 0.0, "constraint distance must be non-negative");
        Self {
            kind: ConstraintKind::Distance,
            anchor_a,
            anchor_b,
            axis: [0.0, 1.0, 0.0],
            distance,
        }
    }

    pub fn revolute(anchor_a: [f32; 3], anchor_b: [f32; 3], axis: [f32; 3]) -> Self {
        assert!(axis != [0.0; 3], "revolute axis must be non-zero");
        Self {
            kind: ConstraintKind::Revolute,
            anchor_a,
            anchor_b,
            axis,
            distance: 0.0,
        }
    }

    pub fn prismatic(anchor_a: [f32; 3], anchor_b: [f32; 3], axis: [f32; 3]) -> Self {
        assert!(axis != [0.0; 3], "prismatic axis must be non-zero");
        Self {
            kind: ConstraintKind::Prismatic,
            anchor_a,
            anchor_b,
            axis,
            distance: 0.0,
        }
    }

    pub fn fixed(anchor_a: [f32; 3], anchor_b: [f32; 3]) -> Self {
        Self {
            kind: ConstraintKind::Fixed,
            anchor_a,
            anchor_b,
            axis: [0.0, 1.0, 0.0],
            distance: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstraintHandle {
    pub id: u32,
    pub generation: u32,
}
