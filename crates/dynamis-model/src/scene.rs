use crate::BodyHandle;
use crate::SoftBodyHandle;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SceneTarget {
    Collider { body: BodyHandle, collider: u32 },
    Particle { body: SoftBodyHandle, particle: u32 },
}

impl SceneTarget {
    pub const fn collider(&self) -> Option<(BodyHandle, u32)> {
        match *self {
            Self::Collider { body, collider } => Some((body, collider)),
            Self::Particle { .. } => None,
        }
    }

    pub const fn particle(&self) -> Option<(SoftBodyHandle, u32)> {
        match *self {
            Self::Particle { body, particle } => Some((body, particle)),
            Self::Collider { .. } => None,
        }
    }

    pub const fn is_body(&self, handle: BodyHandle) -> bool {
        match *self {
            Self::Collider { body, .. } => {
                body.id == handle.id && body.generation == handle.generation
            }
            Self::Particle { .. } => false,
        }
    }

    pub const fn is_soft(&self, handle: SoftBodyHandle) -> bool {
        match *self {
            Self::Particle { body, .. } => {
                body.id == handle.id && body.generation == handle.generation
            }
            Self::Collider { .. } => false,
        }
    }
}
