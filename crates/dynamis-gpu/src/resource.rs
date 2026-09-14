use crate::{GpuSlot, StreamElement, TypedSlot};

const DOMAIN_SHIFT: u32 = 24;
const LOCAL_MASK: u32 = (1 << DOMAIN_SHIFT) - 1;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ResourceId(u32);

impl ResourceId {
    pub const fn new(domain: u32, local: u32) -> Self {
        assert!(domain < (1 << (32 - DOMAIN_SHIFT)));
        assert!(local < (1 << DOMAIN_SHIFT));
        Self((domain << DOMAIN_SHIFT) | local)
    }

    pub const fn domain(self) -> u32 {
        self.0 >> DOMAIN_SHIFT
    }

    pub const fn local(self) -> u32 {
        self.0 & LOCAL_MASK
    }
}

#[derive(Clone, Copy)]
pub enum SlotRef {
    Whole {
        resource: ResourceId,
        element: StreamElement,
    },
    Range {
        resource: ResourceId,
        offset: u64,
        size: u64,
        element: StreamElement,
    },
}

impl SlotRef {
    pub const fn whole(resource: ResourceId, element: StreamElement) -> Self {
        Self::Whole { resource, element }
    }

    pub const fn range(
        resource: ResourceId,
        offset: u64,
        size: u64,
        element: StreamElement,
    ) -> Self {
        Self::Range {
            resource,
            offset,
            size,
            element,
        }
    }

    pub const fn element(self) -> StreamElement {
        match self {
            Self::Whole { element, .. } | Self::Range { element, .. } => element,
        }
    }

    pub fn resolve<R: Resources>(self, resources: &R) -> TypedSlot<'_> {
        match self {
            Self::Whole { resource, .. } => {
                TypedSlot::new(resources.whole(resource), self.element())
            }
            Self::Range {
                resource,
                offset,
                size,
                ..
            } => TypedSlot::new(resources.range(resource, offset, size), self.element()),
        }
    }
}

pub trait Resources {
    fn generation(&self) -> u64;

    fn slots(&self, resource: ResourceId) -> u32;

    fn whole(&self, resource: ResourceId) -> GpuSlot<'_>;

    fn range(&self, resource: ResourceId, offset: u64, size: u64) -> GpuSlot<'_>;
}
