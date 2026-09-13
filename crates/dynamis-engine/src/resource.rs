use dynamis_gpu::GpuSlot;

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
    Whole(ResourceId),
    Range {
        resource: ResourceId,
        offset: u64,
        size: u64,
    },
}

impl SlotRef {
    pub const fn whole(resource: ResourceId) -> Self {
        Self::Whole(resource)
    }

    pub const fn range(resource: ResourceId, offset: u64, size: u64) -> Self {
        Self::Range {
            resource,
            offset,
            size,
        }
    }

    pub fn resolve<R: Resources>(self, resources: &R) -> GpuSlot<'_> {
        match self {
            Self::Whole(resource) => resources.whole(resource),
            Self::Range {
                resource,
                offset,
                size,
            } => resources.range(resource, offset, size),
        }
    }
}

pub trait Resources {
    fn generation(&self) -> u64;

    fn slots(&self, resource: ResourceId) -> u32;

    fn whole(&self, resource: ResourceId) -> GpuSlot<'_>;

    fn range(&self, resource: ResourceId, offset: u64, size: u64) -> GpuSlot<'_>;
}
