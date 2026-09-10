use crate::constant::NO_SLOT;
use bytemuck::{Pod, Zeroable};

const _: () = {
    use std::mem::size_of;
    assert!(size_of::<RowMoveRecord>() == 16);
};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct RowMoveRecord {
    pub row: u32,
    pub source: u32,
    pub fresh: u32,
    pub _pad: u32,
}

impl RowMoveRecord {
    pub fn source(row: u32, source: u32) -> Self {
        Self {
            row,
            source,
            fresh: NO_SLOT,
            _pad: 0,
        }
    }

    pub fn fresh(row: u32, fresh: u32) -> Self {
        Self {
            row,
            source: NO_SLOT,
            fresh,
            _pad: 0,
        }
    }

    pub fn clear(row: u32) -> Self {
        Self {
            row,
            source: NO_SLOT,
            fresh: NO_SLOT,
            _pad: 0,
        }
    }
}
