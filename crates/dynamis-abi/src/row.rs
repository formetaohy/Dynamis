use crate::RowMoveRecord;
use crate::constant::NO_SLOT;

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
