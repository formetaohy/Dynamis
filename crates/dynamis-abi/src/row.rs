use crate::RowMoveRecord;
use crate::constant::NO_SLOT;

impl RowMoveRecord {
    /// The row takes the state the device holds at `source`, which is where that row's identity
    /// answers from.
    pub fn source(row: u32, source: u32) -> Self {
        Self {
            row,
            source,
            fresh: NO_SLOT,
            _pad: 0,
        }
    }

    /// The row takes the state `fresh` answers, because the device holds no state for its identity
    /// yet.
    pub fn fresh(row: u32, fresh: u32) -> Self {
        Self {
            row,
            source: NO_SLOT,
            fresh,
            _pad: 0,
        }
    }
}
