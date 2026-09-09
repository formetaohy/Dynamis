//! The device-authority dispatch table: one entry per indirect dispatch, written by
//! the device from a counter slot so stages run by how much work they actually have.

use bytemuck::{Pod, Zeroable};

/// One indirect dispatch: workgroups per row, rows, depth, and a pad word.
///
/// The arguments are packed exactly as the compute API expects them, and the table
/// buffer carries both storage and indirect usages.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DispatchArgs {
    pub per_row: u32,
    pub rows: u32,
    pub layers: u32,
    pub _pad: u32,
}

/// Bytes one direct argument occupies, also the stride between table slots.
pub const DISPATCH_ARGS_BYTES: u64 = 16;

/// Where one dispatcher lives in the table and what it is built from.
#[derive(Clone, Copy, Debug)]
pub struct Dispatcher {
    /// Counter slot whose count feeds this dispatcher.
    pub counter: u32,
    /// Lanes one workgroup covers, so a count becomes workgroups.
    pub lanes: u32,
    /// Byte offset of this dispatcher's `DispatchArgs` in the table.
    pub offset: u64,
}

impl Dispatcher {
    /// The table slot this dispatcher owns.
    pub fn slot(&self) -> u32 {
        (self.offset / DISPATCH_ARGS_BYTES) as u32
    }
}

/// Builds a dispatcher for the given table slot, counter slot, and workgroup width.
pub const fn dispatch(counter: u32, lanes: u32, slot: u32) -> Dispatcher {
    Dispatcher {
        counter,
        lanes,
        offset: slot as u64 * DISPATCH_ARGS_BYTES,
    }
}
