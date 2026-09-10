use wgpu::{BufferAddress, BufferUsages, Device};

/// One indirect dispatch: workgroups per row, rows, depth, and a pad word.
///
/// The words are packed exactly as the compute API expects them, so the table binds
/// as storage and dispatches from at the same time.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DispatchArgs {
    per_row: u32,
    rows: u32,
    layers: u32,
    _pad: u32,
}

/// Bytes one argument occupies, also the stride between table slots.
pub const DISPATCH_ARGS_BYTES: BufferAddress = size_of::<DispatchArgs>() as BufferAddress;

const _: () = assert!(DISPATCH_ARGS_BYTES == 16);

/// The table the device fills with indirect arguments, one slot per stage.
pub struct DispatchTable {
    buffer: crate::GpuBuffer,
    slots: u32,
}

impl DispatchTable {
    pub fn new(device: &Device, label: &str, slots: u32) -> Self {
        assert!(slots > 0, "a dispatch table needs at least one slot");
        Self {
            buffer: crate::GpuBuffer::new(
                device,
                label,
                slots as BufferAddress * DISPATCH_ARGS_BYTES,
                BufferUsages::STORAGE
                    .union(BufferUsages::INDIRECT)
                    .union(BufferUsages::COPY_DST),
            ),
            slots,
        }
    }

    pub fn buffer(&self) -> &crate::GpuBuffer {
        &self.buffer
    }

    pub fn slots(&self) -> u32 {
        self.slots
    }

    pub fn offset(&self, slot: u32) -> BufferAddress {
        assert!(
            slot < self.slots,
            "dispatch slot {slot} is outside the {} table slots",
            self.slots
        );
        slot as BufferAddress * DISPATCH_ARGS_BYTES
    }
}
