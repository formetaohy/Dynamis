use bytemuck::{Pod, Zeroable};

const _: () = {
    use std::mem::size_of;
    assert!(size_of::<ContactEventRecord>() == 64);
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ContactEventRecord {
    pub kind: u32,
    pub sensor: u32,
    pub first_id: u32,
    pub first_generation: u32,
    pub second_id: u32,
    pub second_generation: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub point: [f32; 3],
    pub _pad2: f32,
    pub normal: [f32; 3],
    pub _pad3: f32,
}
