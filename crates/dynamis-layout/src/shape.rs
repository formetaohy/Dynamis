use bytemuck::{Pod, Zeroable};

const _: () = {
    use std::mem::size_of;
    assert!(size_of::<ShapeSourceRecord>() == 32);
    assert!(size_of::<BvhNodeRecord>() == 48);
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ShapeSourceRecord {
    pub kind: u32,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub triangle_offset: u32,
    pub triangle_count: u32,
    pub node_offset: u32,
    pub node_count: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BvhNodeRecord {
    pub min: [f32; 3],
    pub _pad0: f32,
    pub max: [f32; 3],
    pub _pad1: f32,
    pub left: u32,
    pub right: u32,
    pub leaf: u32,
    pub _pad2: u32,
}
