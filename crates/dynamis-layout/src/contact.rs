use bytemuck::{Pod, Zeroable};

const _: () = {
    use std::mem::size_of;
    assert!(size_of::<PairRecord>() == 8);
    assert!(size_of::<ManifoldPointRecord>() == 32);
    assert!(size_of::<ContactRecord>() == 192);
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PairRecord {
    pub a: u32,
    pub b: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ManifoldPointRecord {
    pub position: [f32; 3],
    pub depth: f32,
    pub accumulated_normal: f32,
    pub accumulated_tangent_1: f32,
    pub accumulated_tangent_2: f32,
    pub _pad0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ContactRecord {
    pub a: u32,
    pub b: u32,
    pub point_count: u32,
    pub sensor: u32,
    pub first_body_id: u32,
    pub second_body_id: u32,
    pub first_generation: u32,
    pub second_generation: u32,
    pub normal: [f32; 3],
    pub _pad0: f32,
    pub friction: f32,
    pub restitution: f32,
    pub rolling_friction: f32,
    pub spin_friction: f32,
    pub points: [ManifoldPointRecord; 4],
}
