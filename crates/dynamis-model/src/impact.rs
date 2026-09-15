use crate::body::BodyHandle;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImpactEvent {
    pub first: BodyHandle,
    pub second: BodyHandle,
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub impulse: f32,
    pub friction_impulse: f32,
    pub step: u64,
}
