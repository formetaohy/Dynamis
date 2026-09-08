#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactEventKind {
    Begin,
    End,
    Persist,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactEvent {
    pub kind: ContactEventKind,
    pub first: crate::body::BodyHandle,
    pub second: crate::body::BodyHandle,
    pub sensor: bool,
    pub point: [f32; 3],
    pub normal: [f32; 3],
}
