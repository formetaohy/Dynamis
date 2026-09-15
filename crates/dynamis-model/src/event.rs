#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactEventKind {
    Begin,
    End,
    Persist,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactEvent {
    pub kind: ContactEventKind,
    pub first: crate::scene::SceneTarget,
    pub second: crate::scene::SceneTarget,
    pub sensor: bool,
    pub point: [f32; 3],
    pub normal: [f32; 3],
}
