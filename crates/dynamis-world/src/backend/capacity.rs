#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamCapacity {
    pub entries: u32,
    pub pairs: u32,
    pub events: u32,
    pub shapes: dynamis_state::ShapeCapacity,
    pub soft: dynamis_soft::SoftCapacity,
}
