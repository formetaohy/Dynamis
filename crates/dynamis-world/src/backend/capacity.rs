#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamCapacity {
    pub state: dynamis_state::ShapeCapacity,
    pub broadphase: dynamis_broadphase::BroadphaseCapacity,
    pub rigid: dynamis_rigid::RigidCapacity,
    pub soft: dynamis_soft::SoftCapacity,
}
