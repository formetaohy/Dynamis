use crate::SurfaceRecord;
use dynamis_model::SurfaceDesc;

impl SurfaceRecord {
    pub fn build(surface: &SurfaceDesc) -> Self {
        Self {
            friction: surface.friction,
            restitution: surface.restitution,
            rolling_friction: surface.rolling_friction,
            spin_friction: surface.spin_friction,
        }
    }

    pub fn desc(&self) -> SurfaceDesc {
        SurfaceDesc {
            friction: self.friction,
            restitution: self.restitution,
            rolling_friction: self.rolling_friction,
            spin_friction: self.spin_friction,
            contact_frequency: f32::INFINITY,
            contact_damping_ratio: 1.0,
        }
    }
}
