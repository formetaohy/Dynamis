use crate::FieldRecord;
use crate::constant::{FIELD_REGION_CUBOID, FIELD_REGION_GLOBAL, FIELD_REGION_SPHERE};
use dynamis_model::{FieldDesc, FieldRegion};

impl FieldRecord {
    pub fn build(field: &FieldDesc) -> Self {
        field.assert_valid();
        let (region, radius, half_extents, orientation) = match field.region {
            FieldRegion::Global => (FIELD_REGION_GLOBAL, 0.0, [0.0; 3], [0.0, 0.0, 0.0, 1.0]),
            FieldRegion::Sphere { radius } => {
                (FIELD_REGION_SPHERE, radius, [0.0; 3], [0.0, 0.0, 0.0, 1.0])
            }
            FieldRegion::Cuboid {
                half_extents,
                orientation,
            } => (FIELD_REGION_CUBOID, 0.0, half_extents, orientation),
        };
        Self {
            position: field.position,
            pull: field.pull,
            half_extents,
            swirl: field.swirl,
            push: field.push,
            radius,
            medium: field.medium,
            linear_drag: field.linear_drag,
            axis: field.axis,
            quadratic_drag: field.quadratic_drag,
            orientation,
            angular_drag: field.angular_drag,
            buoyancy: field.buoyancy,
            region,
            collision_group: field.filter.group(),
            collision_mask: field.filter.mask(),
            _wgsl_pad0: [0; 12],
        }
    }
}
