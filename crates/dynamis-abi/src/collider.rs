use crate::ColliderRecord;
use crate::constant::{
    COLLIDER_SENSOR, EVENT_MODE_BEGIN_END, EVENT_MODE_PERSIST, NO_COLLISION_FILTER, SHAPE_NONE,
};
use crate::shape::ShapeRole;
use dynamis_model::{ColliderDesc, CollisionFilter, ContactEventMode, Shape};

pub fn contact_relaxation(frequency: f32) -> f32 {
    if frequency.is_infinite() {
        return 0.0;
    }
    1.0 / (std::f32::consts::TAU * frequency)
}

pub fn contact_frequency(relaxation: f32) -> f32 {
    if relaxation <= 0.0 {
        return f32::INFINITY;
    }
    1.0 / (std::f32::consts::TAU * relaxation)
}

impl ColliderRecord {
    pub const fn cleared() -> Self {
        Self {
            slot: 0,
            kind: SHAPE_NONE,
            flags: 0,
            radius: 0.0,
            half_height: 0.0,
            impact_force: f32::INFINITY,
            half_extents: [0.0; 3],
            collision_group: NO_COLLISION_FILTER,
            local_offset: [0.0; 3],
            collision_mask: NO_COLLISION_FILTER,
            local_rotation: [0.0, 0.0, 0.0, 1.0],
            friction: 0.0,
            restitution: 0.0,
            source: 0,
            rolling_friction: 0.0,
            scale: [1.0; 3],
            spin_friction: 0.0,
            relaxation: 0.0,
            damping_ratio: 0.0,
        }
    }

    pub fn build(collider: &ColliderDesc, source: u32, slot: u32) -> Self {
        collider.assert_valid();
        let role = ShapeRole::of_shape(&collider.shape);
        let factor = role.scale.dimension_factor(collider.scale);
        let mut flags = if collider.sensor { COLLIDER_SENSOR } else { 0 };
        match collider.events {
            ContactEventMode::None => {}
            ContactEventMode::BeginEnd => flags |= EVENT_MODE_BEGIN_END,
            ContactEventMode::Persist => {
                flags |= EVENT_MODE_BEGIN_END | EVENT_MODE_PERSIST;
            }
        }
        Self {
            slot,
            kind: role.code,
            flags,
            radius: match collider.shape {
                Shape::Sphere { radius } => radius * factor[0],
                Shape::Capsule { radius, .. } | Shape::Cylinder { radius, .. } => {
                    radius * factor[0]
                }
                _ => 0.0,
            },
            half_height: match collider.shape {
                Shape::Capsule { half_height, .. } | Shape::Cylinder { half_height, .. } => {
                    half_height * factor[0]
                }
                _ => 0.0,
            },
            impact_force: collider.impact_force.unwrap_or(f32::INFINITY),
            half_extents: match collider.shape {
                Shape::Cuboid { half_extents } => [
                    half_extents[0] * factor[0],
                    half_extents[1] * factor[1],
                    half_extents[2] * factor[2],
                ],
                _ => [0.0; 3],
            },
            collision_group: collider
                .filter
                .map_or(NO_COLLISION_FILTER, CollisionFilter::group),
            local_offset: collider.offset,
            collision_mask: collider
                .filter
                .map_or(NO_COLLISION_FILTER, CollisionFilter::mask),
            local_rotation: collider.rotation,
            friction: collider.friction,
            restitution: collider.restitution,
            source,
            rolling_friction: collider.rolling_friction,
            scale: role.scale.record_scale(collider.scale),
            spin_friction: collider.spin_friction,
            relaxation: contact_relaxation(collider.contact_frequency),
            damping_ratio: collider.contact_damping_ratio,
        }
    }
}
