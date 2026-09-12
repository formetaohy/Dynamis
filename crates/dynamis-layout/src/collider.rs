use crate::constant::{
    COLLIDER_EVENT_BEGIN_END, COLLIDER_EVENT_PERSIST, COLLIDER_SENSOR, NO_COLLISION_FILTER,
    SHAPE_CAPSULE, SHAPE_CUBOID, SHAPE_CYLINDER, SHAPE_HEIGHTFIELD, SHAPE_HULL, SHAPE_MESH,
    SHAPE_PLANE, SHAPE_SPHERE,
};
use crate::{AabbRecord, ColliderRecord};
use dynamis_model::{ColliderDesc, ContactEventMode, Shape};

impl ColliderRecord {
    pub fn build(collider: &ColliderDesc, source: u32) -> Self {
        let kind = shape_kind(&collider.shape);
        let uniform =
            collider.scale[0] == collider.scale[1] && collider.scale[1] == collider.scale[2];
        let mut flags = if collider.sensor { COLLIDER_SENSOR } else { 0 };
        match collider.events {
            ContactEventMode::None => {}
            ContactEventMode::BeginEnd => flags |= COLLIDER_EVENT_BEGIN_END,
            ContactEventMode::Persist => {
                flags |= COLLIDER_EVENT_BEGIN_END | COLLIDER_EVENT_PERSIST;
            }
        }
        Self {
            kind,
            flags,
            radius: match collider.shape {
                Shape::Sphere { radius } => {
                    if uniform {
                        radius * collider.scale[0]
                    } else {
                        radius
                    }
                }
                Shape::Capsule { radius, .. } | Shape::Cylinder { radius, .. } => {
                    if uniform {
                        radius * collider.scale[0]
                    } else {
                        radius
                    }
                }
                _ => 0.0,
            },
            half_height: match collider.shape {
                Shape::Capsule { half_height, .. } | Shape::Cylinder { half_height, .. } => {
                    if uniform {
                        half_height * collider.scale[0]
                    } else {
                        half_height
                    }
                }
                _ => 0.0,
            },
            half_extents: match collider.shape {
                Shape::Cuboid { half_extents } => [
                    half_extents[0] * collider.scale[0],
                    half_extents[1] * collider.scale[1],
                    half_extents[2] * collider.scale[2],
                ],
                _ => [0.0; 3],
            },
            collision_group: collider.collision_group.unwrap_or(NO_COLLISION_FILTER),
            local_offset: collider.offset,
            collision_mask: collider.collision_mask.unwrap_or(NO_COLLISION_FILTER),
            local_rotation: collider.rotation,
            friction: collider.friction,
            restitution: collider.restitution,
            source,
            rolling_friction: collider.rolling_friction,
            scale: baked_scale(collider, uniform),
            spin_friction: collider.spin_friction,
        }
    }
}

fn baked_scale(collider: &ColliderDesc, uniform: bool) -> [f32; 3] {
    match collider.shape {
        Shape::Cuboid { .. } | Shape::Plane => [1.0; 3],
        _ if uniform => [1.0; 3],
        _ => collider.scale,
    }
}

fn shape_kind(shape: &Shape) -> u32 {
    match shape {
        Shape::Sphere { .. } => SHAPE_SPHERE,
        Shape::Cuboid { .. } => SHAPE_CUBOID,
        Shape::Capsule { .. } => SHAPE_CAPSULE,
        Shape::Cylinder { .. } => SHAPE_CYLINDER,
        Shape::Hull(_) => SHAPE_HULL,
        Shape::Mesh(_) => SHAPE_MESH,
        Shape::HeightField(_) => SHAPE_HEIGHTFIELD,
        Shape::Plane => SHAPE_PLANE,
    }
}

impl AabbRecord {
    pub const fn empty() -> Self {
        Self {
            min: [f32::MAX; 3],
            _pad0: 0.0,
            max: [f32::MIN; 3],
            _pad1: 0.0,
        }
    }
}
