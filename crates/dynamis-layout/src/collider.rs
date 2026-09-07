use crate::constant::{
    COLLIDER_SENSOR, SHAPE_CAPSULE, SHAPE_CUBOID, SHAPE_CYLINDER, SHAPE_HEIGHTFIELD, SHAPE_HULL,
    SHAPE_MESH, SHAPE_PLANE, SHAPE_SPHERE,
};
use bytemuck::{Pod, Zeroable};
use dynamis_model::{ColliderDesc, Shape};

const _: () = {
    use std::mem::size_of;
    assert!(size_of::<ColliderRecord>() == 96);
    assert!(size_of::<AabbRecord>() == 32);
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ColliderRecord {
    pub kind: u32,
    pub flags: u32,
    pub radius: f32,
    pub half_height: f32,
    pub half_extents: [f32; 3],
    pub _pad0: f32,
    pub local_offset: [f32; 3],
    pub _pad1: f32,
    pub local_rotation: [f32; 4],
    pub friction: f32,
    pub restitution: f32,
    pub source: u32,
    pub _pad2: u32,
    pub scale: [f32; 3],
    pub _pad_scale: f32,
}

impl ColliderRecord {
    pub fn build(collider: &ColliderDesc, source: u32) -> Self {
        let kind = shape_kind(&collider.shape);
        let uniform =
            collider.scale[0] == collider.scale[1] && collider.scale[1] == collider.scale[2];
        Self {
            kind,
            flags: if collider.sensor { COLLIDER_SENSOR } else { 0 },
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
            _pad0: 0.0,
            local_offset: collider.offset,
            _pad1: 0.0,
            local_rotation: collider.rotation,
            friction: collider.friction,
            restitution: collider.restitution,
            source,
            _pad2: 0,
            scale: baked_scale(collider, uniform),
            _pad_scale: 0.0,
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AabbRecord {
    pub min: [f32; 3],
    pub _pad0: f32,
    pub max: [f32; 3],
    pub _pad1: f32,
}
