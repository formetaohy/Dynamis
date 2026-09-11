use crate::world::shape_pool::ShapePool;
use dynamis_layout::{
    AabbRecord, ColliderRecord, SHAPE_CAPSULE, SHAPE_CUBOID, SHAPE_CYLINDER, SHAPE_HEIGHTFIELD,
    SHAPE_HULL, SHAPE_MESH, SHAPE_NONE, SHAPE_PLANE, SHAPE_SPHERE,
};
use dynamis_math::{quat_mul, quat_rotate};
use dynamis_model::MAX_COLLIDERS_PER_BODY;

fn abs3(v: [f32; 3]) -> [f32; 3] {
    [v[0].abs(), v[1].abs(), v[2].abs()]
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn static_aabbs(
    position: [f32; 3],
    orientation: [f32; 4],
    colliders: &[ColliderRecord; MAX_COLLIDERS_PER_BODY],
    shapes: &ShapePool,
) -> [AabbRecord; MAX_COLLIDERS_PER_BODY] {
    let mut aabbs = [AabbRecord::empty(); MAX_COLLIDERS_PER_BODY];
    for (index, collider) in colliders.iter().enumerate() {
        if collider.kind == SHAPE_NONE {
            continue;
        }
        aabbs[index] = collider_aabb(position, orientation, collider, shapes);
    }
    aabbs
}

fn collider_aabb(
    position: [f32; 3],
    orientation: [f32; 4],
    collider: &ColliderRecord,
    shapes: &ShapePool,
) -> AabbRecord {
    let rotation = quat_mul(orientation, collider.local_rotation);
    let (center_offset, extent) = match collider.kind {
        SHAPE_PLANE => ([0.0f32; 3], [1e6f32; 3]),
        SHAPE_SPHERE => ([0.0; 3], [collider.radius; 3]),
        SHAPE_CUBOID => {
            let e = collider.half_extents;
            let ex = abs3(quat_rotate(rotation, [e[0], 0.0, 0.0]));
            let ey = abs3(quat_rotate(rotation, [0.0, e[1], 0.0]));
            let ez = abs3(quat_rotate(rotation, [0.0, 0.0, e[2]]));
            (
                [0.0; 3],
                [
                    ex[0] + ey[0] + ez[0],
                    ex[1] + ey[1] + ez[1],
                    ex[2] + ey[2] + ez[2],
                ],
            )
        }
        SHAPE_CAPSULE | SHAPE_CYLINDER => {
            let axis = quat_rotate(rotation, [0.0, 1.0, 0.0]);
            (
                [0.0; 3],
                [
                    axis[0].abs() * collider.half_height + collider.radius,
                    axis[1].abs() * collider.half_height + collider.radius,
                    axis[2].abs() * collider.half_height + collider.radius,
                ],
            )
        }
        SHAPE_HULL | SHAPE_MESH | SHAPE_HEIGHTFIELD => {
            let (local_min, local_max) = shapes.record_by_index(collider.source as usize).bounds;
            let local_center = [
                (local_min[0] + local_max[0]) * 0.5,
                (local_min[1] + local_max[1]) * 0.5,
                (local_min[2] + local_max[2]) * 0.5,
            ];
            let half = [
                (local_max[0] - local_min[0]) * 0.5,
                (local_max[1] - local_min[1]) * 0.5,
                (local_max[2] - local_min[2]) * 0.5,
            ];
            let ex = abs3(quat_rotate(rotation, [half[0], 0.0, 0.0]));
            let ey = abs3(quat_rotate(rotation, [0.0, half[1], 0.0]));
            let ez = abs3(quat_rotate(rotation, [0.0, 0.0, half[2]]));
            (
                quat_rotate(rotation, local_center),
                [
                    ex[0] + ey[0] + ez[0],
                    ex[1] + ey[1] + ez[1],
                    ex[2] + ey[2] + ez[2],
                ],
            )
        }
        _ => ([0.0; 3], [0.0; 3]),
    };
    let offset = quat_rotate(orientation, collider.local_offset);
    let center = add3(add3(position, offset), center_offset);
    let s = collider.scale[0]
        .max(collider.scale[1])
        .max(collider.scale[2]);
    let scaled = [extent[0] * s, extent[1] * s, extent[2] * s];
    AabbRecord {
        min: sub3(center, scaled),
        _pad0: 0.0,
        max: add3(center, scaled),
        _pad1: 0.0,
    }
}
