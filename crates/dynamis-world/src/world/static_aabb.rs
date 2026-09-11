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
    let offset = quat_rotate(orientation, collider.local_offset);
    let center = [
        position[0] + offset[0],
        position[1] + offset[1],
        position[2] + offset[2],
    ];
    let rotation = quat_mul(orientation, collider.local_rotation);
    let extent = match collider.kind {
        SHAPE_PLANE => [1e6f32; 3],
        SHAPE_SPHERE => [collider.radius; 3],
        SHAPE_CUBOID => {
            let e = collider.half_extents;
            let ex = abs3(quat_rotate(rotation, [e[0], 0.0, 0.0]));
            let ey = abs3(quat_rotate(rotation, [0.0, e[1], 0.0]));
            let ez = abs3(quat_rotate(rotation, [0.0, 0.0, e[2]]));
            [
                ex[0] + ey[0] + ez[0],
                ex[1] + ey[1] + ez[1],
                ex[2] + ey[2] + ez[2],
            ]
        }
        SHAPE_CAPSULE | SHAPE_CYLINDER => {
            let axis = quat_rotate(rotation, [0.0, 1.0, 0.0]);
            [
                axis[0].abs() * collider.half_height + collider.radius,
                axis[1].abs() * collider.half_height + collider.radius,
                axis[2].abs() * collider.half_height + collider.radius,
            ]
        }
        SHAPE_HULL | SHAPE_MESH | SHAPE_HEIGHTFIELD => {
            let record = shapes.record_by_index(collider.source as usize);
            let mut min = [f32::MAX; 3];
            let mut max = [f32::MIN; 3];
            for vertex in &shapes.vertices[record.vertex_offset as usize
                ..(record.vertex_offset + record.vertex_count) as usize]
            {
                let rotated = quat_rotate(rotation, [vertex[0], vertex[1], vertex[2]]);
                for axis in 0..3 {
                    min[axis] = min[axis].min(rotated[axis]);
                    max[axis] = max[axis].max(rotated[axis]);
                }
            }
            [
                (max[0] - min[0]) * 0.5,
                (max[1] - min[1]) * 0.5,
                (max[2] - min[2]) * 0.5,
            ]
        }
        _ => [0.0; 3],
    };
    let s = collider.scale[0]
        .max(collider.scale[1])
        .max(collider.scale[2]);
    let scaled = [extent[0] * s, extent[1] * s, extent[2] * s];
    AabbRecord {
        min: [
            center[0] - scaled[0],
            center[1] - scaled[1],
            center[2] - scaled[2],
        ],
        _pad0: 0.0,
        max: [
            center[0] + scaled[0],
            center[1] + scaled[1],
            center[2] + scaled[2],
        ],
        _pad1: 0.0,
    }
}
