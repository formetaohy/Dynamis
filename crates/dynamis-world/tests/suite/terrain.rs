use super::common::{DT, asleep, gravity_config, observed_world, settle_until};
use dynamis_model::{BodyDesc, ColliderDesc, PhysicsConfig, Shape};

fn grid_mesh(size: f32, cells: u32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let slope = 20.0f32.to_radians().tan();
    let step = size / cells as f32;
    let mut vertices = Vec::new();
    for row in 0..=cells {
        for col in 0..=cells {
            let x = -size * 0.5 + col as f32 * step;
            let z = -size * 0.5 + row as f32 * step;
            vertices.push([x, x * slope, z]);
        }
    }
    let mut triangles = Vec::new();
    for row in 0..cells {
        for col in 0..cells {
            let near = row * (cells + 1) + col;
            let far = near + cells + 1;
            triangles.push([near, far, near + 1]);
            triangles.push([near + 1, far, far + 1]);
        }
    }
    (vertices, triangles)
}

fn height_field(size: u32, cells: f32) -> (u32, Vec<f32>, [f32; 2]) {
    let slope = 20.0f32.to_radians().tan();
    let heights = (0..size)
        .flat_map(|row| {
            (0..size).map(move |col| {
                let _ = row;
                (col as f32 - (size / 2) as f32) * slope * cells
            })
        })
        .collect();
    (size, heights, [cells, cells])
}

fn box_on_slope(
    world: &mut dynamis_world::World,
    start: f32,
    slope: f32,
) -> dynamis_model::BodyHandle {
    world.spawn(
        BodyDesc::cuboid([0.4, 0.4, 0.4])
            .position([start, start * slope + 0.5, 0.0])
            .orientation(slope_quaternion())
            .friction(0.8),
    )
}

fn slope_quaternion() -> [f32; 4] {
    let half = 20.0f32.to_radians() * 0.5;
    [0.0, 0.0, half.sin(), half.cos()]
}

fn slope_rise() -> f32 {
    20.0f32.to_radians().tan()
}

#[test]
fn a_box_rests_on_a_mesh_slope() {
    let mut world = observed_world(gravity_config());
    let (vertices, triangles) = grid_mesh(40.0, 8);
    let mesh = world.add_mesh(&vertices, &triangles, None);
    world.spawn(BodyDesc::new(ColliderDesc::new(Shape::mesh(mesh)).friction(0.8)).mass(0.0));
    let start = -5.0;
    let body = box_on_slope(&mut world, start, slope_rise());
    settle_until(&mut world, 600, |world| asleep(world));
    let state = world.read_state(body);
    assert!(
        (state.position[0] - start).abs() < 0.05,
        "a box must hold a 20 degree mesh slope, drifted to x={}",
        state.position[0]
    );
    let surface = state.position[0] * slope_rise();
    assert!(
        state.position[1] - surface > 0.4,
        "the box must rest on the mesh surface, y={}",
        state.position[1]
    );
}

#[test]
fn a_box_rests_on_a_height_field_slope() {
    let mut world = observed_world(gravity_config());
    let (side, heights, cell) = height_field(33, 1.0);
    let field = world.add_height_field(side, side, &heights, cell, None);
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::height_field(field)).friction(0.8))
            .mass(0.0)
            .position([-16.0 * cell[0], 0.0, -16.0 * cell[1]]),
    );
    let start = -5.0;
    let body = box_on_slope(&mut world, start, slope_rise());
    settle_until(&mut world, 600, |world| asleep(world));
    let state = world.read_state(body);
    assert!(
        (state.position[0] - start).abs() < 0.05,
        "a box must hold a 20 degree height field slope, drifted to x={}",
        state.position[0]
    );
    assert!(
        state.position[1] - state.position[0] * slope_rise() > 0.4,
        "the box must rest on the height field surface, y={}",
        state.position[1]
    );
}

#[test]
fn a_scene_contact_answers_the_points_the_body_touches() {
    let mut world = observed_world(gravity_config());
    let (vertices, triangles) = grid_mesh(40.0, 8);
    let mesh = world.add_mesh(&vertices, &triangles, None);
    let floor = world.spawn(BodyDesc::new(ColliderDesc::new(Shape::mesh(mesh))).mass(0.0));
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::cuboid([0.3, 0.3, 0.3]))).position([
            -0.7,
            0.3 * 20.0f32.to_radians().cos() + 0.03,
            0.4,
        ]),
    );
    for _ in 0..30 {
        world.step(DT);
    }
    world.wait();
    let manifold = world
        .inspect_contacts()
        .into_iter()
        .find(|manifold| manifold.first == floor || manifold.second == floor)
        .expect("a box on a mesh must hold a manifold");
    assert!(
        manifold.points.len() >= 2,
        "a box face on a mesh must answer the corners it touches, got {} points",
        manifold.points.len()
    );
    for point in &manifold.points {
        assert!(
            point.depth > -0.05,
            "every point of a resting manifold must reach the surface, got {:?}",
            point
        );
    }
    let lowest = manifold
        .points
        .iter()
        .map(|point| point.position[1])
        .fold(f32::INFINITY, f32::min);
    let highest = manifold
        .points
        .iter()
        .map(|point| point.position[1])
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        highest - lowest > 0.2,
        "the manifold must span the face it answers, got {lowest}..{highest}"
    );
}

#[test]
fn a_ring_shaped_contact_rests_on_a_floor() {
    let params = |angle: f32| (angle.cos(), angle.sin());
    let mut world = observed_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([30.0, 0.5, 30.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let sides = 16u32;
    let mut vertices = Vec::new();
    for level in [-1.0f32, 1.0] {
        for index in 0..sides {
            let (x, z) = params(index as f32 * std::f32::consts::TAU / sides as f32);
            vertices.push([x * 0.5, level, z * 0.5]);
        }
    }
    let mut triangles = Vec::new();
    for index in 0..sides {
        let next = (index + 1) % sides;
        triangles.push([index, sides + index, next]);
        triangles.push([next, sides + index, sides + next]);
        triangles.push([index, sides, next]);
        triangles.push([sides + index, sides + 1, sides + next]);
    }
    let hull = world.add_hull(&vertices, &triangles);
    let body =
        world.spawn(BodyDesc::new(ColliderDesc::new(Shape::hull(hull))).position([0.0, 1.01, 0.0]));
    settle_until(&mut world, 400, |world| asleep(world));
    let state = world.read_state(body);
    assert!(
        (state.position[1] - 1.0).abs() < 0.02,
        "a ring shaped face must rest flat on the floor, y={}",
        state.position[1]
    );
    assert!(
        state.orientation[0].abs() < 0.01 && state.orientation[2].abs() < 0.01,
        "a ring shaped face must not tip over, orientation={:?}",
        state.orientation
    );
}

#[test]
fn a_cylinder_rests_on_a_cuboid_floor() {
    let mut world = observed_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([30.0, 0.5, 30.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let body = world.spawn(BodyDesc::cylinder(0.5, 1.0).position([0.0, 1.01, 0.0]));
    settle_until(&mut world, 400, |world| asleep(world));
    let state = world.read_state(body);
    assert!(
        (state.position[1] - 1.0).abs() < 0.02,
        "a cylinder must rest on its cap, y={}",
        state.position[1]
    );
}

#[test]
fn a_thin_floor_is_answered_by_the_points_that_touch_it() {
    let mut world = observed_world(PhysicsConfig {
        damping: 0.0,
        angular_damping: 0.0,
        ..gravity_config()
    });
    world.spawn(
        BodyDesc::cuboid([4.0, 0.05, 4.0])
            .mass(0.0)
            .position([0.0, -0.05, 0.0]),
    );
    let hull = {
        let steps = 12u32;
        let mut vertices = Vec::new();
        for level in [-0.5f32, 0.5] {
            for index in 0..steps {
                let angle = index as f32 * std::f32::consts::TAU / steps as f32;
                vertices.push([0.5 * angle.cos(), level, 0.5 * angle.sin()]);
            }
        }
        let mut triangles = Vec::new();
        for index in 0..steps {
            let next = (index + 1) % steps;
            triangles.push([index, steps + index, next]);
            triangles.push([next, steps + index, steps + next]);
            triangles.push([index, steps, next]);
            triangles.push([steps + index, steps + 1, steps + next]);
        }
        world.add_hull(&vertices, &triangles)
    };
    let body = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::hull(hull)).friction(0.0))
            .position([0.0, 1.1, 0.0])
            .velocity([0.0, -3.0, 0.0]),
    );
    for _ in 0..120 {
        world.step(DT);
    }
    world.wait();
    let state = world.read_state(body);
    assert!(
        (state.position[1] - 0.5).abs() < 0.05,
        "a struck body must rest on the floor it hit, y={}",
        state.position[1]
    );
}

#[test]
fn a_face_answered_by_more_points_than_a_manifold_holds_stays_spread() {
    let mut world = observed_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([30.0, 0.5, 30.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let steps = 8u32;
    let mut vertices = Vec::new();
    for level in [-1.0f32, 1.0] {
        for index in 0..steps {
            let angle = index as f32 * std::f32::consts::TAU / steps as f32;
            vertices.push([0.5 * angle.cos(), level, 0.5 * angle.sin()]);
        }
    }
    let mut triangles = Vec::new();
    for index in 0..steps {
        let next = (index + 1) % steps;
        triangles.push([index, steps + index, next]);
        triangles.push([next, steps + index, steps + next]);
        triangles.push([index, steps, next]);
        triangles.push([steps + index, steps + 1, steps + next]);
    }
    let hull = world.add_hull(&vertices, &triangles);
    let body = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::hull(hull)).friction(0.8)).position([0.0, 2.0, 0.0]),
    );
    settle_until(&mut world, 300, |world| asleep(world));
    let manifold = world
        .inspect_contacts()
        .into_iter()
        .find(|manifold| manifold.first == body || manifold.second == body)
        .expect("a resting body must hold a manifold");
    assert_eq!(
        manifold.points.len(),
        4,
        "an eight point face must be answered by four points, got {}",
        manifold.points.len()
    );
    let mut closest = f32::INFINITY;
    for first in 0..manifold.points.len() {
        for second in first + 1..manifold.points.len() {
            let delta = [
                manifold.points[first].position[0] - manifold.points[second].position[0],
                manifold.points[first].position[2] - manifold.points[second].position[2],
            ];
            closest = closest.min(delta[0].hypot(delta[1]));
        }
    }
    assert!(
        closest > 0.5,
        "the four points must spread over the face rather than sit beside each other, closest {closest}"
    );
    let state = world.read_state(body);
    assert!(
        (state.position[1] - 1.0).abs() < 0.02,
        "the body must rest flat on the floor, y={}",
        state.position[1]
    );
}
