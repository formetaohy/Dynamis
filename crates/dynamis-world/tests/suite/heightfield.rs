use super::common::{
    DT, asleep, gravity_config, observed_world, settle, settle_until, static_config,
};
use dynamis_abi::COUNTER_CONTACTS;
use dynamis_model::{
    BodyDesc, ColliderDesc, QueryFilter, Shape, SoftBodyDesc, SoftMaterial, SurfaceDesc,
    SurfaceTable,
};
use dynamis_state::ShapeCapacity;
use dynamis_world::World;

const CELL: [f32; 2] = [2.0, 2.0];

fn field(
    world: &mut World,
    rows: u32,
    cols: u32,
    heights: &[f32],
) -> dynamis_model::ShapeSourceHandle {
    world.add_height_field(rows, cols, heights, CELL, None)
}

fn terrain(world: &mut World, rows: u32, cols: u32, heights: &[f32]) -> dynamis_model::BodyHandle {
    let source = field(world, rows, cols, heights);
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::height_field(source)))
            .mass(0.0)
            .friction(0.9),
    )
}

fn flat(rows: u32, cols: u32) -> Vec<f32> {
    vec![0.0; (rows * cols) as usize]
}

fn raise(heights: &mut [f32], cols: u32, row: u32, col: u32, height: f32) {
    heights[(row * cols + col) as usize] = height;
}

fn capacity(world: &mut World) -> ShapeCapacity {
    world.step(DT);
    world.wait();
    world.stream_capacity().state.shapes
}

#[test]
fn a_height_field_keeps_its_grid_instead_of_a_triangle_soup() {
    let mut world = observed_world(static_config());
    let side = 32u32;
    let source = field(&mut world, side, side, &flat(side, side));
    let shapes = capacity(&mut world);
    assert_eq!(
        shapes.vertices,
        side * side,
        "a height field stores one vertex per sample"
    );
    assert!(
        shapes.cells < side * side / 4,
        "a height field without a surface table reserves no cells"
    );

    let vertices = (0..side * side)
        .map(|index| {
            [
                (index % side) as f32 * CELL[0],
                0.0,
                (index / side) as f32 * CELL[1],
            ]
        })
        .collect::<Vec<[f32; 3]>>();
    let mut triangles = Vec::new();
    for row in 0..side - 1 {
        for col in 0..side - 1 {
            let near = row * side + col;
            let far = near + side;
            triangles.push([near, far, near + 1]);
            triangles.push([far, far + 1, near + 1]);
        }
    }
    world.add_mesh(&vertices, &triangles, None);
    let meshed = capacity(&mut world);
    let editor = capacity(&mut world);
    assert!(
        meshed.vertices >= shapes.vertices + side * side,
        "a mesh reserves the samples a grid already carries"
    );
    assert!(
        meshed.triangles > shapes.triangles + side * side,
        "a mesh stores the cells a grid derives, grid {} mesh {}",
        shapes.triangles,
        meshed.triangles
    );
    assert!(
        meshed.nodes > shapes.nodes + side * side,
        "a mesh indexes its cells by a bvh a grid derives, grid {} mesh {}",
        shapes.nodes,
        meshed.nodes
    );

    world.update_height_field(
        source,
        side,
        side,
        &vec![1.0; (side * side) as usize],
        CELL,
        None,
    );
    let edited = capacity(&mut world);
    assert_eq!(
        (edited.vertices, edited.triangles, edited.nodes),
        (editor.vertices, editor.triangles, editor.nodes),
        "a height field edit rewrites its samples in place"
    );
}

#[test]
fn a_height_field_capacity_grows_with_the_edited_grid() {
    let mut world = observed_world(static_config());
    let source = field(&mut world, 2, 2, &flat(2, 2));
    assert_eq!(capacity(&mut world).vertices, 64);
    world.update_height_field(source, 8, 8, &flat(8, 8), CELL, None);
    assert_eq!(
        capacity(&mut world).vertices,
        64,
        "a grid inside the reserved floor keeps its stream"
    );
    world.update_height_field(source, 16, 16, &flat(16, 16), CELL, None);
    assert_eq!(capacity(&mut world).vertices, 256);
}

#[test]
fn a_height_field_sits_and_slides_by_its_cells() {
    let mut world = observed_world(gravity_config());
    let mut heights = Vec::new();
    for _row in 0..4 {
        for col in 0..6 {
            heights.push((5 - col) as f32);
        }
    }
    terrain(&mut world, 4, 6, &heights);
    let ball = world.spawn(BodyDesc::sphere(0.3).position([0.6, 5.0, 3.0]));
    settle_until(&mut world, 240, |world| {
        world.read_state(ball).position[0] > 8.0
    });
    let state = world.read_state(ball);
    assert!(
        state.position[0] > 8.0,
        "a ball must roll downhill across the cells, got {:?}",
        state.position
    );
}

#[test]
fn a_height_field_ray_reaches_every_cell() {
    let mut world = observed_world(static_config());
    let side = 8u32;
    let mut heights = flat(side, side);
    for row in 6..8 {
        for col in 6..8 {
            raise(&mut heights, side, row, col, 3.0);
        }
    }
    terrain(&mut world, side, side, &heights);
    world.step(DT);
    world.wait();

    let ridge = [CELL[0] * 6.5, 3.0, CELL[1] * 6.5];
    let heading = [ridge[0] - 1.0, ridge[1] - 6.0, ridge[2] - 1.0];
    let reach =
        (heading[0] * heading[0] + heading[1] * heading[1] + heading[2] * heading[2]).sqrt();
    let handle = world.ray_query(
        [1.0, 6.0, 1.0],
        [heading[0] / reach, heading[1] / reach, heading[2] / reach],
        reach + 1.0,
        &QueryFilter::default(),
    );
    world.wait();
    let hit = world
        .query_hit(handle)
        .expect("a ray must reach the far ridge");
    assert!(
        (hit.point[1] - 3.0).abs() < 0.2,
        "the ray must land on the raised far cells, got {:?}",
        hit.point
    );
    assert!(
        hit.point[0] > 10.0 && hit.point[2] > 10.0,
        "the ray must land in the far cells, got {:?}",
        hit.point
    );
}

#[test]
fn a_height_field_ray_clears_the_grid_when_it_misses() {
    let mut world = observed_world(static_config());
    let side = 8u32;
    terrain(&mut world, side, side, &flat(side, side));
    world.step(DT);
    world.wait();
    let above = world.ray_query(
        [0.5, 5.0, 0.5],
        [1.0, 0.0, 0.0],
        4.0,
        &QueryFilter::default(),
    );
    world.wait();
    assert!(
        world.query_hit(above).is_none(),
        "a ray above the grid must miss every cell"
    );
    let inside = world.ray_query(
        [0.5, 5.0, 0.5],
        [0.0, -1.0, 0.0],
        10.0,
        &QueryFilter::default(),
    );
    world.wait();
    let hit = world.query_hit(inside).expect("a downward ray must land");
    assert!(
        hit.point[1].abs() < 0.1,
        "the ray must land on the grid surface, got {:?}",
        hit.point
    );
}

#[test]
fn a_height_field_sweep_and_overlap_answer_their_cells() {
    let mut world = observed_world(static_config());
    let side = 6u32;
    let mut heights = flat(side, side);
    for row in 2..4 {
        for col in 2..4 {
            raise(&mut heights, side, row, col, 2.0);
        }
    }
    terrain(&mut world, side, side, &heights);
    world.step(DT);
    world.wait();

    let sweep = world.sweep_query(
        &Shape::sphere(0.25),
        [0.0, 0.0, 0.0, 1.0],
        [5.0, 4.0, 5.0],
        [0.0, -1.0, 0.0],
        10.0,
        &QueryFilter::default(),
    );
    world.wait();
    let hit = world
        .query_hit(sweep)
        .expect("a sweep must find the raised cells");
    assert!(
        hit.point[1] > 1.9,
        "the sweep must stop on the ridge, got {:?}",
        hit.point
    );

    let overlap = world.overlap_query(
        &Shape::sphere(0.5),
        [0.0, 0.0, 0.0, 1.0],
        [5.0, 1.8, 5.0],
        &QueryFilter::default(),
    );
    world.wait();
    assert!(
        world.query_hit(overlap).is_some(),
        "an overlap inside the ridge must reach its cells"
    );

    let clear = world.overlap_query(
        &Shape::sphere(0.2),
        [0.0, 0.0, 0.0, 1.0],
        [5.0, 4.0, 5.0],
        &QueryFilter::default(),
    );
    world.wait();
    assert!(
        world.query_hit(clear).is_none(),
        "an overlap above the ridge must clear it"
    );
}

#[test]
fn a_height_field_scales_its_samples() {
    let mut world = observed_world(gravity_config());
    let side = 4u32;
    let source = field(&mut world, side, side, &vec![1.0; (side * side) as usize]);
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::height_field(source)).scale([1.0, 2.0, 1.0]))
            .mass(0.0),
    );
    let ball = world.spawn(BodyDesc::sphere(0.3).position([3.0, 4.0, 3.0]));
    settle_until(&mut world, 180, |world| asleep(world));
    let rest = world.read_state(ball).position[1];
    assert!(
        (rest - 2.3).abs() < 0.1,
        "a grid scaled twice in height must double its samples, got {rest}"
    );
}

#[test]
fn a_height_field_surface_steers_each_cell() {
    let mut world = observed_world(gravity_config());
    let palette = [
        SurfaceDesc::new().friction(0.0),
        SurfaceDesc::new().friction(0.9),
    ];
    let side = 10u32;
    let mut indices = vec![0u32; ((side - 1) * (side - 1)) as usize];
    for row in 3..(side - 1) {
        for col in 0..(side - 1) {
            indices[(row * (side - 1) + col) as usize] = 1;
        }
    }
    let source = world.add_height_field(
        side,
        side,
        &flat(side, side),
        CELL,
        Some(SurfaceTable::new(&palette, &indices)),
    );
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::height_field(source)).friction(0.9)).mass(0.0),
    );
    assert!(
        capacity(&mut world).cells >= (side - 1) * (side - 1),
        "a height field with a surface table reserves one cell per surface"
    );
    let slider = world.spawn(
        BodyDesc::sphere(0.3)
            .position([2.0, 0.35, 1.0])
            .velocity([4.0, 0.0, 0.0]),
    );
    let gripper = world.spawn(
        BodyDesc::sphere(0.3)
            .position([2.0, 0.35, 10.0])
            .velocity([4.0, 0.0, 0.0]),
    );
    settle(&mut world, 120);
    let slid = world.read_state(slider).velocity[0];
    let gripped = world.read_state(gripper).velocity[0];
    assert!(
        slid > gripped + 1.0,
        "a frictionless cell must let the body slide, slid {slid} gripped {gripped}"
    );
}

#[test]
fn a_rotated_height_field_answers_its_cells() {
    let mut world = observed_world(gravity_config());
    let side = 5u32;
    let source = field(&mut world, side, side, &vec![3.0; (side * side) as usize]);
    let half = std::f32::consts::FRAC_PI_4;
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::height_field(source)))
            .mass(0.0)
            .orientation([0.0, half.sin(), 0.0, half.cos()]),
    );
    let mut spot = None;
    for x in 0..40 {
        for z in 0..40 {
            let point = [-5.0 + x as f32 * 0.5, -5.0 + z as f32 * 0.5];
            let handle = world.ray_query(
                [point[0], 8.0, point[1]],
                [0.0, -1.0, 0.0],
                20.0,
                &QueryFilter::default(),
            );
            world.wait();
            match world.query_hit(handle) {
                Some(hit) if (hit.point[1] - 3.0).abs() < 0.05 => spot = Some(point),
                _ => {}
            }
        }
    }
    let spot = spot.expect("a rotated grid must answer its cells");
    let ball = world.spawn(BodyDesc::sphere(0.3).position([spot[0], 6.0, spot[1]]));
    settle_until(&mut world, 240, |world| asleep(world));
    let rest = world.read_state(ball).position[1];
    assert!(
        (rest - 3.3).abs() < 0.1,
        "a rotated grid must carry the body at its own height, got {rest}"
    );
}

#[test]
fn a_height_field_carries_soft_particles() {
    let mut world = observed_world(gravity_config());
    let side = 4u32;
    terrain(&mut world, side, side, &flat(side, side));
    let mut cloth = SoftBodyDesc::cloth([4, 4], 0.3, SoftMaterial::rigid());
    cloth.position = [1.0, 1.5, 1.0];
    cloth.radius = 0.1;
    let cloth = world.add_soft_body(cloth);
    settle_until(&mut world, 240, |world| {
        world
            .inspect_soft_particles(cloth)
            .iter()
            .all(|particle| particle[1] > -0.2)
    });
    let particles = world.inspect_soft_particles(cloth);
    assert!(
        particles.iter().all(|particle| particle[1] > -0.2),
        "a cloth must rest above the grid, got {particles:?}"
    );
}

#[test]
fn a_height_field_is_measured_by_its_contacts() {
    let mut world = observed_world(gravity_config());
    let side = 4u32;
    terrain(&mut world, side, side, &flat(side, side));
    let ball = world.spawn(BodyDesc::sphere(0.4).position([3.0, 2.0, 3.0]));
    settle(&mut world, 60);
    assert!(
        world.measured()[COUNTER_CONTACTS] > 0,
        "a resting body must publish its contact with the grid"
    );
    let manifold = world
        .inspect_contacts()
        .into_iter()
        .find(|manifold| manifold.first.id == ball.id || manifold.second.id == ball.id)
        .expect("the ball must publish a manifold");
    assert!(
        manifold.points.iter().all(|point| point.depth > -0.2),
        "the manifold must sit on the grid surface, got {:?}",
        manifold.points
    );
    assert!(
        manifold.surface.is_none(),
        "a grid without a surface table answers the collider surface"
    );
}

#[test]
fn a_height_field_refuses_a_grid_of_one_row() {
    let mut world = observed_world(static_config());
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.add_height_field(1, 4, &[0.0; 4], CELL, None)
    }));
    assert!(refused.is_err(), "a height field needs a cell in both axes");
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.add_height_field(2, 2, &[0.0; 3], CELL, None)
    }));
    assert!(
        refused.is_err(),
        "a height field needs one sample per vertex"
    );
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.add_height_field(2, 2, &[0.0; 4], [0.0, 1.0], None)
    }));
    assert!(
        refused.is_err(),
        "a height field needs a positive cell size"
    );
}

#[test]
fn a_height_field_edit_reshapes_every_cell() {
    let mut world = observed_world(static_config());
    let side = 5u32;
    let mut heights = flat(side, side);
    for row in 1..3 {
        for col in 1..3 {
            raise(&mut heights, side, row, col, 1.0);
        }
    }
    let source = field(&mut world, side, side, &heights);
    world.spawn(BodyDesc::new(ColliderDesc::new(Shape::height_field(source))).mass(0.0));
    world.step(DT);
    world.wait();
    let aim = [CELL[0] * 1.5, -1.0, CELL[1] * 1.5];
    let measured = |world: &mut World| {
        let handle = world.ray_query(
            [aim[0], 3.0, aim[2]],
            [0.0, -1.0, 0.0],
            10.0,
            &QueryFilter::default(),
        );
        world.wait();
        world.query_hit(handle).expect("the cell must answer").point[1]
    };
    assert!(
        (measured(&mut world) - 1.0).abs() < 0.1,
        "the raised cell must answer its height"
    );
    world.update_height_field(
        source,
        side,
        side,
        &vec![2.0; (side * side) as usize],
        CELL,
        None,
    );
    assert!(
        (measured(&mut world) - 2.0).abs() < 0.1,
        "an edited grid must answer its new heights"
    );
    world.update_height_field(source, side, 3, &flat(side, 3), CELL, None);
    assert!(
        measured(&mut world).abs() < 0.1,
        "a reshaped grid must answer its new cells"
    );
}
