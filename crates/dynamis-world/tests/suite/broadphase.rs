use super::common::{DT, gravity_config, new_world, settle, settle_until, static_config};
use dynamis_layout::{
    COUNTER_CONTACTS, COUNTER_ENTRIES, COUNTER_PAIRS, COUNTER_RESTING, COUNTER_SPILLOVER_PAIRS,
};
use dynamis_model::{BodyDesc, ColliderDesc, QueryFilter, Shape};

const WIDE: f32 = 30.0;
const CROWD: usize = 64;

fn wide_static_floor(world: &mut dynamis_world::World) {
    world.spawn(
        BodyDesc::cuboid([WIDE, 0.5, WIDE])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
}

fn quarter_turn_z() -> [f32; 4] {
    let half = std::f32::consts::FRAC_PI_4;
    [0.0, 0.0, half.sin(), half.cos()]
}

fn crowd_above(world: &mut dynamis_world::World, height: f32, radius: f32) {
    for index in 0..CROWD {
        let column = (index % 8) as f32 * 2.0 - 7.0;
        let row = (index / 8) as f32 * 3.0 - 10.0;
        world.spawn(BodyDesc::sphere(radius).position([column, height, row]));
    }
}

#[test]
fn a_wide_static_floor_pairs_only_the_bodies_inside_its_bounds() {
    let mut world = new_world(gravity_config());
    wide_static_floor(&mut world);
    crowd_above(&mut world, 40.0, 0.4);
    world.step(DT);
    world.wait();
    let measured = *world.measured();
    assert!(
        measured[COUNTER_CONTACTS] == 0,
        "bodies far above the floor must not touch it, got {} contacts",
        measured[COUNTER_CONTACTS]
    );
    assert!(
        measured[COUNTER_PAIRS] <= CROWD as u32,
        "a distant wide floor must not pair with every body, got {} pairs for {CROWD} bodies",
        measured[COUNTER_PAIRS]
    );
    settle_until(&mut world, 400, |world| {
        world.measured()[COUNTER_CONTACTS] >= CROWD as u32
    });
}

#[test]
fn a_wide_static_floor_still_carries_a_dense_crowd() {
    let mut world = new_world(gravity_config());
    wide_static_floor(&mut world);
    crowd_above(&mut world, 1.2, 0.4);
    settle_until(&mut world, 240, |world| {
        world.measured()[COUNTER_RESTING] >= CROWD as u32
    });
    for handle in world.bodies().to_vec() {
        let state = world.read_state(handle);
        if state.inverse_mass > 0.0 {
            assert!(
                (state.position[1] - 0.4).abs() < 0.25,
                "every body must rest on the floor, got {}",
                state.position[1]
            );
        }
    }
}

#[test]
fn a_body_beyond_a_wide_collider_bounds_never_pairs_with_it() {
    let mut world = new_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([WIDE, 0.5, WIDE])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let distant = world.spawn(BodyDesc::sphere(0.4).position([WIDE + 10.0, 0.0, 0.0]));
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_PAIRS],
        0,
        "a body outside a wide collider bounds must not pair with it"
    );
    assert!(
        world.read_state(distant).position[1] < 0.0,
        "the distant body must keep falling"
    );
}

#[test]
fn a_coarse_collider_links_to_finer_neighbours_across_levels() {
    let mut world = new_world(gravity_config());
    let terrain = world.spawn(
        BodyDesc::cuboid([200.0, 0.5, 200.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let block = world.spawn(BodyDesc::cuboid([1.0, 1.0, 1.0]).position([0.0, 3.0, 0.0]));
    let pebble = world.spawn(BodyDesc::sphere(0.3).position([0.0, 6.0, 0.0]));
    world.step(DT);
    world.wait();
    assert!(
        world.measured()[COUNTER_ENTRIES] > 0,
        "a coarse collider must enter the grid"
    );
    assert_eq!(world.read_state(terrain).position[1], -0.5);
    settle_until(&mut world, 240, |world| world.read_state(block).sleeping);
    assert!(
        (world.read_state(block).position[1] - 1.0).abs() < 0.2,
        "a fine block must rest on a coarse terrain, got {}",
        world.read_state(block).position[1]
    );
    assert!(
        world.read_state(pebble).position[1] > world.read_state(block).position[1],
        "the pebble must rest on the block, got {}",
        world.read_state(pebble).position[1]
    );
}

#[test]
fn a_narrow_ray_reaches_a_coarse_static_collider() {
    let mut world = new_world(gravity_config());
    let terrain = world.spawn(
        BodyDesc::cuboid([200.0, 0.5, 200.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    world.step(DT);
    world.wait();
    let handle = world.ray_query(
        [0.0, 50.0, 0.0],
        [0.0, -1.0, 0.0],
        100.0,
        &QueryFilter::default(),
    );
    world.wait();
    let hit = world
        .query_hit(handle)
        .expect("the ray must reach the terrain");
    assert_eq!(hit.body, terrain, "the ray must report the terrain body");
    assert!(
        (hit.point[1] - 0.0).abs() < 1e-3,
        "the ray must meet the terrain surface, got {}",
        hit.point[1]
    );

    let overlap = world.overlap_query(
        &Shape::sphere(1.0),
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 60.0, 0.0],
        &QueryFilter::default(),
    );
    world.wait();
    assert!(
        world.query_hit(overlap).is_none(),
        "an overlap far above the terrain must miss it"
    );
}

#[test]
fn a_query_on_a_sleeping_world_still_reaches_a_coarse_collider() {
    let mut world = new_world(gravity_config());
    let terrain = world.spawn(
        BodyDesc::cuboid([200.0, 0.5, 200.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.5, 0.0]));
    settle_until(&mut world, 240, |world| world.read_state(ball).sleeping);
    let handle = world.ray_query(
        [6.0, 20.0, 0.0],
        [0.0, -1.0, 0.0],
        40.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    let hit = world
        .query_hit(handle)
        .expect("a sleeping world must still answer queries");
    assert_eq!(hit.body, terrain);
}

#[test]
fn a_coarse_collider_has_an_entry_span_within_budget() {
    let mut world = new_world(gravity_config());
    for index in 0..4 {
        let offset = index as f32 * 400.0;
        world.spawn(
            BodyDesc::cuboid([150.0, 0.5, 150.0])
                .mass(0.0)
                .position([offset, -0.5, 0.0]),
        );
    }
    for index in 0..16 {
        world.spawn(BodyDesc::sphere(0.3).position([index as f32 * 3.0, 8.0, 0.0]));
    }
    world.step(DT);
    world.wait();
    let entries = world.measured()[COUNTER_ENTRIES];
    assert!(
        entries <= 20 * dynamis_layout::MAX_CELLS_PER_COLLIDER,
        "grid entries must stay within the per collider budget, got {entries}"
    );
}

#[test]
fn overlapping_coarse_colliders_repel_each_other() {
    let mut world = new_world(gravity_config());
    world.spawn(
        BodyDesc::new(
            ColliderDesc::new(Shape::cuboid([64.0, 0.5, 64.0]))
                .friction(0.0)
                .restitution(0.0),
        )
        .mass(0.0)
        .position([0.0, -0.5, 0.0]),
    );
    let falling = world.spawn(
        BodyDesc::new(
            ColliderDesc::new(Shape::cuboid([8.0, 8.0, 8.0]))
                .friction(0.0)
                .restitution(0.0),
        )
        .position([0.0, 20.0, 0.0]),
    );
    settle_until(&mut world, 240, |world| world.read_state(falling).sleeping);
    let resting = world.read_state(falling).position[1];
    assert!(
        (resting - 8.0).abs() < 0.3,
        "a coarse body must rest on a coarser floor, got {resting}"
    );
}

#[test]
fn a_teleported_static_collider_pairs_at_its_new_pose() {
    let mut world = new_world(gravity_config());
    let platform = world.spawn(
        BodyDesc::cuboid([0.5, 0.5, 0.5])
            .mass(0.0)
            .position([50.0, 0.5, 0.0]),
    );
    let ball = world.spawn(BodyDesc::sphere(0.25).position([0.0, 3.0, 0.0]));
    settle(&mut world, 30);
    world.set_position(platform, [0.0, 0.5, 0.0]);
    settle_until(&mut world, 180, |world| world.read_state(ball).sleeping);
    let resting = world.read_state(ball).position[1];
    assert!(
        (resting - 1.25).abs() < 0.1,
        "a teleported static collider must catch the fall, ball rests at {resting}"
    );
}

#[test]
fn a_reoriented_static_collider_pairs_at_its_new_pose() {
    let mut world = new_world(static_config());
    let rod = world.spawn(
        BodyDesc::cuboid([2.0, 0.05, 0.05])
            .mass(0.0)
            .position([0.0, 0.0, 3.0]),
    );
    let ball = world.spawn(
        BodyDesc::sphere(0.3)
            .position([0.0, 0.6, 0.0])
            .velocity([0.0, 0.0, 4.0]),
    );
    settle(&mut world, 60);
    assert!(
        world.read_state(ball).position[2] > 3.5,
        "a rod lying flat lets the ball pass above it, got {}",
        world.read_state(ball).position[2]
    );

    world.set_position(ball, [0.0, 0.6, 0.0]);
    world.set_velocity(ball, [0.0, 0.0, 4.0]);
    world.set_orientation(rod, quarter_turn_z());
    settle(&mut world, 60);
    let stopped = world.read_state(ball).position[2];
    assert!(
        stopped < 2.9,
        "an upright rod must stop the ball, got {stopped}"
    );
}

fn sparse_grains(world: &mut dynamis_world::World, side: usize, spacing: f32) {
    for index in 0..side * side * side {
        let x = (index % side) as f32 * spacing;
        let y = (index / side % side) as f32 * spacing;
        let z = (index / (side * side)) as f32 * spacing;
        world.spawn(BodyDesc::sphere(0.06).position([x, y, z]));
    }
}

fn dense_grains(world: &mut dynamis_world::World, side: usize) {
    sparse_grains(world, side, 0.1);
}

fn candidate_pairs(side: usize) -> (usize, u32) {
    let mut world = new_world(static_config());
    dense_grains(&mut world, side);
    for _ in 0..3 {
        world.step(DT);
        world.wait();
    }
    let measured = *world.measured();
    (side * side * side, measured[COUNTER_PAIRS])
}

#[test]
fn a_dense_grain_cluster_pairs_with_its_neighbours_only() {
    let (small_bodies, small_pairs) = candidate_pairs(6);
    let (large_bodies, large_pairs) = candidate_pairs(10);
    assert!(
        small_pairs < 64 * small_bodies as u32,
        "a grain lattice must not pair beyond its cell neighbourhood, got {small_pairs} pairs for {small_bodies} grains"
    );
    assert!(
        large_pairs * 4 < small_pairs * 30,
        "candidate pairs must grow with the grain count, not with the square of the density: {small_bodies} grains produced {small_pairs} pairs, {large_bodies} grains produced {large_pairs}"
    );
}

#[test]
fn a_dense_grain_cluster_stops_spilling_once_the_plan_catches_up() {
    let mut world = new_world(static_config());
    dense_grains(&mut world, 10);
    for _ in 0..3 {
        world.step(DT);
        world.wait();
    }
    assert_eq!(
        world.measured()[COUNTER_SPILLOVER_PAIRS],
        0,
        "a settled grain lattice must fit the planned pair stream"
    );
    for _ in 0..3 {
        world.step(DT);
        world.wait();
    }
    assert_eq!(
        world.measured()[COUNTER_SPILLOVER_PAIRS],
        0,
        "a settled grain lattice must keep fitting the planned pair stream"
    );
}

#[test]
fn a_wide_query_reaches_grains_finer_than_the_query() {
    let mut world = new_world(static_config());
    sparse_grains(&mut world, 6, 1.0);
    world.step(DT);
    world.wait();
    let center = [3.0, 3.0, 3.0];
    let local = world.sphere_query(center, 0.2, &QueryFilter::default());
    let wide = world.cuboid_query(center, [2.5, 2.5, 2.5], &QueryFilter::default());
    world.wait();
    assert!(
        world.query_hit(local).is_some(),
        "a local query must reach the grain it covers"
    );
    assert!(
        world.query_hit(wide).is_some(),
        "a query spanning more cells than the per level budget must still reach the grains inside it"
    );
    assert!(
        !world.query_overflow(local) && !world.query_overflow(wide),
        "a sparse fine grid must answer wide queries without reporting overflow"
    );
}

#[test]
fn a_plane_carries_grains_far_below_its_grid_resolution() {
    let mut world = new_world(gravity_config());
    world.spawn(BodyDesc::new(ColliderDesc::new(Shape::plane())).mass(0.0));
    let far = world.spawn(
        BodyDesc::sphere(0.0005)
            .position([500.0, 0.5, 0.0])
            .mass(1.0),
    );
    settle_until(&mut world, 240, |world| {
        world.read_state(far).sleeping || { world.read_state(far).position[1] < 0.01 }
    });
    assert!(
        world.measured()[COUNTER_CONTACTS] > 0,
        "a plane must pair with grains far smaller than its extent"
    );
    assert!(
        world.read_state(far).position[1] > -0.1,
        "a grain must rest on the plane, got {}",
        world.read_state(far).position[1]
    );
}
