use super::common::{DT, gravity_config, new_world, settle_until};
use dynamis_layout::{COUNTER_CONTACTS, COUNTER_ENTRIES, COUNTER_PAIRS, COUNTER_RESTING};
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
    settle_until(&mut world, 240, |world| {
        world.measured()[COUNTER_CONTACTS] > 0 && world.read_state(block).sleeping
    });
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
    assert!(world.measured()[COUNTER_ENTRIES] > 0 && world.read_state(terrain).position[1] == -0.5);
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
    world.flush_queries();
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
    world.flush_queries();
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
