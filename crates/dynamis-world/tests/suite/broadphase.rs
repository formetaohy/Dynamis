use super::common::{
    DT, assert_unrefused, gravity_config, observed_world, refused, settle, settle_until,
    static_config,
};
use dynamis_abi::{
    COUNTER_CONTACTS, COUNTER_ENTRIES, COUNTER_GRID_EXTENT, COUNTER_GRID_SCALE, COUNTER_PAIRS,
    COUNTER_REFUSED_PAIRS, COUNTER_RESTING, Shortfall,
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
    let mut world = observed_world(gravity_config());
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
    let mut world = observed_world(gravity_config());
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
    let mut world = observed_world(gravity_config());
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
    let mut world = observed_world(gravity_config());
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
    let mut world = observed_world(gravity_config());
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
    assert_eq!(hit.body(), terrain, "the ray must report the terrain body");
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
    let mut world = observed_world(gravity_config());
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
    assert_eq!(hit.body(), terrain);
}

#[test]
fn a_coarse_collider_has_an_entry_span_within_budget() {
    let mut world = observed_world(gravity_config());
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
        entries <= 20 * dynamis_abi::MAX_CELLS_PER_COLLIDER,
        "grid entries must stay within the per collider budget, got {entries}"
    );
}

#[test]
fn overlapping_coarse_colliders_repel_each_other() {
    let mut world = observed_world(gravity_config());
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
    let mut world = observed_world(gravity_config());
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
    let mut world = observed_world(static_config());
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

#[test]
fn a_collider_pair_enters_the_candidate_stream_exactly_once() {
    for offset in [0.05f32, 0.2, 0.35, 0.5, 0.65] {
        let mut world = observed_world(static_config());
        world.spawn(BodyDesc::sphere(0.35).mass(0.0).position([0.0, 0.0, 0.0]));
        world.spawn(BodyDesc::sphere(0.35).position([offset, 0.0, 0.0]));
        world.step(DT);
        world.wait();
        assert_eq!(
            world.measured()[COUNTER_PAIRS],
            1,
            "two overlapping spheres at {offset} span several cells but stay one candidate pair, got {}",
            world.measured()[COUNTER_PAIRS]
        );
        assert_unrefused(&world);
    }
}

#[test]
fn a_chain_of_colliders_emits_one_pair_per_touching_neighbour() {
    let mut world = observed_world(static_config());
    for index in 0..16 {
        world.spawn(BodyDesc::sphere(0.4).position([index as f32 * 0.5, 0.0, 0.0]));
    }
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_PAIRS],
        15,
        "sixteen chained spheres hold fifteen touching neighbours, got {}",
        world.measured()[COUNTER_PAIRS]
    );
}

#[test]
fn a_coarse_collider_emits_one_pair_per_covered_collider() {
    let mut world = observed_world(static_config());
    world.spawn(
        BodyDesc::cuboid([WIDE, 0.5, WIDE])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    for index in 0..CROWD {
        let column = (index % 8) as f32 - 3.5;
        let row = (index / 8) as f32 - 3.5;
        world.spawn(BodyDesc::sphere(0.4).position([column, 0.4, row]));
    }
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_PAIRS],
        CROWD as u32,
        "a wide floor owns exactly one candidate pair per resting body, got {}",
        world.measured()[COUNTER_PAIRS]
    );
}

#[test]
fn a_dense_pile_fits_the_planned_pair_stream_from_its_first_collapse() {
    let mut world = observed_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([WIDE, 0.5, WIDE])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    for index in 0..512 {
        let x = (index % 8) as f32 * 0.8 - 2.8;
        let z = ((index / 8) % 8) as f32 * 0.8 - 2.8;
        let y = 1.0 + (index / 64) as f32 * 0.8;
        world.spawn(BodyDesc::sphere(0.35).position([x, y, z]));
    }
    for _ in 0..120 {
        world.step(DT);
        let refusals = world.refusals();
        assert!(
            refusals.is_empty(),
            "a collapsing pile must never truncate its candidate pairs, capacity {}, refused {refusals:?}",
            world.stream_capacity().broadphase.pairs,
        );
    }
    world.wait();
    let measured = *world.measured();
    assert!(
        measured[COUNTER_CONTACTS] + measured[COUNTER_RESTING] > 0,
        "a collapsing pile must keep its contacts"
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
    let mut world = observed_world(static_config());
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
fn a_dense_grain_cluster_never_spills_its_pair_stream() {
    let mut world = observed_world(static_config());
    dense_grains(&mut world, 10);
    for frame in 0..6 {
        world.step(DT);
        world.wait();
        let refusals = world.refusals();
        assert!(
            refusals.is_empty(),
            "a grain lattice must hold every candidate pair on frame {frame}, refused {refusals:?}",
        );
    }
}

#[test]
fn a_wide_query_reaches_grains_finer_than_the_query() {
    let mut world = observed_world(static_config());
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
    let mut world = observed_world(gravity_config());
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

fn clumped_spheres(
    world: &mut dynamis_world::World,
    side: usize,
    spacing: f32,
    height: f32,
) -> Vec<dynamis_model::BodyHandle> {
    let mut bodies = Vec::with_capacity(side * side * side);
    for index in 0..side * side * side {
        let x = (index % side) as f32 * spacing - 0.5;
        let y = (index / side % side) as f32 * spacing + height;
        let z = (index / (side * side)) as f32 * spacing - 0.5;
        bodies.push(world.spawn(BodyDesc::sphere(0.06).position([x, y, z])));
    }
    bodies
}

#[test]
fn a_saturated_pair_stream_never_stores_more_candidates_than_it_holds() {
    let mut world = observed_world(gravity_config());
    clumped_spheres(&mut world, 10, 0.02, 0.1);
    let mut spill = 0;
    for frame in 0..8 {
        world.step(DT);
        world.wait();
        let measured = *world.measured();
        let capacity = world.stream_capacity().broadphase.pairs;
        let spilled = refused(&world, COUNTER_REFUSED_PAIRS);
        let stored = measured[COUNTER_PAIRS] - spilled;
        assert!(
            stored <= capacity,
            "frame {frame} stored {stored} candidates in a stream of {capacity}"
        );
        spill = spill.max(spilled);
    }
    assert!(
        spill > 0,
        "a clump of overlapping spheres must exceed the pair stream"
    );
    let refusals = world.refusals();
    assert!(
        refusals.is_empty(),
        "the widened stream of {} must serve every candidate pair, refused {refusals:?}",
        world.stream_capacity().broadphase.pairs,
    );
}

#[test]
fn a_fluid_never_drops_grid_entries() {
    let mut world = observed_world(gravity_config());
    wide_static_floor(&mut world);
    let fluid = fluid_lattice(&mut world);
    for step in 0..40 {
        world.step(DT);
        world.wait();
        let entries = world.measured()[COUNTER_ENTRIES];
        let capacity = world.stream_capacity().broadphase.entries;
        assert!(
            entries <= capacity,
            "step {step} emitted {entries} grid entries into a stream of {capacity}"
        );
        assert_eq!(
            world.measured()[dynamis_abi::COUNTER_ENTRY_FAULTS],
            0,
            "a fluid must place every grid entry"
        );
    }
    let lowest = world
        .inspect_soft_particles(fluid)
        .into_iter()
        .fold(f32::MAX, |low, position| low.min(position[1]));
    assert!(
        lowest > -0.1,
        "a fluid must rest on the floor instead of falling through it, lowest {lowest}"
    );
}

const FLUID_SIDE: usize = 8;
const FLUID_SPACING: f32 = 0.3;

fn fluid_lattice(world: &mut dynamis_world::World) -> dynamis_model::SoftBodyHandle {
    let mut particles = Vec::with_capacity(FLUID_SIDE * FLUID_SIDE * FLUID_SIDE);
    for x in 0..FLUID_SIDE {
        for y in 0..FLUID_SIDE {
            for z in 0..FLUID_SIDE {
                particles.push([
                    x as f32 * FLUID_SPACING,
                    0.3 + y as f32 * FLUID_SPACING,
                    z as f32 * FLUID_SPACING,
                ]);
            }
        }
    }
    world.add_soft_body(
        dynamis_model::SoftBodyDesc::fluid(
            particles,
            0.1,
            dynamis_model::FluidMaterial::new(FLUID_SPACING, 1.2 * FLUID_SPACING),
        )
        .position([-1.0, 0.0, -1.0]),
    )
}

#[test]
fn a_truncated_pair_stream_reports_the_counter_it_refused() {
    let mut world = observed_world(gravity_config());
    clumped_spheres(&mut world, 10, 0.02, 0.1);
    world.step(DT);
    world.wait();
    let refusals = world.refusals();
    let refusal = refusals
        .iter()
        .find(|refusal| refusal.slot == COUNTER_REFUSED_PAIRS)
        .unwrap_or_else(|| {
            panic!("a clump must report its truncated pair stream, got {refusals:?}")
        });
    assert_eq!(refusal.counter, "COUNTER_REFUSED_PAIRS");
    assert_eq!(refusal.shortfall, Shortfall::Physics);
    assert!(
        refusal.label.contains("pair"),
        "a refusal must carry the label its declaration names, got {:?}",
        refusal.label
    );
    assert!(refusal.count > 0);
}

#[test]
fn the_grid_resolution_answers_the_shape_set_not_the_poses() {
    let mut world = observed_world(static_config());
    let rod = world.spawn(
        BodyDesc::cuboid([4.0, 0.1, 0.1])
            .mass(0.0)
            .position([0.0, 0.0, 0.0]),
    );
    let bead = world.spawn(BodyDesc::static_sphere(0.25).position([0.0, 0.0, 20.0]));
    world.step(DT);
    world.wait();
    let declared = (
        world.measured()[COUNTER_GRID_SCALE],
        world.measured()[COUNTER_GRID_EXTENT],
    );

    world.set_orientation(rod, quarter_turn_z());
    world.step(DT);
    world.wait();
    assert_eq!(
        (
            world.measured()[COUNTER_GRID_SCALE],
            world.measured()[COUNTER_GRID_EXTENT],
        ),
        declared,
        "rotating a collider must not re-derive the grid resolution"
    );

    world.set_collider(
        bead,
        0,
        ColliderDesc::new(Shape::sphere(0.25)).scale([2.0, 2.0, 2.0]),
    );
    world.step(DT);
    world.wait();
    assert_ne!(
        (
            world.measured()[COUNTER_GRID_SCALE],
            world.measured()[COUNTER_GRID_EXTENT],
        ),
        declared,
        "a rescaled collider must re-derive the grid resolution"
    );
}
