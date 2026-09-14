use super::common::{distance, gravity_config, new_world, settle, settle_until, static_config};
use dynamis_abi::COUNTER_COARSE_NEIGHBOURS;
use dynamis_model::{
    BodyDesc, ColliderDesc, FluidMaterial, PhysicsConfig, Shape, SoftBodyDesc, SoftBodyHandle,
};
use dynamis_world::World;

const SPACING: f32 = 0.3;
const RADIUS: f32 = 0.1;

fn fluid_material() -> FluidMaterial {
    FluidMaterial::new(SPACING, 1.2 * SPACING)
}

fn lattice(extent: [u32; 3], spacing: f32, origin: [f32; 3]) -> Vec<[f32; 3]> {
    let mut particles = Vec::with_capacity((extent[0] * extent[1] * extent[2]) as usize);
    for x in 0..extent[0] {
        for y in 0..extent[1] {
            for z in 0..extent[2] {
                particles.push([
                    origin[0] + x as f32 * spacing,
                    origin[1] + y as f32 * spacing,
                    origin[2] + z as f32 * spacing,
                ]);
            }
        }
    }
    particles
}

fn mean_neighbour_span(positions: &[[f32; 3]]) -> f32 {
    let mut total = 0.0;
    let mut samples = 0.0;
    for (index, position) in positions.iter().enumerate() {
        let mut nearest = f32::MAX;
        for (other_index, other) in positions.iter().enumerate() {
            if index != other_index {
                nearest = nearest.min(distance(*position, *other));
            }
        }
        total += nearest;
        samples += 1.0;
    }
    total / samples
}

fn extent(positions: &[[f32; 3]]) -> [f32; 3] {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for position in positions {
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }
    [max[0] - min[0], max[1] - min[1], max[2] - min[2]]
}

fn fluid_block(world: &mut World, spacing: f32, origin: [f32; 3]) -> SoftBodyHandle {
    world.add_soft_body(SoftBodyDesc::fluid(
        lattice([4, 4, 4], spacing, origin),
        RADIUS,
        fluid_material(),
    ))
}

fn floor(world: &mut World, half: [f32; 3]) -> dynamis_model::BodyHandle {
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::cuboid(half)))
            .mass(0.0)
            .position([0.0, -half[1], 0.0]),
    )
}

#[test]
fn a_compressed_fluid_reaches_its_rest_spacing() {
    let mut world = new_world(static_config());
    let handle = world.add_soft_body(SoftBodyDesc::fluid(
        lattice([3, 3, 3], 0.2, [-0.2, 0.0, -0.2]),
        0.09,
        fluid_material(),
    ));
    settle(&mut world, 120);
    let grew = extent(&world.soft_body_positions(handle));
    settle(&mut world, 120);
    let held = extent(&world.soft_body_positions(handle));
    let span = mean_neighbour_span(&world.soft_body_positions(handle));
    assert!(
        (span - SPACING).abs() < 0.05 * SPACING,
        "an incompressible fluid must relax to its rest spacing {SPACING}, got {span}"
    );
    assert_eq!(
        grew, held,
        "a relaxed fluid must not drift once it reaches its rest spacing"
    );
    assert_eq!(world.measured()[COUNTER_COARSE_NEIGHBOURS], 0);
}

#[test]
fn a_fluid_holds_its_spacing_where_a_plain_particle_cloud_packs_solid() {
    let mut spans = Vec::new();
    for fluid in [false, true] {
        let mut world = new_world(gravity_config());
        floor(&mut world, [1.5, 0.25, 1.5]);
        let particles = lattice([4, 4, 4], SPACING, [-0.45, 0.6, -0.45]);
        let handle = if fluid {
            world.add_soft_body(SoftBodyDesc::fluid(particles, RADIUS, fluid_material()))
        } else {
            world.add_soft_body(SoftBodyDesc::new(particles, Vec::new()).radius(RADIUS))
        };
        settle(&mut world, 300);
        let positions = world.soft_body_positions(handle);
        spans.push(mean_neighbour_span(&positions));
        assert!(
            positions.iter().all(|position| position[1] > 0.0),
            "a settled fluid must rest on the floor"
        );
        assert_eq!(world.measured()[COUNTER_COARSE_NEIGHBOURS], 0);
    }
    assert!(
        spans[1] > 0.85 * SPACING,
        "a fluid must hold its rest spacing, got {}",
        spans[1]
    );
    assert!(
        spans[0] < 0.75 * spans[1],
        "a plain particle cloud packs onto its collision radius while a fluid holds its spacing, {spans:?}"
    );
}

#[test]
fn a_fluid_impact_never_sinks_through_the_floor() {
    let mut world = new_world(gravity_config());
    floor(&mut world, [2.0, 0.25, 2.0]);
    let handle = fluid_block(&mut world, SPACING, [-0.45, 3.0, -0.45]);
    settle(&mut world, 240);
    let positions = world.soft_body_positions(handle);
    let lowest = positions.iter().fold(f32::MAX, |low, at| low.min(at[1]));
    assert!(
        lowest > 0.0,
        "a fluid impact must stay on top of the floor, lowest particle at {lowest}"
    );
    assert_eq!(world.measured()[COUNTER_COARSE_NEIGHBOURS], 0);
}

#[test]
fn a_fluid_pushes_a_body_it_rests_on() {
    let mut world = new_world(gravity_config());
    floor(&mut world, [2.0, 0.25, 2.0]);
    let carrier = world.spawn(BodyDesc::cuboid([0.4, 0.1, 0.4]).position([0.0, 0.1, 0.0]));
    let _fluid = fluid_block(&mut world, SPACING, [-0.45, 1.6, -0.45]);
    settle_until(&mut world, 400, |world| {
        world.read_state(carrier).position[0] > 0.05
    });
}

#[test]
fn identical_fluids_observe_identical_positions() {
    let mut worlds = Vec::new();
    for _ in 0..2 {
        let mut world = new_world(gravity_config());
        floor(&mut world, [1.5, 0.25, 1.5]);
        let handle = fluid_block(&mut world, SPACING, [-0.45, 1.0, -0.45]);
        settle(&mut world, 120);
        worlds.push((world, handle));
    }
    let mut observed = Vec::new();
    for (world, handle) in worlds.iter_mut() {
        observed.push(world.soft_body_positions(*handle));
    }
    let (first, second) = (&observed[0], &observed[1]);
    assert_eq!(
        first, second,
        "identical fluid worlds must observe identical particle positions"
    );
}

#[test]
fn a_removed_fluid_leaves_the_world_steppable() {
    let mut world = new_world(PhysicsConfig::default());
    let handle = fluid_block(&mut world, SPACING, [-0.45, 0.5, -0.45]);
    settle(&mut world, 30);
    world.remove_soft_body(handle);
    assert_eq!(world.soft_body_count(), 0);
    settle(&mut world, 30);
}
