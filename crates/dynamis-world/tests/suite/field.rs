use super::common::{
    DT, asleep, gravity_config, observed_world, settle, settle_until, static_config,
};
use dynamis_model::{
    BodyDesc, CollisionFilter, FieldDesc, FieldHandle, FieldRegion, SoftBodyDesc, SoftBodyHandle,
};
use dynamis_world::World;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn ground(world: &mut World) {
    world.spawn(
        BodyDesc::cuboid([10.0, 0.5, 10.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
}

fn resting_body(world: &mut World) -> dynamis_model::BodyHandle {
    world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.5, 0.0]))
}

fn pressed_body(world: &mut World) -> (dynamis_model::BodyHandle, FieldHandle) {
    ground(world);
    let body = resting_body(world);
    let field = world.add_field(FieldDesc::uniform(
        FieldRegion::sphere(4.0),
        [0.0, -12.0, 0.0],
    ));
    (body, field)
}

#[test]
fn a_rotated_cuboid_field_reaches_the_volume_it_spans() {
    let mut world = observed_world(static_config());
    let quarter_turn = [
        0.0,
        0.0,
        std::f32::consts::FRAC_PI_4.sin(),
        std::f32::consts::FRAC_PI_4.cos(),
    ];
    let inside = world.spawn(BodyDesc::sphere(0.5).position([0.0, 1.5, 0.0]));
    let outside = world.spawn(BodyDesc::sphere(0.5).position([0.0, 3.5, 0.0]));
    world.add_field(FieldDesc::uniform(
        FieldRegion::oriented_cuboid([2.0, 0.5, 0.5], quarter_turn),
        [8.0, 0.0, 0.0],
    ));
    settle(&mut world, 20);
    assert!(
        world.read_state(inside).position[0] > 0.3,
        "a body inside the turned cuboid must answer the field, got {:?}",
        world.read_state(inside).position
    );
    assert_eq!(
        world.read_state(outside).position,
        [0.0, 3.5, 0.0],
        "a body outside the turned cuboid must never answer the field"
    );
}

#[test]
fn a_field_honours_the_filter_of_the_bodies_it_reaches() {
    let mut world = observed_world(static_config());
    let reached = world.spawn(BodyDesc::sphere(0.5).filter(CollisionFilter::new(1, 1)));
    let held = world.spawn(
        BodyDesc::sphere(0.5)
            .position([3.0, 0.0, 0.0])
            .filter(CollisionFilter::new(2, 2)),
    );
    world.add_field(
        FieldDesc::uniform(FieldRegion::Global, [9.0, 0.0, 0.0]).filter(CollisionFilter::new(1, 1)),
    );
    settle(&mut world, 20);
    assert!(
        world.read_state(reached).position[0] > 0.4,
        "a body the filter admits must answer the field, got {:?}",
        world.read_state(reached).position
    );
    assert_eq!(
        world.read_state(held).position,
        [3.0, 0.0, 0.0],
        "a body the filter excludes must never answer the field"
    );
}

#[test]
fn a_uniform_field_accelerates_the_bodies_it_reaches() {
    let mut world = observed_world(static_config());
    let inside = world.spawn(BodyDesc::sphere(0.5));
    let outside = world.spawn(BodyDesc::sphere(0.5).position([6.0, 0.0, 0.0]));
    world.add_field(FieldDesc::uniform(
        FieldRegion::sphere(2.0),
        [10.0, 0.0, 0.0],
    ));
    settle(&mut world, 30);
    let moved = world.read_state(inside);
    assert!(
        moved.position[0] > 1.0,
        "a body inside the field must be pushed along it, got {:?}",
        moved.position
    );
    assert!(moved.velocity[0] > 4.0, "got {:?}", moved.velocity);
    assert_eq!(
        world.read_state(outside).position,
        [6.0, 0.0, 0.0],
        "a body outside the region must never answer the field"
    );
}

#[test]
fn a_drag_field_relaxes_a_body_toward_its_medium() {
    let mut world = observed_world(static_config());
    let body = world.spawn(BodyDesc::sphere(0.5).velocity([5.0, 0.0, 0.0]));
    world.add_field(FieldDesc::wind(FieldRegion::Global, [0.0; 3], 20.0));
    settle(&mut world, 40);
    let state = world.read_state(body);
    assert!(
        state.velocity[0] < 0.2,
        "a drag must relax the speed toward the medium, got {}",
        state.velocity[0]
    );
    assert!(
        state.velocity[0] >= 0.0,
        "a drag must never reverse the speed past the medium, got {}",
        state.velocity[0]
    );
    assert!(
        state.position[0] > 0.2,
        "a drifting body must keep the distance it covered, got {:?}",
        state.position
    );

    let mut world = observed_world(static_config());
    let body = world.spawn(BodyDesc::sphere(0.5).velocity([40.0, 0.0, 0.0]));
    world.add_field(
        FieldDesc::of(FieldRegion::Global)
            .medium([0.0; 3])
            .drag(0.0, 1000.0),
    );
    settle(&mut world, 2);
    assert_eq!(
        world.read_state(body).velocity[0],
        0.0,
        "a quadratic drag at its limit must stop the body without reversing it"
    );
}

#[test]
fn a_buoyant_field_floats_the_body_it_holds() {
    let mut world = observed_world(gravity_config());
    let floats = world.spawn(BodyDesc::sphere(0.5).position([0.0, 5.0, 0.0]));
    let sinks = world.spawn(BodyDesc::sphere(0.5).position([20.0, 5.0, 0.0]));
    world.add_field(FieldDesc::buoyant(FieldRegion::sphere(8.0), 1.5, 0.0));
    settle(&mut world, 30);
    assert!(
        world.read_state(floats).position[1] > 5.5,
        "a medium denser than the body must float it, got {}",
        world.read_state(floats).position[1]
    );
    assert!(
        world.read_state(sinks).position[1] < 4.5,
        "a body outside the medium must keep falling, got {}",
        world.read_state(sinks).position[1]
    );
}

#[test]
fn a_radial_field_pulls_toward_its_position() {
    let mut world = observed_world(static_config());
    let body = world.spawn(BodyDesc::sphere(0.5).position([4.0, 0.0, 0.0]));
    world.add_field(FieldDesc::radial(FieldRegion::sphere(10.0), [0.0; 3], 9.0));
    settle(&mut world, 20);
    let state = world.read_state(body);
    assert!(
        state.position[0] < 4.0 && state.velocity[0] < 0.0,
        "a positive pull must draw the body inward, got {:?} at {:?}",
        state.velocity,
        state.position
    );
}

#[test]
fn a_vortex_field_swirls_about_its_axis() {
    let mut world = observed_world(static_config());
    let body = world.spawn(BodyDesc::sphere(0.5).position([2.0, 0.0, 0.0]));
    world.add_field(FieldDesc::vortex(
        FieldRegion::sphere(10.0),
        [0.0; 3],
        [0.0, 1.0, 0.0],
        2.0,
    ));
    settle(&mut world, 20);
    let state = world.read_state(body);
    assert!(
        state.position[2] < 0.0 && state.velocity[2] < 0.0,
        "a positive swirl about +y must turn the body toward -z, got {:?} at {:?}",
        state.velocity,
        state.position
    );
}

#[test]
fn a_field_refresh_leaves_a_sleeping_body_asleep() {
    let mut world = observed_world(gravity_config());
    let (body, field) = pressed_body(&mut world);
    settle_until(&mut world, 240, |world| asleep(world));
    let resting = world.read_state(body).position;
    let pressed = world.field_desc(field);
    world.update_field(field, pressed);
    settle(&mut world, 12);
    assert_eq!(
        world.read_state(body).position,
        resting,
        "a field refresh must not move a body that sleeps through it"
    );
    assert!(asleep(&world), "a field refresh must not wake the body");
}

#[test]
fn a_changed_field_resumes_the_sleeping_body_it_reaches() {
    let mut world = observed_world(gravity_config());
    let (body, field) = pressed_body(&mut world);
    settle_until(&mut world, 240, |world| asleep(world));
    let resting = world.read_state(body).position;
    world.update_field(
        field,
        FieldDesc::uniform(FieldRegion::sphere(4.0), [0.0, 30.0, 0.0]),
    );
    settle(&mut world, 12);
    assert!(
        world.read_state(body).position[1] > resting[1] + 0.1,
        "a field that changed must resume the body it reaches, got {}",
        world.read_state(body).position[1]
    );
}

#[test]
fn a_new_field_resumes_the_sleeping_body_it_reaches() {
    let mut world = observed_world(gravity_config());
    let (body, _) = pressed_body(&mut world);
    settle_until(&mut world, 240, |world| asleep(world));
    let resting = world.read_state(body).position;
    world.add_field(FieldDesc::uniform(
        FieldRegion::sphere(4.0),
        [0.0, 30.0, 0.0],
    ));
    settle(&mut world, 12);
    assert!(
        world.read_state(body).position[1] > resting[1] + 0.1,
        "a new field must resume the body it reaches, got {}",
        world.read_state(body).position[1]
    );
}

#[test]
fn a_removed_field_wakes_the_sleeping_simulation() {
    let mut world = observed_world(gravity_config());
    let (_, field) = pressed_body(&mut world);
    settle_until(&mut world, 240, |world| asleep(world));
    world.remove_field(field);
    settle(&mut world, 4);
    assert!(
        !asleep(&world),
        "a removed field must wake the simulation it held"
    );
    assert_eq!(world.field_count(), 0);
}

#[test]
fn a_wind_field_carries_the_soft_body_it_reaches() {
    let mut world = observed_world(static_config());
    let net: SoftBodyHandle = world.add_soft_body(SoftBodyDesc::net(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        vec![[0, 1]],
    ));
    world.add_field(FieldDesc::wind(FieldRegion::Global, [6.0, 0.0, 0.0], 6.0));
    settle(&mut world, 60);
    let particles = world.inspect_soft_particles(net);
    assert!(
        particles.iter().all(|particle| particle[0] > 0.5),
        "a wind must carry every particle downwind, got {particles:?}"
    );
    assert!(
        (particles[0][0] - particles[1][0]).abs() < 1.2,
        "a wind must not tear the element that links the particles, got {particles:?}"
    );
}

#[test]
fn a_burst_of_fields_widens_the_field_stream() {
    let mut world = observed_world(static_config());
    let floor = world.stream_capacity().state.fields;
    for _ in 0..=floor {
        world.add_field(FieldDesc::uniform(FieldRegion::Global, [1.0, 0.0, 0.0]));
    }
    assert_eq!(world.field_count(), floor as usize + 1);
    world.step(DT);
    world.wait();
    assert!(
        world.stream_capacity().state.fields > floor,
        "a field burst must widen the field stream past {floor}"
    );
}

#[test]
fn a_snapshot_carries_the_fields_it_captured() {
    let mut world = observed_world(static_config());
    let body = world.spawn(BodyDesc::sphere(0.5));
    world.add_field(FieldDesc::uniform(FieldRegion::Global, [8.0, 0.0, 0.0]));
    settle(&mut world, 6);
    let snapshot = world.snapshot();
    let speed = world.read_state(body).velocity[0];
    settle(&mut world, 6);
    world.restore(&snapshot);
    assert_eq!(world.field_count(), 1);
    world.observe_all_bodies();
    settle(&mut world, 6);
    assert!(
        world.read_state(body).velocity[0] > speed,
        "a restored field must keep accelerating, got {} after {}",
        world.read_state(body).velocity[0],
        speed
    );
}

#[test]
fn identical_fields_observe_identical_positions() {
    fn build(world: &mut World) -> dynamis_model::BodyHandle {
        let body = world.spawn(
            BodyDesc::sphere(0.5)
                .position([2.0, 0.0, 0.0])
                .velocity([0.0, 1.0, 0.0]),
        );
        world.add_field(
            FieldDesc::vortex(FieldRegion::sphere(10.0), [0.0; 3], [0.0, 1.0, 0.0], 3.0)
                .angular_drag(2.0),
        );
        world.add_field(FieldDesc::uniform(
            FieldRegion::sphere(10.0),
            [0.0, 0.0, -4.0],
        ));
        body
    }

    let mut first = observed_world(static_config());
    let first_body = build(&mut first);
    let mut second = observed_world(static_config());
    let second_body = build(&mut second);
    for _ in 0..48 {
        first.step(DT);
        second.step(DT);
    }
    first.wait();
    second.wait();
    assert_eq!(
        world_bits(&first, first_body),
        world_bits(&second, second_body),
        "two worlds driven by the same fields must agree bit for bit"
    );
}

fn world_bits(world: &World, body: dynamis_model::BodyHandle) -> Vec<u32> {
    let state = world.read_state(body);
    state
        .position
        .iter()
        .chain(state.velocity.iter())
        .map(|value| value.to_bits())
        .collect()
}

#[test]
fn a_field_without_an_effect_leaves_the_world_whole() {
    let mut world = observed_world(static_config());
    let body = world.spawn(BodyDesc::sphere(0.5));
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            world.add_field(FieldDesc::of(FieldRegion::Global));
        }))
        .is_err(),
        "a field that declares no effect must be refused"
    );
    assert_eq!(world.field_count(), 0);
    assert!(world.fields().is_empty());
    world.add_field(FieldDesc::uniform(FieldRegion::Global, [4.0, 0.0, 0.0]));
    settle(&mut world, 12);
    assert!(world.read_state(body).velocity[0] > 0.0);
}

#[test]
fn a_swirl_axis_must_be_a_unit_vector() {
    let mut world = observed_world(static_config());
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            world.add_field(FieldDesc::of(FieldRegion::Global).swirl([0.0, 2.0, 0.0], 1.0));
        }))
        .is_err(),
        "a swirl axis must be a unit vector"
    );
    assert_eq!(world.field_count(), 0);
}
