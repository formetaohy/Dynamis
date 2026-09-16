use super::common::{DT, gravity_config, observed_world};
use dynamis_model::{BodyDesc, VehicleDesc, VehicleHandle, VehicleInput, WheelDesc};
use dynamis_world::World;

const RADIUS: f32 = 0.35;
const MASS: f32 = 900.0;

fn ground(world: &mut World) {
    world.spawn(
        BodyDesc::cuboid([60.0, 0.5, 60.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
}

fn car() -> VehicleDesc {
    let wheels = [
        WheelDesc::new([0.8, -0.2, 1.2], RADIUS)
            .steering()
            .driving(),
        WheelDesc::new([-0.8, -0.2, 1.2], RADIUS)
            .steering()
            .driving(),
        WheelDesc::new([0.8, -0.2, -1.2], RADIUS).driving(),
        WheelDesc::new([-0.8, -0.2, -1.2], RADIUS).driving(),
    ]
    .into_iter()
    .map(|wheel| wheel.suspension(0.3, 1.5, 0.7))
    .collect();
    VehicleDesc::new(
        BodyDesc::cuboid([0.9, 0.3, 1.8])
            .mass(MASS)
            .position([0.0, 0.75, 0.0]),
        wheels,
    )
}

fn settled_vehicle(world: &mut World) -> VehicleHandle {
    let vehicle = world.add_vehicle(car());
    for _ in 0..30 {
        world.step(DT);
    }
    world.wait();
    vehicle
}

fn drive(world: &mut World, vehicle: VehicleHandle, input: VehicleInput, frames: usize) {
    for _ in 0..frames {
        world.set_vehicle_input(vehicle, input);
        world.step(DT);
    }
    world.wait();
}

#[test]
fn vehicle_rests_on_its_suspension() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let vehicle = settled_vehicle(&mut world);
    let body = world.vehicle_body(vehicle);
    let state = world.inspect_vehicle_state(vehicle);
    assert_eq!(
        state.wheels_grounded, 4,
        "every wheel must touch flat ground, got {state:?}"
    );
    assert!(
        state.forward_speed.abs() < 0.05,
        "an idle vehicle must not drive itself, got {}",
        state.forward_speed
    );
    let height = world.read_state(body).position[1];
    assert!(
        (0.68..0.80).contains(&height),
        "the chassis must ride at its static height, got {height}"
    );
    let resting = world.read_state(body).position;
    drive(&mut world, vehicle, VehicleInput::IDLE, 30);
    let drifted = world.read_state(body).position;
    assert!(
        (drifted[1] - resting[1]).abs() < 1e-3,
        "a resting vehicle must not sink, got {} then {}",
        resting[1],
        drifted[1]
    );
}

#[test]
fn vehicle_drives_forward() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let vehicle = settled_vehicle(&mut world);
    drive(&mut world, vehicle, VehicleInput::drive(1.0, 0.0), 120);
    let state = world.inspect_vehicle_state(vehicle);
    assert!(
        state.position[2] > 6.0,
        "a throttled vehicle must advance along its forward axis, got {:?}",
        state.position
    );
    assert!(
        state.position[0].abs() < 0.5,
        "a straight vehicle must hold its lane, got {:?}",
        state.position
    );
    assert!(
        state.forward_speed > 6.0,
        "a throttled vehicle must carry forward speed, got {}",
        state.forward_speed
    );
    assert!(
        state.wheels_grounded >= 2,
        "a driving vehicle must stay on the ground, got {state:?}"
    );
}

#[test]
fn vehicle_steers_while_driving() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let vehicle = settled_vehicle(&mut world);
    drive(&mut world, vehicle, VehicleInput::drive(1.0, 1.0), 120);
    let state = world.inspect_vehicle_state(vehicle);
    assert!(
        state.position[0] > 4.0,
        "a right steered vehicle must turn right, got {:?}",
        state.position
    );
    assert!(
        state.position[2] > 2.0,
        "a steered vehicle must keep advancing, got {:?}",
        state.position
    );
}

#[test]
fn vehicle_brakes_to_rest() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let vehicle = settled_vehicle(&mut world);
    drive(&mut world, vehicle, VehicleInput::drive(1.0, 0.0), 120);
    let rolling = world.inspect_vehicle_state(vehicle);
    assert!(
        rolling.forward_speed > 6.0,
        "the vehicle must be rolling, got {}",
        rolling.forward_speed
    );
    let brake = VehicleInput {
        throttle: 0.0,
        steering: 0.0,
        brake: 1.0,
    };
    drive(&mut world, vehicle, brake, 180);
    let state = world.inspect_vehicle_state(vehicle);
    assert!(
        state.forward_speed.abs() < 0.2,
        "braking must stop the vehicle, got {}",
        state.forward_speed
    );
    assert!(
        state.position[2] > rolling.position[2],
        "braking must not undo the drive, got {:?}",
        state.position
    );
}

#[test]
fn vehicle_falls_without_ground() {
    let mut world = observed_world(gravity_config());
    let vehicle = world.add_vehicle(car());
    for _ in 0..30 {
        world.step(DT);
    }
    world.wait();
    let state = world.inspect_vehicle_state(vehicle);
    assert_eq!(
        state.wheels_grounded, 0,
        "an airborne vehicle touches nothing, got {state:?}"
    );
    assert!(
        state.position[1] < 0.0,
        "an airborne vehicle must fall, got {:?}",
        state.position
    );
}

#[test]
fn vehicle_state_is_observable() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let vehicle = world.add_vehicle(car());
    assert!(
        world.try_vehicle_state(vehicle).is_none(),
        "a vehicle state that no run has published must not be invented by the host"
    );
    drive(&mut world, vehicle, VehicleInput::IDLE, 10);
    let observed = world
        .try_vehicle_state(vehicle)
        .expect("wait() must publish the state of a watched vehicle");
    assert_eq!(
        observed.value.position,
        world.inspect_vehicle_state(vehicle).position,
        "the published state must be the device fact the host reads back"
    );
    world.stop_observing_vehicle(vehicle);
    assert!(
        world.try_vehicle_state(vehicle).is_none(),
        "a vehicle stops publishing once its observation ends"
    );
}

#[test]
fn vehicle_survives_a_snapshot_round_trip() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let vehicle = settled_vehicle(&mut world);
    drive(&mut world, vehicle, VehicleInput::drive(1.0, 0.0), 30);
    let snapshot = world.snapshot();
    let before = world.inspect_vehicle_state(vehicle);
    drive(&mut world, vehicle, VehicleInput::drive(1.0, 0.0), 30);
    let advanced = world.inspect_vehicle_state(vehicle);
    assert!(
        advanced.position[2] > before.position[2] + 0.5,
        "the vehicle must keep driving before the restore"
    );
    world.restore(&snapshot);
    let restored = world.inspect_vehicle_state(vehicle);
    assert!(
        (restored.position[2] - before.position[2]).abs() < 1e-3,
        "a restored vehicle must stand where the snapshot caught it, got {:?} vs {:?}",
        restored.position,
        before.position
    );
}

#[test]
fn vehicle_removal_frees_its_chassis() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let vehicle = settled_vehicle(&mut world);
    let body = world.vehicle_body(vehicle);
    world.remove_vehicle(vehicle);
    assert_eq!(world.vehicle_count(), 0, "a removed vehicle is gone");
    assert!(
        !world.bodies().contains(&body),
        "removing a vehicle must remove its chassis"
    );
}

#[test]
#[should_panic(expected = "is the chassis of a vehicle")]
fn removing_a_vehicles_chassis_directly_panics() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let vehicle = settled_vehicle(&mut world);
    let body = world.vehicle_body(vehicle);
    world.remove(body);
}

fn six_wheeler() -> VehicleDesc {
    let wheels = [
        WheelDesc::new([0.8, -0.2, 1.6], RADIUS)
            .steering()
            .driving(),
        WheelDesc::new([-0.8, -0.2, 1.6], RADIUS)
            .steering()
            .driving(),
        WheelDesc::new([0.8, -0.2, 0.0], RADIUS).driving(),
        WheelDesc::new([-0.8, -0.2, 0.0], RADIUS).driving(),
        WheelDesc::new([0.8, -0.2, -1.6], RADIUS).driving(),
        WheelDesc::new([-0.8, -0.2, -1.6], RADIUS).driving(),
    ]
    .into_iter()
    .map(|wheel| wheel.suspension(0.3, 1.5, 0.7))
    .collect();
    VehicleDesc::new(
        BodyDesc::cuboid([0.9, 0.3, 2.2])
            .mass(MASS)
            .position([0.0, 0.75, 0.0]),
        wheels,
    )
}

#[test]
fn a_vehicle_carries_every_declared_wheel() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let vehicle = world.add_vehicle(six_wheeler());
    drive(&mut world, vehicle, VehicleInput::IDLE, 30);
    let state = world.inspect_vehicle_state(vehicle);
    assert_eq!(
        state.wheels_grounded, 6,
        "every declared wheel must touch flat ground, got {state:?}"
    );
    let body = world.vehicle_body(vehicle);
    let height = world.read_state(body).position[1];
    assert!(
        (0.68..0.80).contains(&height),
        "a six wheeled chassis must ride at its static height, got {height}"
    );
    drive(&mut world, vehicle, VehicleInput::drive(1.0, 0.0), 120);
    let state = world.inspect_vehicle_state(vehicle);
    assert!(
        state.position[2] > 6.0,
        "a six wheeled vehicle must drive on every axle, got {:?}",
        state.position
    );
}

#[test]
fn a_new_vehicle_takes_over_the_wheel_span_of_a_retired_one() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let first = settled_vehicle(&mut world);
    world.remove_vehicle(first);
    let second = world.add_vehicle(car());
    drive(&mut world, second, VehicleInput::IDLE, 30);
    let state = world.inspect_vehicle_state(second);
    assert_eq!(
        state.wheels_grounded, 4,
        "a successor must answer with its own wheels, got {state:?}"
    );
    let body = world.vehicle_body(second);
    let height = world.read_state(body).position[1];
    assert!(
        (0.68..0.80).contains(&height),
        "a successor must ride at its static height, got {height}"
    );
    drive(&mut world, second, VehicleInput::drive(1.0, 0.0), 120);
    let state = world.inspect_vehicle_state(second);
    assert!(
        state.position[2] > 6.0,
        "a successor must drive from the inherited wheel span, got {:?}",
        state.position
    );
}
