use super::common::{DT, sim, static_config};
use dynamis_model::{BodyDesc, ContactEventKind};

#[test]
fn sensor_transit_emits_begin_then_end_and_never_blocks() {
    let mut world = sim(4, static_config());
    let sensor = world.spawn(
        BodyDesc::sphere(0.5)
            .sensor(true)
            .mass(0.0)
            .position([0.0, 0.0, 0.0]),
    );
    let ball = world.spawn(
        BodyDesc::sphere(0.2)
            .position([0.0, 0.0, -3.0])
            .velocity([0.0, 0.0, 4.0]),
    );
    let mut begin_found = false;
    let mut end_found = false;
    for _ in 0..90 {
        world.step(DT);
        world.wait();
        for event in world.drain_events() {
            assert!(event.sensor, "sensor events must be flagged");
            assert!(event.first == sensor || event.second == sensor);
            assert!(event.first == ball || event.second == ball);
            match event.kind {
                ContactEventKind::Begin => begin_found = true,
                ContactEventKind::End => end_found = true,
                ContactEventKind::Persist => {}
            }
        }
    }
    assert!(begin_found, "transit must emit a begin event");
    assert!(end_found, "transit must emit an end event");
    let state = world.read_state(ball);
    assert!(
        state.position[0].abs() < 1e-3 && state.position[2] > 0.5,
        "sensor must not deflect the passing ball"
    );
}

#[test]
fn solid_begin_fires_on_landing_and_end_on_removal() {
    let mut world = sim(8, super::common::gravity_config());
    let ground = world.spawn(BodyDesc::static_sphere(10.0).position([0.0, -2.0, 0.0]));
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 10.0, 0.0]));
    let mut began = false;
    for _ in 0..60 {
        world.step(DT);
        world.wait();
        for event in world.drain_events() {
            assert!(!event.sensor, "solid contacts must not be sensors");
            if event.kind == ContactEventKind::Begin
                && (event.first == ground || event.second == ground)
                && (event.first == ball || event.second == ball)
            {
                began = true;
                assert!(
                    event.point[1].abs() > 7.5,
                    "event must carry the contact point"
                );
                assert!(event.normal[1].abs() > 0.9, "event normal must be vertical");
            }
        }
    }
    assert!(began, "landing must emit a begin event");
    world.remove(ball);
    let mut ended = false;
    for _ in 0..4 {
        world.step(DT);
        world.wait();
        for event in world.drain_events() {
            if event.kind == ContactEventKind::End {
                ended = true;
            }
        }
    }
    assert!(
        ended,
        "removal must emit an end event for the vanished contact"
    );
}

#[test]
fn sensor_and_solid_events_carry_distinct_flags() {
    let mut world = sim(8, static_config());
    let sensor = world.spawn(
        BodyDesc::sphere(0.5)
            .sensor(true)
            .mass(0.0)
            .position([0.0, 0.0, 0.0]),
    );
    let solid = world.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 3.0]));
    let ball = world.spawn(
        BodyDesc::sphere(0.2)
            .position([0.0, 0.0, -3.0])
            .velocity([0.0, 0.0, 4.0]),
    );
    let _ = solid;
    let mut seen_sensor = false;
    let mut seen_solid = false;
    for _ in 0..90 {
        world.step(DT);
        world.wait();
        for event in world.drain_events() {
            if event.first == sensor || event.second == sensor {
                assert!(event.sensor);
                seen_sensor = true;
            }
            if event.first == solid || event.second == solid {
                assert!(!event.sensor, "solid events must not be sensor-flagged");
                seen_solid = true;
            }
        }
    }
    let _ = ball;
    assert!(seen_sensor, "sensor contact must be reported");
    assert!(seen_solid, "solid contact must be reported");
}

#[test]
fn live_contacts_survive_row_moves() {
    let mut world = sim(8, super::common::gravity_config());
    let ground = world.spawn(BodyDesc::static_sphere(10.0).position([0.0, -2.0, 0.0]));
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 5.0, 0.0]));
    let touches = |event: &dynamis_model::ContactEvent| {
        [event.first, event.second].contains(&ball) && [event.first, event.second].contains(&ground)
    };
    let mut landed = false;
    for _ in 0..90 {
        world.step(DT);
        world.wait();
        for event in world.drain_events() {
            landed |= event.kind == ContactEventKind::Begin && touches(&event);
        }
        if landed {
            break;
        }
    }
    assert!(landed, "the falling ball must land on the ground");

    let mut restarts = 0;
    let mut tally = |world: &mut dynamis_simulate::Simulation| {
        for event in world.drain_events() {
            if event.kind != ContactEventKind::Persist && touches(&event) {
                restarts += 1;
            }
        }
    };

    let stranger = world.spawn(BodyDesc::sphere(0.5).position([40.0, 40.0, 40.0]));
    world.spawn(BodyDesc::sphere(0.5).position([-40.0, 40.0, 40.0]));
    for _ in 0..6 {
        world.step(DT);
        world.wait();
        tally(&mut world);
    }

    world.remove(stranger);
    for _ in 0..6 {
        world.step(DT);
        world.wait();
        tally(&mut world);
    }

    assert_eq!(
        restarts, 0,
        "moving body rows must neither restart nor end a live contact"
    );
}
