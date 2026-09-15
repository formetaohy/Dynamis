use super::common::{DT, new_world, static_config};
use dynamis_model::BodyHandle;
use dynamis_model::{
    BodyDesc, ColliderDesc, ContactEvent, ContactEventKind, ContactEventMode, FluidMaterial, Shape,
    SoftBodyDesc, SoftBodyHandle,
};

#[test]
fn sensor_transit_emits_begin_then_end_and_never_blocks() {
    let mut world = new_world(static_config());
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
            assert!(event.first.is_body(sensor) || event.second.is_body(sensor));
            assert!(event.first.is_body(ball) || event.second.is_body(ball));
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
    let mut world = new_world(super::common::gravity_config());
    let ground = world.spawn(BodyDesc::static_sphere(10.0).position([0.0, -2.0, 0.0]));
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 10.0, 0.0]));
    let mut began = false;
    for _ in 0..60 {
        world.step(DT);
        world.wait();
        for event in world.drain_events() {
            assert!(!event.sensor, "solid contacts must not be sensors");
            if event.kind == ContactEventKind::Begin
                && (event.first.is_body(ground) || event.second.is_body(ground))
                && (event.first.is_body(ball) || event.second.is_body(ball))
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
    let mut world = new_world(static_config());
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
            if event.first.is_body(sensor) || event.second.is_body(sensor) {
                assert!(event.sensor);
                seen_sensor = true;
            }
            if event.first.is_body(solid) || event.second.is_body(solid) {
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
    let mut world = new_world(super::common::gravity_config());
    let ground = world.spawn(BodyDesc::static_sphere(10.0).position([0.0, -2.0, 0.0]));
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 5.0, 0.0]));
    let touches = |event: &dynamis_model::ContactEvent| {
        (event.first.is_body(ball) || event.second.is_body(ball))
            && (event.first.is_body(ground) || event.second.is_body(ground))
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
    let mut tally = |world: &mut dynamis_world::World| {
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

fn soft_particles(rows: u32, columns: u32) -> Vec<[f32; 3]> {
    let mut particles = Vec::new();
    for row in 0..rows {
        for column in 0..columns {
            particles.push([row as f32 * 0.3, 0.0, column as f32 * 0.3]);
        }
    }
    particles
}

fn soft_solid_contact(
    world: &mut dynamis_world::World,
    soft: SoftBodyHandle,
    body: BodyHandle,
) -> Vec<ContactEvent> {
    world
        .drain_events()
        .into_iter()
        .filter(|event| {
            (event.first.is_soft(soft) && event.second.is_body(body))
                || (event.second.is_soft(soft) && event.first.is_body(body))
        })
        .collect()
}

#[test]
fn a_soft_body_announces_its_landing_and_its_parting() {
    let mut world = new_world(super::common::gravity_config());
    let ground = world.spawn(BodyDesc::sphere(10.0).mass(0.0).position([0.0, -9.0, 0.0]));
    let soft = world.add_soft_body(
        SoftBodyDesc::fluid(soft_particles(2, 2), 0.15, FluidMaterial::new(0.3, 0.36))
            .position([0.0, 2.0, 0.0])
            .events(ContactEventMode::BeginEnd),
    );
    let mut began = false;
    for _ in 0..90 {
        world.step(DT);
        world.wait();
        for event in soft_solid_contact(&mut world, soft, ground) {
            if event.kind == ContactEventKind::Begin {
                began = true;
                assert!(
                    !event.sensor,
                    "a solid contact must not be flagged a sensor"
                );
                assert!(
                    event.first.is_soft(soft) || event.second.is_soft(soft),
                    "the fact must address the soft particle it came from, got {event:?}"
                );
                assert!(
                    (event.point[1] - 1.0).abs() < 0.5,
                    "the fact must carry the contact point, got {event:?}"
                );
                assert!(
                    event.normal[1].abs() > 0.9,
                    "the fact must carry the surface normal"
                );
            }
        }
    }
    assert!(began, "a landing soft body must announce its contact");
    world.remove(ground);
    let mut ended = false;
    for _ in 0..6 {
        world.step(DT);
        world.wait();
        for event in world.drain_events() {
            if event.kind == ContactEventKind::End
                && (event.first.is_soft(soft) || event.second.is_soft(soft))
            {
                ended = true;
            }
        }
    }
    assert!(
        ended,
        "a parted soft body must announce the end of its contact"
    );
}

#[test]
fn a_soft_body_transits_a_sensor_without_being_deflected() {
    let mut world = new_world(super::common::gravity_config());
    let sensor = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(1.0)).sensor(true))
            .mass(0.0)
            .position([0.0, 5.0, 0.0]),
    );
    let soft = world.add_soft_body(
        SoftBodyDesc::fluid(soft_particles(2, 2), 0.15, FluidMaterial::new(0.3, 0.36))
            .position([0.0, 8.0, 0.0])
            .events(ContactEventMode::BeginEnd),
    );
    let mut began = false;
    let mut ended = false;
    for _ in 0..180 {
        world.step(DT);
        world.wait();
        for event in soft_solid_contact(&mut world, soft, sensor) {
            assert!(
                event.sensor,
                "a sensor transit must be flagged, got {event:?}"
            );
            match event.kind {
                ContactEventKind::Begin => began = true,
                ContactEventKind::End => ended = true,
                ContactEventKind::Persist => {}
            }
        }
    }
    assert!(began && ended, "a sensor transit must begin and end");
    let height = world
        .inspect_soft_particles(soft)
        .iter()
        .map(|particle| particle[1])
        .fold(f32::MAX, f32::min);
    assert!(
        height < 3.0,
        "a sensor must not deflect a soft body, lowest particle {height}"
    );
}

#[test]
fn a_soft_body_without_events_stays_silent() {
    let mut world = new_world(super::common::gravity_config());
    let ground = world.spawn(BodyDesc::sphere(10.0).mass(0.0).position([0.0, -9.0, 0.0]));
    let soft = world.add_soft_body(
        SoftBodyDesc::fluid(soft_particles(2, 2), 0.15, FluidMaterial::new(0.3, 0.36))
            .position([0.0, 2.0, 0.0]),
    );
    let mut announced = 0;
    for _ in 0..90 {
        world.step(DT);
        world.wait();
        announced += soft_solid_contact(&mut world, soft, ground).len();
    }
    assert_eq!(
        announced, 0,
        "a soft body that carries no event mode must stay silent"
    );
    assert!(
        (world.inspect_soft_particles(soft)[0][1] - 1.0).abs() < 0.5,
        "the silent soft body must still rest on the ground"
    );
}

#[test]
fn a_soft_body_reports_each_persisting_step_once() {
    let mut world = new_world(super::common::gravity_config());
    let ground = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(10.0)).events(ContactEventMode::Persist))
            .mass(0.0)
            .position([0.0, -9.0, 0.0]),
    );
    let soft = world.add_soft_body(
        SoftBodyDesc::fluid(soft_particles(1, 1), 0.15, FluidMaterial::new(0.3, 0.36))
            .position([0.0, 2.0, 0.0])
            .events(ContactEventMode::Persist),
    );
    let mut persists = 0;
    for _ in 0..150 {
        world.step(DT);
        world.wait();
        let events = soft_solid_contact(&mut world, soft, ground);
        let begins = events
            .iter()
            .filter(|event| event.kind == ContactEventKind::Begin)
            .count();
        let step_persists = events
            .iter()
            .filter(|event| event.kind == ContactEventKind::Persist)
            .count();
        assert!(
            step_persists <= begins.max(1),
            "a persisting contact rides one step at a time, got {step_persists} for {begins} begins"
        );
        persists += step_persists;
    }
    assert!(
        persists > 4,
        "a resting soft body must persist its contact, got {persists}"
    );
    assert!(
        !world.read_state(ground).sleeping,
        "the ground stays a static body"
    );
    assert_eq!(
        world.measured()[dynamis_abi::COUNTER_REFUSED_SOFT_EVENTS],
        0,
        "a soft body must not spill its contact facts"
    );
}

#[test]
fn two_soft_bodies_announce_each_other() {
    let mut world = new_world(super::common::gravity_config());
    world.spawn(BodyDesc::sphere(10.0).mass(0.0).position([0.0, -9.0, 0.0]));
    let lower = world.add_soft_body(
        SoftBodyDesc::fluid(soft_particles(1, 1), 0.15, FluidMaterial::new(0.3, 0.36))
            .position([0.0, 1.0, 0.0])
            .events(ContactEventMode::BeginEnd),
    );
    let upper = world.add_soft_body(
        SoftBodyDesc::fluid(soft_particles(1, 1), 0.15, FluidMaterial::new(0.3, 0.36))
            .position([0.0, 1.25, 0.0])
            .events(ContactEventMode::BeginEnd),
    );
    let mut announced = 0;
    let mut parted = 0;
    for _ in 0..90 {
        world.step(DT);
        world.wait();
        for event in world.drain_events() {
            let (Some((first_body, first_particle)), Some((second_body, second_particle))) =
                (event.first.particle(), event.second.particle())
            else {
                continue;
            };
            assert_ne!(
                first_body, second_body,
                "a soft body must not announce its own particles"
            );
            assert!(
                (first_body == lower && second_body == upper)
                    || (first_body == upper && second_body == lower),
                "the fact must address the two soft bodies that touched, got {event:?}"
            );
            assert_eq!((first_particle, second_particle), (0, 0));
            assert!(!event.sensor);
            match event.kind {
                ContactEventKind::Begin => announced += 1,
                ContactEventKind::End => parted += 1,
                ContactEventKind::Persist => {}
            }
        }
    }
    assert!(
        announced > 0,
        "two stacked soft bodies must announce each other"
    );
    assert!(parted > 0, "the parted pair must announce its parting");
    let lower_height = world.inspect_soft_particles(lower)[0][1];
    let upper_height = world.inspect_soft_particles(upper)[0][1];
    assert!(
        upper_height > lower_height,
        "the announced pair must stay stacked, lower {lower_height} upper {upper_height}"
    );
}
