use dynamis::{BodyDesc, ContactEventKind, GpuContext, PhysicsConfig, QueryFilter, Simulation};
use std::sync::{Mutex, MutexGuard};

const DT: f32 = 1.0 / 60.0;

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn serialized_gpu() -> (MutexGuard<'static, ()>, GpuContext) {
    let guard = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let context = pollster::block_on(GpuContext::new());
    (guard, context)
}

fn static_config() -> PhysicsConfig {
    PhysicsConfig {
        gravity: [0.0, 0.0, 0.0],
        damping: 0.0,
        angular_damping: 0.0,
        ..PhysicsConfig::default()
    }
}

#[test]
fn sensor_begin_and_end_events_fire_on_transit() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let sensor = sim.spawn(
        BodyDesc::sphere(0.5)
            .sensor(true)
            .mass(0.0)
            .position([0.0, 0.0, 0.0]),
    );
    let ball = sim.spawn(
        BodyDesc::sphere(0.2).position([0.0, 0.0, -3.0]).velocity([0.0, 0.0, 4.0]),
    );
    let mut begin_found = false;
    let mut end_found = false;
    for _ in 0..90 {
        sim.step(DT);
        sim.wait();
        for event in sim.drain_events() {
            assert!(event.sensor, "sensor transit events must be flagged");
            let touches = event.first == sensor || event.second == sensor;
            assert!(touches, "events must reference the sensor body");
            let touches_ball = event.first == ball || event.second == ball;
            assert!(touches_ball, "events must reference the moving ball");
            match event.kind {
                ContactEventKind::Begin => begin_found = true,
                ContactEventKind::End => end_found = true,
            }
        }
    }
    assert!(begin_found, "sensor transit must emit a begin event");
    assert!(end_found, "sensor transit must emit an end event");
}

#[test]
fn sensor_does_not_block_motion() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    sim.spawn(
        BodyDesc::sphere(0.5)
            .sensor(true)
            .mass(0.0)
            .position([0.0, 0.0, 0.0]),
    );
    let ball = sim.spawn(
        BodyDesc::sphere(0.2).position([0.0, 0.0, -2.0]).velocity([0.0, 0.0, 2.0]),
    );
    for _ in 0..90 {
        sim.step(DT);
    }
    sim.wait();
    let state = sim.read_state(ball);
    assert!(state.position[0].abs() < 1e-3, "sensor must not deflect the ball");
    assert!(
        state.position[2] > 0.5,
        "ball must pass through the sensor, got z={}",
        state.position[2]
    );
}

#[test]
fn contact_events_fire_on_landing_and_removal() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    let ground = sim.spawn(BodyDesc::static_sphere(10.0).position([0.0, -2.0, 0.0]));
    let ball = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 2.0, 0.0]));
    let mut began = false;
    for _ in 0..60 {
        sim.step(DT);
        sim.wait();
        for event in sim.drain_events() {
            assert!(!event.sensor, "solid contacts must not be sensor events");
            if event.kind == ContactEventKind::Begin
                && (event.first == ground || event.second == ground)
                && (event.first == ball || event.second == ball)
            {
                began = true;
            }
        }
    }
    assert!(began, "landing must emit a begin event");
    sim.remove(ball);
    let mut ended = false;
    for _ in 0..4 {
        sim.step(DT);
        sim.wait();
        for event in sim.drain_events() {
            if event.kind == ContactEventKind::End {
                ended = true;
            }
        }
    }
    assert!(ended, "removal must emit an end event for the vanished contact");
}

#[test]
fn query_filter_ignores_static_bodies() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 2.0]));
    let query = sim.raycast(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        10.0,
        &QueryFilter {
            ignore_static: true,
            ..QueryFilter::default()
        },
    );
    sim.step(DT);
    sim.wait();
    assert_eq!(sim.query_hit(query), None, "static bodies must be filtered out");
}

#[test]
fn multi_hit_query_returns_all_overlaps() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 0.0]));
    sim.spawn(BodyDesc::static_sphere(0.4).position([0.8, 0.0, 0.0]));
    let query = sim.sphere_query(
        [0.4, 0.0, 0.0],
        1.0,
        &QueryFilter {
            max_hits: 4,
            ..QueryFilter::default()
        },
    );
    sim.step(DT);
    sim.wait();
    let hits = sim.query_hits(query);
    assert_eq!(hits.len(), 2, "both spheres must be reported, got {:?}", hits.len());
    assert!(!sim.query_overflow(query));
}
