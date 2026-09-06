use dynamis::{BodyDesc, BodyHandle, GpuContext, PhysicsConfig, Simulation};
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

fn settle(sim: &mut Simulation, frames: usize) {
    for _ in 0..frames {
        sim.step(DT);
    }
    sim.wait();
}

#[test]
fn stationary_body_falls_asleep() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let ball = sim.spawn(BodyDesc::sphere(0.5));
    settle(&mut sim, 40);
    assert!(
        sim.read_state(ball).sleeping,
        "idle body must fall asleep after the sleep time elapses"
    );
}

#[test]
fn moving_body_keeps_awake() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let ball = sim.spawn(BodyDesc::sphere(0.5).velocity([5.0, 0.0, 0.0]));
    settle(&mut sim, 60);
    let state = sim.read_state(ball);
    assert!(
        !state.sleeping,
        "body above the sleep velocity threshold must stay awake"
    );
    assert!(state.position[0] > 3.0, "moving body must keep travelling");
}

#[test]
fn sleep_commands_toggle_state() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let ball = sim.spawn(BodyDesc::sphere(0.5));
    settle(&mut sim, 5);
    sim.sleep(ball);
    settle(&mut sim, 2);
    assert!(
        sim.read_state(ball).sleeping,
        "explicit sleep must put the body to sleep immediately"
    );
    sim.wake(ball);
    settle(&mut sim, 2);
    assert!(
        !sim.read_state(ball).sleeping,
        "explicit wake must revive the body"
    );
}

#[test]
fn patches_and_impulses_wake_sleeping_body() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let ball = sim.spawn(BodyDesc::sphere(0.5));
    settle(&mut sim, 40);
    assert!(sim.read_state(ball).sleeping);

    sim.set_position(ball, [2.0, 0.0, 0.0]);
    settle(&mut sim, 2);
    assert!(
        !sim.read_state(ball).sleeping,
        "position patch must wake a sleeping body"
    );

    settle(&mut sim, 40);
    assert!(sim.read_state(ball).sleeping);
    sim.apply_impulse(ball, [0.0, 0.0, 1.0]);
    settle(&mut sim, 2);
    assert!(
        !sim.read_state(ball).sleeping,
        "impulse must wake a sleeping body"
    );
}

#[test]
fn impact_wakes_sleeping_body() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let ground = sim.spawn(BodyDesc::static_sphere(5.0).position([0.0, -1.0, 0.0]));
    let _ = ground;
    let target = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 4.5, 0.0]));
    settle(&mut sim, 40);
    assert!(
        sim.read_state(target).sleeping,
        "resting target must be asleep before the impact"
    );

    let _striker = sim.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 12.0, 0.0])
            .velocity([0.0, -10.0, 0.0]),
    );
    settle(&mut sim, 60);
    assert!(
        !sim.read_state(target).sleeping,
        "impact must wake the sleeping target"
    );
}

#[test]
fn sleeping_body_behaves_like_static_until_woken() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let target = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 0.0]));
    settle(&mut sim, 40);
    assert!(sim.read_state(target).sleeping);

    let striker = sim.spawn(
        BodyDesc::sphere(0.5)
            .position([-5.0, 0.0, 0.0])
            .velocity([7.0, 0.0, 0.0]),
    );
    settle(&mut sim, 40);
    let striker_x = sim.read_state(striker).position[0];
    assert!(
        striker_x < 0.0,
        "striker must not pass through the sleeping body, got x {striker_x}"
    );
    assert!(
        !sim.read_state(target).sleeping,
        "impact must wake the sleeping target"
    );
}

#[test]
fn constraint_partners_sleep_and_wake_together() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let first = sim.spawn(BodyDesc::sphere(0.3).position([0.0, 0.0, 0.0]));
    let second = sim.spawn(BodyDesc::sphere(0.3).position([2.0, 0.0, 0.0]));
    sim.add_constraint(
        first,
        second,
        dynamis::ConstraintDesc::distance([0.0; 3], [0.0; 3], 2.0),
    );
    settle(&mut sim, 50);
    assert!(
        sim.read_state(first).sleeping && sim.read_state(second).sleeping,
        "linked bodies must share the sleeping state"
    );

    sim.wake(first);
    settle(&mut sim, 3);
    assert!(
        !sim.read_state(first).sleeping && !sim.read_state(second).sleeping,
        "wake must propagate through the constraint island"
    );
}

#[test]
fn distant_idle_bodies_sleep_independently() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let first: BodyHandle = sim.spawn(BodyDesc::sphere(0.3).position([0.0, 0.0, 0.0]));
    let second: BodyHandle = sim.spawn(BodyDesc::sphere(0.3).position([50.0, 0.0, 0.0]));
    settle(&mut sim, 50);
    assert!(sim.read_state(first).sleeping && sim.read_state(second).sleeping);

    sim.wake(first);
    settle(&mut sim, 3);
    assert!(
        !sim.read_state(first).sleeping && sim.read_state(second).sleeping,
        "unconnected bodies must not propagate wake"
    );
}

#[test]
fn active_member_keeps_contact_island_awake() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 16, PhysicsConfig::default());
    let ground = sim.spawn(BodyDesc::static_sphere(5.0).position([0.0, -1.0, 0.0]));
    let _ = ground;
    let lower = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 4.5, 0.0]));
    let upper = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 5.3, 0.0]));
    settle(&mut sim, 50);
    assert!(sim.read_state(lower).sleeping && sim.read_state(upper).sleeping);

    for _ in 0..20 {
        sim.apply_force(upper, [3.0, 0.0, 0.0]);
        sim.step(DT);
    }
    sim.wait();
    let lower_state = sim.read_state(lower);
    let upper_state = sim.read_state(upper);
    assert!(
        !lower_state.sleeping && !upper_state.sleeping,
        "contact island must stay awake while driven"
    );
    assert!(
        upper_state.position[0] > 0.05,
        "driven upper body must keep moving"
    );
}
