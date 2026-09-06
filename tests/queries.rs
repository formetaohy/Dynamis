use dynamis::{BodyDesc, BodyHandle, GpuContext, PhysicsConfig, QueryFilter, Simulation};

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
fn raycast_hits_nearest_body() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let near = sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 2.0]));
    let far = sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 10.0]));
    let query = sim.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 20.0, &QueryFilter::default());
    sim.step(DT);
    sim.wait();
    let hit = sim.query_hit(query).expect("raycast should hit");
    assert_eq!(hit.body, near);
    assert!(
        (hit.distance - 1.5).abs() < 1e-3,
        "expected 1.5, got {}",
        hit.distance
    );
    let _ = far;
}

#[test]
fn raycast_misses_when_out_of_range() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let _target = sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 10.0]));
    let query = sim.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 5.0, &QueryFilter::default());
    sim.step(DT);
    sim.wait();
    assert_eq!(sim.query_hit(query), None, "raycast should miss");
}

#[test]
fn raycast_hits_static_and_dynamic_bodies() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let dynamic = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 8.0]));
    let statics = sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 3.0]));
    let query = sim.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 20.0, &QueryFilter::default());
    sim.step(DT);
    sim.wait();
    let hit = sim.query_hit(query).expect("raycast should hit");
    assert_eq!(hit.body, statics);
    assert!((hit.distance - 2.5).abs() < 1e-3);
    let _ = dynamic;
}

#[test]
fn raycast_misses_when_body_behind_origin() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let _behind = sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, -5.0]));
    let query = sim.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 20.0, &QueryFilter::default());
    sim.step(DT);
    sim.wait();
    assert_eq!(
        sim.query_hit(query),
        None,
        "body behind origin must not be hit"
    );
}

#[test]
fn raycast_origin_inside_body_hits() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let target = sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 0.0]));
    let query = sim.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 20.0, &QueryFilter::default());
    sim.step(DT);
    sim.wait();
    let hit = sim.query_hit(query).expect("origin inside body should hit");
    assert_eq!(hit.body, target);
    assert!(
        (hit.distance - 0.5).abs() < 1e-3,
        "exit distance should equal radius, got {}",
        hit.distance
    );
}

#[test]
fn sphere_query_reports_overlap() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let target = sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 0.0]));
    let query = sim.sphere_query([0.0, 0.0, 0.8], 0.5, &QueryFilter::default());
    sim.step(DT);
    sim.wait();
    let hit = sim
        .query_hit(query)
        .expect("overlapping spheres should hit");
    assert_eq!(hit.body, target);
    assert!(
        (hit.distance - (-0.2)).abs() < 1e-3,
        "overlap depth expected -0.2, got {}",
        hit.distance
    );
}

#[test]
fn sphere_query_misses_when_separated() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let _target = sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 0.0]));
    let query = sim.sphere_query([0.0, 0.0, 5.0], 0.5, &QueryFilter::default());
    sim.step(DT);
    sim.wait();
    assert_eq!(sim.query_hit(query), None, "separated spheres must not hit");
}

#[test]
fn query_result_carries_simulation_step() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let _target = sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 2.0]));
    let query = sim.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 10.0, &QueryFilter::default());
    sim.step(DT);
    sim.wait();
    let hit = sim.query_hit(query).expect("raycast should hit");
    assert_eq!(hit.step, 0);
}

#[test]
fn raycast_queries_resolve_on_latest_state() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let target = sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 2.0]));
    sim.step(DT);
    sim.wait();
    sim.set_position(target, [0.0, 0.0, 30.0]);
    let query = sim.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 10.0, &QueryFilter::default());
    sim.step(DT);
    sim.wait();
    assert_eq!(
        sim.query_hit(query),
        None,
        "query must run against the post-teleport state"
    );
}

#[test]
fn batched_queries_resolve_in_submission_order() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let near = sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 2.0]));
    let far = sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 8.0]));
    let first = sim.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 30.0, &QueryFilter::default());
    let second = sim.raycast([0.0, 0.0, 9.5], [0.0, 0.0, -1.0], 30.0, &QueryFilter::default());
    sim.step(DT);
    sim.wait();
    assert_eq!(sim.query_hit(first).expect("first should hit").body, near);
    assert_eq!(sim.query_hit(second).expect("second should hit").body, far);
}

#[test]
fn query_results_stay_readable_across_steps() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 2, static_config());
    let target = sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 2.0]));
    let query = sim.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 10.0, &QueryFilter::default());
    sim.step(DT);
    sim.wait();
    let hit = sim.query_hit(query).expect("first should hit");
    assert_eq!(hit.body, target);
    for _ in 0..5 {
        sim.step(DT);
    }
    sim.wait();
    assert_eq!(
        sim.query_hit(query),
        Some(hit),
        "result must persist until its slot is reused"
    );
}

#[test]
fn stale_query_handle_panics() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 2, static_config());
    let _target = sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 2.0]));
    let first = sim.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 10.0, &QueryFilter::default());
    sim.step(DT);
    sim.wait();
    let _ = sim.query_hit(first);
    let query_capacity = sim.capacity() * 2;
    for _ in 0..query_capacity {
        sim.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 10.0, &QueryFilter::default());
    }
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sim.query_hit(first);
        }))
        .is_err()
    );
}

#[test]
fn query_ring_exhausted_panics() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 2, static_config());
    let query_capacity = sim.capacity() * 2;
    for _ in 0..query_capacity {
        sim.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 10.0, &QueryFilter::default());
    }
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sim.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 10.0, &QueryFilter::default());
        }))
        .is_err()
    );
}

#[test]
fn raycast_panics_on_non_positive_distance() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sim.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0, &QueryFilter::default());
        }))
        .is_err()
    );
}

#[test]
fn raycast_panics_on_zero_direction() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sim.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 10.0, &QueryFilter::default());
        }))
        .is_err()
    );
}

#[test]
fn bullet_sweep_prevents_tunneling() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let wall = sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 5.0]));
    let _target = sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 12.0]));

    let mut muzzle: [f32; 3] = [0.0, 0.0, 0.0];
    let mut stopped: Option<BodyHandle> = None;
    for _ in 0..10 {
        let travel = 3.0;
        let query = sim.raycast(muzzle, [0.0, 0.0, 1.0], travel, &QueryFilter::default());
        sim.step(DT);
        sim.wait();
        match sim.query_hit(query) {
            Some(hit) => {
                muzzle[2] += hit.distance;
                stopped = Some(hit.body);
                break;
            }
            None => {
                muzzle[2] += travel;
            }
        }
    }
    assert_eq!(
        stopped,
        Some(wall),
        "bullet must stop at the wall, not tunnel"
    );
    assert!(
        (muzzle[2] - 4.5).abs() < 1e-3,
        "bullet should rest at wall surface, got {}",
        muzzle[2]
    );
}
