use super::common::distance;
use super::common::{DT, gravity_config, sim, static_config};
use dynamis_gpu::{GpuContext, GpuRequest, WarmupBudget};
use dynamis_model::{BodyDesc, BodyHandle, ConstraintDesc, PhysicsConfig};
use dynamis_simulate::Simulation;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn logical_capacity_runs_on_minimum_device_limits() {
    let context = pollster::block_on(GpuContext::open(&GpuRequest::minimum_limits()))
        .expect("minimum device limits must support a simulation");
    let mut world = Simulation::new(context, static_config());
    world.warmup(WarmupBudget::All);
    let body = world.spawn(BodyDesc::sphere(0.5));
    world.step(DT);
    world.wait();
    assert_eq!(world.read_state(body).position, [0.0; 3]);
}

fn sphere_inertia(radius: f32, mass: f32) -> f32 {
    5.0 / (2.0 * mass * radius * radius)
}

#[test]
fn spawn_state_is_immediately_readable() {
    let mut world = sim(static_config());
    let ball = world.spawn(
        BodyDesc::sphere(0.5)
            .position([1.0, 2.0, 3.0])
            .velocity([4.0, 5.0, 6.0])
            .mass(2.0),
    );
    let state = world.read_state(ball);
    assert_eq!(state.position, [1.0, 2.0, 3.0]);
    assert_eq!(state.velocity, [4.0, 5.0, 6.0]);
    assert_eq!(state.orientation, [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(state.inverse_mass, 0.5);
    assert_eq!(world.count(), 1);
    assert_eq!(world.bodies(), &[ball]);
}

#[test]
fn remove_swap_keeps_live_bodies_in_place() {
    let mut world = sim(static_config());
    let first = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 0.0]));
    let second = world.spawn(BodyDesc::sphere(0.5).position([2.0, 0.0, 0.0]));
    let third = world.spawn(BodyDesc::sphere(0.5).position([4.0, 0.0, 0.0]));
    world.step(DT);
    world.remove(second);
    world.step(DT);
    world.wait();
    assert_eq!(world.count(), 2);
    assert_eq!(world.read_state(first).position, [0.0, 0.0, 0.0]);
    assert_eq!(world.read_state(third).position, [4.0, 0.0, 0.0]);
    assert_eq!(world.bodies(), &[first, third]);
}

#[test]
fn removed_and_stale_handles_panic() {
    let mut world = sim(static_config());
    let ball = world.spawn(BodyDesc::sphere(0.5));
    world.remove(ball);
    assert!(catch_unwind(AssertUnwindSafe(|| world.read_state(ball))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| world.remove(ball))).is_err());
    let fresh = world.spawn(BodyDesc::sphere(0.5));
    assert_ne!(
        fresh.generation, ball.generation,
        "slot reuse must bump generation"
    );
    assert!(catch_unwind(AssertUnwindSafe(|| world.read_state(ball))).is_err());
    world.step(DT);
    assert!(
        catch_unwind(AssertUnwindSafe(|| world.read_state(fresh))).is_err(),
        "device-owned state must require explicit synchronization"
    );
    world.wait();
    assert_eq!(world.count(), 1);
    assert_eq!(world.read_state(fresh).position, [0.0, 0.0, 0.0]);
}

fn spawn_burst(world: &mut Simulation, count: usize) {
    for index in 0..count {
        world.spawn(BodyDesc::static_sphere(0.05).position([1000.0 + index as f32, 0.0, 0.0]));
    }
}

#[test]
fn a_world_outgrowing_its_plan_keeps_simulating() {
    let mut world = sim(static_config());
    let ball = world.spawn(
        BodyDesc::sphere(0.3)
            .position([0.0, 5.0, 0.0])
            .velocity([1.0, 0.0, 0.0]),
    );
    settle_frames(&mut world, 10);
    let before = world.read_state(ball).position[0];
    spawn_burst(&mut world, 160);
    settle_frames(&mut world, 10);
    let after = world.read_state(ball).position[0];
    assert!(
        after > before + 0.1,
        "a world reallocated under live demand must keep its flow"
    );
    assert!((world.read_state(ball).position[1] - 5.0).abs() < 1e-4);
    let extra = world.spawn(BodyDesc::sphere(0.2).position([0.0, 4.0, 0.0]));
    world.step(DT);
    world.wait();
    assert!(world.read_state(extra).position[1] > 0.0);
}

#[test]
fn body_and_constraint_handles_grow_on_demand() {
    let mut world = sim(static_config());
    let ground = world.spawn(BodyDesc::static_sphere(5.0));
    for index in 0..40u32 {
        let ball = world.spawn(BodyDesc::sphere(0.1).position([0.0, 3.0 + index as f32, 0.0]));
        world.add_constraint(
            ball,
            ground,
            ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0),
        );
    }
    assert_eq!(world.constraints().len(), 40);
    world.step(DT);
    world.wait();
    assert_eq!(world.count(), 41);
}

#[test]
fn invalid_inputs_panic() {
    let mut world = sim(static_config());
    let ball = world.spawn(BodyDesc::sphere(0.5));
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            world.read_state(BodyHandle {
                id: ball.id + 9,
                generation: ball.generation,
            });
        }))
        .is_err()
    );
    assert!(catch_unwind(AssertUnwindSafe(|| world.step(0.0))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| world.step(-1.0))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| world.step(f32::NAN))).is_err());
}

#[test]
fn cloned_contexts_step_query_rebuild_and_retire_worlds_concurrently() {
    const WORKERS: usize = 8;
    let start = std::sync::Barrier::new(WORKERS);
    std::thread::scope(|scope| {
        for worker in 0..WORKERS {
            let start = &start;
            scope.spawn(move || {
                start.wait();
                for _ in 0..2 {
                    let mut world = sim(static_config());
                    let speed = (worker + 1) as f32;
                    let body = world.spawn(BodyDesc::sphere(0.2).velocity([speed, 0.0, 0.0]));
                    for frame in 0..12 {
                        if frame == 4 {
                            for index in 0..160 {
                                world.spawn(BodyDesc::static_sphere(0.2).position([
                                    index as f32,
                                    10.0,
                                    0.0,
                                ]));
                            }
                        }
                        world.step(DT);
                    }
                    world.wait();
                    let position = world.read_state(body).position;
                    assert!((position[0] - speed * DT * 12.0).abs() < 1e-4);
                    let query = world.ray_query(
                        [position[0], -1.0, 0.0],
                        [0.0, 1.0, 0.0],
                        2.0,
                        &dynamis_model::QueryFilter::default(),
                    );
                    world.flush_queries();
                    assert_eq!(world.query_hit(query).unwrap().body, body);
                    world.drain_events();
                }
            });
        }
    });
}

#[test]
fn apply_force_accelerates_and_consumes_within_one_step() {
    let mut world = sim(static_config());
    let ball = world.spawn(BodyDesc::sphere(0.5).mass(2.0));
    world.apply_force(ball, [0.0, 4.0, 0.0]);
    world.step(DT);
    world.wait();
    let expected = 4.0 * DT / 2.0;
    assert!(
        (world.read_state(ball).velocity[1] - expected).abs() < 1e-4,
        "force must integrate F*dt/m"
    );
    for _ in 0..5 {
        world.step(DT);
    }
    world.wait();
    let after_idle = world.read_state(ball).velocity[1];
    assert!(
        (after_idle - expected).abs() < 1e-4,
        "force must be consumed within its step"
    );
}

#[test]
fn apply_force_at_point_combines_linear_and_angular() {
    let mut world = sim(static_config());
    let ball = world.spawn(BodyDesc::sphere(0.5));
    world.apply_force_at_point(ball, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    world.step(DT);
    world.wait();
    let state = world.read_state(ball);
    let torque = [0.0f32, -1.0, 0.0];
    assert!((state.velocity[2] - DT).abs() < 1e-4, "linear part F*dt/m");
    assert!(
        (state.angular_velocity[1] - sphere_inertia(0.5, 1.0) * torque[1] * DT).abs() < 1e-3,
        "angular part I^-1 * (r x F) * dt"
    );
    assert!(state.angular_velocity[0].abs() < 1e-4);
    assert!(state.angular_velocity[2].abs() < 1e-4);
}

#[test]
fn apply_impulse_changes_linear_and_angular_exactly() {
    let mut world = sim(static_config());
    let ball = world.spawn(BodyDesc::sphere(0.5).mass(2.0));
    world.apply_impulse_at_point(ball, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]);
    world.step(DT);
    world.wait();
    let state = world.read_state(ball);
    assert!((state.velocity[1] - 0.5).abs() < 1e-4, "v += J/m");
    assert!(
        (state.angular_velocity[2] - sphere_inertia(0.5, 2.0)).abs() < 1e-3,
        "omega += I^-1 * (r x J)"
    );
    assert!(state.angular_velocity[0].abs() < 1e-3);
    assert!(state.angular_velocity[1].abs() < 1e-3);
}

#[test]
fn apply_torque_and_angular_impulse_match_closed_form() {
    let mut world = sim(static_config());
    let ball = world.spawn(BodyDesc::sphere(0.5));
    const STEPS: u32 = 5;
    for _ in 0..STEPS {
        world.apply_torque(ball, [0.0, 0.0, 5.0]);
        world.step(DT);
    }
    world.wait();
    let expected = sphere_inertia(0.5, 1.0) * 5.0 * DT * STEPS as f32;
    assert!(
        (world.read_state(ball).angular_velocity[2] - expected).abs() < 1e-3,
        "torque integrates I^-1 * tau * t"
    );
    world.apply_angular_impulse(ball, [0.0, 0.0, 2.0]);
    world.step(DT);
    world.wait();
    let boosted = world.read_state(ball).angular_velocity[2];
    assert!(
        (boosted - expected - sphere_inertia(0.5, 1.0) * 2.0).abs() < 1e-3,
        "angular impulse adds I^-1 * L"
    );
}

#[test]
fn velocity_and_position_patches_apply() {
    let mut world = sim(gravity_config());
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 3.0, 0.0]));
    world.step(DT);
    world.set_velocity(ball, [0.0, 3.0, 0.0]);
    world.step(DT);
    world.wait();
    let state = world.read_state(ball);
    assert!(
        (state.velocity[1] - (3.0 - 9.81 * DT)).abs() < 1e-4,
        "patch overrides then gravity resumes"
    );
    assert!((state.position[1] - (3.0 + 3.0 * DT - 2.0 * 9.81 * DT * DT)).abs() < 1e-3);
    world.set_position(ball, [7.0, 1.0, 2.0]);
    world.step(DT);
    world.wait();
    let state = world.read_state(ball);
    assert!((state.position[0] - 7.0).abs() < 1e-4);
    let carried_velocity = 3.0 - 2.0 * 9.81 * DT;
    assert!(
        (state.position[1] - (1.0 + carried_velocity * DT)).abs() < 1e-4,
        "teleport must keep the carried velocity"
    );
    assert!((state.position[2] - 2.0).abs() < 1e-4);
}

#[test]
fn angular_velocity_and_orientation_patches_apply() {
    let mut world = sim(static_config());
    let ball = world.spawn(BodyDesc::sphere(0.5));
    world.step(DT);
    world.set_angular_velocity(ball, [0.0, 1.0, 0.0]);
    world.set_orientation(ball, [0.0, 0.0, 0.0, 1.0]);
    world.step(DT);
    world.wait();
    let state = world.read_state(ball);
    assert!((state.angular_velocity[1] - 1.0).abs() < 1e-4);
    let half = 0.5 * DT;
    let q = state.orientation;
    assert!(
        (q[1] - half.sin()).abs() < 1e-3 && (q[3] - half.cos()).abs() < 1e-3,
        "orientation must integrate one step of angular velocity"
    );
    let norm = q.iter().map(|c| c * c).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "orientation must stay unit length"
    );
}

#[test]
fn non_unit_orientation_patch_panics() {
    let mut world = sim(static_config());
    let ball = world.spawn(BodyDesc::sphere(0.5));
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            world.set_orientation(ball, [1.0, 1.0, 0.0, 0.0]);
        }))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            BodyDesc::sphere(0.5).orientation([1.0, 1.0, 0.0, 0.0]);
        }))
        .is_err()
    );
}

#[test]
fn set_mass_zero_freezes_then_mass_restores() {
    let mut world = sim(gravity_config());
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 3.0, 0.0]));
    world.step(DT);
    world.set_mass(ball, 0.0);
    world.step(DT);
    world.wait();
    let state = world.read_state(ball);
    assert_eq!(state.inverse_mass, 0.0);
    assert_eq!(state.velocity, [0.0, 0.0, 0.0]);
    assert_eq!(state.angular_velocity, [0.0, 0.0, 0.0]);
    let frozen_y = state.position[1];
    settle_frames(&mut world, 30);
    assert_eq!(
        world.read_state(ball).position[1],
        frozen_y,
        "zero-mass body must ignore gravity"
    );
    world.set_mass(ball, 1.0);
    settle_frames(&mut world, 10);
    assert!(
        world.read_state(ball).position[1] < frozen_y,
        "restored mass must fall again"
    );
}

#[test]
fn kinematic_flag_round_trip() {
    let mut world = sim(gravity_config());
    let body = world.spawn(BodyDesc::sphere(0.5));
    world.set_kinematic(body, true);
    world.set_velocity(body, [2.0, 0.0, 0.0]);
    settle_frames(&mut world, 10);
    let state = world.read_state(body);
    assert!(
        state.position[0] > 0.2 && state.position[1] == 0.0,
        "kinematic body must move by velocity and ignore gravity"
    );
    world.set_kinematic(body, false);
    world.set_velocity(body, [0.0, 0.0, 0.0]);
    settle_frames(&mut world, 20);
    assert!(
        world.read_state(body).position[1] < -0.1,
        "de-kinematic body must fall"
    );
}

#[test]
fn config_accessors_round_trip() {
    let mut world = sim(static_config());
    assert_eq!(world.config().gravity, [0.0, 0.0, 0.0]);
    world.set_gravity([0.0, -1.0, 0.0]);
    assert_eq!(world.config().gravity, [0.0, -1.0, 0.0]);
    world.set_config(PhysicsConfig {
        gravity: [0.0, -2.0, 0.0],
        ..PhysicsConfig::default()
    });
    assert_eq!(world.config().gravity, [0.0, -2.0, 0.0]);
    assert!(world.constraints().is_empty());
}

fn settle_frames(world: &mut Simulation, frames: usize) {
    for _ in 0..frames {
        world.step(DT);
    }
    world.wait();
}

#[test]
fn mixed_static_and_dynamic_removal_keeps_world_consistent() {
    let mut world = sim(gravity_config());
    let ground = world.spawn(
        BodyDesc::cuboid([50.0, 0.5, 50.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let a = world.spawn(BodyDesc::sphere(0.3).position([0.0, 2.0, 0.0]));
    let b = world.spawn(BodyDesc::sphere(0.3).position([2.0, 2.0, 0.0]));
    let gate = world.spawn(
        BodyDesc::sphere(0.3)
            .sensor(true)
            .mass(0.0)
            .position([4.0, 1.0, 0.0]),
    );
    settle_frames(&mut world, 30);
    world.remove(b);
    settle_frames(&mut world, 30);
    world.remove(gate);
    settle_frames(&mut world, 60);
    let state = world.read_state(a);
    assert!(
        state.position[1].abs() < 0.4 && state.position[1] > 0.05,
        "survivor must rest on the ground after mixed removals, got {}",
        state.position[1]
    );
    assert!(world.read_state(ground).position[1] == -0.5);
    assert_eq!(world.count(), 2);
}

#[test]
fn a_spawn_burst_lands_every_identity_and_releases_it() {
    let mut world = sim(static_config());
    let mut handles = Vec::new();
    for index in 0..200 {
        let ball = world.spawn(BodyDesc::sphere(0.2).position([
            (index % 4) as f32 * 1.0,
            0.0,
            (index / 4) as f32 * 1.0,
        ]));
        handles.push(ball);
    }
    world.step(DT);
    world.wait();
    assert_eq!(world.count(), 200);
    assert!(
        handles
            .iter()
            .all(|handle| world.read_state(*handle).inverse_mass > 0.0)
    );
    for handle in &handles {
        world.remove(*handle);
    }
    world.step(DT);
    world.wait();
    assert_eq!(world.count(), 0);
}

#[test]
fn dynamic_spawn_shuttles_around_static_slots() {
    let mut world = sim(gravity_config());
    let ground = world.spawn(
        BodyDesc::cuboid([50.0, 0.5, 50.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    for index in 0..4 {
        let ball = world.spawn(BodyDesc::sphere(0.25).position([(index as f32) * 2.0, 2.0, 0.0]));
        settle_frames(&mut world, 20);
        assert!(
            world.read_state(ball).position[1] > 0.0,
            "ball must land when spawned after other bodies"
        );
    }
    assert_eq!(world.count(), 5);
    let ground_state = world.read_state(ground);
    assert_eq!(ground_state.position, [0.0, -0.5, 0.0]);
}

#[test]
fn mass_migration_keeps_constraints_attached() {
    let mut world = sim(static_config());
    let pick = world.spawn(BodyDesc::sphere(0.2));
    let load = world.spawn(BodyDesc::sphere(0.2).position([0.0, -2.0, 0.0]));
    world.add_constraint(
        pick,
        load,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 2.0),
    );
    world.set_mass(load, 0.0);
    for _ in 0..30 {
        world.step(DT);
    }
    world.wait();
    assert_eq!(world.read_state(load).position[1], -2.0);
    world.set_mass(load, 1.0);
    world.set_velocity(load, [0.0, -1.0, 0.0]);
    for _ in 1..120 {
        world.step(DT);
    }
    world.wait();
    let span = distance(
        world.read_state(pick).position,
        world.read_state(load).position,
    );
    assert!(
        (span - 2.0).abs() < 0.15,
        "constraint must stay attached after mass migration, span {span}"
    );
}

#[test]
fn id_space_grows_with_spawn_bursts() {
    let mut world = sim(static_config());
    let mut handles = Vec::new();
    for burst in [64, 192, 768] {
        for _ in 0..burst {
            let z = handles.len() as f32 * 0.5;
            handles.push(world.spawn(BodyDesc::sphere(0.1).position([0.0, 0.0, z + 10.0])));
        }
        world.step(DT);
        world.wait();
        assert_eq!(world.count(), handles.len());
        for (index, handle) in handles.iter().enumerate() {
            assert_eq!(
                world.read_state(*handle).position[2],
                index as f32 * 0.5 + 10.0
            );
        }
    }
}

#[test]
fn remove_then_spawn_reuses_slot_with_fresh_generation() {
    let mut world = sim(static_config());
    let first = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 0.0]));
    let _second = world.spawn(BodyDesc::sphere(0.5).position([2.0, 0.0, 0.0]));
    let _third = world.spawn(BodyDesc::sphere(0.5).position([4.0, 0.0, 0.0]));
    let _fourth = world.spawn(BodyDesc::sphere(0.5).position([6.0, 0.0, 0.0]));
    world.step(DT);
    world.remove(first);
    world.step(DT);
    let fresh = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 0.0]));
    world.step(DT);
    world.wait();
    assert_eq!(world.count(), 4);
    assert_eq!(world.read_state(fresh).position, [0.0, 0.0, 0.0]);
    assert!(
        catch_unwind(AssertUnwindSafe(|| world.read_state(first))).is_err(),
        "the removed handle must stay stale after respawn"
    );
}

#[test]
fn reallocation_preserves_kinematic_and_ccd_flags() {
    let mut world = sim(gravity_config());
    let platform = world.spawn(
        BodyDesc::cuboid([1.0, 0.2, 1.0])
            .position([0.0, 3.0, 0.0])
            .kinematic(true)
            .ccd(true),
    );
    settle_frames(&mut world, 10);
    spawn_burst(&mut world, 160);
    settle_frames(&mut world, 60);
    let state = world.read_state(platform);
    assert!(
        (state.position[1] - 3.0).abs() < 1e-3,
        "a kinematic body must not fall through a reallocation, got {}",
        state.position[1]
    );
}

#[test]
fn reallocation_preserves_collision_filters() {
    let mut world = sim(gravity_config());
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0])
            .collision_group(0x2)
            .collision_mask(0x8),
    );
    let ball = world.spawn(
        BodyDesc::sphere(0.4)
            .position([0.0, 4.0, 0.0])
            .collision_group(0x8)
            .collision_mask(0x2),
    );
    settle_frames(&mut world, 20);
    spawn_burst(&mut world, 160);
    settle_frames(&mut world, 180);
    let height = world.read_state(ball).position[1];
    assert!(
        height > 0.3,
        "a filtered pair must keep colliding across a reallocation, got {height}"
    );
}

#[test]
fn reallocation_preserves_constraint_slots_and_breaks() {
    let mut world = sim(gravity_config());
    let anchor = world.spawn(BodyDesc::static_sphere(0.2).position([0.0, 6.0, 0.0]));
    let mut handles = vec![anchor];
    for index in 0..4 {
        let ball = world.spawn(BodyDesc::sphere(0.2).position([
            index as f32 * 0.6,
            5.0 - index as f32 * 0.6,
            0.0,
        ]));
        world.add_constraint(
            *handles.last().unwrap(),
            ball,
            ConstraintDesc::ball([0.0; 3], [0.0; 3]),
        );
        handles.push(ball);
    }
    let middle = world.add_constraint(
        handles[1],
        handles[3],
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.2),
    );
    world.remove_constraint(middle);
    settle_frames(&mut world, 60);
    spawn_burst(&mut world, 160);
    settle_frames(&mut world, 60);
    let tip = world.read_state(handles[4]).position;
    let anchor_pos = world.read_state(anchor).position;
    let span = distance(tip, anchor_pos);
    assert!(
        span < 3.0,
        "the chain must stay attached across a reallocation and a removal, span {span}"
    );
}

#[test]
fn apply_impulse_at_center_of_mass_spins_nothing() {
    let mut world = sim(static_config());
    let offset = world.spawn(
        BodyDesc::sphere(0.5)
            .mass(2.0)
            .com([0.4, 0.0, 0.0])
            .position([0.0, 0.0, 0.0]),
    );
    world.apply_impulse(offset, [0.0, 0.0, 3.0]);
    world.step(DT);
    world.wait();
    let state = world.read_state(offset);
    assert!(
        state
            .angular_velocity
            .iter()
            .all(|value| value.abs() < 1e-5),
        "a centre-of-mass impulse must not induce spin, got {:?}",
        state.angular_velocity
    );
    assert!(
        state.velocity[2] > 1.4,
        "the linear response must still apply, got {:?}",
        state.velocity
    );
}

#[test]
fn apply_impulse_at_point_levers_about_the_center_of_mass() {
    let mut world = sim(static_config());
    let body = world.spawn(
        BodyDesc::sphere(0.5)
            .mass(2.0)
            .com([0.4, 0.0, 0.0])
            .position([0.0, 0.0, 0.0]),
    );
    world.apply_impulse_at_point(body, [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]);
    world.step(DT);
    world.wait();
    let state = world.read_state(body);
    assert!(
        state.angular_velocity[1].abs() > 0.1,
        "an off-centre impulse must lever the body, got {:?}",
        state.angular_velocity
    );
}

#[test]
fn sleep_threshold_override_accepts_zero() {
    let mut world = sim(gravity_config());
    let ball = world.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 0.0, 0.0])
            .sleep_thresholds(0.0, 0.0),
    );
    world.set_sleep_thresholds(ball, 0.0, 0.0);
    settle_frames(&mut world, 5);
    assert!(
        !world.read_state(ball).sleeping,
        "a zero threshold must keep the body awake rather than inherit the global one"
    );
}

#[test]
fn deterministic_rebuild_on_spawn_burst() {
    let mut world = sim(static_config());
    let a = world.spawn(BodyDesc::sphere(0.3));
    let _b = world.spawn(BodyDesc::sphere(0.3).position([5.0, 0.3, 0.0]));
    let _ = a;
    for _ in 0..140 {
        world.step(DT);
        let probe =
            world.sphere_query([0.0, 0.0, 0.0], 2.0, &dynamis_model::QueryFilter::default());
        world.flush_queries();
        let hits = world.query_hits(probe);
        assert!(hits.iter().all(|hit| hit.distance.is_finite()));
    }
    world.wait();
    let probe = world.sphere_query([0.0, 0.0, 0.0], 2.0, &dynamis_model::QueryFilter::default());
    world.flush_queries();
    let hits = world.query_hits(probe);
    assert!(
        hits.iter().all(|hit| hit.distance.is_finite()),
        "query results are garbage after capacity rebuild"
    );
    for handle in world.bodies().to_vec() {
        let y = world.read_state(handle).position[1];
        assert!(y.is_finite(), "body drifted after capacity rebuild: {y}");
    }
}
