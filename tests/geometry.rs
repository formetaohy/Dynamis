use dynamis::{
    BodyDesc, ColliderDesc, GpuContext, PhysicsConfig, QueryFilter, Shape, Simulation,
};
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

fn flat_floor(sim: &mut Simulation) {
    let vertices = vec![[-30.0f32, 0.0, -30.0], [30.0, 0.0, -30.0], [30.0, 0.0, 30.0], [-30.0, 0.0, 30.0]];
    let triangles = vec![[0u32, 2, 1], [0, 3, 2]];
    let floor = sim.add_mesh(&vertices, &triangles);
    sim.spawn(BodyDesc::new(ColliderDesc::new(Shape::mesh(floor))).mass(0.0));
}

fn settle(sim: &mut Simulation, frames: usize) {
    for _ in 0..frames {
        sim.step(DT);
    }
    sim.wait();
}

#[test]
fn cylinder_rests_on_ground() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    flat_floor(&mut sim);
    let cylinder = sim.spawn(BodyDesc::cylinder(0.5, 1.0).position([0.0, 10.0, 0.0]));
    settle(&mut sim, 120);
    let y = sim.read_state(cylinder).position[1];
    assert!(
        (y - 1.0).abs() < 0.05,
        "cylinder must rest at y=1.0, got {y}"
    );
}

#[test]
fn hull_falls_and_rests() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    let vertices = [
        [-1.0f32, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let triangles = [
        [0u32, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [1, 5, 4],
        [1, 4, 0],
        [2, 6, 5],
        [2, 5, 1],
        [3, 7, 6],
        [3, 6, 2],
        [0, 4, 7],
        [0, 7, 3],
    ];
    let source = sim.add_hull(&vertices, &triangles);
    flat_floor(&mut sim);
    let hull = sim.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::hull(source))).position([0.0, 10.0, 0.0]),
    );
    settle(&mut sim, 200);
    let y = sim.read_state(hull).position[1];
    assert!(
        (y - 1.0).abs() < 0.05,
        "hull cube must rest at y=1.0, got {y}"
    );
}

#[test]
fn mesh_collider_blocks_falling_ball() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    let vertices = vec![
        [-5.0f32, 0.0, -5.0],
        [5.0, 0.0, -5.0],
        [5.0, 0.0, 5.0],
        [-5.0, 0.0, 5.0],
    ];
    let triangles = vec![[0u32, 2, 1], [0, 3, 2]];
    let floor = sim.add_mesh(&vertices, &triangles);
    sim.spawn(BodyDesc::new(ColliderDesc::new(Shape::mesh(floor))).mass(0.0));
    let ball = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 5.0, 0.0]));
    settle(&mut sim, 120);
    let y = sim.read_state(ball).position[1];
    assert!(
        (y - 0.5).abs() < 0.05,
        "ball must rest on the mesh floor at y=0.5, got {y}"
    );
}

#[test]
fn height_field_collider_blocks_falling_ball() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    let heights = vec![0.0f32; 9];
    let source = sim.add_height_field(3, 3, &heights, [2.0, 2.0]);
    sim.spawn(BodyDesc::new(ColliderDesc::new(Shape::height_field(source))).mass(0.0));
    let ball = sim.spawn(BodyDesc::sphere(0.5).position([1.0, 5.0, 1.0]));
    settle(&mut sim, 120);
    let y = sim.read_state(ball).position[1];
    assert!(
        (y - 0.5).abs() < 0.05,
        "ball must rest on the height field at y=0.5, got {y}"
    );
}

#[test]
fn compound_collider_rests_stably() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    flat_floor(&mut sim);
    let compound = sim.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(0.5)))
            .collider(ColliderDesc::new(Shape::sphere(0.5)).offset([0.0, 0.8, 0.0]))
            .position([0.0, 10.0, 0.0]),
    );
    settle(&mut sim, 120);
    let y = sim.read_state(compound).position[1];
    assert!(
        y > 0.1 && y < 3.0,
        "compound must be caught by the floor without sinking, got {y}"
    );
}

#[test]
fn set_collider_switches_compound_slot() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let body = sim.spawn(BodyDesc::sphere(0.3));
    sim.set_collider(body, 0, ColliderDesc::new(Shape::sphere(0.9)));
    sim.step(DT);
    sim.wait();
    let query = sim.raycast([0.0, 2.0, 0.0], [0.0, -1.0, 0.0], 10.0, &QueryFilter::default());
    sim.step(DT);
    sim.wait();
    let hit = sim.query_hit(query).expect("enlarged collider must be hit");
    assert_eq!(hit.body, body);
    assert!(
        (hit.distance - 1.1).abs() < 1e-3,
        "enlarged collider must be reachable at 1.1, got {}",
        hit.distance
    );
}

#[test]
fn box_query_detects_overlap() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    sim.spawn(BodyDesc::static_sphere(1.0).position([0.0, 0.0, 0.0]));
    let query = sim.box_query(
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        &QueryFilter::default(),
    );
    sim.step(DT);
    sim.wait();
    assert!(
        sim.query_hit(query).is_some(),
        "box query must detect the overlapping sphere"
    );
}

#[test]
fn sweep_query_detects_blocking_wall() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let wall = sim.spawn(BodyDesc::static_sphere(2.0).position([0.0, 0.0, 10.0]));
    let query = sim.sweep_query(
        &Shape::sphere(0.3),
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        30.0,
        &QueryFilter::default(),
    );
    sim.step(DT);
    sim.wait();
    let hit = sim.query_hit(query).expect("sweep must hit the wall");
    assert_eq!(hit.body, wall);
    assert!(
        (hit.distance - 7.7).abs() < 0.05,
        "sweep must stop at the wall surface at 7.7, got {}",
        hit.distance
    );
}

#[test]
fn deep_penetrating_hull_escapes_box() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    let ground = sim.spawn(BodyDesc::cuboid([20.0, 1.0, 20.0]).mass(0.0).position([0.0, -1.0, 0.0]));
    let _ = ground;
    let vertices = vec![
        [-0.5f32, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5],
        [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ];
    let triangles = vec![
        [0u32, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ];
    let source = sim.add_hull(&vertices, &triangles);
    let hull = sim.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::hull(source)))
            .position([0.0, 0.3, 0.0])
            .restitution(0.0),
    );
    settle(&mut sim, 90);
    let y = sim.read_state(hull).position[1];
    assert!(
        y > 0.3 && y < 2.0,
        "embedded hull must be expelled to the box top by the penetration solver, got {y}"
    );
}

#[test]
fn mesh_plane_blocks_body_from_below() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    flat_floor(&mut sim);
    let ball = sim.spawn(
        BodyDesc::sphere(0.5).position([0.0, -3.0, 0.0]).velocity([0.0, 10.0, 0.0]).restitution(0.0),
    );
    let mut peak = f32::MIN;
    for _ in 0..40 {
        sim.step(DT);
        sim.wait();
        peak = peak.max(sim.read_state(ball).position[1]);
    }
    assert!(
        peak <= -0.45,
        "mesh plane must stop the body on its lower side, peak reached {peak}"
    );
}

#[test]
fn height_field_blocks_body_from_below() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    let heights = vec![0.0f32; 9];
    let source = sim.add_height_field(3, 3, &heights, [2.0, 2.0]);
    sim.spawn(BodyDesc::new(ColliderDesc::new(Shape::height_field(source))).mass(0.0));
    let ball = sim.spawn(
        BodyDesc::sphere(0.5).position([0.0, -3.0, 0.0]).velocity([0.0, 10.0, 0.0]).restitution(0.0),
    );
    let mut peak = f32::MIN;
    for _ in 0..40 {
        sim.step(DT);
        sim.wait();
        peak = peak.max(sim.read_state(ball).position[1]);
    }
    assert!(
        peak <= -0.45,
        "height field must stop the body on its lower side, peak reached {peak}"
    );
}
