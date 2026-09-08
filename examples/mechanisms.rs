use dynamis::{BodyDesc, BodyHandle, ConstraintDesc, ConstraintMotor, DofDesc, Simulation};
use dynamis_example_common as common;
use dynamis_example_render::{App, AppContext, Color, Geometry, MeshId, Transform, Vec3};

const ARM_REACH: f32 = 2.0;

struct Example {
    simulator: common::physics::Simulator,
    bodies: Vec<(BodyHandle, MeshId)>,
    orbit: common::camera::Orbit,
    spawned: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    App::new(
        "dynamis mechanisms",
        Example {
            simulator: common::physics::Simulator::with_capacity(32),
            bodies: Vec::new(),
            orbit: common::camera::Orbit::new(0.7, 0.42, 26.0, Vec3::new(0.0, 4.0, 0.0)),
            spawned: 0,
        },
    )
    .on_startup(setup)
    .on_update(update)
    .run()
}

fn setup(ctx: &mut AppContext, example: &mut Example) {
    common::scene::setup_scene(ctx, &mut example.simulator.simulation);
    spawn_servo_arm(ctx, &mut example.simulator.simulation, &mut example.bodies);
    spawn_cone_pendulum(ctx, &mut example.simulator.simulation, &mut example.bodies);
    spawn_slider(ctx, &mut example.simulator.simulation, &mut example.bodies);
}

fn spawn_visual(
    ctx: &mut AppContext,
    simulation: &mut Simulation,
    bodies: &mut Vec<(BodyHandle, MeshId)>,
    desc: BodyDesc,
    shape: HandleKind,
    color: Color,
) -> BodyHandle {
    let (geometry, position) = match shape {
        HandleKind::Sphere(radius) => (
            Geometry::Sphere { radius },
            Transform::from_xyz(desc.position[0], desc.position[1], desc.position[2]),
        ),
        HandleKind::Cuboid(half_extents) => (
            Geometry::Cuboid { half_extents },
            Transform::from_xyz(desc.position[0], desc.position[1], desc.position[2]),
        ),
    };
    let handle = simulation.spawn(desc);
    let mesh = ctx.spawn(&geometry, common::scene::standard_material(color), position);
    bodies.push((handle, mesh));
    handle
}

enum HandleKind {
    Sphere(f32),
    Cuboid([f32; 3]),
}

fn spawn_servo_arm(
    ctx: &mut AppContext,
    simulation: &mut Simulation,
    bodies: &mut Vec<(BodyHandle, MeshId)>,
) {
    let base = spawn_visual(
        ctx,
        simulation,
        bodies,
        BodyDesc::static_sphere(0.3).position([2.5, 3.0, 0.0]),
        HandleKind::Sphere(0.3),
        Color::srgb(0.72, 0.74, 0.78),
    );
    let arm = spawn_visual(
        ctx,
        simulation,
        bodies,
        BodyDesc::cuboid([ARM_REACH, 0.06, 0.06]).position([2.5 + ARM_REACH, 3.0, 0.0]),
        HandleKind::Cuboid([ARM_REACH, 0.06, 0.06]),
        Color::srgb(0.97, 0.5, 0.3),
    );
    let dofs = [
        DofDesc::locked(),
        DofDesc::locked(),
        DofDesc::locked(),
        DofDesc::free(),
        DofDesc::free(),
        DofDesc::driven(ConstraintMotor {
            target_velocity: 0.0,
            max_force: 4.0,
            target_position: Some(1.9),
            stiffness: 0.12,
            damping: 0.35,
        }),
    ];
    let desc = ConstraintDesc::six_dof(
        [0.0; 3],
        [ARM_REACH, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    )
    .dofs(dofs);
    simulation.add_constraint(base, arm, desc);
}

fn spawn_cone_pendulum(
    ctx: &mut AppContext,
    simulation: &mut Simulation,
    bodies: &mut Vec<(BodyHandle, MeshId)>,
) {
    let pivot = spawn_visual(
        ctx,
        simulation,
        bodies,
        BodyDesc::static_sphere(0.32).position([-3.0, 11.0, 0.0]),
        HandleKind::Sphere(0.32),
        Color::srgb(0.72, 0.74, 0.78),
    );
    let bob = spawn_visual(
        ctx,
        simulation,
        bodies,
        BodyDesc::sphere(0.45)
            .position([-3.0, 11.0 - 2.0, 0.0])
            .velocity([1.2, 0.0, 0.0])
            .friction(0.4),
        HandleKind::Sphere(0.45),
        Color::srgb(0.35, 0.8, 0.95),
    );
    simulation.add_constraint(
        pivot,
        bob,
        ConstraintDesc::cone([0.0; 3], [0.0, -2.0, 0.0], [0.0, -1.0, 0.0], 0.35),
    );
}

fn spawn_slider(
    ctx: &mut AppContext,
    simulation: &mut Simulation,
    bodies: &mut Vec<(BodyHandle, MeshId)>,
) {
    let anchor = spawn_visual(
        ctx,
        simulation,
        bodies,
        BodyDesc::static_sphere(0.28).position([-7.0, 4.5, 0.0]),
        HandleKind::Sphere(0.28),
        Color::srgb(0.72, 0.74, 0.78),
    );
    let slider = spawn_visual(
        ctx,
        simulation,
        bodies,
        BodyDesc::sphere(0.4).position([-7.0, 4.5, 0.0]),
        HandleKind::Sphere(0.4),
        Color::srgb(0.6, 0.8, 0.35),
    );
    simulation.add_constraint(
        anchor,
        slider,
        ConstraintDesc::prismatic([0.0; 3], [0.0; 3], [1.0, 0.0, 0.0])
            .servo(1.6, 0.2, 0.4)
            .motor_force(6.0),
    );
}

fn update(ctx: &mut AppContext, example: &mut Example) {
    if ctx.input.left_pressed {
        let index = example.spawned;
        let ball = example
            .simulator
            .simulation
            .spawn(BodyDesc::sphere(0.3).position([index as f32 * 1.4, 3.0, 2.5]));
        let mesh = ctx.spawn(
            &Geometry::Sphere { radius: 0.3 },
            common::scene::standard_material(common::scene::indexed_color(index)),
            Transform::from_xyz(index as f32 * 1.4, 3.0, 2.5),
        );
        example.bodies.push((ball, mesh));
        example.spawned += 1;
    }
    if ctx.input.right_pressed {
        example.simulator.simulation.grow(example.spawned + 24);
    }

    common::physics::advance_physics(ctx, &mut example.simulator);
    common::physics::sync_visuals(ctx, &example.simulator, &example.bodies);
    common::camera::orbit_camera(ctx, &mut example.orbit);
}
