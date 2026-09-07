use dynamis::{BodyDesc, BodyHandle, ConstraintDesc, Shape, Simulation};
use dynamis_example_common as common;
use dynamis_example_render::{App, AppContext, Color, MeshId, Transform, Vec3};

struct Example {
    simulator: common::physics::Simulator,
    bodies: Vec<(BodyHandle, MeshId)>,
    orbit: common::camera::Orbit,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    App::new(
        "dynamis joints",
        Example {
            simulator: common::physics::Simulator::with_capacity(64),
            bodies: Vec::new(),
            orbit: common::camera::Orbit::new(0.7, 0.42, 42.0, Vec3::new(0.0, 3.0, 0.0)),
        },
    )
    .on_startup(setup)
    .on_update(update)
    .run()
}

fn setup(ctx: &mut AppContext, example: &mut Example) {
    common::scene::setup_scene(ctx, &mut example.simulator.simulation);
    spawn_chain(ctx, &mut example.simulator.simulation, &mut example.bodies);
    spawn_pendulum(ctx, &mut example.simulator.simulation, &mut example.bodies);
    spawn_motor(ctx, &mut example.simulator.simulation, &mut example.bodies);
}

fn spawn_visual(
    ctx: &mut AppContext,
    simulation: &mut Simulation,
    bodies: &mut Vec<(BodyHandle, MeshId)>,
    desc: BodyDesc,
    shape: Shape,
    color: Color,
) -> BodyHandle {
    let position = desc.position;
    let handle = simulation.spawn(desc);
    let mesh = ctx.spawn(
        &common::scene::geometry_from_shape(&shape),
        common::scene::standard_material(color),
        Transform::from_xyz(position[0], position[1], position[2]),
    );
    bodies.push((handle, mesh));
    handle
}

fn spawn_chain(
    ctx: &mut AppContext,
    simulation: &mut Simulation,
    bodies: &mut Vec<(BodyHandle, MeshId)>,
) {
    let radius = 0.34;
    let spacing = 1.1;
    let anchor = spawn_visual(
        ctx,
        simulation,
        bodies,
        BodyDesc::static_sphere(0.42).position([0.0, 13.0, 0.0]),
        Shape::sphere(0.42),
        Color::srgb(0.72, 0.74, 0.78),
    );
    let mut previous = anchor;
    for index in 0..9 {
        let x = 0.4 + index as f32 * 0.3;
        let y = 13.0 - (index + 1) as f32 * spacing;
        let handle = spawn_visual(
            ctx,
            simulation,
            bodies,
            BodyDesc::sphere(radius).position([x, y, 0.0]).friction(0.4),
            Shape::sphere(radius),
            common::scene::indexed_color(index),
        );
        simulation.add_constraint(
            previous,
            handle,
            ConstraintDesc::ball([0.0, -spacing / 2.0, 0.0], [0.0, spacing / 2.0, 0.0]),
        );
        previous = handle;
    }
}

fn spawn_pendulum(
    ctx: &mut AppContext,
    simulation: &mut Simulation,
    bodies: &mut Vec<(BodyHandle, MeshId)>,
) {
    let pivot = spawn_visual(
        ctx,
        simulation,
        bodies,
        BodyDesc::static_sphere(0.35).position([-8.5, 10.0, 0.0]),
        Shape::sphere(0.35),
        Color::srgb(0.72, 0.74, 0.78),
    );
    let bob = spawn_visual(
        ctx,
        simulation,
        bodies,
        BodyDesc::sphere(0.55)
            .position([-8.5, 7.0, 0.0])
            .velocity([2.0, 0.0, 0.0])
            .friction(0.5),
        Shape::sphere(0.55),
        Color::srgb(0.25, 0.85, 0.6),
    );
    simulation.add_constraint(
        pivot,
        bob,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 3.0),
    );
}

fn spawn_motor(
    ctx: &mut AppContext,
    simulation: &mut Simulation,
    bodies: &mut Vec<(BodyHandle, MeshId)>,
) {
    let axle = spawn_visual(
        ctx,
        simulation,
        bodies,
        BodyDesc::static_sphere(0.35).position([8.5, 10.5, 0.0]),
        Shape::sphere(0.35),
        Color::srgb(0.72, 0.74, 0.78),
    );
    let blade = spawn_visual(
        ctx,
        simulation,
        bodies,
        BodyDesc::cuboid([1.3, 0.14, 0.14])
            .position([8.5, 10.5, 0.0])
            .friction(0.5),
        Shape::cuboid([1.3, 0.14, 0.14]),
        Color::srgb(0.9, 0.3, 0.3),
    );
    simulation.add_constraint(
        axle,
        blade,
        ConstraintDesc::revolute([0.0; 3], [-1.3, 0.0, 0.0], [0.0, 0.0, 1.0]).motor(2.5),
    );
}

fn update(ctx: &mut AppContext, example: &mut Example) {
    common::physics::advance_physics(ctx, &mut example.simulator);
    common::physics::sync_visuals(ctx, &example.simulator, &example.bodies);
    common::camera::orbit_camera(ctx, &mut example.orbit);
    ctx.hud = format!(
        "bodies: {}   constraints: {}   fps: {:.0}\nball chain   distance pendulum   revolute motor\nright-drag: orbit   wheel: zoom",
        example.simulator.simulation.count(),
        example.simulator.simulation.constraints().len(),
        1.0 / ctx.time.delta_secs().max(1e-6),
    );
}
