use dynamis::{BodyDesc, BodyHandle, Shape, Simulation};
use dynamis_example_common as common;
use dynamis_example_render::{App, AppContext, MeshId, Transform, Vec3};

const BODY_COUNT: usize = 128;

struct Example {
    dynamics: common::physics::Dynamics,
    bodies: Vec<(BodyHandle, MeshId)>,
    orbit: common::camera::Orbit,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    App::new(
        "dynamis falling shapes",
        Example {
            dynamics: common::physics::Dynamics::with_capacity(BODY_COUNT + 4),
            bodies: Vec::new(),
            orbit: common::camera::Orbit::new(0.7, 0.42, 42.0, Vec3::new(0.0, 3.0, 0.0)),
        },
    )
    .on_startup(setup)
    .on_update(update)
    .run()
}

fn setup(ctx: &mut AppContext, example: &mut Example) {
    common::scene::setup_scene(ctx, &mut example.dynamics.simulation);
    spawn_collection(ctx, &mut example.dynamics.simulation, &mut example.bodies);
}

fn spawn_collection(
    ctx: &mut AppContext,
    simulation: &mut Simulation,
    bodies: &mut Vec<(BodyHandle, MeshId)>,
) {
    for index in 0..BODY_COUNT {
        let radius = 0.35 + (index % 5) as f32 * 0.07;
        let (desc, shape, color) = match index % 4 {
            0 => (
                BodyDesc::sphere(radius),
                Shape::sphere(radius),
                common::scene::indexed_color(index),
            ),
            1 => {
                let half = [radius, radius * 0.9, radius * 1.1];
                (
                    BodyDesc::cuboid(half),
                    Shape::cuboid(half),
                    common::scene::indexed_color(index),
                )
            }
            2 => {
                let narrow = radius * 0.8;
                (
                    BodyDesc::capsule(narrow, radius),
                    Shape::capsule(narrow, radius),
                    common::scene::indexed_color(index + 2),
                )
            }
            _ => {
                let narrow = radius * 0.8;
                (
                    BodyDesc::cylinder(narrow, radius),
                    Shape::cylinder(narrow, radius),
                    common::scene::indexed_color(index + 4),
                )
            }
        };
        let x = (index % 16) as f32 * 1.7 - 12.75;
        let z = (index / 16) as f32 * 1.7 - 6.0;
        let y = 5.0 + (index % 13) as f32 * 1.5;
        let spin = 0.4 + (index % 7) as f32 * 0.25;
        let handle = simulation.spawn(
            desc.position([x, y, z])
                .angular_velocity([spin, spin * 0.7, spin * 0.5])
                .restitution(0.2 + (index % 4) as f32 * 0.08)
                .friction(0.7),
        );
        let mesh = ctx.spawn(
            &common::scene::geometry_from_shape(&shape),
            common::scene::standard_material(color),
            Transform::from_xyz(x, y, z),
        );
        bodies.push((handle, mesh));
    }
}

fn update(ctx: &mut AppContext, example: &mut Example) {
    common::physics::advance_physics(ctx, &mut example.dynamics);
    common::physics::sync_visuals(ctx, &example.dynamics, &example.bodies);
    common::camera::orbit_camera(ctx, &mut example.orbit);
    let sleeping = example
        .dynamics
        .simulation
        .bodies()
        .iter()
        .filter(|handle| example.dynamics.simulation.read_state(**handle).sleeping)
        .count();
    ctx.hud = format!(
        "bodies: {}   sleeping: {}   fps: {:.0}\nright-drag: orbit   wheel: zoom",
        example.dynamics.simulation.count(),
        sleeping,
        1.0 / ctx.time.delta_secs().max(1e-6),
    );
}
