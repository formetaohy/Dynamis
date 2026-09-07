use dynamis::{BodyDesc, BodyHandle, ContactEventKind, Shape, Simulation};
use dynamis_example_common as common;
use dynamis_example_render::{App, AppContext, Color, Material, MeshId, Transform, Vec3};

const BALL_COUNT: usize = 40;
const GATE_Y: [f32; 3] = [3.0, 6.0, 9.0];

struct GateState {
    body: BodyHandle,
    mesh: MeshId,
    entered: usize,
    flash: f32,
}

struct Example {
    dynamics: common::physics::Dynamics,
    bodies: Vec<(BodyHandle, MeshId)>,
    gates: Vec<GateState>,
    orbit: common::camera::Orbit,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    App::new(
        "dynamis sensors",
        Example {
            dynamics: common::physics::Dynamics::with_capacity(80),
            bodies: Vec::new(),
            gates: Vec::new(),
            orbit: common::camera::Orbit::new(0.7, 0.42, 42.0, Vec3::new(0.0, 3.0, 0.0)),
        },
    )
    .on_startup(setup)
    .on_update(update)
    .run()
}

fn setup(ctx: &mut AppContext, example: &mut Example) {
    common::scene::setup_scene(ctx, &mut example.dynamics.simulation);
    spawn_gates(ctx, &mut example.dynamics.simulation, &mut example.gates);
    spawn_balls(ctx, &mut example.dynamics.simulation, &mut example.bodies);
}

fn spawn_gates(ctx: &mut AppContext, simulation: &mut Simulation, gates: &mut Vec<GateState>) {
    for y in GATE_Y {
        let shape = Shape::cuboid([5.0, 0.25, 5.0]);
        let body = simulation.spawn(
            BodyDesc::cuboid([5.0, 0.25, 5.0])
                .sensor(true)
                .mass(0.0)
                .position([0.0, y, 0.0]),
        );
        let mesh = ctx.spawn(
            &common::scene::geometry_from_shape(&shape),
            Material {
                base_color: Color::srgba(0.2, 0.8, 0.95, 0.25),
                roughness: 1.0,
                unlit: true,
                ..Default::default()
            },
            Transform::from_xyz(0.0, y, 0.0),
        );
        gates.push(GateState {
            body,
            mesh,
            entered: 0,
            flash: 0.0,
        });
    }
}

fn spawn_balls(
    ctx: &mut AppContext,
    simulation: &mut Simulation,
    bodies: &mut Vec<(BodyHandle, MeshId)>,
) {
    for index in 0..BALL_COUNT {
        let radius = 0.3 + (index % 4) as f32 * 0.06;
        let x = (index % 10) as f32 * 1.2 - 5.4;
        let z = (index / 10) as f32 * 1.2 - 1.8;
        let y = 14.0 + (index % 5) as f32 * 0.9;
        let handle = simulation.spawn(
            BodyDesc::sphere(radius)
                .position([x, y, z])
                .restitution(0.25)
                .friction(0.8),
        );
        let mesh = ctx.spawn(
            &common::scene::geometry_from_shape(&Shape::sphere(radius)),
            common::scene::standard_material(common::scene::indexed_color(index)),
            Transform::from_xyz(x, y, z),
        );
        bodies.push((handle, mesh));
    }
}

fn update(ctx: &mut AppContext, example: &mut Example) {
    common::physics::advance_physics(ctx, &mut example.dynamics);
    collect_events(&mut example.dynamics, &mut example.gates);
    update_gates(ctx, &mut example.gates);
    common::physics::sync_visuals(ctx, &example.dynamics, &example.bodies);
    common::camera::orbit_camera(ctx, &mut example.orbit);
    let counts = example
        .gates
        .iter()
        .map(|gate| gate.entered.to_string())
        .collect::<Vec<_>>()
        .join(" / ");
    ctx.hud = format!(
        "sensor gates entered: {counts}   fps: {:.0}\nright-drag: orbit   wheel: zoom",
        1.0 / ctx.time.delta_secs().max(1e-6),
    );
}

fn collect_events(dynamics: &mut common::physics::Dynamics, gates: &mut [GateState]) {
    for event in dynamics.simulation.drain_events() {
        if !event.sensor {
            continue;
        }
        for gate in &mut *gates {
            if (event.first == gate.body || event.second == gate.body)
                && event.kind == ContactEventKind::Begin
            {
                gate.entered += 1;
                gate.flash = 1.0;
            }
        }
    }
}

fn update_gates(ctx: &mut AppContext, gates: &mut [GateState]) {
    for gate in gates {
        gate.flash = (gate.flash - ctx.time.delta_secs() * 1.5).max(0.0);
        let light = 0.25 + gate.flash * 0.75;
        ctx.mesh_material(gate.mesh).base_color =
            Color::srgba(0.2 * light, 0.8 * light, 0.95 * light, 0.3);
    }
}
