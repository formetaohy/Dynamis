use dynamis::{BodyDesc, BodyHandle, GpuContext, PhysicsConfig, Shape, Simulation, WarmupBudget};
use dynamis_example_render::{
    App, AppContext, Color, EulerRot, Geometry, Material, MeshId, Quat, Transform, Vec3,
};
use std::thread::JoinHandle;

const PHYSICS_STEP: f32 = 1.0 / 60.0;
const MAX_SUBSTEPS: u32 = 8;
const BODY_COUNT: usize = 128;
const GROUND_HALF: f32 = 30.0;

struct Example {
    simulation: Simulation,
    bodies: Vec<(BodyHandle, MeshId)>,
    compiler: Option<JoinHandle<()>>,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let gpu = pollster::block_on(GpuContext::new());
    let simulation = Simulation::new(gpu, BODY_COUNT + 4, PhysicsConfig::default());
    let compiler = {
        let gpu = simulation.gpu().clone();
        std::thread::spawn(move || {
            gpu.warmup(WarmupBudget::All);
        })
    };
    App::new(
        "dynamis falling",
        Example {
            simulation,
            bodies: Vec::new(),
            compiler: Some(compiler),
        },
    )
    .on_startup(setup)
    .on_update(update)
    .run()
}

fn setup(ctx: &mut AppContext, example: &mut Example) {
    setup_scene(ctx, &mut example.simulation);
    ctx.camera.eye = Vec3::new(29.33, 20.13, 13.1);
    ctx.camera.target = Vec3::new(0.0, 3.0, 0.0);
    spawn_collection(ctx, &mut example.simulation, &mut example.bodies);
}

fn update(ctx: &mut AppContext, example: &mut Example) {
    if !example.simulation.is_warm() {
        ctx.hud = "compiling shaders".to_owned();
        return;
    }
    if let Some(compiler) = example.compiler.take() {
        compiler.join().expect("shader compilation thread panicked");
    }
    example
        .simulation
        .update(ctx.time.delta_secs(), PHYSICS_STEP, MAX_SUBSTEPS);
    example.simulation.synchronize_states();
    for (body, mesh) in &example.bodies {
        let state = example.simulation.read_state(*body);
        let transform = ctx.mesh_transform(*mesh);
        transform.translation = Vec3::from(state.position);
        transform.rotation = Quat::from_xyzw(
            state.orientation[0],
            state.orientation[1],
            state.orientation[2],
            state.orientation[3],
        );
        transform.scale = Vec3::ONE;
    }
    let sleeping = example
        .bodies
        .iter()
        .filter(|(body, _)| example.simulation.read_state(*body).sleeping)
        .count();
    ctx.hud = format!(
        "bodies: {}   sleeping: {}   fps: {:.0}",
        example.simulation.count(),
        sleeping,
        1.0 / ctx.time.delta_secs().max(1e-6),
    );
}

fn setup_scene(ctx: &mut AppContext, simulation: &mut Simulation) {
    simulation.spawn(
        BodyDesc::cuboid([GROUND_HALF, 0.5, GROUND_HALF])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    ctx.spawn(
        &Geometry::Cuboid {
            half_extents: [GROUND_HALF, 0.5, GROUND_HALF],
        },
        Material {
            base_color: Color::srgb(0.2, 0.23, 0.26),
            roughness: 0.95,
            ..Default::default()
        },
        Transform::from_xyz(0.0, -0.5, 0.0),
    );
    ctx.lights.direction = Quat::from_euler(EulerRot::XYZ, -0.9, 0.7, 0.0) * Vec3::NEG_Z;
    ctx.lights.color = Color::srgb(1.0, 0.94, 0.84);
    ctx.lights.intensity = 1.6;
    ctx.lights.ambient = Color::srgb(0.72, 0.74, 0.78);
    ctx.lights.ambient_intensity = 0.35;
}

fn spawn_collection(
    ctx: &mut AppContext,
    simulation: &mut Simulation,
    bodies: &mut Vec<(BodyHandle, MeshId)>,
) {
    for index in 0..BODY_COUNT {
        let radius = 0.35 + (index % 5) as f32 * 0.07;
        let (desc, shape) = match index % 4 {
            0 => (BodyDesc::sphere(radius), Shape::sphere(radius)),
            1 => {
                let half = [radius, radius * 0.9, radius * 1.1];
                (BodyDesc::cuboid(half), Shape::cuboid(half))
            }
            2 => {
                let narrow = radius * 0.8;
                (
                    BodyDesc::capsule(narrow, radius),
                    Shape::capsule(narrow, radius),
                )
            }
            _ => {
                let narrow = radius * 0.8;
                (
                    BodyDesc::cylinder(narrow, radius),
                    Shape::cylinder(narrow, radius),
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
            &geometry_of(&shape),
            Material {
                base_color: indexed_color(index),
                roughness: 0.55,
                ..Default::default()
            },
            Transform::from_xyz(x, y, z),
        );
        bodies.push((handle, mesh));
    }
}

fn geometry_of(shape: &Shape) -> Geometry {
    match *shape {
        Shape::Sphere { radius } => Geometry::Sphere { radius },
        Shape::Cuboid { half_extents } => Geometry::Cuboid { half_extents },
        Shape::Capsule {
            radius,
            half_height,
        } => Geometry::Capsule {
            radius,
            half_height,
        },
        Shape::Cylinder {
            radius,
            half_height,
        } => Geometry::Cylinder {
            radius,
            half_height,
        },
        Shape::Hull(_) | Shape::Mesh(_) | Shape::HeightField(_) => {
            panic!("vertex-sourced shapes have no render counterpart")
        }
        Shape::Plane => panic!("plane shapes have no render counterpart"),
    }
}

fn indexed_color(index: usize) -> Color {
    const PALETTE: [[u8; 3]; 8] = [
        [224, 108, 117],
        [97, 175, 239],
        [152, 195, 121],
        [229, 192, 123],
        [198, 120, 221],
        [86, 182, 194],
        [209, 154, 102],
        [92, 143, 253],
    ];
    let [red, green, blue] = PALETTE[index % PALETTE.len()];
    Color::srgb_u8(red, green, blue)
}
