use dynamis_example_common as common;
use bevy::prelude::*;
use common::{
    camera::orbit_camera,
    physics::{Dynamics, PhysicsBody, advance_physics, sync_visuals},
    scene::{indexed_color, primitive_mesh, setup_scene, standard_material},
};
use dynamis::{BodyDesc, Shape, Simulation};


const BODY_COUNT: usize = 128;

#[derive(Component)]
struct Hud;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "dynamis falling shapes".into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup)
        .add_systems(Update, (advance_physics, sync_visuals, orbit_camera, update_hud).chain())
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut dynamics = Dynamics::with_capacity(BODY_COUNT + 4);
    setup_scene(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut dynamics.simulation,
    );
    spawn_collection(
        &mut commands,
        &mut dynamics.simulation,
        &mut meshes,
        &mut materials,
    );
    commands.insert_resource(dynamics);
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: FontSize::Px(18.0),
            ..default()
        },
        TextColor(Color::srgb_u8(230, 232, 235)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
        Hud,
    ));
}

fn spawn_collection(
    commands: &mut Commands,
    simulation: &mut Simulation,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    for index in 0..BODY_COUNT {
        let radius = 0.35 + (index % 5) as f32 * 0.07;
        let (desc, shape, color) = match index % 4 {
            0 => (
                BodyDesc::sphere(radius),
                Shape::sphere(radius),
                indexed_color(index),
            ),
            1 => {
                let half = [radius, radius * 0.9, radius * 1.1];
                (
                    BodyDesc::cuboid(half),
                    Shape::cuboid(half),
                    indexed_color(index),
                )
            }
            2 => {
                let narrow = radius * 0.8;
                (
                    BodyDesc::capsule(narrow, radius),
                    Shape::capsule(narrow, radius),
                    indexed_color(index + 2),
                )
            }
            _ => {
                let narrow = radius * 0.8;
                (
                    BodyDesc::cylinder(narrow, radius),
                    Shape::cylinder(narrow, radius),
                    indexed_color(index + 4),
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
        commands.spawn((
            Mesh3d(primitive_mesh(&shape, meshes)),
            MeshMaterial3d(standard_material(materials, color)),
            Transform::from_xyz(x, y, z),
            PhysicsBody(handle),
        ));
    }
}

fn update_hud(time: Res<Time>, dynamics: Res<Dynamics>, mut hud: Single<&mut Text, With<Hud>>) {
    let sleeping = dynamics
        .simulation
        .bodies()
        .iter()
        .filter(|handle| dynamics.simulation.read_state(**handle).sleeping)
        .count();
    hud.0 = format!(
        "bodies: {}   sleeping: {}   fps: {:.0}\nright-drag: orbit   wheel: zoom",
        dynamics.simulation.count(),
        sleeping,
        1.0 / time.delta_secs().max(1e-6),
    );
}
