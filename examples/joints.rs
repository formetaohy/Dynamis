use dynamis_example_common as common;
use bevy::prelude::*;
use common::{
    camera::orbit_camera,
    physics::{Dynamics, PhysicsBody, advance_physics, sync_visuals},
    scene::{indexed_color, primitive_mesh, setup_scene, standard_material},
};
use dynamis::{BodyDesc, BodyHandle, ConstraintDesc, Shape, Simulation};


#[derive(Component)]
struct Hud;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "dynamis joints".into(),
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
    let mut dynamics = Dynamics::with_capacity(64);
    setup_scene(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut dynamics.simulation,
    );
    spawn_chain(
        &mut commands,
        &mut dynamics.simulation,
        &mut meshes,
        &mut materials,
    );
    spawn_pendulum(
        &mut commands,
        &mut dynamics.simulation,
        &mut meshes,
        &mut materials,
    );
    spawn_motor(
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

fn spawn_visual(
    commands: &mut Commands,
    simulation: &mut Simulation,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    desc: BodyDesc,
    shape: Shape,
    color: Color,
) -> BodyHandle {
    let position = desc.position;
    let handle = simulation.spawn(desc);
    commands.spawn((
        Mesh3d(primitive_mesh(&shape, meshes)),
        MeshMaterial3d(standard_material(materials, color)),
        Transform::from_xyz(position[0], position[1], position[2]),
        PhysicsBody(handle),
    ));
    handle
}

fn spawn_chain(
    commands: &mut Commands,
    simulation: &mut Simulation,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let radius = 0.34;
    let spacing = 1.1;
    let anchor = spawn_visual(
        commands,
        simulation,
        meshes,
        materials,
        BodyDesc::static_sphere(0.42).position([0.0, 13.0, 0.0]),
        Shape::sphere(0.42),
        Color::srgb(0.72, 0.74, 0.78),
    );
    let mut previous = anchor;
    for index in 0..9 {
        let x = 0.4 + index as f32 * 0.3;
        let y = 13.0 - (index + 1) as f32 * spacing;
        let handle = spawn_visual(
            commands,
            simulation,
            meshes,
            materials,
            BodyDesc::sphere(radius).position([x, y, 0.0]).friction(0.4),
            Shape::sphere(radius),
            indexed_color(index),
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
    commands: &mut Commands,
    simulation: &mut Simulation,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let pivot = spawn_visual(
        commands,
        simulation,
        meshes,
        materials,
        BodyDesc::static_sphere(0.35).position([-8.5, 10.0, 0.0]),
        Shape::sphere(0.35),
        Color::srgb(0.72, 0.74, 0.78),
    );
    let bob = spawn_visual(
        commands,
        simulation,
        meshes,
        materials,
        BodyDesc::sphere(0.55)
            .position([-8.5, 7.0, 0.0])
            .velocity([2.0, 0.0, 0.0])
            .friction(0.5),
        Shape::sphere(0.55),
        Color::srgb(0.25, 0.85, 0.6),
    );
    simulation.add_constraint(pivot, bob, ConstraintDesc::distance([0.0; 3], [0.0; 3], 3.0));
}

fn spawn_motor(
    commands: &mut Commands,
    simulation: &mut Simulation,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let axle = spawn_visual(
        commands,
        simulation,
        meshes,
        materials,
        BodyDesc::static_sphere(0.35).position([8.5, 10.5, 0.0]),
        Shape::sphere(0.35),
        Color::srgb(0.72, 0.74, 0.78),
    );
    let blade = spawn_visual(
        commands,
        simulation,
        meshes,
        materials,
        BodyDesc::cuboid([1.3, 0.14, 0.14]).position([8.5, 10.5, 0.0]).friction(0.5),
        Shape::cuboid([1.3, 0.14, 0.14]),
        Color::srgb(0.9, 0.3, 0.3),
    );
    simulation.add_constraint(
        axle,
        blade,
        ConstraintDesc::revolute([0.0; 3], [-1.3, 0.0, 0.0], [0.0, 0.0, 1.0]).motor(2.5),
    );
}

fn update_hud(time: Res<Time>, dynamics: Res<Dynamics>, mut hud: Single<&mut Text, With<Hud>>) {
    hud.0 = format!(
        "bodies: {}   constraints: {}   fps: {:.0}\nball chain   distance pendulum   revolute motor\nright-drag: orbit   wheel: zoom",
        dynamics.simulation.count(),
        dynamics.simulation.constraints().len(),
        1.0 / time.delta_secs().max(1e-6),
    );
}
