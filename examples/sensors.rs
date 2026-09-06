use dynamis_example_common as common;
use bevy::prelude::*;
use common::{
    camera::orbit_camera,
    physics::{Dynamics, PhysicsBody, advance_physics, sync_visuals},
    scene::{indexed_color, primitive_mesh, setup_scene, standard_material},
};
use dynamis::{BodyDesc, BodyHandle, ContactEventKind, Shape, Simulation};


const BALL_COUNT: usize = 40;
const GATE_Y: [f32; 3] = [3.0, 6.0, 9.0];

#[derive(Resource)]
struct Gates {
    states: Vec<GateState>,
}

struct GateState {
    body: BodyHandle,
    material: Handle<StandardMaterial>,
    entered: usize,
    flash: f32,
}

#[derive(Component)]
struct Hud;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "dynamis sensors".into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                advance_physics,
                collect_events,
                update_gates,
                sync_visuals,
                orbit_camera,
                update_hud,
            )
                .chain(),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut dynamics = Dynamics::with_capacity(80);
    setup_scene(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut dynamics.simulation,
    );
    let mut gates = Gates { states: Vec::new() };
    spawn_gates(
        &mut commands,
        &mut dynamics.simulation,
        &mut meshes,
        &mut materials,
        &mut gates,
    );
    spawn_balls(
        &mut commands,
        &mut dynamics.simulation,
        &mut meshes,
        &mut materials,
    );
    commands.insert_resource(dynamics);
    commands.insert_resource(gates);
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

fn spawn_gates(
    commands: &mut Commands,
    simulation: &mut Simulation,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    gates: &mut Gates,
) {
    for y in GATE_Y {
        let body = simulation.spawn(
            BodyDesc::cuboid([5.0, 0.25, 5.0])
                .sensor(true)
                .mass(0.0)
                .position([0.0, y, 0.0]),
        );
        let material = materials.add(StandardMaterial {
            base_color: Color::srgba(0.2, 0.8, 0.95, 0.25),
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            ..default()
        });
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(10.0, 0.5, 10.0))),
            MeshMaterial3d(material.clone()),
            Transform::from_xyz(0.0, y, 0.0),
            PhysicsBody(body),
        ));
        gates.states.push(GateState {
            body,
            material,
            entered: 0,
            flash: 0.0,
        });
    }
}

fn spawn_balls(
    commands: &mut Commands,
    simulation: &mut Simulation,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
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
        commands.spawn((
            Mesh3d(primitive_mesh(&Shape::sphere(radius), meshes)),
            MeshMaterial3d(standard_material(materials, indexed_color(index))),
            Transform::from_xyz(x, y, z),
            PhysicsBody(handle),
        ));
    }
}

fn collect_events(mut dynamics: ResMut<Dynamics>, mut gates: ResMut<Gates>) {
    for event in dynamics.simulation.drain_events() {
        if !event.sensor {
            continue;
        }
        for gate in &mut gates.states {
            if (event.first == gate.body || event.second == gate.body)
                && event.kind == ContactEventKind::Begin
            {
                gate.entered += 1;
                gate.flash = 1.0;
            }
        }
    }
}

fn update_gates(
    time: Res<Time>,
    mut gates: ResMut<Gates>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for gate in &mut gates.states {
        gate.flash = (gate.flash - time.delta_secs() * 1.5).max(0.0);
        let light = 0.25 + gate.flash * 0.75;
        if let Some(mut material) = materials.get_mut(&gate.material) {
            material.base_color = Color::srgba(0.2 * light, 0.8 * light, 0.95 * light, 0.3);
        }
    }
}

fn update_hud(time: Res<Time>, gates: Res<Gates>, mut hud: Single<&mut Text, With<Hud>>) {
    let counts = gates
        .states
        .iter()
        .map(|gate| gate.entered.to_string())
        .collect::<Vec<_>>()
        .join(" / ");
    hud.0 = format!(
        "sensor gates entered: {counts}   fps: {:.0}\nright-drag: orbit   wheel: zoom",
        1.0 / time.delta_secs().max(1e-6),
    );
}
