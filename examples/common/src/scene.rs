use crate::camera::Orbit;
use crate::physics::PhysicsBody;
use bevy::prelude::*;
use dynamis::{BodyDesc, Shape, Simulation};

pub const GROUND_HALF: f32 = 30.0;

#[derive(Component)]
pub struct MaterialHandle(pub Handle<StandardMaterial>);

pub fn setup_scene(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    simulation: &mut Simulation,
) {
    let ground = simulation.spawn(
        BodyDesc::cuboid([GROUND_HALF, 0.5, GROUND_HALF])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(GROUND_HALF * 2.0, 1.0, GROUND_HALF * 2.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.23, 0.26),
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::from_xyz(0.0, -0.5, 0.0),
        PhysicsBody(ground),
    ));
    commands.spawn((
        Camera3d::default(),
        AmbientLight {
            color: Color::WHITE,
            brightness: 90.0,
            ..default()
        },
        Transform::default(),
        Orbit {
            yaw: 0.7,
            pitch: 0.42,
            distance: 42.0,
            target: Vec3::new(0.0, 3.0, 0.0),
        },
    ));
    commands.spawn((
        DirectionalLight {
            color: Color::srgb_u8(255, 240, 214),
            illuminance: 9000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, 0.7, 0.0)),
    ));
}

pub fn indexed_color(index: usize) -> Color {
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

pub fn primitive_mesh(shape: &Shape, meshes: &mut Assets<Mesh>) -> Handle<Mesh> {
    match *shape {
        Shape::Sphere { radius } => meshes.add(Sphere::new(radius)),
        Shape::Box { half_extents } => meshes.add(Cuboid::new(
            half_extents[0] * 2.0,
            half_extents[1] * 2.0,
            half_extents[2] * 2.0,
        )),
        Shape::Capsule {
            radius,
            half_height,
        } => meshes.add(Capsule3d::new(radius, half_height * 2.0)),
        Shape::Cylinder {
            radius,
            half_height,
        } => meshes.add(Cylinder::new(radius, half_height * 2.0)),
        Shape::Hull(_) | Shape::Mesh(_) | Shape::HeightField(_) => {
            panic!("vertex-sourced shapes have no bevy counterpart")
        }
    }
}

pub fn standard_material(
    materials: &mut Assets<StandardMaterial>,
    color: Color,
) -> Handle<StandardMaterial> {
    materials.add(StandardMaterial {
        base_color: color,
        perceptual_roughness: 0.55,
        ..default()
    })
}
