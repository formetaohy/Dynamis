use dynamis_example_common as common;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use common::{
    camera::orbit_camera,
    physics::{Dynamics, PhysicsBody, advance_physics, sync_visuals},
    scene::{MaterialHandle, indexed_color, primitive_mesh, setup_scene, standard_material},
};
use dynamis::{BodyDesc, BodyHandle, QueryFilter, QueryHandle, Shape, Simulation};
use std::collections::HashMap;


const QUERY_SETTLE: f32 = 0.5;
const HIGHLIGHT_SECONDS: f32 = 1.5;

#[derive(Resource, Default)]
struct Interactions {
    pending: Vec<PendingQuery>,
    highlights: Vec<(BodyHandle, f32)>,
    last_ray: Option<RayRecord>,
}

struct PendingQuery {
    handle: QueryHandle,
    submitted: f32,
    kind: QueryKind,
}

#[derive(Clone, Copy)]
enum QueryKind {
    Ray { origin: [f32; 3], direction: [f32; 3] },
    Sphere,
}

struct RayRecord {
    origin: [f32; 3],
    direction: [f32; 3],
    point: [f32; 3],
    normal: [f32; 3],
    hit: bool,
}

#[derive(Resource, Default)]
struct BodyEntities(HashMap<u32, Entity>);

#[derive(Component)]
struct Hud;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "dynamis queries".into(),
                ..default()
            }),
            ..default()
        }))
        .init_resource::<Interactions>()
        .init_resource::<BodyEntities>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                advance_physics,
                sync_visuals,
                orbit_camera,
                handle_input,
                resolve_queries,
                highlight,
                draw_queries,
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
    let mut dynamics = Dynamics::with_capacity(120);
    setup_scene(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut dynamics.simulation,
    );
    let mut entities = BodyEntities::default();
    spawn_towers(
        &mut commands,
        &mut dynamics.simulation,
        &mut meshes,
        &mut materials,
        &mut entities,
    );
    spawn_rollers(
        &mut commands,
        &mut dynamics.simulation,
        &mut meshes,
        &mut materials,
        &mut entities,
    );
    commands.insert_resource(dynamics);
    commands.insert_resource(entities);
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

fn spawn_towers(
    commands: &mut Commands,
    simulation: &mut Simulation,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    entities: &mut BodyEntities,
) {
    for tower in 0..3 {
        for level in 0..6 {
            let half = [0.55, 0.32, 0.55];
            let x = (tower as f32 - 1.0) * 5.0;
            let y = 0.32 + level as f32 * 0.66;
            let z = (level % 2) as f32 * 0.3 - 0.15;
            let handle = simulation.spawn(
                BodyDesc::cuboid(half)
                    .position([x, y, z])
                    .friction(0.7)
                    .restitution(0.05),
            );
            let entity = spawn_entity(
                commands,
                meshes,
                materials,
                handle,
                Shape::cuboid(half),
                [x, y, z],
                indexed_color(tower * 2 + level),
            );
            entities.0.insert(handle.id, entity);
        }
    }
}

fn spawn_rollers(
    commands: &mut Commands,
    simulation: &mut Simulation,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    entities: &mut BodyEntities,
) {
    for index in 0..6 {
        let radius = 0.3 + (index % 3) as f32 * 0.08;
        let x = -3.5 + index as f32 * 1.4;
        let y = 1.5 + (index % 2) as f32 * 0.8;
        let z = 4.5;
        let handle = simulation.spawn(
            BodyDesc::sphere(radius)
                .position([x, y, z])
                .velocity([1.0 + (index % 3) as f32 * 0.5, 0.0, 0.0])
                .restitution(0.3)
                .friction(0.5),
        );
        let entity = spawn_entity(
            commands,
            meshes,
            materials,
            handle,
            Shape::sphere(radius),
            [x, y, z],
            indexed_color(index + 4),
        );
        entities.0.insert(handle.id, entity);
    }
}

fn spawn_entity(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    handle: BodyHandle,
    shape: Shape,
    position: [f32; 3],
    color: Color,
) -> Entity {
    let material = standard_material(materials, color);
    commands
        .spawn((
            Mesh3d(primitive_mesh(&shape, meshes)),
            MeshMaterial3d(material.clone()),
            MaterialHandle(material),
            Transform::from_xyz(position[0], position[1], position[2]),
            PhysicsBody(handle),
        ))
        .id()
}

fn handle_input(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    time: Res<Time>,
    mut interactions: ResMut<Interactions>,
    mut dynamics: ResMut<Dynamics>,
) {
    let Ok(window) = windows.single() else { return };
    let Ok((camera, view)) = cameras.single() else { return };
    let Some(cursor) = window.cursor_position() else { return };
    let Ok(ray) = camera.viewport_to_world(view, cursor) else { return };
    let origin = ray.origin.to_array();
    let direction = ray.direction.as_vec3().to_array();
    if buttons.just_pressed(MouseButton::Left) {
        let handle = dynamics
            .simulation
            .raycast(origin, direction, 120.0, &QueryFilter::default());
        interactions.pending.push(PendingQuery {
            handle,
            submitted: time.elapsed_secs(),
            kind: QueryKind::Ray { origin, direction },
        });
    }
    if buttons.just_pressed(MouseButton::Right) {
        let center = [
            origin[0] + direction[0] * 14.0,
            origin[1] + direction[1] * 14.0,
            origin[2] + direction[2] * 14.0,
        ];
        let handle = dynamics.simulation.sphere_query(
            center,
            3.0,
            &QueryFilter {
                max_hits: 64,
                ..QueryFilter::default()
            },
        );
        interactions.pending.push(PendingQuery {
            handle,
            submitted: time.elapsed_secs(),
            kind: QueryKind::Sphere,
        });
    }
}

fn resolve_queries(
    time: Res<Time>,
    mut dynamics: ResMut<Dynamics>,
    mut interactions: ResMut<Interactions>,
) {
    let now = time.elapsed_secs();
    let pending = std::mem::take(&mut interactions.pending);
    for pending in pending {
        if now - pending.submitted < QUERY_SETTLE {
            interactions.pending.push(pending);
            continue;
        }
        match pending.kind {
            QueryKind::Ray { origin, direction } => {
                let simulation = &mut dynamics.simulation;
                match simulation.query_hit(pending.handle) {
                    Some(hit) => {
                        let state = simulation.read_state(hit.body);
                        let mass = if state.inverse_mass > 0.0 {
                            1.0 / state.inverse_mass
                        } else {
                            0.0
                        };
                        let power = 3.0 + mass * 1.5;
                        let impulse = [
                            direction[0] * power,
                            direction[1] * power,
                            direction[2] * power,
                        ];
                        simulation.apply_impulse_at_point(hit.body, impulse, hit.point);
                        interactions.last_ray = Some(RayRecord {
                            origin,
                            direction,
                            point: hit.point,
                            normal: hit.normal,
                            hit: true,
                        });
                    }
                    None => {
                        interactions.last_ray = Some(RayRecord {
                            origin,
                            direction,
                            point: [
                                origin[0] + direction[0] * 120.0,
                                origin[1] + direction[1] * 120.0,
                                origin[2] + direction[2] * 120.0,
                            ],
                            normal: [0.0; 3],
                            hit: false,
                        });
                    }
                }
            }
            QueryKind::Sphere => {
                for hit in dynamics.simulation.query_hits(pending.handle) {
                    interactions.highlights.push((hit.body, now + HIGHLIGHT_SECONDS));
                }
            }
        }
    }
}

fn highlight(
    time: Res<Time>,
    mut interactions: ResMut<Interactions>,
    bodies: Res<BodyEntities>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    visuals: Query<&MaterialHandle>,
) {
    let now = time.elapsed_secs();
    let mut index = 0;
    while index < interactions.highlights.len() {
        let (body, until) = interactions.highlights[index];
        if now >= until {
            set_emissive(body, 0.0, &bodies, &visuals, &mut materials);
            interactions.highlights.swap_remove(index);
            continue;
        }
        let intensity = ((until - now) / HIGHLIGHT_SECONDS).min(1.0);
        set_emissive(body, intensity, &bodies, &visuals, &mut materials);
        index += 1;
    }
}

fn set_emissive(
    body: BodyHandle,
    intensity: f32,
    bodies: &BodyEntities,
    visuals: &Query<&MaterialHandle>,
    materials: &mut Assets<StandardMaterial>,
) {
    let Some(&entity) = bodies.0.get(&body.id) else { return };
    let Ok(handle) = visuals.get(entity) else { return };
    let Some(mut material) = materials.get_mut(&handle.0) else { return };
    material.emissive = LinearRgba::new(intensity * 0.4, intensity * 0.9, intensity * 0.25, 1.0);
}

fn draw_queries(
    mut gizmos: Gizmos,
    interactions: Res<Interactions>,
    bodies: Res<BodyEntities>,
    transforms: Query<&GlobalTransform, With<PhysicsBody>>,
) {
    if let Some(ray) = &interactions.last_ray {
        gizmos.ray(
            Vec3::from(ray.origin),
            Vec3::from(ray.direction) * 120.0,
            Color::srgb(1.0, 0.8, 0.2),
        );
        gizmos.sphere(
            Isometry3d::from_translation(Vec3::from(ray.point)),
            0.22,
            Color::srgb(1.0, 0.35, 0.2),
        );
        if ray.hit {
            gizmos.ray(
                Vec3::from(ray.point),
                Vec3::from(ray.normal) * 1.5,
                Color::srgb(0.3, 1.0, 0.9),
            );
        }
    }
    for (body, _) in &interactions.highlights {
        let Some(&entity) = bodies.0.get(&body.id) else { continue };
        let Ok(transform) = transforms.get(entity) else { continue };
        gizmos.sphere(
            Isometry3d::from_translation(transform.translation()),
            0.85,
            Color::srgb(0.3, 0.9, 0.4),
        );
    }
}

fn update_hud(
    time: Res<Time>,
    interactions: Res<Interactions>,
    mut hud: Single<&mut Text, With<Hud>>,
) {
    hud.0 = format!(
        "left click: raycast + impulse   right click: sphere query\nactive: {}   fps: {:.0}",
        interactions.highlights.len(),
        1.0 / time.delta_secs().max(1e-6),
    );
}
