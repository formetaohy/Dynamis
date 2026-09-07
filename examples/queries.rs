use dynamis::{BodyDesc, BodyHandle, QueryFilter, QueryHandle, Shape, Simulation};
use dynamis_example_common as common;
use dynamis_example_render::{App, AppContext, Color, MeshId, Transform, Vec3};
use std::collections::HashMap;

const QUERY_SETTLE: f32 = 0.5;
const HIGHLIGHT_SECONDS: f32 = 1.5;

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
    Ray {
        origin: [f32; 3],
        direction: [f32; 3],
    },
    Sphere,
}

struct RayRecord {
    origin: [f32; 3],
    direction: [f32; 3],
    point: [f32; 3],
    normal: [f32; 3],
    hit: bool,
}

struct Example {
    dynamics: common::physics::Dynamics,
    bodies: Vec<(BodyHandle, MeshId)>,
    entities: HashMap<u32, MeshId>,
    interactions: Interactions,
    orbit: common::camera::Orbit,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    App::new(
        "dynamis queries",
        Example {
            dynamics: common::physics::Dynamics::with_capacity(120),
            bodies: Vec::new(),
            entities: HashMap::new(),
            interactions: Interactions {
                pending: Vec::new(),
                highlights: Vec::new(),
                last_ray: None,
            },
            orbit: common::camera::Orbit::new(0.7, 0.42, 42.0, Vec3::new(0.0, 3.0, 0.0)),
        },
    )
    .on_startup(setup)
    .on_update(update)
    .run()
}

fn setup(ctx: &mut AppContext, example: &mut Example) {
    common::scene::setup_scene(ctx, &mut example.dynamics.simulation);
    spawn_towers(
        ctx,
        &mut example.dynamics.simulation,
        &mut example.bodies,
        &mut example.entities,
    );
    spawn_rollers(
        ctx,
        &mut example.dynamics.simulation,
        &mut example.bodies,
        &mut example.entities,
    );
}

fn spawn_towers(
    ctx: &mut AppContext,
    simulation: &mut Simulation,
    bodies: &mut Vec<(BodyHandle, MeshId)>,
    entities: &mut HashMap<u32, MeshId>,
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
            let mesh = spawn_entity(
                ctx,
                Shape::cuboid(half),
                [x, y, z],
                common::scene::indexed_color(tower * 2 + level),
            );
            bodies.push((handle, mesh));
            entities.insert(handle.id, mesh);
        }
    }
}

fn spawn_rollers(
    ctx: &mut AppContext,
    simulation: &mut Simulation,
    bodies: &mut Vec<(BodyHandle, MeshId)>,
    entities: &mut HashMap<u32, MeshId>,
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
        let mesh = spawn_entity(
            ctx,
            Shape::sphere(radius),
            [x, y, z],
            common::scene::indexed_color(index + 4),
        );
        bodies.push((handle, mesh));
        entities.insert(handle.id, mesh);
    }
}

fn spawn_entity(ctx: &mut AppContext, shape: Shape, position: [f32; 3], color: Color) -> MeshId {
    ctx.spawn(
        &common::scene::geometry_from_shape(&shape),
        common::scene::standard_material(color),
        Transform::from_xyz(position[0], position[1], position[2]),
    )
}

fn update(ctx: &mut AppContext, example: &mut Example) {
    common::physics::advance_physics(ctx, &mut example.dynamics);
    common::physics::sync_visuals(ctx, &example.dynamics, &example.bodies);
    common::camera::orbit_camera(ctx, &mut example.orbit);
    handle_input(ctx, example);
    resolve_queries(ctx, example);
    highlight(ctx, example);
    draw_queries(ctx, example);
    ctx.hud = format!(
        "left click: raycast + impulse   right click: sphere query\nactive: {}   fps: {:.0}",
        example.interactions.highlights.len(),
        1.0 / ctx.time.delta_secs().max(1e-6),
    );
}

fn handle_input(ctx: &mut AppContext, example: &mut Example) {
    let Some((cursor_x, cursor_y)) = ctx.input.cursor else {
        return;
    };
    let (origin, direction) = ctx.ray_from_screen(cursor_x, cursor_y);
    let origin = origin.to_array();
    let direction = direction.to_array();
    if ctx.input.left_pressed {
        let handle = example.dynamics.simulation.ray_query(
            origin,
            direction,
            120.0,
            &QueryFilter::default(),
        );
        example.interactions.pending.push(PendingQuery {
            handle,
            submitted: ctx.time.elapsed_secs(),
            kind: QueryKind::Ray { origin, direction },
        });
    }
    if ctx.input.right_pressed {
        let center = [
            origin[0] + direction[0] * 14.0,
            origin[1] + direction[1] * 14.0,
            origin[2] + direction[2] * 14.0,
        ];
        let handle = example.dynamics.simulation.sphere_query(
            center,
            3.0,
            &QueryFilter {
                max_hits: 64,
                ..QueryFilter::default()
            },
        );
        example.interactions.pending.push(PendingQuery {
            handle,
            submitted: ctx.time.elapsed_secs(),
            kind: QueryKind::Sphere,
        });
    }
}

fn resolve_queries(ctx: &mut AppContext, example: &mut Example) {
    let now = ctx.time.elapsed_secs();
    let pending = std::mem::take(&mut example.interactions.pending);
    for pending in pending {
        if now - pending.submitted < QUERY_SETTLE {
            example.interactions.pending.push(pending);
            continue;
        }
        match pending.kind {
            QueryKind::Ray { origin, direction } => {
                let simulation = &mut example.dynamics.simulation;
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
                        example.interactions.last_ray = Some(RayRecord {
                            origin,
                            direction,
                            point: hit.point,
                            normal: hit.normal,
                            hit: true,
                        });
                    }
                    None => {
                        example.interactions.last_ray = Some(RayRecord {
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
                for hit in example.dynamics.simulation.query_hits(pending.handle) {
                    example
                        .interactions
                        .highlights
                        .push((hit.body, now + HIGHLIGHT_SECONDS));
                }
            }
        }
    }
}

fn highlight(ctx: &mut AppContext, example: &mut Example) {
    let now = ctx.time.elapsed_secs();
    let mut index = 0;
    while index < example.interactions.highlights.len() {
        let (body, until) = example.interactions.highlights[index];
        if now >= until {
            set_emissive(ctx, example, body, 0.0);
            example.interactions.highlights.swap_remove(index);
            continue;
        }
        let intensity = ((until - now) / HIGHLIGHT_SECONDS).min(1.0);
        set_emissive(ctx, example, body, intensity);
        index += 1;
    }
}

fn set_emissive(ctx: &mut AppContext, example: &Example, body: BodyHandle, intensity: f32) {
    let Some(&mesh) = example.entities.get(&body.id) else {
        return;
    };
    ctx.mesh_material(mesh).emissive =
        Color::srgb(intensity * 0.4, intensity * 0.9, intensity * 0.25);
}

fn draw_queries(ctx: &mut AppContext, example: &Example) {
    if let Some(ray) = &example.interactions.last_ray {
        ctx.gizmo_ray(
            Vec3::from(ray.origin),
            Vec3::from(ray.direction) * 120.0,
            Color::srgb(1.0, 0.8, 0.2),
        );
        ctx.gizmo_sphere(Vec3::from(ray.point), 0.22, Color::srgb(1.0, 0.35, 0.2));
        if ray.hit {
            ctx.gizmo_ray(
                Vec3::from(ray.point),
                Vec3::from(ray.normal) * 1.5,
                Color::srgb(0.3, 1.0, 0.9),
            );
        }
    }
    for (body, _) in &example.interactions.highlights {
        let Some(&mesh) = example.entities.get(&body.id) else {
            continue;
        };
        let center = ctx.mesh_transform(mesh).translation;
        ctx.gizmo_sphere(center, 0.85, Color::srgb(0.3, 0.9, 0.4));
    }
}
