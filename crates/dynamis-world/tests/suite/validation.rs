use super::common::{DT, observed_world, static_config};
use dynamis_model::{
    BodyDesc, CharacterDesc, ColliderDesc, ConstraintDesc, PhysicsConfig, QueryFilter, Shape,
    SoftBodyDesc, SurfaceDesc, SurfaceTable, VehicleDesc, WheelDesc,
};
use dynamis_world::World;
use std::panic::{AssertUnwindSafe, catch_unwind};

const NAN: f32 = f32::NAN;
const INF: f32 = f32::INFINITY;

fn refuses(what: &str, world: &mut World, attempt: impl FnOnce(&mut World)) {
    assert!(
        catch_unwind(AssertUnwindSafe(|| attempt(world))).is_err(),
        "{what} must be refused"
    );
}

fn mesh() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    (
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
    )
}

type Attempt = Box<dyn FnOnce(&mut World)>;
type Case = (&'static str, Attempt);

#[test]
fn a_refused_input_leaves_the_world_whole() {
    let mut world = observed_world(static_config());
    let body = world.spawn(BodyDesc::sphere(0.5));
    let other = world.spawn(BodyDesc::sphere(0.5).position([0.0, 3.0, 0.0]));
    let joint = world.add_constraint(body, other, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
    world.step(DT);
    world.wait();
    let held = world.read_state(body);
    let bodies = world.bodies().len();
    let constraints = world.constraints().len();
    let joint_desc = world.constraint_desc(joint);

    let mut palette = SurfaceDesc::new();
    palette.friction = NAN;
    let surface = [palette];
    let raws: Vec<Case> = vec![
        (
            "a nan body position",
            Box::new(move |world: &mut World| world.set_position(body, [NAN, 0.0, 0.0])),
        ),
        (
            "a non-unit body orientation",
            Box::new(move |world: &mut World| world.set_orientation(body, [NAN, 0.0, 0.0, 1.0])),
        ),
        (
            "an infinite body velocity",
            Box::new(move |world: &mut World| world.set_velocity(body, [INF, 0.0, 0.0])),
        ),
        (
            "a nan body angular velocity",
            Box::new(move |world: &mut World| world.set_angular_velocity(body, [NAN, 0.0, 0.0])),
        ),
        (
            "a nan force",
            Box::new(move |world: &mut World| world.apply_force(body, [NAN, 0.0, 0.0])),
        ),
        (
            "a nan force point",
            Box::new(move |world: &mut World| {
                world.apply_force_at_point(body, [1.0, 0.0, 0.0], [NAN, 0.0, 0.0])
            }),
        ),
        (
            "a nan torque",
            Box::new(move |world: &mut World| world.apply_torque(body, [0.0, NAN, 0.0])),
        ),
        (
            "a nan impulse",
            Box::new(move |world: &mut World| world.apply_impulse(body, [0.0, 0.0, NAN])),
        ),
        (
            "a nan angular impulse",
            Box::new(move |world: &mut World| world.apply_angular_impulse(body, [NAN; 3])),
        ),
        (
            "an infinite mass",
            Box::new(move |world: &mut World| world.set_mass(body, INF)),
        ),
        (
            "a nan center of mass",
            Box::new(move |world: &mut World| world.set_com(body, [NAN; 3])),
        ),
        (
            "a nan inertia tensor",
            Box::new(move |world: &mut World| world.set_inertia(body, [NAN; 6])),
        ),
        (
            "a nan gravity scale",
            Box::new(move |world: &mut World| world.set_gravity_scale(body, NAN)),
        ),
        (
            "a nan sleep threshold",
            Box::new(move |world: &mut World| world.set_sleep_thresholds(body, NAN, 0.0)),
        ),
        (
            "a nan friction",
            Box::new(move |world: &mut World| world.set_friction(body, NAN)),
        ),
        (
            "a zero contact frequency",
            Box::new(move |world: &mut World| world.set_contact_frequency(body, 0.0)),
        ),
        (
            "a nan collider",
            Box::new(move |world: &mut World| {
                let collider = ColliderDesc::new(Shape::sphere(0.5)).restitution(NAN);
                world.set_collider(body, 0, collider);
            }),
        ),
        (
            "a nan collider added",
            Box::new(move |world: &mut World| {
                let collider = ColliderDesc::new(Shape::sphere(0.5)).friction(NAN);
                world.add_collider(body, collider);
            }),
        ),
        (
            "a nan shape",
            Box::new(move |world: &mut World| world.set_shape(body, Shape::Sphere { radius: NAN })),
        ),
        (
            "a nan joint axis",
            Box::new(move |world: &mut World| {
                world.add_constraint(
                    body,
                    other,
                    ConstraintDesc::revolute([0.0; 3], [0.0; 3], [NAN, 0.0, 0.0]),
                );
            }),
        ),
        (
            "a nan joint patch",
            Box::new(move |world: &mut World| {
                world.update_constraint(joint, ConstraintDesc::ball([NAN, 0.0, 0.0], [0.0; 3]));
            }),
        ),
        (
            "a nan mesh vertex",
            Box::new(move |world: &mut World| {
                let (mut vertices, triangles) = mesh();
                vertices[0][1] = NAN;
                world.add_mesh(&vertices, &triangles, None);
            }),
        ),
        (
            "a mesh triangle outside its vertices",
            Box::new(move |world: &mut World| {
                let (vertices, mut triangles) = mesh();
                triangles[0][2] = vertices.len() as u32;
                world.add_mesh(&vertices, &triangles, None);
            }),
        ),
        (
            "a nan surface",
            Box::new(move |world: &mut World| {
                let (vertices, triangles) = mesh();
                let indices = [0u32; 4];
                let table = SurfaceTable::new(&surface, &indices);
                world.add_mesh(&vertices, &triangles, Some(table));
            }),
        ),
        (
            "a nan height field sample",
            Box::new(move |world: &mut World| {
                let heights = [0.0, 0.0, 0.0, NAN];
                world.add_height_field(2, 2, &heights, [1.0, 1.0], None);
            }),
        ),
        (
            "a nan soft particle",
            Box::new(move |world: &mut World| {
                let mut desc = SoftBodyDesc::net(vec![[0.0; 3], [0.5, 0.0, 0.0]], vec![[0, 1]]);
                desc.particles[0] = [NAN, 0.0, 0.0];
                world.add_soft_body(desc);
            }),
        ),
        (
            "a soft body without one inverse mass per particle",
            Box::new(move |world: &mut World| {
                let mut desc = SoftBodyDesc::net(vec![[0.0; 3], [0.5, 0.0, 0.0]], vec![[0, 1]]);
                desc.inverse_masses = vec![1.0, INF];
                world.add_soft_body(desc);
            }),
        ),
        (
            "a nan soft particle position",
            Box::new(move |world: &mut World| {
                let soft = world.add_soft_body(SoftBodyDesc::net(
                    vec![[0.0; 3], [0.5, 0.0, 0.0]],
                    vec![[0, 1]],
                ));
                world.set_soft_particle_position(soft, 0, [NAN, 0.0, 0.0]);
            }),
        ),
        (
            "a nan ray direction",
            Box::new(move |world: &mut World| {
                world.ray_query([0.0; 3], [NAN, 0.0, 0.0], 1.0, &QueryFilter::default());
            }),
        ),
        (
            "a nan query radius",
            Box::new(move |world: &mut World| {
                world.sphere_query([0.0; 3], NAN, &QueryFilter::default());
            }),
        ),
        (
            "a nan query half extent",
            Box::new(move |world: &mut World| {
                world.cuboid_query([0.0; 3], [1.0, NAN, 1.0], &QueryFilter::default());
            }),
        ),
        (
            "a nan sweep direction",
            Box::new(move |world: &mut World| {
                world.sweep_query(
                    &Shape::sphere(0.5),
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0; 3],
                    [NAN, 0.0, 0.0],
                    1.0,
                    &QueryFilter::default(),
                );
            }),
        ),
        (
            "a nan overlap shape",
            Box::new(move |world: &mut World| {
                world.overlap_query(
                    &Shape::Sphere { radius: NAN },
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0; 3],
                    &QueryFilter::default(),
                );
            }),
        ),
        (
            "a nan gravity",
            Box::new(move |world: &mut World| world.set_gravity([0.0, NAN, 0.0])),
        ),
        (
            "an infinite time scale",
            Box::new(move |world: &mut World| world.set_time_scale(INF)),
        ),
        (
            "a nan timestep",
            Box::new(move |world: &mut World| world.step(NAN)),
        ),
        (
            "a nan frame duration",
            Box::new(move |world: &mut World| world.update(NAN, DT, 4)),
        ),
        (
            "an infinite substep duration",
            Box::new(move |world: &mut World| world.update(DT, INF, 4)),
        ),
        (
            "a nan character position",
            Box::new(move |world: &mut World| {
                world.add_character([NAN, 0.0, 0.0], CharacterDesc::default());
            }),
        ),
        (
            "a nan wheel anchor",
            Box::new(move |world: &mut World| {
                let mut wheel = WheelDesc::new([0.0; 3], 0.3);
                wheel.anchor = [NAN, 0.0, 0.0];
                world.add_vehicle(VehicleDesc::new(
                    BodyDesc::cuboid([0.9, 0.3, 1.8]),
                    vec![wheel],
                ));
            }),
        ),
        (
            "an inadmissible config",
            Box::new(move |world: &mut World| {
                world.set_config(PhysicsConfig {
                    relaxation: 0.0,
                    ..PhysicsConfig::default()
                });
            }),
        ),
    ];

    for (what, attempt) in raws {
        refuses(what, &mut world, attempt);
        world.step(DT);
        world.wait();
        let state = world.read_state(body);
        assert_eq!(state.position, held.position, "{what} moved the body");
        assert_eq!(state.velocity, held.velocity, "{what} moved the body");
        assert_eq!(state.com, held.com, "{what} moved the body");
        assert_eq!(
            world.constraint_desc(joint),
            joint_desc,
            "{what} patched the joint"
        );
        assert_eq!(world.bodies().len(), bodies, "{what} leaked a body");
        assert_eq!(
            world.constraints().len(),
            constraints,
            "{what} leaked a constraint"
        );
        assert!(
            world.refusals().is_empty(),
            "{what} refused work the device cannot lose"
        );
    }
}

#[test]
fn a_legal_extreme_still_reaches_the_simulation() {
    let mut world = observed_world(PhysicsConfig {
        gravity: [0.0, 0.0, 0.0],
        damping: 0.0,
        angular_damping: 0.0,
        slop: 0.0,
        contact_margin: 0.0,
        restitution_threshold: 0.0,
        settle_velocity: 0.0,
        ..PhysicsConfig::default()
    });
    let body = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::capsule(0.5, 0.0)))
            .restitution(1.0)
            .contact_frequency(INF)
            .gravity_scale(0.0)
            .com([0.0, 0.0, 0.0]),
    );
    world.set_gravity([0.0, -0.0, 0.0]);
    world.set_time_scale(1.0);
    world.set_sleep_thresholds(body, 0.0, 0.0);
    world.apply_force(body, [0.0; 3]);
    world.step(DT);
    world.wait();
    assert_eq!(world.read_state(body).position, [0.0; 3]);
    assert!(world.refusals().is_empty());
}
