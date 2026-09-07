use bytemuck::Zeroable;
use dynamis_layout::{
    BODY_CCD, BODY_KINEMATIC, BodyCommandRecord, COLLIDER_SENSOR, COMMAND_ANGULAR_IMPULSE,
    COMMAND_CONSTRAINT_ADD, COMMAND_CONSTRAINT_REMOVE, COMMAND_FORCE, COMMAND_FORCE_AT_POINT,
    COMMAND_IMPULSE, COMMAND_PATCH, COMMAND_REMOVE, COMMAND_SLEEP, COMMAND_TORQUE, COMMAND_WAKE,
    CONSTRAINT_BALL, CONSTRAINT_DISABLE_COLLISIONS, CONSTRAINT_DISTANCE, CONSTRAINT_FIXED,
    CONSTRAINT_HAS_LIMIT, CONSTRAINT_HAS_MOTOR, CONSTRAINT_IS_SPRING, CONSTRAINT_PRISMATIC,
    CONSTRAINT_REVOLUTE, ColliderRecord, ConstraintCommandRecord, ConstraintRecord, DispatchArgs,
    FILTER_IGNORE_KINEMATIC, FILTER_IGNORE_SENSORS, FILTER_IGNORE_SLEEPING, FILTER_IGNORE_STATIC,
    PATCH_POSITION, PATCH_VELOCITY, QUERY_BOX, QUERY_RAY, QUERY_SPHERE, QUERY_SWEEP, QueryRecord,
    RigidBodyRecord, SHAPE_BOX, SHAPE_CAPSULE, SHAPE_CYLINDER, SHAPE_HEIGHTFIELD, SHAPE_HULL,
    SHAPE_MESH, SHAPE_SPHERE, SimParamsRecord,
};
use dynamis_model::{BodyDesc, ColliderDesc, ConstraintDesc, PhysicsConfig, QueryFilter, Shape};
use std::panic::catch_unwind;

#[test]
fn rigid_body_record_encodes_desc_fields() {
    let desc = BodyDesc::sphere(0.5)
        .mass(2.0)
        .collision_group(7)
        .collision_mask(3)
        .kinematic(true)
        .ccd(true)
        .restitution(0.4)
        .friction(0.6);
    let record = RigidBodyRecord::build(&desc, 5, 9, [1.0, 2.0, 3.0]);
    assert_eq!(record.inverse_mass, 0.0, "kinematic mass is infinite");
    assert_eq!(record.body_id, 5);
    assert_eq!(record.generation, 9);
    assert_eq!(record.collider_count, 1);
    assert_eq!(record.collision_group, 7);
    assert_eq!(record.collision_mask, 3);
    assert_eq!(record.restitution, 0.4);
    assert_eq!(record.friction, 0.6);
    assert_eq!(record.flags & BODY_KINEMATIC, BODY_KINEMATIC);
    assert_eq!(record.flags & BODY_CCD, BODY_CCD);
    assert_eq!(record.inverse_inertia_body, [1.0, 2.0, 3.0]);

    let dynamic = RigidBodyRecord::build(&BodyDesc::sphere(0.5).mass(2.0), 0, 1, [0.0; 3]);
    assert_eq!(dynamic.inverse_mass, 0.5);
}

#[test]
fn collider_record_encodes_every_shape_kind() {
    let sphere = ColliderRecord::build(
        &ColliderDesc::new(Shape::sphere(0.4))
            .friction(0.3)
            .restitution(0.8),
        0,
    );
    assert_eq!(sphere.kind, SHAPE_SPHERE);
    assert_eq!(sphere.radius, 0.4);
    assert_eq!(sphere.friction, 0.3);
    assert_eq!(sphere.restitution, 0.8);
    assert_eq!(sphere.flags & COLLIDER_SENSOR, 0);

    let box_record = ColliderRecord::build(&ColliderDesc::new(Shape::cuboid([1.0, 2.0, 3.0])), 0);
    assert_eq!(box_record.kind, SHAPE_BOX);
    assert_eq!(box_record.half_extents, [1.0, 2.0, 3.0]);

    let capsule = ColliderRecord::build(&ColliderDesc::new(Shape::capsule(0.3, 1.0)), 0);
    assert_eq!(capsule.kind, SHAPE_CAPSULE);
    assert_eq!(capsule.radius, 0.3);
    assert_eq!(capsule.half_height, 1.0);

    let cylinder = ColliderRecord::build(&ColliderDesc::new(Shape::cylinder(0.5, 2.0)), 0);
    assert_eq!(cylinder.kind, SHAPE_CYLINDER);
    assert_eq!(cylinder.radius, 0.5);
    assert_eq!(cylinder.half_height, 2.0);

    let hull = ColliderRecord::build(
        &ColliderDesc::new(Shape::hull(dynamis_model::ShapeSourceHandle {
            id: 4,
            generation: 2,
        })),
        0,
    );
    assert_eq!(hull.kind, SHAPE_HULL);
    let mesh = ColliderRecord::build(
        &ColliderDesc::new(Shape::mesh(dynamis_model::ShapeSourceHandle {
            id: 4,
            generation: 2,
        })),
        0,
    );
    assert_eq!(mesh.kind, SHAPE_MESH);
    let field = ColliderRecord::build(
        &ColliderDesc::new(Shape::height_field(dynamis_model::ShapeSourceHandle {
            id: 4,
            generation: 2,
        })),
        0,
    );
    assert_eq!(field.kind, SHAPE_HEIGHTFIELD);

    let sensor = ColliderRecord::build(&ColliderDesc::new(Shape::sphere(0.5)).sensor(true), 0);
    assert_eq!(sensor.flags & COLLIDER_SENSOR, COLLIDER_SENSOR);
}

#[test]
fn constraint_record_encodes_kinds_and_options() {
    let ball = ConstraintRecord::build(
        &ConstraintDesc::ball([1.0, 0.0, 0.0], [0.0, 2.0, 0.0]),
        3,
        4,
    );
    assert_eq!(ball.kind, CONSTRAINT_BALL);
    assert_eq!(ball.a, 3);
    assert_eq!(ball.b, 4);
    assert_eq!(ball.anchor_a, [1.0, 0.0, 0.0]);
    assert_eq!(ball.anchor_b, [0.0, 2.0, 0.0]);
    assert_eq!(
        ball.flags & CONSTRAINT_DISABLE_COLLISIONS,
        CONSTRAINT_DISABLE_COLLISIONS
    );

    let distance = ConstraintRecord::build(
        &ConstraintDesc::distance([0.0; 3], [0.0; 3], 2.5).disable_collisions(false),
        0,
        1,
    );
    assert_eq!(distance.kind, CONSTRAINT_DISTANCE);
    assert_eq!(distance.distance, 2.5);
    assert_eq!(distance.flags & CONSTRAINT_DISABLE_COLLISIONS, 0);

    let revolute = ConstraintRecord::build(
        &ConstraintDesc::revolute([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0]).limit(-0.5, 0.5),
        0,
        1,
    );
    assert_eq!(revolute.kind, CONSTRAINT_REVOLUTE);
    assert_eq!(revolute.axis_a, [0.0, 0.0, 1.0]);
    assert_ne!(revolute.flags & CONSTRAINT_HAS_LIMIT, 0);
    assert_eq!(revolute.limit_min, -0.5);
    assert_eq!(revolute.limit_max, 0.5);

    let prismatic = ConstraintRecord::build(
        &ConstraintDesc::prismatic([0.0; 3], [0.0; 3], [1.0, 0.0, 0.0])
            .limit(0.0, 3.0)
            .motor(2.0),
        0,
        1,
    );
    assert_eq!(prismatic.kind, CONSTRAINT_PRISMATIC);
    assert_ne!(prismatic.flags & CONSTRAINT_HAS_LIMIT, 0);
    assert_ne!(prismatic.flags & CONSTRAINT_HAS_MOTOR, 0);
    assert_eq!(prismatic.motor_speed, 2.0);

    let fixed = ConstraintRecord::build(&ConstraintDesc::fixed([0.0; 3], [0.0; 3]), 0, 1);
    assert_eq!(fixed.kind, CONSTRAINT_FIXED);

    let spring = ConstraintRecord::build(
        &ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0).spring(3.0, 0.7),
        0,
        1,
    );
    assert_ne!(spring.flags & CONSTRAINT_IS_SPRING, 0);
    assert_eq!(spring.spring_frequency, 3.0);
    assert_eq!(spring.spring_damping_ratio, 0.7);
}

#[test]
fn sim_params_record_maps_config() {
    let config = PhysicsConfig {
        gravity: [0.0, -9.81, 3.0],
        damping: 0.5,
        angular_damping: 0.25,
        solve_iterations: 7,
        position_iterations: 3,
        relaxation: 0.4,
        slop: 0.01,
        restitution_threshold: 2.0,
        max_velocity: 50.0,
        max_angular_velocity: 100.0,
        broadphase_cell_size: 4.0,
        sleep_velocity: 0.3,
        sleep_angular_velocity: 0.4,
        sleep_time: 0.8,
        wake_velocity: 0.6,
    };
    let record = SimParamsRecord::new(&config, 1.0 / 60.0, 11, 2);
    assert_eq!(record.gravity, [0.0, -9.81, 3.0, 0.0]);
    assert_eq!(record.dt, 1.0 / 60.0);
    assert_eq!(record.damping, 0.5);
    assert_eq!(record.angular_damping, 0.25);
    assert_eq!(record.body_count, 11);
    assert_eq!(record.constraint_count, 2);
    assert_eq!(record.solve_iterations, 7);
    assert_eq!(record.position_iterations, 3);
    assert_eq!(record.relaxation, 0.4);
    assert_eq!(record.slop, 0.01);
    assert_eq!(record.restitution_threshold, 2.0);
    assert_eq!(record.max_velocity, 50.0);
    assert_eq!(record.max_angular_velocity, 100.0);
    assert_eq!(record.grid_cell_size, 4.0);
    assert_eq!(record.sleep_velocity, 0.3);
    assert_eq!(record.sleep_angular_velocity, 0.4);
    assert_eq!(record.sleep_time, 0.8);
    assert_eq!(record.wake_velocity, 0.6);
}

#[test]
fn query_record_encodes_kinds_and_filters() {
    let filter = QueryFilter {
        group: 5,
        mask: 9,
        ignore_sensors: true,
        ignore_sleeping: false,
        ignore_static: true,
        ignore_kinematic: true,
        max_hits: 3,
    };
    let ray = QueryRecord::ray([0.0, 1.0, 2.0], [0.0, 0.0, 1.0], 8.0, &filter);
    assert_eq!(ray.kind, QUERY_RAY);
    assert_eq!(ray.origin, [0.0, 1.0, 2.0]);
    assert_eq!(ray.direction, [0.0, 0.0, 1.0]);
    assert_eq!(ray.extent, 8.0);
    assert_eq!(
        ray.filter_flags,
        FILTER_IGNORE_SENSORS | FILTER_IGNORE_STATIC | FILTER_IGNORE_KINEMATIC
    );
    assert_eq!(ray.group, 5);
    assert_eq!(ray.mask, 9);
    assert_eq!(ray.max_hits, 3);

    let sphere = QueryRecord::sphere([1.0; 3], 0.7, &QueryFilter::default());
    assert_eq!(sphere.kind, QUERY_SPHERE);
    assert_eq!(sphere.shape_kind, SHAPE_SPHERE);
    assert_eq!(sphere.radius, 0.7);
    assert_eq!(sphere.extent, 0.7);

    let box_query = QueryRecord::box_query([0.0; 3], [1.0, 2.0, 3.0], &QueryFilter::default());
    assert_eq!(box_query.kind, QUERY_BOX);
    assert_eq!(box_query.shape_kind, SHAPE_BOX);
    assert_eq!(box_query.half_extents, [1.0, 2.0, 3.0]);

    let sweep = QueryRecord::sweep(
        &Shape::capsule(0.4, 1.0),
        [0.0, 0.0, 0.0, 1.0],
        [0.0; 3],
        [0.0, 1.0, 0.0],
        5.0,
        &QueryFilter::default(),
    );
    assert_eq!(sweep.kind, QUERY_SWEEP);
    assert_eq!(sweep.shape_kind, SHAPE_CAPSULE);
    assert_eq!(sweep.radius, 0.4);
    assert_eq!(sweep.half_height, 1.0);
    assert_eq!(sweep.extent, 5.0);

    let sleep_filter = QueryFilter {
        ignore_sleeping: true,
        ..QueryFilter::default()
    };
    let sleeping = QueryRecord::ray([0.0; 3], [0.0, 0.0, 1.0], 1.0, &sleep_filter);
    assert_eq!(
        sleeping.filter_flags,
        FILTER_IGNORE_SENSORS | FILTER_IGNORE_SLEEPING,
        "default filters skip sensors too"
    );
    assert!(
        catch_unwind(|| {
            QueryRecord::sweep(
                &Shape::mesh(dynamis_model::ShapeSourceHandle {
                    id: 0,
                    generation: 1,
                }),
                [0.0, 0.0, 0.0, 1.0],
                [0.0; 3],
                [0.0, 1.0, 0.0],
                5.0,
                &QueryFilter::default(),
            );
        })
        .is_err()
    );
}

#[test]
fn body_commands_encode_their_payloads() {
    let body = RigidBodyRecord::build(&BodyDesc::sphere(0.5), 1, 1, [0.0; 3]);
    let add = BodyCommandRecord::add(2, body, [ColliderRecord::zeroed(); 4]);
    assert_eq!(add.kind, 0);
    assert_eq!(add.slot, 2);
    assert_eq!(add.aux, 0);

    let remove = BodyCommandRecord::remove(3, 7);
    assert_eq!(remove.kind, COMMAND_REMOVE);
    assert_eq!(remove.slot, 3);
    assert_eq!(remove.extra, 7);

    let patch = BodyCommandRecord::patch(
        1,
        PATCH_POSITION | PATCH_VELOCITY,
        body,
        [ColliderRecord::zeroed(); 4],
        2,
    );
    assert_eq!(patch.kind, COMMAND_PATCH);
    assert_eq!(patch.extra, PATCH_POSITION | PATCH_VELOCITY);
    assert_eq!(patch.aux, 2);

    let force = BodyCommandRecord::force(4, [1.0, 2.0, 3.0]);
    assert_eq!(force.kind, COMMAND_FORCE);
    assert_eq!(force.body.force, [1.0, 2.0, 3.0]);

    let force_at = BodyCommandRecord::force_at_point(4, [0.0, 0.0, 1.0], [5.0, 0.0, 0.0]);
    assert_eq!(force_at.kind, COMMAND_FORCE_AT_POINT);
    assert_eq!(force_at.body.position, [5.0, 0.0, 0.0]);

    let torque = BodyCommandRecord::torque(4, [0.0, 0.0, 1.0]);
    assert_eq!(torque.kind, COMMAND_TORQUE);

    let impulse = BodyCommandRecord::impulse(4, [1.0, 0.0, 0.0]);
    assert_eq!(impulse.kind, COMMAND_IMPULSE);
    assert_eq!(impulse.body.velocity, [1.0, 0.0, 0.0]);

    let impulse_at = BodyCommandRecord::impulse_at_point(4, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]);
    assert_eq!(impulse_at.extra, 1);
    assert_eq!(impulse_at.body.position, [1.0, 0.0, 0.0]);

    let angular = BodyCommandRecord::angular_impulse(4, [0.0, 0.0, 1.0]);
    assert_eq!(angular.kind, COMMAND_ANGULAR_IMPULSE);
    assert_eq!(angular.body.angular_velocity, [0.0, 0.0, 1.0]);

    assert_eq!(BodyCommandRecord::sleep(4).kind, COMMAND_SLEEP);
    assert_eq!(BodyCommandRecord::wake(4).kind, COMMAND_WAKE);
}

#[test]
fn constraint_command_and_dispatch_args_encode() {
    let record = ConstraintRecord::build(&ConstraintDesc::ball([0.0; 3], [0.0; 3]), 1, 2);
    let add = ConstraintCommandRecord::add(3, record);
    assert_eq!(add.kind, COMMAND_CONSTRAINT_ADD);
    assert_eq!(add.slot, 3);
    let remove = ConstraintCommandRecord::remove(3);
    assert_eq!(remove.kind, COMMAND_CONSTRAINT_REMOVE);

    assert_eq!(DispatchArgs::none().x, 0);
    assert_eq!(DispatchArgs::none().y, 1);
    assert_eq!(DispatchArgs::sized(42).x, 42);
}
