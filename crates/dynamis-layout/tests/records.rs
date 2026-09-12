use dynamis_layout::{
    BODY_CCD, BODY_KINEMATIC, BodyDescriptorRecord, BodyEditRecord, BodyEditRunRecord,
    BodyStateRecord, COLLIDER_SENSOR, CONSTRAINT_BALL, CONSTRAINT_DISABLE_COLLISIONS,
    CONSTRAINT_DISTANCE, CONSTRAINT_FIXED, CONSTRAINT_GEAR, CONSTRAINT_HAS_BREAK,
    CONSTRAINT_HAS_LIMIT, CONSTRAINT_HAS_MOTOR, CONSTRAINT_HAS_SWING, CONSTRAINT_IS_SPRING,
    CONSTRAINT_PRISMATIC, CONSTRAINT_PULLEY, CONSTRAINT_REVOLUTE, ColliderRecord,
    ConstraintDescriptorRecord, EDIT_ANGULAR_IMPULSE, EDIT_FORCE, EDIT_FORCE_AT_POINT,
    EDIT_IMPULSE, EDIT_IMPULSE_AT_POINT, EDIT_PATCH, EDIT_SLEEP, EDIT_TORQUE, EDIT_WAKE,
    FILTER_IGNORE_KINEMATIC, FILTER_IGNORE_SENSORS, FILTER_IGNORE_SLEEPING, FILTER_IGNORE_STATIC,
    OVERRIDE_SLEEP_ANGULAR, OVERRIDE_SLEEP_LINEAR, PATCH_POSITION, PATCH_VELOCITY, QUERY_CUBOID,
    QUERY_RAY, QUERY_SPHERE, QUERY_SWEEP, QueryRecord, RowMoveRecord, RowStreams, SHAPE_CAPSULE,
    SHAPE_CUBOID, SHAPE_CYLINDER, SHAPE_HEIGHTFIELD, SHAPE_HULL, SHAPE_MESH, SHAPE_PLANE,
    SHAPE_SPHERE, StepParamsRecord,
};
use dynamis_model::{
    BodyDesc, ColliderDesc, ConstraintDesc, MassProperties, PhysicsConfig, QueryFilter, Shape,
};
use std::panic::catch_unwind;

#[test]
fn body_descriptor_encodes_the_host_owned_half() {
    let desc = BodyDesc::sphere(0.5)
        .mass(2.0)
        .collision_group(7)
        .collision_mask(3)
        .kinematic(true)
        .ccd(true)
        .sleep_thresholds(0.1, 0.2);
    let record = BodyDescriptorRecord::build(
        &desc,
        MassProperties {
            com: [0.5, 0.0, 0.0],
            inverse_inertia: [1.0, 0.0, 0.0, 2.0, 0.0, 3.0],
        },
        &PhysicsConfig::default(),
    );
    assert_eq!(record.inverse_mass, 0.0, "kinematic mass is infinite");
    assert_eq!(record.collision_group, 7);
    assert_eq!(record.collision_mask, 3);
    assert_eq!(record.flags & BODY_KINEMATIC, BODY_KINEMATIC);
    assert_eq!(record.flags & BODY_CCD, BODY_CCD);
    assert_eq!(
        record.flags & (OVERRIDE_SLEEP_LINEAR | OVERRIDE_SLEEP_ANGULAR),
        OVERRIDE_SLEEP_LINEAR | OVERRIDE_SLEEP_ANGULAR
    );
    assert_eq!(record.sleep_velocity, 0.1);
    assert_eq!(record.sleep_angular_velocity, 0.2);
    assert_eq!(record.com, [0.5, 0.0, 0.0]);
    assert_eq!(record.inverse_inertia, [1.0, 0.0, 0.0, 2.0, 0.0, 3.0]);

    let dynamic = BodyDescriptorRecord::build(
        &BodyDesc::sphere(0.5).mass(2.0),
        MassProperties::zeroed(),
        &PhysicsConfig::default(),
    );
    assert_eq!(dynamic.inverse_mass, 0.5);
    assert_eq!(dynamic.flags & OVERRIDE_SLEEP_LINEAR, 0);
}

#[test]
fn body_state_starts_from_the_desc() {
    let desc = BodyDesc::sphere(0.5)
        .position([1.0, 2.0, 3.0])
        .velocity([0.0, 1.0, 0.0])
        .angular_velocity([1.0, 0.0, 0.0]);
    let state = BodyStateRecord::initial(&desc, 5, 9);
    assert_eq!(state.position, [1.0, 2.0, 3.0]);
    assert_eq!(state.prev_position, [1.0, 2.0, 3.0]);
    assert_eq!(state.velocity, [0.0, 1.0, 0.0]);
    assert_eq!(state.angular_velocity, [1.0, 0.0, 0.0]);
    assert_eq!(state.body_id, 5);
    assert_eq!(state.generation, 9);
    assert_eq!(state.sleeping, 0);
    assert_eq!(state.force, [0.0; 3]);
    assert_eq!(state.torque, [0.0; 3]);
}

#[test]
fn collider_record_encodes_every_shape_kind() {
    let sphere = ColliderRecord::build(
        &ColliderDesc::new(Shape::sphere(0.4))
            .friction(0.3)
            .restitution(0.8),
        0,
        0,
    );
    assert_eq!(sphere.kind, SHAPE_SPHERE);
    assert_eq!(sphere.radius, 0.4);
    assert_eq!(sphere.friction, 0.3);
    assert_eq!(sphere.restitution, 0.8);
    assert_eq!(sphere.flags & COLLIDER_SENSOR, 0);

    let box_record =
        ColliderRecord::build(&ColliderDesc::new(Shape::cuboid([1.0, 2.0, 3.0])), 0, 0);
    assert_eq!(box_record.kind, SHAPE_CUBOID);
    assert_eq!(box_record.half_extents, [1.0, 2.0, 3.0]);

    let capsule = ColliderRecord::build(&ColliderDesc::new(Shape::capsule(0.3, 1.0)), 0, 0);
    assert_eq!(capsule.kind, SHAPE_CAPSULE);
    assert_eq!(capsule.radius, 0.3);
    assert_eq!(capsule.half_height, 1.0);

    let cylinder = ColliderRecord::build(&ColliderDesc::new(Shape::cylinder(0.5, 2.0)), 0, 0);
    assert_eq!(cylinder.kind, SHAPE_CYLINDER);
    assert_eq!(cylinder.radius, 0.5);
    assert_eq!(cylinder.half_height, 2.0);

    let hull = ColliderRecord::build(
        &ColliderDesc::new(Shape::hull(dynamis_model::ShapeSourceHandle {
            id: 4,
            generation: 2,
        })),
        0,
        0,
    );
    assert_eq!(hull.kind, SHAPE_HULL);
    let mesh = ColliderRecord::build(
        &ColliderDesc::new(Shape::mesh(dynamis_model::ShapeSourceHandle {
            id: 4,
            generation: 2,
        })),
        0,
        0,
    );
    assert_eq!(mesh.kind, SHAPE_MESH);
    let field = ColliderRecord::build(
        &ColliderDesc::new(Shape::height_field(dynamis_model::ShapeSourceHandle {
            id: 4,
            generation: 2,
        })),
        0,
        0,
    );
    assert_eq!(field.kind, SHAPE_HEIGHTFIELD);

    let sensor = ColliderRecord::build(&ColliderDesc::new(Shape::sphere(0.5)).sensor(true), 0, 0);
    assert_eq!(sensor.flags & COLLIDER_SENSOR, COLLIDER_SENSOR);

    let filtered = ColliderRecord::build(
        &ColliderDesc::new(Shape::sphere(0.5))
            .collision_group(0x8)
            .collision_mask(0x4)
            .rolling_friction(0.3)
            .spin_friction(0.6),
        0,
        0,
    );
    assert_eq!(filtered.collision_group, 0x8);
    assert_eq!(filtered.collision_mask, 0x4);
    assert_eq!(filtered.rolling_friction, 0.3);
    assert_eq!(filtered.spin_friction, 0.6);
    let inherited = ColliderRecord::build(&ColliderDesc::new(Shape::sphere(0.5)), 0, 0);
    assert_eq!(inherited.collision_group, u32::MAX);
    assert_eq!(inherited.collision_mask, u32::MAX);
}

#[test]
fn constraint_record_encodes_kinds_and_options() {
    let ball = ConstraintDescriptorRecord::build(
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

    let distance = ConstraintDescriptorRecord::build(
        &ConstraintDesc::distance([0.0; 3], [0.0; 3], 2.5).disable_collisions(false),
        0,
        1,
    );
    assert_eq!(distance.kind, CONSTRAINT_DISTANCE);
    assert_eq!(distance.distance, 2.5);
    assert_eq!(distance.flags & CONSTRAINT_DISABLE_COLLISIONS, 0);

    let revolute = ConstraintDescriptorRecord::build(
        &ConstraintDesc::revolute([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0]).limit(-0.5, 0.5),
        0,
        1,
    );
    assert_eq!(revolute.kind, CONSTRAINT_REVOLUTE);
    assert_eq!(revolute.axis_a, [0.0, 0.0, 1.0]);
    assert_ne!(revolute.flags & CONSTRAINT_HAS_LIMIT, 0);
    assert_eq!(revolute.limit_min, -0.5);
    assert_eq!(revolute.limit_max, 0.5);

    let prismatic = ConstraintDescriptorRecord::build(
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

    let fixed = ConstraintDescriptorRecord::build(&ConstraintDesc::fixed([0.0; 3], [0.0; 3]), 0, 1);
    assert_eq!(fixed.kind, CONSTRAINT_FIXED);

    let spring = ConstraintDescriptorRecord::build(
        &ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0).spring(3.0, 0.7),
        0,
        1,
    );
    assert_ne!(spring.flags & CONSTRAINT_IS_SPRING, 0);
    assert_eq!(spring.spring_frequency, 3.0);
    assert_eq!(spring.spring_damping_ratio, 0.7);
}

#[test]
fn step_params_record_maps_config() {
    let config = PhysicsConfig {
        gravity: [0.0, -9.81, 3.0],
        damping: 0.5,
        angular_damping: 0.25,
        solve_iterations: 7,
        position_iterations: 5,
        relaxation: 0.4,
        slop: 0.01,
        contact_margin: 0.03,
        restitution_threshold: 2.0,
        max_velocity: 50.0,
        max_angular_velocity: 100.0,
        broadphase_cell_size: 4.0,
        sleep_velocity: 0.3,
        sleep_angular_velocity: 0.4,
        sleep_time: 0.8,
        wake_velocity: 0.6,
        friction_combine: dynamis_model::MaterialCombine::Min,
        restitution_combine: dynamis_model::MaterialCombine::Average,
    };
    let record = StepParamsRecord::new(
        &config,
        1.0 / 60.0,
        dynamis_layout::FrameCounts {
            dynamic_bodies: 9,
            bodies: 11,
            colliders: 13,
            constraints: 2,
        },
        RowStreams {
            edit_runs: 5,
            body_moves: 4,
            constraint_moves: 1,
        },
        3,
    );
    assert_eq!(record.gravity, [0.0, -9.81, 3.0, 0.0]);
    assert_eq!(record.dt, 1.0 / 60.0);
    assert_eq!(record.damping, 0.5);
    assert_eq!(record.angular_damping, 0.25);
    assert_eq!(record.dynamic_count, 9);
    assert_eq!(record.collider_count, 13);
    assert_eq!(record.edit_run_count, 5);
    assert_eq!(record.body_move_count, 4);
    assert_eq!(record.constraint_move_count, 1);
    assert_eq!(record.event_slot, 3);
    assert_eq!(record.body_count, 11);
    assert_eq!(record.constraint_count, 2);
    assert_eq!(record.solve_iterations, 7);
    assert_eq!(record.position_iterations, 5);
    assert_eq!(record.relaxation, 0.4);
    assert_eq!(record.slop, 0.01);
    assert_eq!(record.contact_margin, 0.03);
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
        exclude: None,
        include: None,
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

    let cuboid_query = QueryRecord::cuboid([0.0; 3], [1.0, 2.0, 3.0], &QueryFilter::default());
    assert_eq!(cuboid_query.kind, QUERY_CUBOID);
    assert_eq!(cuboid_query.shape_kind, SHAPE_CUBOID);
    assert_eq!(cuboid_query.half_extents, [1.0, 2.0, 3.0]);

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
fn body_edits_encode_their_payloads() {
    let state = BodyStateRecord::initial(&BodyDesc::sphere(0.5), 1, 1);
    let patch = BodyEditRecord::patch(PATCH_POSITION | PATCH_VELOCITY, state);
    assert_eq!(patch.kind, EDIT_PATCH);
    assert_eq!(patch.mask, PATCH_POSITION | PATCH_VELOCITY);
    assert_eq!(patch.state.position, state.position);
    assert_eq!(patch.state.velocity, state.velocity);

    let force = BodyEditRecord::force([1.0, 2.0, 3.0]);
    assert_eq!(force.kind, EDIT_FORCE);
    assert_eq!(force.state.force, [1.0, 2.0, 3.0]);

    let force_at = BodyEditRecord::force_at_point([0.0, 0.0, 1.0], [5.0, 0.0, 0.0]);
    assert_eq!(force_at.kind, EDIT_FORCE_AT_POINT);
    assert_eq!(force_at.state.force, [0.0, 0.0, 1.0]);
    assert_eq!(force_at.state.position, [5.0, 0.0, 0.0]);

    let torque = BodyEditRecord::torque([0.0, 0.0, 1.0]);
    assert_eq!(torque.kind, EDIT_TORQUE);
    assert_eq!(torque.state.torque, [0.0, 0.0, 1.0]);

    let impulse = BodyEditRecord::impulse([1.0, 0.0, 0.0]);
    assert_eq!(impulse.kind, EDIT_IMPULSE);
    assert_eq!(impulse.state.velocity, [1.0, 0.0, 0.0]);

    let impulse_at = BodyEditRecord::impulse_at_point([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]);
    assert_eq!(impulse_at.kind, EDIT_IMPULSE_AT_POINT);
    assert_eq!(impulse_at.state.velocity, [0.0, 1.0, 0.0]);
    assert_eq!(impulse_at.state.position, [1.0, 0.0, 0.0]);

    let angular = BodyEditRecord::angular_impulse([0.0, 0.0, 1.0]);
    assert_eq!(angular.kind, EDIT_ANGULAR_IMPULSE);
    assert_eq!(angular.state.angular_velocity, [0.0, 0.0, 1.0]);

    assert_eq!(BodyEditRecord::sleep().kind, EDIT_SLEEP);
    assert_eq!(BodyEditRecord::wake().kind, EDIT_WAKE);
}

#[test]
fn row_moves_encode_their_source() {
    let moved = RowMoveRecord::source(7, 3);
    assert_eq!((moved.row, moved.source, moved.fresh), (7, 3, u32::MAX));
    let fresh = RowMoveRecord::fresh(7, 2);
    assert_eq!((fresh.row, fresh.source, fresh.fresh), (7, u32::MAX, 2));
    let cleared = RowMoveRecord::clear(7);
    assert_eq!(
        (cleared.row, cleared.source, cleared.fresh),
        (7, u32::MAX, u32::MAX)
    );
}

#[test]
fn body_edit_runs_address_their_rows() {
    let run = BodyEditRunRecord::new(7, 3, 2);
    assert_eq!(run.row, 7);
    assert_eq!(run.first, 3);
    assert_eq!(run.len, 2);
}

#[test]
fn constraint_command_encodes() {
    let record = ConstraintDescriptorRecord::build(&ConstraintDesc::ball([0.0; 3], [0.0; 3]), 1, 2);
    assert_eq!(record.kind, CONSTRAINT_BALL);
    assert_eq!(record.a, 1);
    assert_eq!(record.b, 2);
}

#[test]
fn collider_record_applies_scale_and_plane_kind() {
    let scaled = ColliderRecord::build(
        &ColliderDesc::new(Shape::cuboid([1.0, 2.0, 3.0])).scale([2.0, 1.0, 0.5]),
        0,
        0,
    );
    assert_eq!(scaled.half_extents, [2.0, 2.0, 1.5]);
    assert_eq!(scaled.scale, [1.0, 1.0, 1.0]);
    let non_uniform = ColliderRecord::build(
        &ColliderDesc::new(Shape::sphere(0.5)).scale([2.0, 1.0, 0.5]),
        0,
        0,
    );
    assert_eq!(non_uniform.radius, 0.5);
    assert_eq!(non_uniform.scale, [2.0, 1.0, 0.5]);
    let uniform = ColliderRecord::build(
        &ColliderDesc::new(Shape::sphere(0.5)).scale([2.0, 2.0, 2.0]),
        0,
        0,
    );
    assert_eq!(uniform.radius, 1.0);
    assert_eq!(uniform.scale, [1.0, 1.0, 1.0]);
    let plane = ColliderRecord::build(&ColliderDesc::new(Shape::plane()), 0, 0);
    assert_eq!(plane.kind, SHAPE_PLANE);
}

#[test]
fn constraint_record_encodes_swing_break_gear_pulley() {
    let swinging = ConstraintDescriptorRecord::build(
        &ConstraintDesc::ball([0.0; 3], [0.0; 3])
            .limit(-0.5, 0.5)
            .swing(0.7, 0.9),
        0,
        1,
    );
    assert_ne!(swinging.flags & CONSTRAINT_HAS_LIMIT, 0);
    assert_ne!(swinging.flags & CONSTRAINT_HAS_SWING, 0);
    assert_eq!(swinging.swing_a, 0.7);
    assert_eq!(swinging.swing_b, 0.9);

    let motor = ConstraintDescriptorRecord::build(
        &ConstraintDesc::revolute([0.0; 3], [0.0; 3], [0.0, 1.0, 0.0])
            .motor(3.0)
            .motor_force(40.0)
            .break_threshold(100.0, 5.0),
        0,
        1,
    );
    assert_eq!(motor.motor_speed, 3.0);
    assert_eq!(motor.motor_max_force, 40.0);
    assert_ne!(motor.flags & CONSTRAINT_HAS_BREAK, 0);
    assert_eq!(motor.break_force, 100.0);
    assert_eq!(motor.break_torque, 5.0);

    let gear = ConstraintDescriptorRecord::build(
        &ConstraintDesc::gear([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 2.5),
        0,
        1,
    );
    assert_eq!(gear.kind, CONSTRAINT_GEAR);
    assert_eq!(gear.axis_a, [0.0, 1.0, 0.0]);
    assert_eq!(gear.axis_b, [0.0, 0.0, 1.0]);
    assert_eq!(gear.gear_ratio, 2.5);

    let pulley = ConstraintDescriptorRecord::build(
        &ConstraintDesc::pulley(
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            4.0,
        ),
        0,
        1,
    );
    assert_eq!(pulley.kind, CONSTRAINT_PULLEY);
    assert_eq!(pulley.distance, 4.0);
    assert_eq!(pulley.pulley_fixed_a, [0.0, 2.0, 0.0]);
    assert_eq!(pulley.pulley_fixed_b, [3.0, 2.0, 0.0]);
    let dual_axis = ConstraintDescriptorRecord::build(
        &ConstraintDesc::revolute([0.0; 3], [0.0; 3], [0.0, 1.0, 0.0]).axis_b([0.0, 0.0, 1.0]),
        0,
        1,
    );
    assert_eq!(dual_axis.axis_b, [0.0, 0.0, 1.0]);
}
